"""Phase I: frequency-aware estimation of the initial Gaussian population (methodology Section 3.2, Stages 2-6).

Implementation choices where the methodology leaves the method open (each one is a P1Config field):

  Stage 2  Scale space sigma_1 < ... < sigma_S from sigma_min to sigma_max (in units of the smallest voxel spacing),
           ratio scale_step. Band D_s = V_s - V_{s+1}; local energy E_s = G_rho * D_s^2, rho = rho_factor sigma_{s+1};
           weights w_s = E_s^gamma / sum_j E_j^gamma. k_s = peak frequency of the difference-of-Gaussians band, in
           cycles per unit (exact for the filter):  k_s = sqrt(ln(sigma_{s+1}^2 / sigma_s^2) / (2 pi^2 (sigma_{s+1}^2 -
           sigma_s^2))). lambda = lam_scale / k, lam_scale a calibration constant fitted on synthetic sinusoids
           (1.078: the filter-bank estimate is about 7% short on pure sinusoids).
  Stage 3  J_s = G_rho * (grad V_s grad V_s^T), trace-normalised, aggregated with w_s into J_bar.
  Stage 4  Directional wavelength: k(x; u)^2 = sum_s w_s k_s^2 u^T T_hat_s u, T_s = G_rho * (grad D_s grad D_s^T),
           T_hat_s = T_s / tr T_s; for a plane wave of wave vector k n this gives k |cos(u, n)|.
           Candidate parents: a regular grid of spacing h = parent_spacing x the smallest voxel spacing over the valid
           domain, isotropic std h / (2b), axes = eigenvectors of J_bar (u_1 = strongest gradient direction).
           Violation ratio q_ik(x) = s_ik / (c_s lambda_k(x; u_ik) + eps).
  Stage 5  Omega_i sampled by support_samples points uniform in the Mahalanobis ball of radius support_radius (the
           centre always included); statistic: p90 (default) | p75 | p95 | max | mean | centre (linear interpolation);
           parents with fewer than min_valid valid samples are not subdivided.
  Stage 6  n_ik = max(1, min(n_max, ceil(q_hat_ik / tau_q))); per-parent budget max_children; global budget
           max_gaussians (a GPU-memory safety limit, parents by decreasing violation). Children at
           t_k(j) = ((j + 1/2) / n_ik - 1/2) 2 b s_ik with covariance R diag((s / n)^2) R^T.
           Child amplitude = V(child centre) (2b / sqrt(2 pi))^3 (the children tile a grid of spacing 2 b sigma);
           the pipeline then applies one global least-squares amplitude scale.

Ablation switches (used in the paper's Tables 1, 2 and 4): frequency = isotropic, structure = none,
subdivision = isotropic | structure_energy, target_count > 0 (bisection on tau_q to reach a given count).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .field import Timer
from .volume import blur3d, inside, sample

DEFAULT_LAM_SCALE = 1.0781238227406251


@dataclass
class P1Config:
    # Stages 2-3
    sigma_min: float = 0.5            # in units of the smallest voxel spacing
    sigma_max: float = 16.0
    scale_step: float = math.sqrt(2.0)
    rho_factor: float = 2.0
    gamma: float = 2.0
    eps_w: float = 1e-12
    eps_k: float = 1e-9
    eps_trace: float = 1e-12
    lam_scale: float = DEFAULT_LAM_SCALE
    # Stages 4-6
    parent_spacing: float = 10.0      # in units of the smallest voxel spacing
    frequency: str = "directional"    # directional | isotropic
    structure: str = "tensor"         # tensor | none
    aggregation: str = "p90"          # centre | mean | max | p75 | p90 | p95
    subdivision: str = "anisotropic"  # anisotropic | isotropic | structure_energy
    c_s: float = 1.0
    eps_lam: float = 1e-9
    support_radius: float = 2.0
    support_samples: int = 64
    min_valid: int = 8
    tau_q: float = 0.1
    n_max: int = 4
    max_children: int = 64
    max_gaussians: int = 2_000_000    # safety limit (GPU memory), not a target count
    b: float = 1.0
    target_count: int = 0             # > 0: bisection on tau_q to reach this count (ablations only)
    chunk: int = 131072
    seed: int = 0

    @property
    def percentile(self):
        return {"p75": 75.0, "p90": 90.0, "p95": 95.0}.get(self.aggregation, 90.0)


# ---------------------------------------------------------------------------------------------------------------------
# Small tensor helpers
# ---------------------------------------------------------------------------------------------------------------------
def band_frequency(sa, sb):
    """Peak frequency (cycles / unit) of the difference of Gaussians G_sa - G_sb, sa < sb."""
    return math.sqrt(math.log(sb ** 2 / sa ** 2) / (2 * math.pi ** 2 * (sb ** 2 - sa ** 2)))


def sym6(g):
    """The 6 entries xx, xy, yy, xz, yz, zz of g g^T for a gradient (gx, gy, gz)."""
    gx, gy, gz = g
    return torch.stack([gx * gx, gx * gy, gy * gy, gx * gz, gy * gz, gz * gz])


def mat3(t6):
    """(6, ...) -> (..., 3, 3)."""
    xx, xy, yy, xz, yz, zz = t6
    return torch.stack([torch.stack([xx, xy, xz], -1), torch.stack([xy, yy, yz], -1), torch.stack([xz, yz, zz], -1)], -2)


# ---------------------------------------------------------------------------------------------------------------------
# Stages 2-3
# ---------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def analyse(vol, cfg, timer=None):
    """Returns lam_bar (1, Z, H, W), G (6, ...) the directional frequency tensor (k(x; u)^2 = u^T G u), J (6, ...) the
    aggregated normalised structure tensor, and the band table."""
    timer = timer or Timer()
    dz, dy, dx = vol.spacing()
    unit = min(dz, dy, dx)
    dev = vol.gt.device
    with timer("1_preprocessing"):
        V = torch.from_numpy(vol.analysis_volume).to(dev)
    sig = [cfg.sigma_min * unit]
    while sig[-1] * cfg.scale_step <= cfg.sigma_max * unit + 1e-9:
        sig.append(sig[-1] * cfg.scale_step)
    blur = lambda v, s: blur3d(v, (s / dz, s / dy, s / dx))
    shape = V.shape
    Wsum = torch.zeros((1,) + shape, device=dev)
    Kacc, Gacc, Jacc = torch.zeros_like(Wsum), torch.zeros((6,) + shape, device=dev), torch.zeros((6,) + shape, device=dev)
    bands = []
    with timer("2_scale_space_frequency"):
        cur = blur(V[None], sig[0])
    for sa, sb in zip(sig[:-1], sig[1:]):
        rho = cfg.rho_factor * sb
        with timer("2_scale_space_frequency"):
            nxt = blur(cur, math.sqrt(sb ** 2 - sa ** 2))
            D = (cur - nxt)[0]
            E = blur(D[None] ** 2, rho)
            ks = band_frequency(sa, sb)
            w = E ** cfg.gamma
            Kacc += w * ks
            Wsum += w
        with timer("4_directional_wavelength"):
            T = blur(sym6(torch.gradient(D, spacing=(dz, dy, dx))[::-1]), rho)
            T = T / (T[0] + T[2] + T[5] + 1e-30)
            Gacc += w * ks ** 2 * T
        with timer("3_structure_tensor"):
            J = blur(sym6(torch.gradient(cur[0], spacing=(dz, dy, dx))[::-1]), rho)
            J = J / (J[0] + J[2] + J[5] + cfg.eps_trace)
            Jacc += w * J
        bands.append({"sigma_a": sa, "sigma_b": sb, "k_s": ks, "wavelength": 1 / ks})
        cur = nxt
        del D, E, T, J
    with timer("2_scale_space_frequency"):
        den = Wsum + cfg.eps_w * float(Wsum.max()) + 1e-30
        lam_bar = cfg.lam_scale / (Kacc / den + cfg.eps_k)
    return {"lam_bar": lam_bar, "G": Gacc / den / cfg.lam_scale ** 2, "J": Jacc / den, "bands": bands, "sigmas": sig}


# ---------------------------------------------------------------------------------------------------------------------
# Stages 4-6
# ---------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def parents(vol, fields, cfg):
    """Stage 4 candidates: grid of spacing h over the valid domain, std h / (2b), axes from J_bar (or the volume axes
    when structure = "none"). Also returns the normalised eigenvalues of J_bar (for subdivision = structure_energy)."""
    dev = vol.gt.device
    h = cfg.parent_spacing * min(vol.spacing())
    ax = [torch.arange(h / 2, n * d, h, device=dev) for n, d in ((vol.W, vol.dx), (vol.H, vol.dy), (vol.Z, vol.dz))]
    X, Y, Zc = torch.meshgrid(*ax, indexing="ij")
    mu = torch.stack([X.flatten(), Y.flatten(), Zc.flatten()], 1)
    mu = mu[inside(mu[:, None], vol)[:, 0]]
    J = mat3(sample(fields["J"], mu[:, None], vol)[:, :, 0])
    ev, Q = torch.linalg.eigh(J)
    ev, Q = ev.flip(-1).clamp(min=0), Q.flip(-1)                                 # u_1 = strongest gradient direction
    Q[:, :, 2] *= torch.sign(torch.linalg.det(Q))[:, None]                       # right-handed
    if cfg.structure == "none":
        Q = torch.eye(3, device=dev).expand(len(mu), 3, 3).contiguous()
    s = torch.full((len(mu), 3), h / (2 * cfg.b), device=dev)
    return mu, Q, s, ev / ev.sum(1, keepdim=True).clamp(min=1e-30)


@torch.no_grad()
def violation(mu, R, s, fields, vol, cfg, seed):
    """Stages 4-5: q_hat (N, 3), the per-axis violation ratio aggregated over the support Omega_i."""
    dev = mu.device
    g = torch.Generator(device=dev).manual_seed(seed)
    out = torch.zeros(len(mu), 3, device=dev)
    S = 1 if cfg.aggregation == "centre" else cfg.support_samples
    for a in range(0, len(mu), cfg.chunk):
        m, Rc, sc = mu[a:a + cfg.chunk], R[a:a + cfg.chunk], s[a:a + cfg.chunk]
        n = len(m)
        d = torch.randn(n, S, 3, device=dev, generator=g)
        d = d / d.norm(dim=-1, keepdim=True) * cfg.support_radius * torch.rand(n, S, 1, device=dev, generator=g) ** (1 / 3)
        d[:, 0] = 0.0                                                            # the centre is always a sample
        pts = m[:, None] + torch.einsum("nij,nsj->nsi", Rc * sc[:, None, :], d)
        ok = inside(pts, vol)
        if cfg.frequency == "isotropic":
            lam = sample(fields["lam_bar"], pts, vol)[0][..., None].expand(n, S, 3)
        else:
            Gm = mat3(sample(fields["G"], pts, vol))
            k2 = torch.einsum("nij,nsik,nkj->nsj", Rc, Gm, Rc).clamp(min=0)      # u_k^T G u_k
            lam = 1.0 / (k2.sqrt() + cfg.eps_k)
        q = sc[:, None, :] / (cfg.c_s * lam + cfg.eps_lam)
        if cfg.aggregation == "centre":
            qh = q[:, 0]
            enough = ok[:, 0]
        else:
            q = torch.where(ok[..., None], q, torch.full_like(q, float("nan")))
            if cfg.aggregation == "mean":
                qh = torch.nanmean(q, 1)
            elif cfg.aggregation == "max":
                qh = torch.where(torch.isnan(q), torch.full_like(q, -1.0), q).max(1).values
            else:
                qh = torch.nanquantile(q, cfg.percentile / 100.0, dim=1, interpolation="linear")
            enough = ok.sum(1) >= cfg.min_valid
        out[a:a + n] = torch.where(enough[:, None], torch.nan_to_num(qh, nan=0.0), torch.zeros_like(qh))
    return out


def _per_parent_budget(n, max_children):
    while True:
        over = n.prod(1) > max_children
        if not over.any():
            return n
        ax = torch.argmax(torch.where(over[:, None], n, torch.zeros_like(n)), 1)
        n[over, ax[over]] -= 1


def _global_budget(n, priority, max_gaussians, n_current):
    """Keeps the subdivisions of the parents with the largest priority while the total count stays within the limit."""
    N = n.prod(1)
    order = torch.argsort(priority, descending=True)
    allowed = torch.zeros_like(N, dtype=torch.bool)
    allowed[order] = torch.cumsum(N[order] - 1, 0) <= max_gaussians - n_current
    n[~allowed] = 1
    return n


def counts(q, cfg, tau, n_current, max_gaussians=None):
    """Stage 6 subdivision counts n (N, 3) for threshold tau."""
    limit = cfg.max_gaussians if max_gaussians is None else max_gaussians
    n = torch.clamp(torch.ceil(q / tau), min=1, max=cfg.n_max).long()
    if cfg.subdivision == "isotropic":
        n = n.max(1, keepdim=True).values.expand(-1, 3).contiguous()
        while True:                                                              # keep isotropic under the budget
            over = n.prod(1) > cfg.max_children
            if not over.any():
                break
            n[over] -= 1
        return _global_budget(n, q.max(1).values, limit, n_current)
    priority = n.max(1).values.float()                                           # before the per-parent budget
    n = _per_parent_budget(n, cfg.max_children)
    return _global_budget(n, priority, limit, n_current)


@torch.no_grad()
def children(mu, R, s, n, b):
    """Stage 6: child centres on the grid t_k(j) = ((j + 0.5) / n - 0.5) 2 b s_k, scales s / n, the parent's frame.
    Returns (parent index, centres, scales, offsets t / s)."""
    N = n.prod(1)
    parent = torch.repeat_interleave(torch.arange(len(mu), device=mu.device), N)
    local = torch.arange(len(parent), device=mu.device) - torch.repeat_interleave(torch.cumsum(N, 0) - N, N)
    k = n[parent].float()
    k1, k2 = n[parent, 1], n[parent, 2]
    j = torch.stack([local // (k1 * k2), (local // k2) % k1, local % k2], 1).float()
    sp = s[parent]
    t = ((j + 0.5) / k - 0.5) * 2 * b * sp
    return parent, mu[parent] + torch.einsum("nij,nj->ni", R[parent], t), sp / k, t / sp


@torch.no_grad()
def run(vol, cfg, fields=None, timer=None):
    """Phase I, cycle 0. Returns (init: xyz, scales, R, amp, parent, n; stats with stage timings; analysis fields)."""
    timer = timer or Timer()
    if fields is None:
        fields = analyse(vol, cfg, timer)
    with timer("4_gaussian_matching"):
        mu, R, s, dn = parents(vol, fields, cfg)
    with timer("5_support_aggregation"):
        q = dn.sqrt() if cfg.subdivision == "structure_energy" else violation(mu, R, s, fields, vol, cfg, cfg.seed)
    with timer("6_subdivision"):
        tau = cfg.tau_q
        if cfg.target_count > 0:                                                 # bisection on log tau
            lo, hi = math.log(1e-4), math.log(1e3)
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                m = int(counts(q, cfg, math.exp(mid), len(mu)).prod(1).sum())
                if m > cfg.target_count:
                    lo = mid
                else:
                    hi = mid
            tau = math.exp(hi)
        n = counts(q, cfg, tau, len(mu))
        parent, c_mu, c_s, _ = children(mu, R, s, n, cfg.b)
        V = torch.from_numpy(vol.analysis_volume).to(vol.gt.device)
        amp = sample(V[None], c_mu[:, None], vol)[0, :, 0] * (2 * cfg.b / math.sqrt(2 * math.pi)) ** 3
    stats = {"parents": len(mu), "gaussians": int(len(c_mu)), "tau": tau, "subdivided_parents": int((n.prod(1) > 1).sum()),
             "n_hist_axis": [torch.bincount(n[:, k], minlength=cfg.n_max + 1)[1:].tolist() for k in range(3)],
             "q_hat_percentiles": {str(p): torch.quantile(q.max(1).values[:1_000_000], p / 100).item() for p in (10, 50, 90, 99)},
             "timing": dict(timer.t), "phase1_seconds": sum(timer.t.values())}
    return {"xyz": c_mu, "scales": c_s, "R": R[parent], "amp": amp, "parent": parent, "n": n}, stats, fields
