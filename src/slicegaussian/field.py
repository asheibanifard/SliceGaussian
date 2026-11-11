"""The additive Gaussian field (methodology Section 3.1) and its evaluators.

    V_hat(x) = sum_i alpha_i exp(-1/2 (x - mu_i)^T Sigma_i^-1 (x - mu_i)),   Sigma_i = R_i diag(s_i^2) R_i^T,

alpha_i a peak amplitude (no normalisation, no opacity, no alpha compositing).

Evaluation at the voxel centres: on the plane z = z_k a 3D Gaussian factorises exactly into a 1D Gaussian in z times a
2D Gaussian in the plane, so rendering every plane z_k evaluates V_hat at every voxel centre. The fused CUDA kernel
(cuda/fieldcut.py) does the cut, the culling, the rasterisation and the analytic backward pass. Each Gaussian is
truncated at RADIUS_SIGMA standard deviations and plane contributions below CULL_EPS are skipped; the truncation error
is checked against reference_volume() in tests/test_math.py.

Parameterisation: centre mu (physical units), log-scales (s = exp), unit quaternion (normalised on use, so R is always a
rotation and Sigma is symmetric positive-definite), amplitude alpha (unconstrained).
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

RADIUS_SIGMA = 3.0       # truncation of each Gaussian, in standard deviations
CULL_EPS = 1e-5          # plane contributions with |alpha exp(-dz^2 / 2c)| below this are skipped


# ---------------------------------------------------------------------------------------------------------------------
# Rotations and covariances
# ---------------------------------------------------------------------------------------------------------------------
def quat_to_rot(q):
    """Quaternions (w, x, y, z) (N, 4), normalised on use -> rotation matrices (N, 3, 3)."""
    w, x, y, z = F.normalize(q, dim=1).unbind(1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).view(-1, 3, 3)


def rot_to_quat(R):
    """Rotation matrices (N, 3, 3) -> unit quaternions (w, x, y, z) (N, 4), numerically stable branch per row."""
    m = R
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = torch.zeros(len(m), 4, device=m.device)
    c0 = tr > 0
    c1 = (~c0) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    c2 = (~c0) & (~c1) & (m[:, 1, 1] > m[:, 2, 2])
    c3 = ~(c0 | c1 | c2)
    S = torch.sqrt((tr + 1.0).clamp(min=1e-12)) * 2
    q[c0] = torch.stack([0.25 * S, (m[:, 2, 1] - m[:, 1, 2]) / S, (m[:, 0, 2] - m[:, 2, 0]) / S, (m[:, 1, 0] - m[:, 0, 1]) / S], 1)[c0]
    S = torch.sqrt((1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2]).clamp(min=1e-12)) * 2
    q[c1] = torch.stack([(m[:, 2, 1] - m[:, 1, 2]) / S, 0.25 * S, (m[:, 0, 1] + m[:, 1, 0]) / S, (m[:, 0, 2] + m[:, 2, 0]) / S], 1)[c1]
    S = torch.sqrt((1.0 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2]).clamp(min=1e-12)) * 2
    q[c2] = torch.stack([(m[:, 0, 2] - m[:, 2, 0]) / S, (m[:, 0, 1] + m[:, 1, 0]) / S, 0.25 * S, (m[:, 1, 2] + m[:, 2, 1]) / S], 1)[c2]
    S = torch.sqrt((1.0 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1]).clamp(min=1e-12)) * 2
    q[c3] = torch.stack([(m[:, 1, 0] - m[:, 0, 1]) / S, (m[:, 0, 2] + m[:, 2, 0]) / S, (m[:, 1, 2] + m[:, 2, 1]) / S, 0.25 * S], 1)[c3]
    return F.normalize(q, dim=1)


def covariance(R, s):
    """Sigma = R diag(s^2) R^T, (N, 3, 3), computed elementwise (faster than a batched GEMM of 3 x 3 matrices)."""
    M = R * s[:, None, :]
    return (M[:, :, None, :] * M[:, None, :, :]).sum(-1)


# ---------------------------------------------------------------------------------------------------------------------
# Gaussian field
# ---------------------------------------------------------------------------------------------------------------------
class GaussianField:
    """M Gaussians (mu, Sigma, alpha). Trainable parameters: xyz, log_scale, quat, amp."""
    PARAMS = ("xyz", "log_scale", "quat", "amp")

    def __init__(self, xyz, scales, R, amp, device="cuda"):
        q = rot_to_quat(R) if R.dim() == 3 else R
        init = {"xyz": xyz, "log_scale": scales.clamp(min=1e-12).log(), "quat": q, "amp": amp}
        self.p = {k: nn.Parameter(v.float().contiguous().to(device)) for k, v in init.items()}
        self.opt = None

    def __len__(self):
        return self.p["xyz"].shape[0]

    @property
    def xyz(self):
        return self.p["xyz"]

    @property
    def scales(self):
        return self.p["log_scale"].exp()

    @property
    def rotation(self):
        return quat_to_rot(self.p["quat"])

    @property
    def amp(self):
        return self.p["amp"]

    def covariance(self):
        return covariance(self.rotation, self.scales)

    def setup_optimizer(self, lrs, extent, iterations, final_ratio=0.1):
        """Adam; the position rate is multiplied by the scene extent (3DGS / GaussianPile convention); every rate
        decays exponentially to final_ratio over the iterations."""
        self.lr0 = {k: lrs[k] * (extent if k == "xyz" else 1.0) for k in self.PARAMS}
        fused = all(p.is_cuda for p in self.p.values())
        self.opt = torch.optim.Adam([{"params": [self.p[k]], "lr": self.lr0[k], "name": k} for k in self.PARAMS],
                                    lr=0.0, eps=1e-15, fused=fused)
        self.final_ratio, self.iterations = final_ratio, iterations

    def set_lr(self, it):
        f = self.final_ratio ** min(it / max(self.iterations, 1), 1.0)
        for g in self.opt.param_groups:
            g["lr"] = self.lr0[g["name"]] * f

    def state_dict(self):
        return {"xyz": self.xyz.detach().cpu(), "scales": self.scales.detach().cpu(),
                "quat": F.normalize(self.p["quat"].detach(), dim=1).cpu(), "amp": self.amp.detach().cpu()}

    @staticmethod
    def from_state(d, device="cuda"):
        return GaussianField(d["xyz"].to(device), d["scales"].to(device), d["quat"].to(device), d["amp"].to(device))

    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path, device="cuda"):
        return GaussianField.from_state(torch.load(path, map_location="cpu"), device)


# ---------------------------------------------------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------------------------------------------------
def render_planes(field, vol, ks, radius_sigma=None, cull_eps=None):
    """V_hat at all voxel centres of the planes ks: (len(ks), H, W), differentiable. Only the Gaussians whose
    radius_sigma slab reaches one of the planes are passed to the kernel."""
    from .cuda import fieldcut
    r = RADIUS_SIGMA if radius_sigma is None else radius_sigma
    e = CULL_EPS if cull_eps is None else cull_eps
    zs = [(k + 0.5) * vol.dz for k in ks]
    R, sc = field.rotation, field.scales
    with torch.no_grad():
        z = torch.as_tensor(zs, device=field.xyz.device, dtype=field.xyz.dtype)
        czz = ((R[:, 2, :] * sc) ** 2).sum(1)
        idx = torch.nonzero(((z[None] - field.xyz[:, 2:3]).abs() <= r * czz.sqrt()[:, None]).any(1)).squeeze(1)
    if len(idx) == 0:
        return torch.zeros(len(ks), vol.H, vol.W, device=field.xyz.device)
    return fieldcut.render(field.xyz[idx], R[idx], sc[idx], field.amp[idx], zs, vol.H, vol.W, vol.dx, vol.dy, r, e)


@torch.no_grad()
def render_volume(field, vol, chunk=16, **kw):
    """The whole volume (Z, H, W), in chunks of `chunk` planes."""
    return torch.cat([render_planes(field, vol, list(range(a, min(a + chunk, vol.Z))), **kw)
                      for a in range(0, vol.Z, chunk)])


@torch.no_grad()
def reference_volume(xyz, scales, R, amp, vol, dtype=torch.float64, vchunk=65536, gchunk=4096):
    """Direct evaluation of the field formula at every voxel centre, without truncation (float64). For validation on
    small volumes only: cost O(voxels x Gaussians)."""
    X = vol.centres().reshape(-1, 3).to(dtype)
    mu, A = xyz.to(dtype), amp.to(dtype)
    P = torch.linalg.inv(covariance(R.to(dtype), scales.to(dtype)))
    out = torch.zeros(len(X), dtype=dtype, device=X.device)
    for a in range(0, len(X), vchunk):
        x = X[a:a + vchunk]
        acc = torch.zeros(len(x), dtype=dtype, device=X.device)
        for g in range(0, len(mu), gchunk):
            d = x[:, None, :] - mu[None, g:g + gchunk]
            q = torch.einsum("vgi,gij,vgj->vg", d, P[g:g + gchunk], d)
            acc += (A[None, g:g + gchunk] * torch.exp(-0.5 * q)).sum(1)
        out[a:a + vchunk] = acc
    return out.view(vol.Z, vol.H, vol.W)


@torch.no_grad()
def calibrate_amplitude(field, vol):
    """Global least-squares amplitude scale c on the valid voxels (alpha <- c alpha); returns c."""
    pred = render_volume(field, vol)[vol.valid_ids]
    gt = vol.gt[vol.valid_ids]
    c = float((pred * gt).sum() / (pred * pred).sum().clamp(min=1e-20))
    field.p["amp"].mul_(c)
    return c


class Timer:
    """Accumulates wall-clock time per named stage (CUDA-synchronised)."""

    def __init__(self):
        self.t = {}

    def __call__(self, name):
        timer = self

        class _T:
            def __enter__(self_):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self_.t0 = time.perf_counter()

            def __exit__(self_, *a):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timer.t[name] = timer.t.get(name, 0.0) + time.perf_counter() - self_.t0
        return _T()
