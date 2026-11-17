"""SAD-GS (Lyu et al., SIGGRAPH 2026) adapted to slice-based scalar volumes: structure-aware densification with the
structure tensor computed on the 3D volume.

    python baselines/sadgs3d/sadgs3d.py VOLUME --spacing DZ,DY,DX --out DIR [--seed 0]

Source: the official SAD-GS code (SADGS repository, commit 029814f): train.py (densification block),
utils/freq_utils.py (update_freq_stats_online), utils/loss_utils.py (get_multiscale_structure_tensor_v1) and
scene/gaussian_model.py (densify_and_prune_structgs, densify_and_split_structgs, densify_and_clone_structgs,
reset_opacity). See README.md for what is kept, translated and removed.

Base model: the slice-adapted 3DGS of this paper's Table 1 (freqpile/experiments/final_experiments_3dgs.py), unchanged:
10^6 random initial Gaussians with a grey value each, slice rendering by masking |z - z_k| > 3 sigma_z and orthographic
alpha compositing (gsplat), L1 + 0.2 D-SSIM, Adam with GaussianPile's learning rates. Only the density control differs.
Scored with slicegaussian.metrics.whole on the valid slices, as every method of the paper.
"""
import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT.parent / "freqpile"


# ---------------------------------------------------------------------------------------------------------------------
# SAD-GS hyperparameters (official defaults: arguments/__init__.py and train.py of commit 029814f)
# ---------------------------------------------------------------------------------------------------------------------
SADGS = dict(
    densify_from_iter=500, densify_until_iter=15_000, densification_interval=500,   # 500: every released SAD-GS config
    # 3DGS's opacity reset (x0.1 every 3000 iterations) is not used: with one slice per iteration a Gaussian is
    # supervised only when its own slice is drawn, its opacity cannot recover before the next densification, and the
    # min-opacity prune then removes the population (observed); the base model has no reset either.
    # opacity_reset_interval = 3000 is kept only as the start of the screen-size prune (size_threshold in train.py).
    opacity_reset_interval=3000, opacity_reset_decay=None, prune_iterations=(4000, 8000),
    st_levels=4, st_base_sigma=1.0, st_octave_step=1.5, st_power_factor=3.0,      # get_multiscale_structure_tensor_v1
    tau_high=1.0, tau_low=0.1, split_ratio_threshold=0.8, prune_ratio_threshold=0.8,
    freq_opacity_threshold=0.05, dense=0.001, importance_score_threshold=0.5,
    # Gradient thresholds. SAD-GS sets them on 3DGS's screen-space gradients: grad_thresh = grad_abs_thresh = 2e-4
    # (3DGS's standard), freq_grad_threshold = 2e-5 (0.1x) and the split gate 1e-5 (0.05x). The base model of this
    # baseline (the slice-adapted 3DGS of Table 1) densifies at 5e-5 on the 2D-mean gradient in pixels, so SAD-GS's
    # thresholds are kept in the same ratio to that standard threshold, in pixels.
    grad_thresh=5e-5, grad_abs_thresh=5e-5, freq_grad_threshold=0.1 * 5e-5, split_grad_min=0.05 * 5e-5,
    min_opacity=0.1, max_screen_size=20, max_world_frac=0.1, opacity_cap=0.8, ks_scale_power=1.0,
    # SAD-GS accumulates per-view statistics every 10 iterations, when nearly every Gaussian is visible in each view.
    # Here a Gaussian is visible only on its own few slices, so the views are accumulated at every iteration.
    stats_every=1,
)


# ---------------------------------------------------------------------------------------------------------------------
# 3D multiscale structure tensor (get_multiscale_structure_tensor_v1, 2D -> 3D)
# ---------------------------------------------------------------------------------------------------------------------
def _blur(v, sigma_px, dx, dz):
    """3D Gaussian blur of (Z, H, W) with std sigma_px in-plane pixels, i.e. sigma_px * dx physical in every axis."""
    from slicegaussian.volume import blur3d
    return blur3d(v[None], (sigma_px * dx / dz, sigma_px, sigma_px))[0]


def _sym6(gx, gy, gz):
    return torch.stack([gx * gx, gx * gy, gy * gy, gx * gz, gy * gz, gz * gz])


@torch.no_grad()
def structure_tensor_3d(V, dz, dx, levels, base_sigma, octave_step, power_factor):
    """(6, Z, H, W) tensor in cycles^2 per in-plane pixel^2 (xx, xy, yy, xz, yz, zz). As v1: a base tensor of the
    image smoothed at base_sigma; per level i, the band response |V_i - V_{i+1}| (smoothed at 2 sigma_i for i > 0)
    weights the base tensor integrated at 3 sigma_i and normalised to unit trace, scaled by band_freq_i^2 =
    octave_step^(-2i); the levels are averaged with weights band^power_factor. Units: every scale is in in-plane
    pixels and converted to the physical spacing, gradients are per in-plane pixel (dV / (dx physical))."""
    bl = lambda v, s: _blur(v, s, dx, dz)
    gz, gy, gx = torch.gradient(bl(V, base_sigma), spacing=(dz / dx, 1.0, 1.0))
    base = _sym6(gx, gy, gz)
    del gx, gy, gz
    acc = torch.zeros_like(base)
    wsum = torch.zeros_like(V)
    cur, s_acc = V, 0.0
    for i in range(levels):
        band_freq = 1.0 / octave_step ** i
        target = base_sigma * octave_step ** i
        nxt = bl(cur, math.sqrt(max(1e-6, target ** 2 - s_acc ** 2)))
        band = (cur - nxt).abs()
        if i > 0:
            band = bl(band, 2.0 * target)
        S = torch.stack([bl(c, 3.0 * target) for c in base])
        S = S / (S[0] + S[2] + S[5] + 1e-6)
        w = band ** power_factor
        acc += S * w * band_freq ** 2
        wsum += w
        cur, s_acc = nxt, target
        del S, band
    return acc / (wsum + 1e-6)


def largest_eigenvalue(t6):
    """Largest eigenvalue of symmetric 3x3 matrices given as (6, n) = (xx, xy, yy, xz, yz, zz); closed form
    (trigonometric solution of the characteristic cubic), the 3D counterpart of lambda1 in update_freq_stats_online."""
    xx, xy, yy, xz, yz, zz = t6.double()
    q = (xx + yy + zz) / 3
    p1 = xy ** 2 + xz ** 2 + yz ** 2
    p2 = (xx - q) ** 2 + (yy - q) ** 2 + (zz - q) ** 2 + 2 * p1
    p = (p2 / 6).clamp(min=1e-30).sqrt()
    bxx, byy, bzz = (xx - q) / p, (yy - q) / p, (zz - q) / p
    bxy, bxz, byz = xy / p, xz / p, yz / p
    detb = bxx * (byy * bzz - byz ** 2) - bxy * (bxy * bzz - byz * bxz) + bxz * (bxy * byz - byy * bxz)
    phi = torch.acos((detb / 2).clamp(-1, 1)) / 3
    return (q + 2 * p * torch.cos(phi)).float()


# ---------------------------------------------------------------------------------------------------------------------
# Model with SAD-GS's per-Gaussian statistics
# ---------------------------------------------------------------------------------------------------------------------
def build_model_class(mod):
    base = mod.Model3DGS

    class ModelSADGS(base):
        def _reset_stats(self):
            super()._reset_stats()
            n, dev = self.num, self.xyz.device
            self.grad_abs_accum = torch.zeros(n, device=dev)
            self.view_count = torch.zeros(n, device=dev)
            self.high_count = torch.zeros(n, device=dev)
            self.low_count = torch.zeros(n, device=dev)
            self.max_eta = torch.zeros(n, 3, device=dev)

        @torch.no_grad()
        def add_stats_abs(self, means2d_grad, means2d_absgrad, radii):
            super().add_stats(means2d_grad, radii)
            vis = radii > 0
            self.grad_abs_accum[vis] += means2d_absgrad[vis].norm(dim=-1)

        def _keep(self, keep):
            for n in ("grad_abs_accum", "view_count", "high_count", "low_count", "max_eta"):
                setattr(self, n, getattr(self, n)[keep])

        @torch.no_grad()
        def prune(self, mask):
            super().prune(mask)
            self._keep(~mask)

        @torch.no_grad()
        def append(self, ext):
            super().append(ext)
            k = ext["xyz"].shape[0]
            dev = self.xyz.device
            for n in ("grad_abs_accum", "view_count", "high_count", "low_count"):
                setattr(self, n, torch.cat([getattr(self, n), torch.zeros(k, device=dev)]))
            self.max_eta = torch.cat([self.max_eta, torch.zeros(k, 3, device=dev)])

        @torch.no_grad()
        def set_alpha(self, alpha):
            """Replace the opacities (logits) and clear their Adam moments, as replace_tensor_to_optimizer."""
            new = {n: {"value": getattr(self, n), "state": (lambda s: s)} for n in self.PARAMS}
            new["alpha_logit"] = {"value": torch.logit(alpha.clamp(1e-6, 1 - 1e-6)), "state": torch.zeros_like}
            self._replace(new)

    return ModelSADGS


# ---------------------------------------------------------------------------------------------------------------------
# update_freq_stats_online, one slice = one view
# ---------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def update_freq_stats(model, idx, grad2d_ndc, st, vol_shape, spacing, P):
    """For the Gaussians rendered on this slice (idx) that are opaque enough and have a 2D-mean gradient above the
    threshold: sample the structure tensor at a random point of the Gaussian's 1-sigma ellipsoid (3D counterpart of
    the jitter inside the projected ellipse), eta_k = (axis length in pixels) / (shortest local wavelength) with the
    wavelength 1 / sqrt(largest eigenvalue) ("wavelength" mode, the default), and count the view as high / low."""
    if idx.numel() == 0:
        return
    dz, dy, dx = spacing
    Z, H, W = vol_shape
    active = (model.alphas[idx] > P["freq_opacity_threshold"]) & (grad2d_ndc.norm(dim=-1) > P["freq_grad_threshold"])
    g = idx[active]
    if g.numel() == 0:
        return
    s = model.scales[g]
    R = mod_quat_to_rot(model.quats[g])
    pts = model.xyz[g] + torch.einsum("nij,nj->ni", R, s * torch.randn_like(s))
    grid = torch.stack([pts[:, 0] / dx / (W - 1), pts[:, 1] / dy / (H - 1), pts[:, 2] / dz / (Z - 1)], -1) * 2 - 1
    S = F.grid_sample(st[None], grid.view(1, -1, 1, 1, 3), align_corners=True)[0, :, :, 0, 0]   # (6, n), zero outside
    lam1 = largest_eigenvalue(S).clamp(min=0)
    wavelength_min = 1.0 / (lam1.sqrt() + 1e-5)                                    # in-plane pixels
    eta = (s / dx) / wavelength_min[:, None]                                       # (n, 3)
    model.view_count[g] += 1
    model.max_eta[g] = torch.maximum(model.max_eta[g], eta)
    e = eta.amax(1)
    model.high_count[g[e > P["tau_high"]]] += 1
    model.low_count[g[e <= P["tau_low"]]] += 1


def mod_quat_to_rot(q):
    from gaussianpile.reference import quat_to_rot
    return quat_to_rot(q)


# ---------------------------------------------------------------------------------------------------------------------
# densify_and_prune_structgs (+ split / clone)
# ---------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def densify_and_prune(model, extent, P, size_check, stats, cap):
    den = model.denom.clamp_min(1)
    grads, grads_abs = model.grad_accum / den, model.grad_abs_accum / den          # pixels
    valid = model.view_count > 0
    high = torch.where(valid, model.high_count / model.view_count.clamp_min(1), torch.zeros_like(grads))
    low = torch.where(valid, model.low_count / model.view_count.clamp_min(1), torch.zeros_like(grads))
    split_mask = (high > P["split_ratio_threshold"]) & (grads >= P["split_grad_min"])
    prune_mask = (low > P["prune_ratio_threshold"]) & valid
    smax = model.scales.max(-1).values
    clone_q, split_q = smax <= P["dense"] * extent, smax > P["dense"] * extent
    final_split = (split_mask | (grads_abs >= P["grad_abs_thresh"])) & split_q
    final_clone = (split_mask | (grads >= P["grad_thresh"])) & clone_q
    metric = model.view_count > P["importance_score_threshold"]
    clone, split = metric & final_clone, metric & final_split
    if cap:
        # Budget (not part of SAD-GS): if this step would exceed the cap, keep the candidates with the largest growth
        # demand first -- splits by max_k eta, then clones by gradient -- while their added Gaussians fit.
        kall = torch.sqrt(model.max_eta.clamp(min=1.0)).ceil().long().clamp(min=1).prod(1)
        cost = torch.where(split, kall - 1, clone.long())
        room = cap - model.num
        if int(cost.sum()) > room:
            stats["cap_hits"] += 1
            score = torch.where(split, 1e6 + model.max_eta.amax(1), torch.where(clone, grads, torch.full_like(grads, -1)))
            order = torch.argsort(score, descending=True)
            fits = torch.cumsum(cost[order], 0) <= max(room, 0)
            keep = torch.zeros_like(split)
            keep[order[fits]] = True
            split, clone = split & keep, clone & keep
    ks = torch.sqrt(model.max_eta[split].clamp(min=1.0)).ceil().long().clamp(min=1)
    n_children = int(ks.prod(1).sum())
    stats["cloned"] += int(clone.sum())
    stats["split"] += int(split.sum())
    stats["children"] += n_children
    n0 = model.num
    ext = {n: getattr(model, n)[clone].clone() for n in model.PARAMS}
    if split.any():
        rep = ks.prod(1)
        s = model.scales[split]
        kr = ks.repeat_interleave(rep, 0).float()
        start = torch.cumsum(rep, 0) - rep
        j = torch.arange(int(rep.sum()), device=s.device) - start.repeat_interleave(rep)
        iz = j % kr[:, 2].long()
        iy = (j // kr[:, 2].long()) % kr[:, 1].long()
        ix = j // (kr[:, 1].long() * kr[:, 2].long())
        grid = torch.stack([ix.float() - (kr[:, 0] - 1) / 2, iy.float() - (kr[:, 1] - 1) / 2,
                            iz.float() - (kr[:, 2] - 1) / 2], 1)
        s_child = s.repeat_interleave(rep, 0) / kr ** P["ks_scale_power"]
        off = (s.repeat_interleave(rep, 0) / kr) * 12 ** 0.5 * grid
        Rm = mod_quat_to_rot(model.quats[split]).repeat_interleave(rep, 0)
        children = {"xyz": model.xyz[split].repeat_interleave(rep, 0) + torch.einsum("nij,nj->ni", Rm, off),
                    "log_scale": s_child.log()}
        for n in model.PARAMS:
            if n not in children:
                children[n] = getattr(model, n)[split].repeat_interleave(rep, 0)
        for n in model.PARAMS:
            ext[n] = torch.cat([ext[n], children[n]], 0)
    model.append(ext)
    drop = torch.zeros(model.num, dtype=torch.bool, device=model.xyz.device)
    drop[:n0] = split | prune_mask
    drop |= model.alphas < P["min_opacity"]
    if size_check:
        drop |= model.max_radii > P["max_screen_size"]
        drop |= model.scales.max(-1).values > P["max_world_frac"] * extent
    stats["pruned"] += int(drop.sum()) - int(split.sum())
    model.prune(drop)
    model.set_alpha(model.alphas.clamp(max=P["opacity_cap"]))
    model._reset_stats()


# ---------------------------------------------------------------------------------------------------------------------
# Rendering: the 3DGS baseline's slice renderer, also returning the absolute 2D-mean gradient (gsplat absgrad)
# ---------------------------------------------------------------------------------------------------------------------
def make_render(mod):
    def render(model, k, dz, dx, H, W):
        z = k * dz
        sz = mod.covariance_zz(model).sqrt()
        idx = torch.nonzero((model.xyz[:, 2] - z).abs() <= mod.EXTENT_SIGMA * sz).squeeze(1)
        if idx.numel() == 0:
            return torch.zeros(H, W, device="cuda"), idx, None, None
        depth0 = float(model.xyz[:, 2].min().detach()) - 10.0 * dz
        viewmat = torch.eye(4, device="cuda")[None]
        viewmat[0, 2, 3] = -depth0
        K = torch.tensor([[1.0 / dx, 0.0, 0.5], [0.0, 1.0 / dx, 0.5], [0.0, 0.0, 1.0]], device="cuda")[None]
        img, _, info = mod.rasterization(model.xyz[idx], model.quats[idx], model.scales[idx], model.alphas[idx],
                                         model.color[idx][:, None], viewmat, K, W, H, camera_model="ortho",
                                         packed=False, near_plane=1e-3, far_plane=1e10, render_mode="RGB",
                                         absgrad=True)
        return img[0, ..., 0], idx, info["means2d"], info["radii"]
    return render


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("volume")
    p.add_argument("--spacing", required=True, help="dz,dy,dx")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations", type=int, default=30000)
    p.add_argument("--initial-n", type=int, default=1_000_000)
    p.add_argument("--log-every", type=int, default=5000)
    p.add_argument("--cap", type=int, default=6_000_000, help="GPU-memory safety limit (not part of SAD-GS)")
    a = p.parse_args()
    spacing = tuple(float(s) for s in a.spacing.split(","))
    sys.path[:0] = [str(REF / "experiments"), str(REF / "standalone"), str(ROOT.parent / "gaussianpile")]
    import random
    from slicegaussian import load_volume, metrics
    mod = importlib.import_module("final_experiments_3dgs")
    mod.fg.SPACING = spacing
    from gaussianpile import GaussianModel
    from gaussianpile.losses import gaussianpile_loss
    P = dict(SADGS)
    random.seed(a.seed), np.random.seed(a.seed), torch.manual_seed(a.seed)
    vol = load_volume(a.volume, spacing)
    a.out.mkdir(parents=True, exist_ok=True)
    stack = mod.hs.Stack(a.volume, spacing, 0)
    Z, H, W = stack.Z, stack.H, stack.W
    dz, dy, dx = spacing
    hi = [(W - 1) * dx, (H - 1) * dy, (Z - 1) * dz]
    extent = float(np.linalg.norm(np.array(hi)) / 2)
    Model = build_model_class(mod)
    g = GaussianModel.random_init(a.initial_n, [0.0, 0.0, 0.0], hi, device="cuda", seed=a.seed)
    ijk = (g.xyz.detach() / torch.tensor([dx, dy, dz], device="cuda")).round().long()
    ijk = torch.minimum(ijk.clamp_min(0), torch.tensor([W - 1, H - 1, Z - 1], device="cuda"))
    model = Model(g.xyz.detach(), g.log_scale.detach(), g.rot.detach(), g.alpha_logit.detach(),
                  stack.slices[ijk[:, 2], ijk[:, 1], ijk[:, 0]].clone())
    del g
    model.setup_optimizer(extent, max_iters=a.iterations)
    render = make_render(mod)
    valid = [k for k in range(Z) if k in set(vol.valid_ids)]

    @torch.no_grad()
    def evaluate():
        pred = torch.stack([render(model, k, dz, dx, H, W)[0] for k in range(Z)]).clamp(0, 1)
        return metrics.whole(None, vol, full=True, pred=pred), pred

    torch.cuda.synchronize()
    t0 = time.time()
    st = structure_tensor_3d(stack.slices.float(), dz, dx, P["st_levels"], P["st_base_sigma"], P["st_octave_step"],
                             P["st_power_factor"])
    torch.cuda.synchronize()
    st_seconds = time.time() - t0
    (a.out / "config.json").write_text(json.dumps({
        "method": "SAD-GS (3D structure tensor) on the slice-adapted 3DGS", "volume": a.volume, "spacing": spacing,
        "seed": a.seed, "iterations": a.iterations, "initial_gaussians": a.initial_n, "cap": a.cap, "sadgs": P,
        "source": "SADGS commit 029814f"}, indent=1))
    stats = {"cloned": 0, "split": 0, "children": 0, "pruned": 0, "cap_hits": 0}
    trace, eval_seconds = [], 0.0
    for it in range(1, a.iterations + 1):
        model.update_lr(it)
        k = valid[int(torch.randint(len(valid), (1,)))]
        img, idx, means2d, radii = render(model, k, dz, dx, H, W)
        if means2d is not None:
            means2d.retain_grad()
        loss = gaussianpile_loss(img, stack.slices[k], mod.fg.LAMBDA_SSIM)
        loss.backward()
        with torch.no_grad():
            if it < P["densify_until_iter"] and means2d is not None and means2d.grad is not None:
                gr, ga = means2d.grad[0], means2d.absgrad[0]                      # pixels, as the base model
                rr = radii[0].float().amax(-1) if radii.dim() == 3 else radii[0].float()
                gf, af = torch.zeros(model.num, 2, device="cuda"), torch.zeros(model.num, 2, device="cuda")
                rf = torch.zeros(model.num, device="cuda")
                gf[idx], af[idx], rf[idx] = gr, ga, rr
                model.add_stats_abs(gf, af, rf)
                if it % P["stats_every"] == 0:
                    vis = rr > 0
                    update_freq_stats(model, idx[vis], gr[vis], st, (Z, H, W), spacing, P)
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            if it < P["densify_until_iter"]:
                if it > P["densify_from_iter"] and it % P["densification_interval"] == 0:
                    densify_and_prune(model, extent, P, it > P["opacity_reset_interval"], stats, a.cap)
                if P["opacity_reset_decay"] and it % P["opacity_reset_interval"] == 0:
                    model.set_alpha(model.alphas * P["opacity_reset_decay"])
            if it in P["prune_iterations"]:
                model.prune(model.alphas < P["min_opacity"])
        if it % a.log_every == 0 or it == a.iterations:
            torch.cuda.synchronize()
            t_e = time.time()
            m, _ = evaluate()
            row = {"iteration": it, "seconds": t_e - t0 - eval_seconds, "gaussians": model.num, "loss": float(loss),
                   "psnr": m["psnr"], "ssim": m["ssim"], "mae": m["mae"], **stats}
            trace.append(row)
            print(json.dumps(row), flush=True)
            torch.cuda.synchronize()
            eval_seconds += time.time() - t_e
    torch.cuda.synchronize()
    seconds = time.time() - t0 - eval_seconds
    final, pred = evaluate()
    torch.save(model.state(), a.out / "model.pt")
    np.save(a.out / "prediction.npy", pred.float().cpu().numpy())
    (a.out / "metrics.json").write_text(json.dumps({
        "train_seconds": seconds, "structure_tensor_seconds": st_seconds, "gaussians": model.num,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9, "densification": stats, "final": final,
        "trace": trace}, indent=1))
    print(f"done: {model.num} Gaussians, {seconds:.1f} s, psnr {final['psnr']:.3f}", flush=True)


if __name__ == "__main__":
    main()
