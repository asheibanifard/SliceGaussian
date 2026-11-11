"""Phase II (methodology Section 3.3): optimization of a fixed Gaussian population, slice-based.

    L = L_vox + beta_g L_grad + beta_m L_multi + beta_r L_reg

Protocol: one slice per iteration, drawn uniformly from the valid slices; every term is evaluated on that rendered
slice (2D):
  L_vox   = mean |V - V_hat| over the pixels of the slice.
  L_grad  = mean ||grad_xy V - grad_xy V_hat||_2, in-plane forward differences in physical units (dv / dx, dv / dy).
  L_multi = (1/S) sum_s mean |G_s * V - G_s * V_hat|, in-plane 2D Gaussian smoothing (replicate padding) at
            sigma_s = multi_sigmas x the in-plane spacing (2 and 4: 8 and 16 nm on EM)              [multi = "smooth"]
            or the band-pass alternative D_s = G_{s-1} - G_s on the same scales (G_0 = identity)  [multi = "band"].
  L_reg   = (1/M) sum_i |alpha_i|: the model has a single amplitude per Gaussian (rather than putting the penalty on
            a separate opacity; methodology 3.3 calls the regularised quantity a_i).
No splitting or pruning. The anisotropy ratio s_max / s_min is clamped at max_anisotropy every 10 steps (float32
covariance accuracy).
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import metrics, snapshots
from .field import render_planes


@dataclass
class P2Config:
    iterations: int = 10000
    slab: int = 1                       # planes per iteration: one slice per iteration (protocol)
    beta_g: float = 0.1
    beta_m: float = 0.1
    beta_r: float = 0.1
    multi: str = "smooth"               # smooth | band | none
    multi_sigmas: str = "2,4"           # in units of the in-plane spacing (8, 16 nm on EM, as before)
    lr_xyz: float = 1.6e-5              # x scene extent
    lr_log_scale: float = 1e-3
    lr_quat: float = 2e-4
    lr_amp: float = 2.5e-4
    final_ratio: float = 0.1
    max_anisotropy: float = 1000.0
    eval_at: str = "0,100,250,500,1000,2000,3000,5000,7500,10000,15000,20000,30000"
    eval_every: int = 500             # whole-volume evaluation (excluded from the timing)
    full_every: int = 5000              # full metrics (3D SSIM, gradient and band errors) here, at 0 and at the end
    snapshot_every: int = 0             # > 0: save GT | prediction | difference images of 3 fixed slices
    seed: int = 0


_KERNELS = {}


def _kernel(sigma, device):
    key = (round(sigma, 6), device)
    if key not in _KERNELS:
        r = max(1, int(4 * sigma + 0.5))
        x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        _KERNELS[key] = (k / k.sum(), r)
    return _KERNELS[key]


def _blur2d(v, sx, sy):
    """In-plane separable Gaussian smoothing of (P, H, W), sigmas in pixels, replicate padding."""
    x = v[:, None]
    k, r = _kernel(sx, v.device)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    k, r = _kernel(sy, v.device)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return x[:, 0]


def losses(pred, gt, valid, field, vol, cfg):
    """pred, gt (P, H, W): the rendered slice(s); every term is 2D, per slice. valid (P,) bool."""
    pred, gt = pred[valid], gt[valid]
    _, dy, dx = vol.spacing()
    parts = {"vox": (gt - pred).abs().mean()}
    L = parts["vox"]
    if cfg.beta_g > 0:
        d = lambda v: torch.stack([(v[:, :-1, 1:] - v[:, :-1, :-1]) / dx, (v[:, 1:, :-1] - v[:, :-1, :-1]) / dy], -1)
        with torch.no_grad():
            dg = d(gt)
        parts["grad"] = (dg - d(pred)).norm(dim=-1).mean()
        L = L + cfg.beta_g * parts["grad"]
    if cfg.beta_m > 0 and cfg.multi != "none":
        sig = [float(s) for s in cfg.multi_sigmas.split(",")]                   # in in-plane pixels
        sy = lambda s: s * dx / dy
        terms = []
        if cfg.multi == "smooth":
            for s in sig:
                with torch.no_grad():
                    ta = _blur2d(gt, s, sy(s))
                terms.append((ta - _blur2d(pred, s, sy(s))).abs().mean())
        else:
            with torch.no_grad():
                ga = [gt] + [_blur2d(gt, s, sy(s)) for s in sig]
            prev = pred
            for i, s in enumerate(sig):
                nb = _blur2d(pred, s, sy(s))
                terms.append(((ga[i] - ga[i + 1]) - (prev - nb)).abs().mean())
                prev = nb
        parts["multi"] = sum(terms) / len(terms)
        L = L + cfg.beta_m * parts["multi"]
    if cfg.beta_r > 0:
        parts["reg"] = field.amp.abs().mean()
        L = L + cfg.beta_r * parts["reg"]
    return L, parts


@torch.no_grad()
def clamp_anisotropy(field, ratio):
    ls = field.p["log_scale"]
    top = ls.max(1, keepdim=True).values
    ls.copy_(torch.maximum(ls, top - math.log(ratio)))


def train(field, vol, cfg, log=None, extra_eval=None, snapshot_dir=None, stop=None, it_offset=0, sec_offset=0.0):
    """Optimizes field in place. Returns the trace: one row per evaluation, with the whole-volume metrics, the
    elapsed optimization time (evaluations excluded), the loss parts and the peak GPU memory."""
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    lrs = {"xyz": cfg.lr_xyz, "log_scale": cfg.lr_log_scale, "quat": cfg.lr_quat, "amp": cfg.lr_amp}
    field.setup_optimizer(lrs, vol.extent, cfg.iterations, cfg.final_ratio)
    starts = [k for k in range(0, vol.Z - cfg.slab + 1) if any(vol.valid_z[k:k + cfg.slab].tolist())]
    at = sorted({int(x) for x in cfg.eval_at.split(",") if int(x) <= cfg.iterations} |
                set(range(0, cfg.iterations + 1, cfg.eval_every)) | {cfg.iterations})
    trace, opt_seconds = [], 0.0
    snap_ks = snapshots.default_slices(vol)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def evaluate(it, parts):
        full = it == 0 or it == cfg.iterations or (it + it_offset) % cfg.full_every == 0
        row = {"iteration": it + it_offset, "seconds": opt_seconds + sec_offset, "gaussians": len(field),
               **{f"loss_{k}": float(v) for k, v in parts.items()}, **metrics.whole(field, vol, full=full),
               "peak_vram_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0}
        if extra_eval:
            row.update(extra_eval(field, vol))
        if snapshot_dir and cfg.snapshot_every > 0 and (it + it_offset) % cfg.snapshot_every == 0:
            snapshots.save(field, vol, snap_ks, it + it_offset, row["psnr"], snapshot_dir)
        trace.append(row)
        if log:
            log(row)

    if it_offset == 0:
        evaluate(0, {})
    for it in range(1, cfg.iterations + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        field.set_lr(it)
        k0 = rng.choice(starts)
        ks = list(range(k0, k0 + cfg.slab))
        pred = render_planes(field, vol, ks)
        L, parts = losses(pred, vol.gt[ks], vol.valid_z[ks], field, vol, cfg)
        L.backward()
        field.opt.step()
        field.opt.zero_grad(set_to_none=True)
        if it % 10 == 0 or it == cfg.iterations:
            clamp_anisotropy(field, cfg.max_anisotropy)
        torch.cuda.synchronize()
        opt_seconds += time.perf_counter() - t0
        if it in at:
            evaluate(it, {"total": L, **parts})
            if stop is not None and stop(trace):
                break
    return trace
