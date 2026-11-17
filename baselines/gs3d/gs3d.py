"""3DGS adapted to slice-based volume reconstruction: random initialisation and 3DGS's adaptive density control on
SliceGaussian's own field, renderer, loss and optimisation settings. Only the population strategy differs from ours.

    python baselines/gs3d/gs3d.py VOLUME --spacing DZ,DY,DX --out DIR [--config configs/paper.json] [--seed 0]

Same as SliceGaussian (configs/paper.json): the additive field of Gaussians, the fused slice-cutting renderer,
L = L_vox + 0.1 L_grad + 0.1 L_multi + 0.1 L_reg on one valid slice per iteration, Adam with the same learning rates,
three cycles of 10,000 iterations with a fresh optimiser and decay to 0.1x in each, the scale-ratio clamp, scoring.

From 3DGS (Kerbl et al. 2023, official defaults): random initialisation with 100,000 points (its random-init setting),
isotropic scales from the mean distance to the 3 nearest neighbours, identity rotations; density control from iteration
500 to 15,000 every 100 iterations on the view-space positional gradient averaged over the views (slices) where the
Gaussian is rendered, threshold 2e-4 in normalised device coordinates; clone if the largest scale is <= 0.01 x scene
extent, else split into 2 children sampled from the parent's Gaussian with scales / 1.6; prune at a low weight, and
after the first opacity-reset interval (3,000) Gaussians wider than 20 px on screen or 0.1 x extent in the world.

Translations to an additive scalar field: 3DGS's opacity is the amplitude here (split children copy it, as 3DGS
copies opacity; a clone and its parent each take half, which leaves the additive field unchanged); the opacity prune (< 0.005) becomes |amplitude| < 0.005;
there is no opacity reset (no opacity). The initial amplitudes are the volume values at the points with one global
least-squares scale, as SliceGaussian's children. Safety limit on the count: the same 10^6 as SliceGaussian's
re-estimation (GPU memory); a step that would exceed it adds only the Gaussians with the largest gradients that fit.
"""
import argparse
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from slicegaussian import Config, load_volume, metrics
from slicegaussian.field import GaussianField, calibrate_amplitude, quat_to_rot, render_planes, RADIUS_SIGMA
from slicegaussian.phase2 import clamp_anisotropy, losses
from slicegaussian.volume import sample

GS = dict(n_init=100_000, densify_from=500, densify_until=15_000, densify_every=100, grad_threshold=2e-4,
          percent_dense=0.01, min_amp=0.005, size_check_from=3000, max_screen_px=20, max_world_frac=0.1)


@torch.no_grad()
def knn3_scale(xyz, max_pairs=2.5e8):
    """sqrt(mean squared distance to the 3 nearest neighbours), as simple-knn in 3DGS (exact, chunked)."""
    out = torch.empty(len(xyz), device=xyz.device)
    chunk = max(1, int(max_pairs // len(xyz)))
    for a in range(0, len(xyz), chunk):
        d = torch.cdist(xyz[a:a + chunk], xyz)
        d2 = d.topk(4, largest=False).values[:, 1:] ** 2
        out[a:a + chunk] = d2.mean(1).clamp(min=1e-12).sqrt()
    return out


@torch.no_grad()
def random_field(vol, n, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    hi = torch.tensor([vol.W * vol.dx, vol.H * vol.dy, vol.Z * vol.dz])
    xyz = (torch.rand(n, 3, generator=g) * hi).to(vol.device)
    s = knn3_scale(xyz)[:, None].expand(n, 3).contiguous()
    R = torch.eye(3, device=vol.device).expand(n, 3, 3).contiguous()
    V = torch.from_numpy(vol.analysis_volume).to(vol.device)
    amp = sample(V[None], xyz[:, None], vol)[0, :, 0]
    f = GaussianField(xyz, s, R, amp, device=vol.device)
    calibrate_amplitude(f, vol)
    return f


class Density:
    """3DGS's densification statistics and operations on a GaussianField whose Adam state is kept consistent."""

    def __init__(self, field, vol, P, cap):
        self.f, self.vol, self.P, self.cap = field, vol, P, cap
        self.reset()

    def reset(self):
        n, dev = len(self.f), self.f.xyz.device
        self.accum = torch.zeros(n, device=dev)
        self.denom = torch.zeros(n, device=dev)
        self.max_r = torch.zeros(n, device=dev)

    @torch.no_grad()
    def add(self, k):
        """After backward on slice k: the Gaussians rendered on it (the renderer's 3-sigma slab test) and their
        view-space positional gradient in NDC (pixel gradient x W/2, H/2), and their on-screen radius in pixels."""
        f, v = self.f, self.vol
        z = (k + 0.5) * v.dz
        R, sc = f.rotation, f.scales
        czz = ((R[:, 2, :] * sc) ** 2).sum(1)
        vis = (z - f.xyz[:, 2]).abs() <= RADIUS_SIGMA * czz.sqrt()
        g = f.p["xyz"].grad
        if g is None:
            return
        gx = g[vis, 0] * v.dx * (v.W / 2)
        gy = g[vis, 1] * v.dy * (v.H / 2)
        self.accum[vis] += torch.sqrt(gx ** 2 + gy ** 2)
        self.denom[vis] += 1
        cxx = ((R[vis, 0, :] * sc[vis]) ** 2).sum(1)
        cyy = ((R[vis, 1, :] * sc[vis]) ** 2).sum(1)
        r = RADIUS_SIGMA * torch.maximum(cxx, cyy).sqrt() / v.dx
        self.max_r[vis] = torch.maximum(self.max_r[vis], r)

    @torch.no_grad()
    def _replace(self, new):
        """Swap the parameter tensors; Adam moments follow (rows kept / zero rows for new Gaussians)."""
        opt = self.f.opt
        for gr in opt.param_groups:
            name = gr["name"]
            old = gr["params"][0]
            p = torch.nn.Parameter(new[name]["value"].contiguous())
            st = opt.state.pop(old, None)
            if st:
                for key in ("exp_avg", "exp_avg_sq"):
                    st[key] = new[name]["state"](st[key])
                opt.state[p] = st
            gr["params"][0] = p
            self.f.p[name] = p

    @torch.no_grad()
    def densify_and_prune(self, it, extent, stats):
        f, P = self.f, self.P
        grads = self.accum / self.denom.clamp(min=1)
        cand = grads >= P["grad_threshold"]
        smax = f.scales.max(1).values
        clone = cand & (smax <= P["percent_dense"] * extent)
        split = cand & (smax > P["percent_dense"] * extent)
        room = self.cap - len(f)
        if int(clone.sum()) + int(split.sum()) > room:                       # safety limit (not part of 3DGS)
            stats["cap_hits"] += 1
            score = torch.where(clone | split, grads, torch.full_like(grads, -1.0))
            keep = torch.zeros_like(cand)
            if room > 0:
                keep[score.topk(room).indices] = True
            clone, split = clone & keep, split & keep
        stats["cloned"] += int(clone.sum())
        stats["split"] += int(split.sum())
        # Clone: 3DGS copies the Gaussian with its opacity, which alpha compositing saturates. In an additive field a
        # copy doubles the local intensity, so parent and clone each take half the amplitude (the field is unchanged).
        f.p["amp"].data[clone] *= 0.5
        ext = {k: f.p[k][clone].detach().clone() for k in f.PARAMS}
        if split.any():
            s = f.scales[split]
            R = f.rotation[split]
            for _ in range(2):
                off = torch.einsum("nij,nj->ni", R, torch.randn_like(s) * s)
                part = {"xyz": f.xyz[split] + off, "log_scale": (s / 1.6).log(),
                        "quat": f.p["quat"][split], "amp": f.amp[split]}
                for k in f.PARAMS:
                    ext[k] = torch.cat([ext[k], part[k].detach()])
        n0 = len(f)
        m = ext["xyz"].shape[0]
        self._replace({k: {"value": torch.cat([f.p[k].detach(), ext[k]]),
                           "state": (lambda st, m=m: torch.cat([st, torch.zeros((m,) + st.shape[1:], device=st.device,
                                                                                  dtype=st.dtype)]))}
                       for k in f.PARAMS})
        self.accum = torch.cat([self.accum, torch.zeros(m, device=self.accum.device)])
        self.denom = torch.cat([self.denom, torch.zeros(m, device=self.denom.device)])
        self.max_r = torch.cat([self.max_r, torch.zeros(m, device=self.max_r.device)])
        drop = torch.zeros(len(f), dtype=torch.bool, device=f.xyz.device)
        drop[:n0] = split
        drop |= f.amp.abs() < P["min_amp"]
        if it > P["size_check_from"]:
            drop |= self.max_r > P["max_screen_px"]
            drop |= f.scales.max(1).values > P["max_world_frac"] * extent
        stats["pruned"] += int(drop.sum()) - int(split.sum())
        keep = ~drop
        self._replace({k: {"value": f.p[k].detach()[keep], "state": (lambda st, keep=keep: st[keep])} for k in f.PARAMS})
        self.reset()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("volume")
    p.add_argument("--spacing", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", default="configs/paper.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cap", type=int, default=None, help="default: the config's re-estimation limit (10^6)")
    p.add_argument("--grad-threshold", type=float, default=None,
                   help="densification threshold (3DGS: 2e-4); tuned on the phantom, see README")
    a = p.parse_args()
    cfg = Config.load(a.config)
    cfg.seed = a.seed
    cap = a.cap or cfg.reestimation.budget
    P = dict(GS)
    if a.grad_threshold is not None:
        P["grad_threshold"] = a.grad_threshold
    random.seed(a.seed), np.random.seed(a.seed), torch.manual_seed(a.seed)
    vol = load_volume(a.volume, tuple(float(s) for s in a.spacing.split(",")))
    a.out.mkdir(parents=True, exist_ok=True)
    extent = vol.extent                                                   # as Phase II (cfg.p2.extent = "volume")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    field = random_field(vol, P["n_init"], a.seed)
    dens = Density(field, vol, P, cap)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    (a.out / "config.json").write_text(json.dumps({"method": "3DGS density control on the SliceGaussian field",
        "volume": a.volume, "spacing": vol.spacing(), "seed": a.seed, "cap": cap, "gs": P,
        "p2": vars(cfg.p2) if not hasattr(cfg.p2, "__dataclass_fields__") else {k: getattr(cfg.p2, k) for k in cfg.p2.__dataclass_fields__},
        "cycles": cfg.reestimation.cycles, "cycle_iters": cfg.reestimation.cycle_iters}, indent=1, default=str))
    stats = {"cloned": 0, "split": 0, "pruned": 0, "cap_hits": 0}
    rng = random.Random(a.seed)
    starts = [k for k in range(vol.Z) if bool(vol.valid_z[k])]
    lrs = {"xyz": cfg.p2.lr_xyz, "log_scale": cfg.p2.lr_log_scale, "quat": cfg.p2.lr_quat, "amp": cfg.p2.lr_amp}
    trace, it = [], 0
    for c in range(cfg.reestimation.cycles):
        field.setup_optimizer(lrs, extent, cfg.reestimation.cycle_iters, cfg.p2.final_ratio)
        for j in range(1, cfg.reestimation.cycle_iters + 1):
            it += 1
            torch.cuda.synchronize()
            ts = time.perf_counter()
            field.set_lr(j)
            k = rng.choice(starts)
            pred = render_planes(field, vol, [k])
            L, parts = losses(pred, vol.gt[[k]], vol.valid_z[[k]], field, vol, cfg.p2)
            L.backward()
            if it < P["densify_until"]:
                dens.add(k)
            field.opt.step()
            field.opt.zero_grad(set_to_none=True)
            if it % 10 == 0:
                clamp_anisotropy(field, cfg.p2.max_anisotropy)
            if P["densify_from"] < it < P["densify_until"] and it % P["densify_every"] == 0:
                dens.densify_and_prune(it, extent, stats)
            torch.cuda.synchronize()
            seconds += time.perf_counter() - ts
            if it % 2500 == 0:
                m = metrics.whole(field, vol, full=False)
                row = {"iteration": it, "seconds": seconds, "gaussians": len(field), "psnr": m["psnr"], **stats}
                trace.append(row)
                print(json.dumps(row), flush=True)
    final = metrics.whole(field, vol, full=True)
    field.save(a.out / "model.pt")
    (a.out / "final.json").write_text(json.dumps({**final, "gaussians": len(field), "total_seconds": seconds,
                                                  "densification": stats, "trace": trace}, indent=1))
    print(f"done: {len(field)} Gaussians, {seconds:.1f} s, psnr {final['psnr']:.3f}", flush=True)


if __name__ == "__main__":
    main()
