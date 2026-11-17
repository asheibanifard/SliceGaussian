"""Our Phase II on another method's final Gaussians: the method's result taken as a Phase I population, scored before
and after one Phase II of 30,000 iterations (no re-estimation), the protocol of the upper block of Table 1.

    micromamba run -n gaussian-volume python scripts/phase2_on_population.py --method gaussianpile|3dgs|sadgs3d \
        --model MODEL.pt --volume data/cremi/cremi_a.tif --spacing 40,4,4 --out DIR

Conversion to SliceGaussian's additive field (our coordinates put voxel centres at (i + 1/2) h; GaussianPile's and
3DGS's at i h, so every centre is shifted by half a voxel):
  * GaussianPile (sigma_z = 0, "paper" normalisation): its slice is sum_i alpha_i N_i G_i(x, y, z_k) with the per-Gaussian
    constant N_i = 1 / sqrt(det S_i), S_i the in-plane conditional covariance at fixed z in pixel units (its Eq. 8-10).
    Our amplitude alpha_i N_i reproduces it exactly (checked: "converted" equals the method's own score).
  * 3DGS / SAD-GS (alpha compositing): there is no exact additive equivalent; amplitude = opacity x intensity followed by
    one global least-squares amplitude scale, as for SliceGaussian's own populations. The method's own score is
    reported next to the converted one.
Writes result.json (own, converted, after Phase II scores; counts; seconds) and model.pt.
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT.parent / "freqpile"
sys.path[:0] = [str(REF / "experiments"), str(REF / "standalone"), str(ROOT.parent / "gaussianpile")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=["gaussianpile", "3dgs", "sadgs3d"])
    p.add_argument("--model", required=True)
    p.add_argument("--volume", required=True)
    p.add_argument("--spacing", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", default="configs/paper.json")
    p.add_argument("--iterations", type=int, default=30000)
    a = p.parse_args()
    from gaussianpile.reference import quat_to_rot
    from slicegaussian import Config, load_volume, metrics, phase2
    from slicegaussian.field import GaussianField, calibrate_amplitude
    from slicegaussian.pipeline import seed_everything
    cfg = Config.load(a.config)
    seed_everything(0)
    vol = load_volume(a.volume, tuple(float(s) for s in a.spacing.split(",")))
    st = torch.load(a.model, map_location="cuda")
    q = torch.nn.functional.normalize(st["rot"], dim=-1)
    R = quat_to_rot(q)
    s = st["log_scale"].exp()
    alpha = torch.sigmoid(st["alpha_logit"])
    xyz = st["xyz"] + 0.5 * torch.tensor([vol.dx, vol.dy, vol.dz], device="cuda")
    if a.method == "gaussianpile":
        C = R @ torch.diag_embed(s ** 2) @ R.transpose(1, 2)
        S2 = C[:, :2, :2] - C[:, :2, 2:3] @ C[:, 2:3, :2] / C[:, 2:3, 2:3]          # conditional on z
        S2 = S2 / torch.tensor([[vol.dx * vol.dx, vol.dx * vol.dy], [vol.dy * vol.dx, vol.dy * vol.dy]], device="cuda")
        amp = alpha / torch.linalg.det(S2).clamp(min=1e-20).sqrt()
    else:
        amp = alpha * st["color"].reshape(-1)
    field = GaussianField(xyz, s, R, amp, device="cuda")
    calibrated = False
    if a.method != "gaussianpile":
        calibrate_amplitude(field, vol)
        calibrated = True
    own_file = Path(a.model).parent / "metrics.json"
    own = json.loads(own_file.read_text()) if own_file.exists() else {}
    own = own.get("train") or own.get("final") or {}
    before = metrics.whole(field, vol, full=True)
    print(f"{a.method}: own {own.get('psnr', float('nan')):.3f}  converted {before['psnr']:.3f}  N {len(field):,}",
          flush=True)
    p2 = replace(cfg.p2, iterations=a.iterations, seed=0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    trace = phase2.train(field, vol, p2)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    after = metrics.whole(field, vol, full=True)
    print(f"{a.method}: after Phase II {after['psnr']:.3f}  ({seconds:.0f} s)", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    field.save(a.out / "model.pt")
    keep = ("psnr", "mae", "ssim")
    (a.out / "result.json").write_text(json.dumps({
        "method": a.method, "model": a.model, "volume": a.volume, "gaussians": len(field),
        "conversion": "exact (alpha N)" if a.method == "gaussianpile" else "opacity x intensity + global LS scale",
        "own": {k: own[k] for k in keep if k in own}, "converted": {k: before[k] for k in keep},
        "after_phase2": {k: after[k] for k in keep}, "phase2_seconds": seconds, "iterations": a.iterations,
        "calibrated": calibrated, "trace_psnr": [(r["iteration"], r["psnr"]) for r in trace]}, indent=1))


if __name__ == "__main__":
    main()
