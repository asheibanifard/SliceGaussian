"""Loss ablation and coefficient sweep with the raw-parameter CUDA kernels (runs/loss_fast), seeds 0-2.

    micromamba run -n gaussian-volume python scripts/summarize_loss_fast.py

Complete loss: runs/remeasure_fast/<volume>/seed_<s> (CREMI A/B/C seeds 0-2, EPFL seed 0) and runs/loss_fast/epfl/full
(EPFL seeds 1-2). Per run: PSNR, MAE, SSIM, final count, fitting time, P99 |alpha|, standard deviation of the PSNR over
the last 2,000 iterations, PSNR against the volume smoothed by one voxel, and beta (share of the noise reproduced).
Writes runs/loss_fast/summary.json (per run) and prints mean +- std over seeds.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "runs/loss_fast"
VOLS = {"cremi_a": ("data/cremi/cremi_a.tif", (40.0, 4.0, 4.0)), "cremi_b": ("data/cremi/cremi_b.tif", (40.0, 4.0, 4.0)),
        "cremi_c": ("data/cremi/cremi_c.tif", (40.0, 4.0, 4.0)), "epfl": ("data/epfl/epfl_ca1.tif", (5.0, 5.0, 5.0))}
ABLATION = ["vox", "vox_grad", "vox_multi", "vox_reg", "full"]
SWEEP = ["grad0.3", "grad1", "multi0.3", "multi1", "reg0.3", "reg1"]


def run_dir(vol, variant, seed):
    if variant == "full":
        if vol == "epfl" and seed > 0:
            return R / "epfl/full" / f"seed_{seed}"
        return ROOT / "runs/remeasure_fast" / vol / f"seed_{seed}"
    return R / vol / variant / f"seed_{seed}"


def measure(d, vol, V, D):
    from slicegaussian.field import GaussianField, render_volume
    f = json.loads((d / "final.json").read_text())
    last = f["iteration"] - 2000
    tr = [float(r["psnr"]) for r in csv.DictReader(open(d / "trace.csv")) if r["psnr"] and int(r["iteration"]) > last]
    st = torch.load(d / "model.pt", map_location="cuda")
    field = GaussianField.load(d / "model.pt")
    with torch.no_grad():
        P = render_volume(field, vol).clamp(0, 1)
    n = V - D
    return {"psnr": f["psnr"], "mae": f["mae"], "ssim": f["ssim"], "gaussians": f["gaussians"],
            "seconds": f["total_seconds"], "p99_amp": float(st["amp"].abs().float().quantile(0.99)),
            "psnr_std_last2k": float(np.std(tr)),
            "psnr_smoothed": float(-10 * torch.log10(((P - D) ** 2).mean())),
            "beta": float(((P - D) * n).sum() / (n ** 2).sum())}


def main():
    from slicegaussian import load_volume
    from slicegaussian.volume import blur3d
    rows = []
    for vol, (path, sp) in VOLS.items():
        volume = load_volume(str(ROOT / path), sp)
        V = volume.gt
        D = blur3d(V[None], (1.0, 1.0, 1.0))[0]
        variants = ABLATION + (SWEEP if vol in ("cremi_a", "epfl") else [])
        for k in variants:
            for s in range(3):
                d = run_dir(vol, k, s)
                if not (d / "final.json").exists():
                    print("missing", d)
                    continue
                r = {"volume": vol, "variant": k, "seed": s, **measure(d, volume, V, D)}
                rows.append(r)
        del volume, V, D
        torch.cuda.empty_cache()
    (R / "summary.json").write_text(json.dumps(rows, indent=1))
    keys = ["psnr", "mae", "ssim", "gaussians", "seconds", "p99_amp", "psnr_std_last2k", "psnr_smoothed", "beta"]
    for vol in VOLS:
        print(f"\n== {vol}")
        for k in ABLATION + SWEEP:
            sel = [r for r in rows if r["volume"] == vol and r["variant"] == k]
            if not sel:
                continue
            out = []
            for key in keys:
                a = np.array([r[key] for r in sel], float)
                out.append(f"{key} {a.mean():.4g}±{a.std(ddof=1) if len(a) > 1 else 0:.2g}")
            print(f"{k:10s} n={len(sel)} " + "  ".join(out))


if __name__ == "__main__":
    main()
