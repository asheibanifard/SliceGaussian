"""Score the noise-study models (runs/noise/cremi_a_n*) against the clean CREMI A volume as well as the noisy input.

    python scripts/score_noise_clean.py [--root runs/noise]

The noisy volumes are the min-max normalised clean volume plus Gaussian noise, clipped to [0, 1], so the clean
reference (data/cremi_noise/cremi_a_clean_normalised.npy) is on the same intensity scale. Every model is rendered once
on all slices and scored with slicegaussian.metrics.scores against both references. Writes <root>/clean_scores.csv.
"""
import argparse
import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPACING = (40.0, 4.0, 4.0)


def render_slicegaussian(path, vol):
    from slicegaussian.field import GaussianField, render_volume
    return render_volume(GaussianField.load(path), vol)


def render_3dgs(path, vol):
    ref = ROOT.parent / "freqpile"
    sys.path[:0] = [str(ref / "experiments"), str(ref / "standalone"), str(ROOT.parent / "gaussianpile")]
    mod = importlib.import_module("final_experiments_3dgs")
    mod.fg.SPACING = SPACING
    s = {k: v.cuda() for k, v in torch.load(path, map_location="cpu").items()}
    model = mod.Model3DGS(s["xyz"], s["log_scale"], s["rot"], s["alpha_logit"], s["color"])
    with torch.no_grad():
        return mod.render_stack(model, vol.Z, vol.dz, vol.dx, vol.H, vol.W)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=ROOT / "runs" / "noise")
    a = p.parse_args()
    from slicegaussian import load_volume, metrics
    clean = torch.from_numpy(np.load(ROOT / "data" / "cremi_noise" / "cremi_a_clean_normalised.npy")).cuda()
    rows = []
    for vdir in sorted(a.root.glob("cremi_a_n*")):
        vol = load_volume(str(ROOT / "data" / "cremi_noise" / f"{vdir.name}.tif"), SPACING)
        noisy = vol.gt
        ids = vol.valid_ids
        data = metrics.scores(noisy[ids], clean[ids], vol)
        rows.append({"volume": vdir.name, "variant": "noisy input", "gaussians": "",
                     "psnr_clean": data["psnr"], "ssim_clean": data["ssim"], "mae_clean": data["mae"],
                     "psnr_noisy": "", "ssim_noisy": "", "mae_noisy": ""})
        for run in sorted(d for d in vdir.iterdir() if (d / "model.pt").exists()):
            is_3dgs = run.name == "3dgs"
            pred = (render_3dgs if is_3dgs else render_slicegaussian)(run / "model.pt", vol)
            c = metrics.scores(pred[ids], clean[ids], vol)
            n = metrics.scores(pred[ids], noisy[ids], vol)
            count = int(torch.load(run / "model.pt", map_location="cpu")["xyz"].shape[0])
            rows.append({"volume": vdir.name, "variant": run.name, "gaussians": count,
                         "psnr_clean": c["psnr"], "ssim_clean": c["ssim"], "mae_clean": c["mae"],
                         "psnr_noisy": n["psnr"], "ssim_noisy": n["ssim"], "mae_noisy": n["mae"]})
            print(f"{vdir.name:16} {run.name:12} G {count:>9,}  clean {c['psnr']:.2f} dB / {c['ssim']:.4f}  "
                  f"noisy {n['psnr']:.2f} dB", flush=True)
    with (a.root / "clean_scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
