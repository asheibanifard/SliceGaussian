"""Render the saved Gaussian models of the BBBC032 comparison to prediction.npy (Z, H, W) next to each model.

    micromamba run -n gaussian-volume python scripts/render_bbbc032_predictions.py

The INR baselines already write prediction.npy. Each rendered volume is re-scored with slicegaussian.metrics.whole and
must reproduce the score saved at the end of training.
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT.parent / "freqpile"
sys.path[:0] = [str(REF / "experiments"), str(REF / "standalone"), str(ROOT.parent / "gaussianpile")]
SPACING = (500.0, 101.0, 101.0)


def main():
    from slicegaussian import load_volume, metrics
    from slicegaussian.field import GaussianField, render_volume
    for v in ["nuclei", "membrane"]:
        vol = load_volume(str(ROOT / f"data/bbbc032/bbbc032_{v}.tif"), SPACING)
        for m in ["slicegaussian", "gaussianpile", "3dgs"]:
            d = ROOT / f"runs/bbbc032/{v}/{m}/seed_0"
            with torch.no_grad():
                if m == "slicegaussian":
                    pred = render_volume(GaussianField.load(d / "model.pt"), vol)
                    saved = json.loads((d / "final.json").read_text())
                elif m == "gaussianpile":
                    mod = importlib.import_module("final_experiments_gaussianpile")
                    mod.SPACING = SPACING
                    g = mod.GaussianModel.from_state(torch.load(d / "model.pt"))
                    cams = mod.stack_cameras(vol.Z, vol.H, vol.W, SPACING)
                    pred = mod.render_stack(g.means, g.scales, g.quats, g.alphas, cams, mod.SETTINGS)
                    saved = json.loads((d / "metrics.json").read_text())["train"]
                else:
                    mod = importlib.import_module("final_experiments_3dgs")
                    mod.fg.SPACING = SPACING
                    g = mod.Model3DGS.from_state(torch.load(d / "model.pt"))
                    pred = mod.render_stack(g, vol.Z, vol.dz, vol.dx, vol.H, vol.W)
                    saved = json.loads((d / "metrics.json").read_text())["train"]
            s = metrics.whole(None, vol, full=True, pred=pred)
            np.save(d / "prediction.npy", pred.float().cpu().numpy())
            print(f"{v} {m}: psnr {s['psnr']:.3f} (saved {saved['psnr']:.3f}), ssim {s['ssim']:.4f}", flush=True)


if __name__ == "__main__":
    main()
