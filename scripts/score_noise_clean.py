"""Score the CREMI A noise runs (runs/noise/cremi_a_n*) against the clean volume, and against the noisy one.

    micromamba run -n gaussian-volume python scripts/score_noise_clean.py

The noisy volumes were made from the min-max normalised clean CREMI A by adding Gaussian noise and clipping to [0, 1]
(data/cremi_noise/provenance.json); they span exactly 0..1, so they normalise to the clean intensity scale and the
clean reference is data/cremi_noise/cremi_a_clean_normalised.npy. Writes runs/noise/clean_scores.json.
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
SPACING = (40.0, 4.0, 4.0)


def psnr(a, b):
    return float(-10 * torch.log10(((a - b) ** 2).mean()))


def main():
    from slicegaussian import load_volume
    from slicegaussian.field import GaussianField, render_volume
    clean = torch.from_numpy(np.load(ROOT / "data/cremi_noise/cremi_a_clean_normalised.npy")).float().cuda()
    out = {}
    for n in ("0.02", "0.05"):
        vol = load_volume(str(ROOT / f"data/cremi_noise/cremi_a_n{n}.tif"), SPACING)
        noisy = vol.gt
        out[n] = {"noisy_vs_clean": psnr(noisy, clean)}
        for d in sorted((ROOT / f"runs/noise/cremi_a_n{n}").iterdir()):
            if not d.is_dir():
                continue
            with torch.no_grad():
                if (d / "final.json").exists():                      # SliceGaussian
                    pred = render_volume(GaussianField.load(d / "model.pt"), vol)
                    count = json.loads((d / "final.json").read_text())["gaussians"]
                elif (d / "metrics.json").exists() and (d / "model.pt").exists():   # 3DGS
                    mod = importlib.import_module("final_experiments_3dgs")
                    mod.fg.SPACING = SPACING
                    g = mod.Model3DGS.from_state(torch.load(d / "model.pt"))
                    pred = mod.render_stack(g, vol.Z, vol.dz, vol.dx, vol.H, vol.W)
                    count = json.loads((d / "metrics.json").read_text())["gaussians"]
                else:
                    continue
            pred = pred.clamp(0, 1)
            out[n][d.name] = {"gaussians": count, "psnr_vs_noisy": psnr(pred, noisy), "psnr_vs_clean": psnr(pred, clean)}
            r = out[n][d.name]
            print(f"noise {n} {d.name:11s} N {count:>9,}  vs noisy {r['psnr_vs_noisy']:6.2f}  vs clean {r['psnr_vs_clean']:6.2f}",
                  flush=True)
        print(f"noise {n} (the noisy input itself vs clean: {out[n]['noisy_vs_clean']:.2f} dB)")
    (ROOT / "runs/noise/clean_scores.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
