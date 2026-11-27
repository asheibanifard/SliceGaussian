"""Noise share beta and PSNR against the volume smoothed by one voxel, for runs not covered by analyse_noise_fit.py:
CoordNet and the matched-count baselines on EPFL CA1 and SNEMI3D. Same definitions as analyse_noise_fit.py
(D = 3D Gaussian smoothing with sigma = 1 voxel; beta = <P - D, V - D> / <V - D, V - D> on the valid slices).

    micromamba run -n gaussian-volume python scripts/noise_share_extra.py

Matched-count GaussianPile and 3DGS models are rendered (and saved as prediction.npy); each prediction must reproduce
its recorded PSNR within 0.02 dB. Writes runs/matched_count/noise_analysis.json.
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
VOLUMES = {"epfl": ("data/epfl/epfl_ca1.tif", (5.0, 5.0, 5.0)), "snemi3d": ("data/snemi3d/snemi3d.tif", (29.0, 6.0, 6.0))}


def psnr(a, b):
    return float(-10 * torch.log10(((a - b) ** 2).mean()))


@torch.no_grad()
def prediction(d, m, vol, spacing):
    f = d / "prediction.npy"
    if f.exists():
        return torch.from_numpy(np.load(f).astype(np.float32).reshape(vol.shape)).cuda()
    if m == "gaussianpile":
        mod = importlib.import_module("final_experiments_gaussianpile")
        mod.SPACING = spacing
        g = mod.GaussianModel.from_state(torch.load(d / "model.pt"))
        pred = mod.render_stack(g.means, g.scales, g.quats, g.alphas, mod.stack_cameras(vol.Z, vol.H, vol.W, spacing),
                                mod.SETTINGS)
    else:
        mod = importlib.import_module("final_experiments_3dgs")
        mod.fg.SPACING = spacing
        pred = mod.render_stack(mod.Model3DGS.from_state(torch.load(d / "model.pt")), vol.Z, vol.dz, vol.dx, vol.H, vol.W)
    np.save(f, pred.float().cpu().numpy())
    return pred.float()


def main():
    from slicegaussian import load_volume, metrics
    from slicegaussian.volume import blur3d
    out = {}
    for v, (path, spacing) in VOLUMES.items():
        vol = load_volume(str(ROOT / path), spacing)
        ids, V = vol.valid_ids, vol.gt
        D = blur3d(V[None], (1.0, 1.0, 1.0))[0]
        noise = (V - D)[ids]
        runs = {"coordnet": ROOT / f"runs/neural/{v}/coordnet/seed_0"}
        runs.update({f"{m}_matched": ROOT / f"runs/matched_count/{v}/{m}/seed_0" for m in ("gaussianpile", "3dgs", "sadgs3d")})
        out[v] = {}
        for name, d in runs.items():
            P = prediction(d, name.replace("_matched", ""), vol, spacing).clamp(0, 1)
            rec = json.loads((d / ("score.json" if name == "coordnet" else "metrics.json")).read_text())
            rec = (rec.get("final") or rec.get("train") or rec)["psnr"]
            got = metrics.whole(None, vol, full=False, pred=P)["psnr"]
            assert abs(got - rec) < 0.02, (v, name, got, rec)
            beta = float(((P[ids] - D[ids]) * noise).sum() / (noise ** 2).sum())
            out[v][name] = {"psnr_raw": got, "psnr_ref1.0": psnr(P[ids], D[ids]), "noise_reproduced": beta}
            print(f"{v:8s} {name:20s} raw {got:6.2f}  smoothed {out[v][name]['psnr_ref1.0']:6.2f}  beta {beta:.3f}", flush=True)
    (ROOT / "runs/matched_count/noise_analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
