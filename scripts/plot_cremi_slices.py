"""Slice comparison of the methods of Table 1 on one CREMI volume: reconstruction and difference from the ground truth.

    micromamba run -n gaussian-volume python scripts/plot_cremi_slices.py [--volume A] [--slice 62]

Every reconstruction is the one scored in Table 1 (seed 0): GaussianPile and 3DGS re-rendered from their saved models
(CREMI C: the valid-slice retrains), the INRs from their saved predictions, ours from the seed-0 model of the complete
method. Each is re-scored on the whole volume and printed next to its Table 1 value as a check.
Writes paper-cvpr2027/figures/fig_cremi_<v>_slices.pdf (and .png).
"""
import argparse
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT.parent / "freqpile"
RES = REF / "results/paper"
sys.path[:0] = [str(REF / "experiments"), str(REF / "standalone"), str(ROOT.parent / "gaussianpile")]
SPACING = (40.0, 4.0, 4.0)
TABLE1 = {"GaussianPile": (32.33, 31.11, 31.57), "3DGS": (37.43, 36.53, 36.72), "NeurComp": (27.72, 25.52, 24.80),
          "CoordNet": (35.30, 31.09, 30.96), "INIF": (28.69, 27.76, 26.65), "Ours": (39.05, 37.29, 39.65)}


def predictions(v, vol):
    from slicegaussian.field import GaussianField, render_volume
    l = v.lower()
    gp = (ROOT / f"runs/baselines_cremi/cremi_{l}/gaussianpile_validonly/seed_0/model.pt" if v == "C"
          else RES / f"CREMI_{v}/final_experiments_gaussianpile/seed_0/model.pt")
    g3 = (ROOT / f"runs/baselines_cremi/cremi_{l}/3dgs_validonly/seed_0/model.pt" if v == "C"
          else RES / f"CREMI_{v}/final_experiments_3dgs/seed_0/model.pt")
    out = {}
    with torch.no_grad():
        mod = importlib.import_module("final_experiments_gaussianpile")
        mod.SPACING = SPACING
        g = mod.GaussianModel.from_state(torch.load(gp))
        cams = mod.stack_cameras(vol.Z, vol.H, vol.W, SPACING)
        out["GaussianPile"] = mod.render_stack(g.means, g.scales, g.quats, g.alphas, cams, mod.SETTINGS)
        m3 = importlib.import_module("final_experiments_3dgs")
        m3.fg.SPACING = SPACING
        out["3DGS"] = m3.render_stack(m3.Model3DGS.from_state(torch.load(g3)), vol.Z, vol.dz, vol.dx, vol.H, vol.W)
        for name, d in (("NeurComp", "neurcomp"), ("CoordNet", "coordnet"), ("INIF", "inif")):
            p = np.load(RES / f"CREMI_{v}/{d}/seed_0/prediction.npy").astype(np.float32).reshape(vol.shape)
            out[name] = torch.from_numpy(p).cuda()
        f = GaussianField.load(RES / f"final-experiment-2/CREMI_{v}/slicegaussian/seed_0/model.pt")
        out["Ours"] = render_volume(f, vol)
    return {k: x.float().clamp(0, 1) for k, x in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--volume", default="A")
    p.add_argument("--slice", type=int, default=62)
    a = p.parse_args()
    from slicegaussian import load_volume, metrics
    v = a.volume
    vol = load_volume(str(ROOT / f"data/cremi/cremi_{v.lower()}.tif"), SPACING)
    preds = predictions(v, vol)
    for k, x in preds.items():
        print(f"{k:12s} whole-volume PSNR {metrics.whole(None, vol, full=False, pred=x)['psnr']:.2f}  "
              f"(Table 1: {TABLE1[k]['ABC'.index(v)]:.2f})", flush=True)
    gt = vol.gt[a.slice].cpu().numpy()
    cols = [("Ground truth", None)] + [(k, preds[k][a.slice].cpu().numpy()) for k in TABLE1]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                         "mathtext.fontset": "dejavuserif"})
    fig, ax = plt.subplots(2, len(cols), figsize=(2.0 * len(cols), 4.3), gridspec_kw=dict(wspace=0.03, hspace=0.18))
    for c, (name, img) in enumerate(cols):
        ax[0, c].imshow(gt if img is None else img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax[0, c].set_title(name, fontsize=15, fontweight="bold" if name == "Ours" else "normal")
        if img is None:
            ax[1, c].imshow(np.zeros_like(gt), cmap="RdBu_r", vmin=-0.1, vmax=0.1)
            ax[1, c].text(0.5, 0.5, "difference\n$\\pm$0.1", ha="center", va="center", fontsize=13,
                          transform=ax[1, c].transAxes)
        else:
            d = img - gt
            ax[1, c].imshow(d, cmap="RdBu_r", vmin=-0.1, vmax=0.1, interpolation="nearest")
            ax[1, c].set_xlabel(f"{-10 * np.log10(np.mean(d ** 2)):.2f} dB", fontsize=14, labelpad=2)
    for x in ax.ravel():
        x.set_xticks([]), x.set_yticks([])
    out = ROOT / f"paper-cvpr2027/figures/fig_cremi_{v.lower()}_slices"
    fig.savefig(f"{out}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{out}.png", bbox_inches="tight", pad_inches=0.02, dpi=100)


if __name__ == "__main__":
    main()
