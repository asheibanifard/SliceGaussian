"""Table and slice figure for the BBBC032 comparison (seed 0): all methods scored from prediction.npy alike.

    micromamba run -n gaussian-volume python scripts/summarize_bbbc032.py [--slice K]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPACING = (500.0, 101.0, 101.0)
METHODS = [("gaussianpile", "GaussianPile"), ("3dgs", "3DGS"), ("neurcomp", "NeurComp"), ("coordnet", "CoordNet"),
           ("inif", "INIF"), ("slicegaussian", "Ours")]


def size_and_time(v, m):
    d = ROOT / f"runs/bbbc032/{v}/{m}/seed_0"
    if m == "slicegaussian":
        f = json.loads((d / "final.json").read_text())
        return f["gaussians"], None, f["total_seconds"]
    j = json.loads((d / "metrics.json").read_text())
    return j.get("gaussians"), j.get("parameters"), j["train_seconds"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slice", type=int, default=62)
    p.add_argument("--out", type=Path, default=ROOT / "runs/bbbc032/figure")
    a = p.parse_args()
    from slicegaussian import load_volume, metrics
    rows, preds, gts = [], {}, {}
    for v in ["nuclei", "membrane"]:
        vol = load_volume(str(ROOT / f"data/bbbc032/bbbc032_{v}.tif"), SPACING)
        gts[v] = vol.gt.cpu().numpy()
        for m, name in METHODS:
            pr = np.load(ROOT / f"runs/bbbc032/{v}/{m}/seed_0/prediction.npy").astype(np.float32).reshape(gts[v].shape)
            preds[v, m] = pr
            s = metrics.whole(None, vol, full=True, pred=torch.from_numpy(pr).cuda())
            g, n, t = size_and_time(v, m)
            rmse = float(np.sqrt(np.mean((np.clip(pr, 0, 1)[vol.valid_ids] - gts[v][vol.valid_ids]) ** 2)))
            rows.append({"volume": v, "method": name, "mae": s["mae"], "rmse": rmse, "ssim": s["ssim"],
                         "psnr": s["psnr"], "gaussians": g, "parameters": n, "seconds": t})
            print(rows[-1], flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "bbbc032_table.json").write_text(json.dumps(rows, indent=1))

    k = a.slice
    cols = [("gt", "Ground truth")] + [(m, n) for m, n in METHODS]
    fig, ax = plt.subplots(2, len(cols), figsize=(2.0 * len(cols), 4.5), gridspec_kw=dict(wspace=0.03, hspace=0.16))
    for r, v in enumerate(["nuclei", "membrane"]):
        gt = gts[v][k]
        for c, (m, name) in enumerate(cols):
            img = gt if m == "gt" else np.clip(preds[v, m][k], 0, 1)
            ax[r, c].imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            if m != "gt":
                psnr = -10 * np.log10(np.mean((img - gt) ** 2))
                ax[r, c].set_xlabel(f"{psnr:.2f} dB", fontsize=15, labelpad=2)
            if r == 0:
                ax[0, c].set_title(name, fontsize=16)
        ax[r, 0].set_ylabel({"nuclei": "Nuclei", "membrane": "Membrane"}[v], fontsize=16)
    for x in ax.ravel():
        x.set_xticks([]), x.set_yticks([])
    fig.savefig(a.out / "fig_bbbc032_slices.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(a.out / "fig_bbbc032_slices.png", bbox_inches="tight", pad_inches=0.02, dpi=110)


if __name__ == "__main__":
    main()
