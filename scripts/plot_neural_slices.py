"""Slices of every method on EPFL CA1 and SNEMI3D: reconstruction and difference from the measured volume.

    micromamba run -n gaussian-volume python scripts/plot_neural_slices.py --out Journal/figures/fig_neural_slices.pdf

Every reconstruction is the saved prediction of the run reported in the paper (runs/neural/<volume>/<method>/seed_0,
seed 0). Each is re-scored on the whole volume and must reproduce its recorded PSNR within 0.02 dB. Image tiles are
embedded at >= 1000 pixels per inch without interpolation (pixel replication).
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
VOLUMES = [("EPFL CA1", "epfl", "data/epfl/epfl_ca1.tif", (5.0, 5.0, 5.0), 62),
           ("SNEMI3D", "snemi3d", "data/snemi3d/snemi3d.tif", (29.0, 6.0, 6.0), 50)]
METHODS = [("SliceGaussian", "slicegaussian"), ("GaussianPile", "gaussianpile"), ("3DGS", "3dgs"),
           ("SAD-GS", "sadgs3d"), ("NeurComp", "neurcomp"), ("INIF", "inif"), ("CoordNet", "coordnet")]
DIFF = 0.15


def recorded(d, m):
    for name in ("final.json", "score.json", "metrics.json"):
        f = d / name
        if f.exists():
            x = json.loads(f.read_text())
            x = x.get("final") or x.get("train") or x
            if "psnr" in x:
                return x["psnr"]
    raise FileNotFoundError(d)


def up(img, inches, dpi=1000):
    k = int(np.ceil(inches * dpi / img.shape[1]))
    return np.kron(img, np.ones((k, k), img.dtype))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--preview", default="")
    a = p.parse_args()
    from slicegaussian import load_volume, metrics
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "serif", "font.serif": ["Times New Roman", "TeX Gyre Termes"],
                         "mathtext.fontset": "stix", "font.size": 7})
    cols = 1 + len(METHODS)
    W = 7.16
    t = (W - 0.30) / cols                                     # tile size (inches)
    block = 0.17 + 0.13 + t + 0.03 + t + 0.15                 # volume title, column titles, two rows, labels
    H = len(VOLUMES) * block + 0.04
    fig = plt.figure(figsize=(W, H))
    top = H - 0.02
    for title, key, path, spacing, k in VOLUMES:
        vol = load_volume(str(ROOT / path), spacing)
        gt = vol.gt[k].cpu().numpy()
        rows = [("Volume", gt, None)]
        for name, m in METHODS:
            d = ROOT / "runs/neural" / key / m / "seed_0"
            pred = torch.from_numpy(np.load(d / "prediction.npy").astype(np.float32).reshape(vol.shape)).cuda().clamp(0, 1)
            score, rec = metrics.whole(None, vol, full=False, pred=pred)["psnr"], recorded(d, m)
            print(f"{key:8s} {name:14s} {score:6.2f} (recorded {rec:6.2f})", flush=True)
            assert abs(score - rec) < 0.02, (key, m, score, rec)
            rows.append((name, pred[k].cpu().numpy(), score))
        fig.text(0.02 / W, (top - 0.14) / H, f"{title}, slice {k}", fontsize=8, fontweight="bold", va="bottom")
        ys = [top - 0.17 - 0.13 - t, top - 0.17 - 0.13 - 2 * t - 0.03]
        for r, y0 in enumerate(ys):
            for c, (name, img, score) in enumerate(rows):
                x0 = 0.28 + c * t
                ax = fig.add_axes([x0 / W, y0 / H, (t - 0.02) / W, (t - 0.02) / H])
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_linewidth(0.5)
                if r == 0:
                    ax.imshow(up(img, t), cmap="gray", vmin=0, vmax=1, interpolation="none")
                    ax.set_title(name, fontsize=7, pad=2, fontweight="bold" if name == "SliceGaussian" else "normal")
                elif score is None:
                    ax.axis("off")
                    ax.text(0.5, 0.5, f"colour range\n$\\pm${DIFF}", ha="center", va="center", fontsize=6.5,
                            transform=ax.transAxes)
                else:
                    ax.imshow(up(img - gt, t), cmap="RdBu_r", vmin=-DIFF, vmax=DIFF, interpolation="none")
                    sl = -10 * np.log10(np.mean((img - gt) ** 2))
                    ax.set_xlabel(f"{sl:.2f} dB", fontsize=6.5, labelpad=1.5)
            fig.text(0.16 / W, (y0 + t / 2) / H, ["reconstruction", "difference"][r], rotation=90, ha="center",
                     va="center", fontsize=6.5)
        top -= block
    fig.savefig(a.out, dpi=1000)
    if a.preview:
        fig.savefig(a.preview, dpi=200)


if __name__ == "__main__":
    main()
