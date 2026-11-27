"""Rate-distortion on CREMI A/B/C, EPFL CA1 and SNEMI3D,
from the measured, decoded archives of runs/rate_distortion.

    python scripts/plot_rate_distortion.py --out Journal/figures/fig_rate_distortion.pdf

Every point is a file that was written, decoded and scored (scripts/rd_measure_*.py). For each Gaussian method, stored with the
GaussianPile codec (scripts/rd_measure_gpcodec.py), the curve joins its non-dominated points over all settings
(SliceGaussian: tau_q; baselines: initial population) and codec precisions; for the three neural representations
together, the non-dominated points are shown unconnected.
Rate: bits per voxel of the file (the uncompressed 8-bit volume is 8 bpv). Prints a summary table.
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
VOLUMES = [
    ("cremi_a", "CREMI A", 125),
    ("cremi_b", "CREMI B", 125),
    ("cremi_c", "CREMI C (valid slices)", 125),
    ("epfl", "EPFL CA1", 125),
    ("snemi3d", "SNEMI3D", 100),
]
# fixed categorical order (validated, light surface); marker shapes as secondary encoding
SERIES = [
    ("slicegaussian", "SliceGaussian", "#257de0", "o"),
    ("gaussianpile", "GaussianPile", "#f47230", "s"),
    ("3dgs", "3DGS", "#00b894", "^"),
    ("hevc", "HEVC", "#f2a600", "D"),
    ("inr", "Neural (NeurComp, INIF, CoordNet)", "#ee7fac", "v"),
]
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e6e6e6"


def points(v, Z):
    """Gaussian methods: the GaussianPile codec (runs/rate_distortion/encoded_gpcodec); HEVC and neural
    representations: their own files (runs/rate_distortion/encoded)."""
    out = {}
    for f in glob.glob(str(ROOT / f"runs/rate_distortion/encoded/{v}/*/*/points.json")):
        for r in json.load(open(f)):
            if r.get("status") != "ok" or r["method"] in ("slicegaussian", "gaussianpile", "3dgs"):
                continue
            m = "inr" if r["method"] in ("neurcomp", "inif", "coordnet") else r["method"]
            out.setdefault(m, []).append((8 * r["encoded_bytes"] / (Z * 256 * 256), r["decoded_psnr"]))
    for f in glob.glob(str(ROOT / f"runs/rate_distortion/encoded_gpcodec/{v}/*/*/points.json")):
        for r in json.load(open(f)):
            out.setdefault(r["method"], []).append((8 * r["encoded_bytes"] / (Z * 256 * 256), r["decoded_psnr"]))
    return out


def front(p):
    f, best = [], -1e9
    for x, y in sorted(p):
        if y > best:
            f.append((x, y))
            best = y
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview", default="")
    a = ap.parse_args()
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "TeX Gyre Termes"],
            "mathtext.fontset": "stix",
            "font.size": 7.5,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig, axes = plt.subplots(1, 5, figsize=(7.16, 2.05), constrained_layout=True)
    for ax, (v, title, Z) in zip(axes, VOLUMES):
        P = points(v, Z)
        for key, name, col, mk in SERIES:
            if key not in P:
                continue
            f = front(P[key])
            ax.plot(
                *zip(*f),
                color=col,
                lw=0 if key == "inr" else 1.3,
                marker=mk,
                ms=3.6 if key == "inr" else 3.2,
                mec="white",
                mew=0.5,
                label=name,
                zorder=3,
            )
        ax.axvline(8, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(0.06, 120)
        ax.set_xticks([0.1, 1, 10, 100])
        ax.set_xticklabels(["0.1", "1", "10", "100"])
        ax.set_title(title, fontsize=7.5, color=INK, pad=3)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("bits per voxel", labelpad=1)
        ax.tick_params(labelsize=6.5)
        print(v, {k: [(round(x, 2), round(y, 2)) for x, y in front(p)] for k, p in P.items()})
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].text(8, axes[0].get_ylim()[0] + 0.3, " raw 8-bit", fontsize=6, color=MUTED, va="bottom")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=5, frameon=False, fontsize=7, handlelength=2.2)
    fig.savefig(a.out)
    if a.preview:
        fig.savefig(a.preview, dpi=220)


if __name__ == "__main__":
    main()
