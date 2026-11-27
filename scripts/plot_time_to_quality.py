"""PSNR against wall time on CREMI A / B / C: SliceGaussian (current implementation) and GaussianPile, fresh processes.

    python scripts/plot_time_to_quality.py --out Journal/figures/fig_time_quality.pdf

SliceGaussian: runs/wallclock_fast/cremi_<v>/slicegaussian; GaussianPile: runs/wallclock/cremi_<v>/gaussianpile (its code
is unchanged). Both evaluated every 100 iterations; the clock includes imports, loading, evaluations and re-estimation.
Also prints the first wall time at which each quality threshold is reached.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE = "#257de0", "#f47230"  # categorical slots 1 and 2 (validated, light surface)
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e6e6e6"
THRESHOLDS = (28, 30, 32, 34, 36, 38)


def curve(path):
    rows = [json.loads(l) for l in open(path / "progress.jsonl")]
    rows = [r for r in rows if "psnr" in r and "wall_seconds" in r]
    return [r["wall_seconds"] for r in rows], [r["psnr"] for r in rows]


def first(t, p, thr):
    return next((a for a, b in zip(t, p) if b >= thr), None)


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
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.15), sharey=True, constrained_layout=True)
    table = {}
    for ax, v in zip(axes, "abc"):
        sg = curve(ROOT / f"runs/wallclock_fast/cremi_{v}/slicegaussian")
        gp = curve(ROOT / f"runs/wallclock/cremi_{v}/gaussianpile")
        for (t, p), c in ((gp, ORANGE), (sg, BLUE)):
            ax.plot(t, p, color=c, lw=1.6, solid_joinstyle="round")
            ax.plot(t[-1], p[-1], "o", ms=4.5, color=c, mec="white", mew=1.0, zorder=5)
        ax.text(
            sg[0][-1] + 8, sg[1][-1] + 0.3, f"SliceGaussian\n{sg[1][-1]:.2f} dB, {sg[0][-1]:.0f} s", color=INK, fontsize=6.5, va="bottom", ha="left"
        )
        ax.text(gp[0][-1] + 8, gp[1][-1] - 0.4, f"GaussianPile\n{gp[1][-1]:.2f} dB,\n{gp[0][-1]:.0f} s", color=INK, fontsize=6.5, va="top", ha="left")
        ax.set_title(f"CREMI {v.upper()}" + (" (valid slices)" if v == "c" else ""), fontsize=8, color=INK)
        ax.set_xlim(0, 420)
        ax.set_ylim(20, 42)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("wall time (s)")
        table[v] = {thr: (first(*sg, thr), first(*gp, thr)) for thr in THRESHOLDS}
    axes[0].set_ylabel("PSNR (dB)")
    fig.savefig(a.out)
    if a.preview:
        fig.savefig(a.preview, dpi=220)
    for thr in THRESHOLDS:
        cells = [" / ".join("NR" if x is None else f"{x:.1f}" for x in table[v][thr]) for v in "abc"]
        print(thr, " | ".join(cells))


if __name__ == "__main__":
    main()
