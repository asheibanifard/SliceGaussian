"""Figure for the two-phase framework (main paper, Table 2): PSNR of each Phase I population before and after one
Phase II, CREMI A / B / C, as a dumbbell per estimator; for other methods' final Gaussians, also their own result.

    micromamba run -n gaussian-volume python scripts/plot_framework.py
Writes paper-cvpr2027/figures/fig_framework.pdf (and .png).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
BEFORE, AFTER, OWN = "#86b6ef", "#2a78d6", "#8a8984"              # blue 250 / blue 450 / neutral
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# (label, before A/B/C, after A/B/C, own result A/B/C or None); values from Table 2 and Table 1
N = None
ROWS = [                                   # CREMI A, B, C, EPFL, SNEMI3D
    ("Uniform grid", (22.73, 22.96, 23.40, N, N), (36.96, 35.29, 36.60, N, N), None),
    ("Random positions", (12.78, 11.49, 12.31, N, N), (37.83, 36.37, 38.11, N, N), None),
    ("Frequency-only", (26.28, 24.27, 25.23, N, N), (38.67, 37.24, 39.32, N, N), None),
    ("Structure-only", (23.04, 22.31, 21.75, N, N), (38.66, 36.61, 38.74, N, N), None),
    ("Phase I (ours)", (22.97, 21.95, 21.46, 14.81, 17.06), (38.80, 37.00, 39.29, 25.95, 32.92), None),
    ("GaussianPile, final", (32.31, 31.08, 31.55, 26.95, 32.21), (37.96, 36.81, 37.70, 27.46, 34.32),
     (32.33, 31.11, 31.57, 26.96, 32.22)),
    ("3DGS, final*", (13.61, 12.53, 12.87, 11.66, 12.75), (38.82, 37.85, 38.82, 28.06, 35.00),
     (37.43, 36.53, 36.72, 27.83, 34.24)),
    ("SAD-GS, final*", (12.65, 11.91, 12.18, 11.53, 12.58), (39.18, 38.25, 39.32, 28.31, 35.13),
     (37.68, 36.84, 37.03, 28.01, 34.43)),
]
TITLES = ["CREMI A", "CREMI B", "CREMI C (valid slices)", "EPFL CA1", "SNEMI3D"]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 8, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2,
                     "ytick.color": INK, "xtick.labelsize": 7.5, "ytick.labelsize": 8})


def main():
    fig, axs = plt.subplots(1, 5, figsize=(9.6, 2.35), sharey=True, gridspec_kw={"wspace": 0.06})
    n = len(ROWS)
    ys = [n - 1 - i + (0.0 if i < 5 else -0.5) for i in range(n)]       # gap between the two blocks
    for v, ax in enumerate(axs):
        for (lab, b, a, own), y in zip(ROWS, ys):
            if a[v] is None:
                continue
            ax.plot([b[v], a[v]], [y, y], color=AFTER, lw=1.0, alpha=0.55, zorder=1, solid_capstyle="round")
            ax.scatter(b[v], y, s=26, facecolor="white", edgecolor=BEFORE, linewidth=1.4, zorder=3)
            ax.scatter(a[v], y, s=30, color=AFTER, edgecolor="white", linewidth=0.8, zorder=4)
            if own:
                ax.scatter(own[v], y, s=30, marker="x", color=OWN, linewidth=1.3, zorder=5)
            if True:
                ax.annotate(f"{a[v]:.1f}", (a[v], y), xytext=(5, 0), textcoords="offset points", va="center",
                            fontsize=7, color=INK2)
        ax.axhline(ys[4] - 0.75, color=GRID, lw=0.8, zorder=0)
        ax.set_xlim(10, 45)
        ax.set_xticks([10, 20, 30, 40])
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(TITLES[v], fontsize=8.5, color=INK, pad=3)
        ax.set_xlabel("3D PSNR (dB)", fontsize=8, color=INK2, labelpad=1)
    axs[0].set_yticks(ys)
    axs[0].set_yticklabels([r[0] for r in ROWS])
    for t in axs[0].get_yticklabels():
        if "ours" in t.get_text():
            t.set_fontweight("bold")
    handles = [Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor=BEFORE, markeredgewidth=1.4,
                      label="Before Phase II (population as estimated)"),
               Line2D([], [], ls="", marker="o", color=AFTER, label="After one Phase II (30,000 iterations)"),
               Line2D([], [], ls="", marker="x", color=OWN, markeredgewidth=1.3, label="Method's own trained result")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 1.07), fontsize=7.5,
               handletextpad=0.3, columnspacing=1.4)
    out = ROOT / "paper-cvpr2027/figures/fig_framework"
    fig.savefig(f"{out}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{out}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)


if __name__ == "__main__":
    main()
