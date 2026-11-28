"""Answer to "Does Phase I estimate the local wavelength correctly?" (supplement, Sec. S2), drawn from the results of
scripts/wavelength_diagnostics.py.

    python scripts/plot_wavelength_answer.py --csv runs/wavelength_diag/wavelength_diagnostics.csv \
        --out paper-cvpr2027/figures/fig_wavelength_answer.pdf

(a) Sinusoids: estimate over truth (correct: 1). (b) Membranes: Gaussian size requested across the membrane over the
membrane's standard deviation, s*/w (correct: the same value for every membrane). (c) Empty background between noisy
membranes: estimated wavelength against the 200 nm below which a 20 nm parent is split (correct: no splitting).
Dots: median over the evaluated voxels; bars: 10th-90th percentile. Where a row pools orientations, the dot is the
median of the medians and the bar spans the lowest 10th to the highest 90th percentile. Orange marks the cases the
text reports as biased, each with its reason.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OK, BAD = "#2a78d6", "#eb6834"            # validated categorical pair (light surface)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(path):
    rows = defaultdict(list)
    for r in csv.DictReader(open(path)):
        rows[(r["experiment"], r["case"], r["quantity"])].append(r)
    return rows


def pick(rows, exp, case_prefix, quantity, scale=1.0, exact=False):
    """Pooled (median, p10, p90) over the rows whose case starts with (or, if exact, equals) case_prefix."""
    sel = [r for (e, c, q), rs in rows.items() if e == exp and (c == case_prefix if exact else c.startswith(case_prefix))
           and q == quantity for r in rs]
    assert sel, (exp, case_prefix, quantity)
    m = np.median([float(r["median"]) for r in sel]) * scale
    return m, min(float(r["p10"]) for r in sel) * scale, max(float(r["p90"]) for r in sel) * scale


def dotplot(ax, items, xlabel, ref=None, band=None, band_label=None, log=False, xlim=None):
    n = len(items)
    ys = np.arange(n)[::-1]
    if band:
        ax.axvspan(*band, color=OK, alpha=0.10, lw=0, zorder=0)
        if band_label:
            ax.text(band[1], n - 0.1, " " + band_label, ha="left", va="bottom", fontsize=6.3, color=MUTED)
    if ref is not None:
        ax.axvline(ref, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    group_prev = None
    for y, it in zip(ys, items):
        label, (m, lo, hi), note, group = it
        c = BAD if note else OK
        ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(m, y, "o", ms=4.5, color=c, mec="white", mew=0.8, zorder=3)
        if note:
            right = (np.log(hi) if log else hi) > (np.log(xlim[0]) + 0.55 * (np.log(xlim[1]) - np.log(xlim[0])) if log
                                                    else xlim[0] + 0.55 * (xlim[1] - xlim[0]))
            if right:
                ax.annotate(note, (m, y), xytext=(0, 4), textcoords="offset points", va="bottom", ha="center",
                            fontsize=6.3, color=INK)
            else:
                ax.annotate(note, (hi, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=6.3,
                            color=INK)
        if group != group_prev and group_prev is not None:
            ax.axhline(y + 0.5, color=GRID, lw=0.8, zorder=0)
        group_prev = group
    ax.set_yticks(ys)
    ax.set_yticklabels([it[0] for it in items])
    ax.set_ylim(-0.6, n - 0.4 + 0.9)
    if log:
        ax.set_xscale("log")
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    style(ax)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelcolor=INK, width=0.6, length=2.5)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=ROOT / "runs/wavelength_diag/wavelength_diagnostics.csv")
    p.add_argument("--out", type=Path, default=ROOT / "paper-cvpr2027/figures/fig_wavelength_answer.pdf")
    p.add_argument("--dpi", type=int, default=1000)
    a = p.parse_args()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5})
    R = load(a.csv)
    ratio = lambda exp, case, q, truth: pick(R, exp, case, q, 1.0 / truth)

    sinus = [(f"(A) in-plane, {l} nm", ratio("A", f"in-plane lambda={l} ", "lambda along n", l),
              "beyond the scales" if l == 256 else None, "A1") for l in (16, 32, 64, 128, 256)]
    sinus += [(f"(A) along $z$, {l} nm", ratio("A", f"along z lambda={l}", "lambda along n", l),
               "beyond the scales" if l > 160 else None, "A2") for l in (160, 320, 640)]
    sinus += [(f"(B) cylinder, $R/\\lambda$ {lab}", ratio("B", f'cylinder lambda=64 R/lambda in [{lo}', "radial lambda", 64),
               None, "B1") for lo, lab in (("0.5", "0.5–1"), ("1", "1–2"), ("2", "2–4"))]
    sinus += [("(B) sphere, in-plane", ratio("B", "sphere lambda=64 equatorial", "radial lambda", 64), None, "B2"),
              ("(B) sphere, along $z$", ratio("B", "sphere lambda=64 polar", "radial lambda", 64),
               "below the axial Nyquist limit", "B2")]
    sinus += [(f"(C) crossing at {phi}°", ratio("C", f"crossing phi={phi} ", "lambda along n1", 64),
               "averaged: too long" if phi > 30 else None, "C") for phi in (30, 60, 90)]

    memb = [(f"(D) planar, $w$ = {w} nm", pick(R, "D", f"planar w={w} ", "s*/w across"), None, "D1") for w in (4, 8, 16)]
    memb += [(f"(D) cylinder, $R$ = {r} nm", pick(R, "D", f"cylinder w=8 R={r}", "s*/w across"), None, "D2")
             for r in (16, 32, 64, 128)]
    memb += [(f"(D) crossing at {phi}°, near", pick(R, "D", f"X-crossing w=8 phi={phi} near", "s*/w across n1"),
              "too large" if phi > 30 else None, "D3") for phi in (30, 60, 90)]
    memb += [("(D) crossing, far from it", pick(R, "D", "X-crossing w=8 phi=", "s*/w across n1"), None, "D3")]
    # the pooled "far" row must use only the far cases
    far = [r for (e, c, q), rs in R.items() if e == "D" and c.endswith("far") for r in rs]
    memb[-1] = (memb[-1][0], (np.median([float(r["median"]) for r in far]), min(float(r["p10"]) for r in far),
                              max(float(r["p90"]) for r in far)), None, "D3")
    memb += [(f"(E) on membrane, noise {sd.rstrip('.0') or '0'}", pick(R, "E", f"noise sd={sd}", "s*/w across (membrane)", exact=True), None, "E")
             for sd in ("0.0", "0.02", "0.05", "0.1")]
    planar = [m for lab, (m, lo, hi), _, g in memb if g == "D1"]

    fig = plt.figure(figsize=(6.9, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.85, 0.55], wspace=1.05, left=0.175, right=0.985, top=0.83,
                          bottom=0.13)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))
    dotplot(ax1, sinus, r"estimate / truth, $\widehat{\lambda}/\lambda$ (log)", ref=1.0, band=(0.9, 1.1),
            band_label="±10%", log=True, xlim=(0.3, 12))
    ax1.set_xticks([0.3, 0.5, 1, 2, 5, 10])
    ax1.set_xticklabels(["0.3", "0.5", "1", "2", "5", "10"])
    ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_title("(a) Sinusoids (A–C):\nis the wavelength right?\ncorrect = 1", loc="left", color=INK)
    dotplot(ax2, memb, r"requested size / membrane s.d., $s^{*}/w$", band=(min(planar), max(planar)),
            band_label="planar range", xlim=(0.4, 0.95))
    ax2.set_title("(b) Membranes (D–E):\nis the requested size consistent?\ncorrect = the same for every membrane",
                  loc="left", color=INK)

    sds = ("0.0", "0.02", "0.05", "0.1")
    x = np.arange(len(sds))
    along = [pick(R, "E", f"noise sd={sd}", "background lambda along t (nm)", exact=True) for sd in sds]
    across = [pick(R, "E", f"noise sd={sd}", "background lambda along n (nm)", exact=True) for sd in sds]
    ax3.axhspan(10, 200, color=BAD, alpha=0.08, lw=0, zorder=0)
    ax3.axhline(200, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax3.text(x[-1] + 0.4, 215, "200 nm:\nsplit threshold", ha="right", va="bottom", fontsize=6.3, color=MUTED)
    ax3.text(x[0] - 0.3, 28, "spurious\nsplitting", ha="left", va="center", fontsize=6.3, color=BAD)
    ax3.plot(x, [c[0] for c in across], color="#a8a7a2", lw=0.8, marker="o", ms=3, zorder=1)
    ax3.text(x[0] - 0.3, across[0][0] * 0.72, "across: set by\nmembrane spacing", fontsize=5.8, color=MUTED,
             va="top")
    ax3.plot(x, [c[0] for c in along], color=MUTED, lw=0.8, zorder=1)
    for xi, (mi, l, h) in zip(x, along):
        c = OK if mi >= 200 else BAD
        ax3.plot([xi, xi], [l, h], color=c, lw=2, solid_capstyle="round", zorder=2)
        ax3.plot(xi, mi, "o", ms=4.5, color=c, mec="white", mew=0.8, zorder=3)
    ax3.text(x[0] + 0.12, along[0][0], "along the\nmembranes", fontsize=6.3, color=INK, va="center")
    ax3.set_yscale("log")
    ax3.set_ylim(20, 30000)
    ax3.set_yticks([20, 50, 200, 1000, 5000, 20000])
    ax3.set_yticklabels(["20", "50", "200", "1k", "5k", "20k"])
    ax3.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax3.set_xticks(x)
    ax3.set_xticklabels(["0", "0.02", "0.05", "0.1"])
    ax3.set_xlim(-0.4, len(sds) - 0.5)
    ax3.set_xlabel("noise s.d. (contrast 1)")
    ax3.set_ylabel(r"background $\widehat{\lambda}$ (nm, log)")
    ax3.set_title("(c) Background (E):\nnoise-made detail?\ncorrect = long", loc="left",
                  color=INK)
    style(ax3)
    ax3.grid(axis="y", color=GRID, lw=0.6)
    ax3.grid(axis="x", visible=False)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
