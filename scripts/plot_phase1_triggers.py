"""Where Phase I triggers subdivision on the synthetic volumes of Sec. S2: each parent on the shown slice, coloured by the
number of children Phase I gives it (n_1 n_2 n_3; 1 = not subdivided, left uncoloured), over the slice itself.

    python scripts/plot_phase1_triggers.py --out paper-cvpr2027/figures/fig_phase1_triggers.pdf

Phase I runs unchanged with the settings of configs/paper.json (cycle 0, before any optimisation). Parents lie on a
grid of 40 nm; the panels show the parents whose centre lies on the shown slice (x-y panels) or on the grid row nearest
the shown cut (x-z panels). Volumes, slices and panel order are those of scripts/plot_synthetic_slices.py.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
# sequential blue ramp (validated default palette, light steps 200 -> 650); one colour per children-count bin
RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281"]
BINS = [2, 3, 5, 9, 17, 33, 65]                    # bins: 2 | 3-4 | 5-8 | 9-16 | 17-32 | 33-64
BIN_LABELS = ["2", "3–4", "5–8", "9–16", "17–32", "33–64"]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "paper-cvpr2027/figures/fig_phase1_triggers.pdf")
    p.add_argument("--config", default=str(ROOT / "configs/paper.json"))
    p.add_argument("--dpi", type=int, default=1000)
    a = p.parse_args()
    from slicegaussian import phase1
    from slicegaussian.volume import Volume
    ps = _load("plot_synthetic_slices")
    cfg = phase1.P1Config(**json.load(open(a.config))["p1"])
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "font.size": 7})
    cmap = ListedColormap(RAMP)
    norm = BoundaryNorm(BINS, cmap.N)
    panels = ps.volumes()
    order = [0, 1, 2, 3, 4, 5, 11, 6, 7, 8, 9, 10]
    fig, axes = plt.subplots(3, 4, figsize=(6.9, 6.1), constrained_layout=True)
    summary = []
    for ax, i in zip(axes.flat, order):
        fam, title, par, v, view = panels[i]
        vol = Volume(v.astype(np.float32), ps.wd.SPACING, name="synthetic", detect_invalid=False)
        init, stats, _ = phase1.run(vol, cfg)
        n = init["n"].cpu().numpy()
        # parent centres in the order used by run(): recompute the grid (identical to phase1.parents)
        h = cfg.parent_spacing * min(vol.spacing())
        xs = np.arange(h / 2, vol.W * vol.dx, h)
        ys = np.arange(h / 2, vol.H * vol.dy, h)
        zs = np.arange(h / 2, vol.Z * vol.dz, h)
        X, Y, Zc = np.meshgrid(xs, ys, zs, indexing="ij")
        P = np.stack([X.ravel(), Y.ravel(), Zc.ravel()], 1)
        assert len(P) == len(n), (len(P), len(n))       # every grid point is inside the synthetic volumes
        N = n.prod(1)
        img, ext = ps.cut(v, view)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest")
        if view == "xy":
            zc = (ps.ZC + 0.5) * vol.dz
            sel = np.abs(P[:, 2] - zc) < h / 2
            cx, cy = P[sel, 0], P[sel, 1]
        else:
            yc = (ps.NY // 2 + 0.5) * vol.dy
            yrow = ys[np.argmin(np.abs(ys - yc))]
            z0 = (ps.ZC - int(round(ps.NX * ps.DX / ps.DZ / 2))) * vol.dz    # top of the cropped cut
            sel = (np.abs(P[:, 1] - yrow) < 1e-6) & (P[:, 2] >= z0) & (P[:, 2] < z0 + ext[2])
            cx, cy = P[sel, 0], P[sel, 2] - z0
        Ns = N[sel]
        for x, y, k in zip(cx, cy, Ns):
            if k > 1:
                ax.add_patch(Rectangle((x - h / 2, y - h / 2), h, h, facecolor=cmap(norm(k)), alpha=0.55, lw=0))
        frac = float((Ns > 1).mean())                # share of the parents on the shown slice
        frac_vol = float((N > 1).mean())
        summary.append({"panel": f"{fam} {title} {par}", "subdivided_share_slice": frac,
                        "subdivided_share_volume": frac_vol, "gaussians": stats["gaussians"],
                        "parents": stats["parents"]})
        ax.set_title(f"({fam}) {title}\n{par}" + (" · $x$–$z$ cut" if view == "xz" else ""), fontsize=8, pad=3)
        ax.text(0.03, 0.97, f"{100 * frac:.0f}% of parents split", transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
                color="white", bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.7))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
        print(f"{title:26s} {par:26s} parents split {100 * frac:5.1f}%   Gaussians {stats['gaussians']:,}", flush=True)
        del vol, init
        torch.cuda.empty_cache()
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.035, pad=0.015, aspect=45,
                      ticks=[(BINS[k] + BINS[k + 1]) / 2 for k in range(len(RAMP))])
    cb.ax.set_xticklabels(BIN_LABELS)
    cb.set_label("children per parent given by Phase I, before any optimisation (uncoloured: parent not subdivided)")
    cb.outline.set_linewidth(0.4)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi)
    (a.out.with_suffix(".json")).write_text(json.dumps(summary, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
