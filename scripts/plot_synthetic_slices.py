"""Slices of the synthetic volumes used to test Phase I's directional wavelength (supplement, Sec. S2).

    python scripts/plot_synthetic_slices.py --out paper-cvpr2027/figures/fig_synthetic_slices.pdf

The volumes are built with the exact formulas and geometry of scripts/wavelength_diagnostics.py (CREMI geometry:
125 x 256 x 256 voxels at 40 x 4 x 4 nm). In-plane panels show the central slice (x-y). Panels marked x-z show a
vertical cut through the centre, cropped to the central 1,024 nm along z so that both axes are at the same physical
scale; the 40 nm slice spacing is visible there as horizontal bands.
"""
import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wd", ROOT / "scripts/wavelength_diagnostics.py")
wd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wd)                                  # module-level grid only; main() is not run
X, Y, Z, C = wd.X, wd.Y, wd.Z, wd.CENTRE
proj, inplane = wd.proj, wd.inplane
NZ, NY, NX = wd.SHAPE
DZ, DY, DX = wd.SPACING
ZC = NZ // 2                                                 # central slice


def sheets(d, w, period=256.0):
    r = (d + period / 2) % period - period / 2
    return np.exp(-0.5 * (r / w) ** 2)


def volumes():
    R = np.sqrt((X - C[0]) ** 2 + (Y - C[1]) ** 2 + (Z - C[2]) ** 2)
    Rc = np.sqrt((X - C[0]) ** 2 + (Y - C[1]) ** 2)
    n30 = inplane(30)
    d1, d2 = np.abs(proj(inplane(0))), np.abs(proj(inplane(60)))
    membrane = sheets(proj(n30), 8)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(wd.SHAPE)
    # (family letter, title, parameters, volume, view)
    return [
        ("A", "Plane sinusoid", r"$\lambda$ = 64 nm, 30°", 0.5 + 0.5 * np.cos(2 * np.pi * proj(n30) / 64), "xy"),
        ("A", "Plane sinusoid along $z$", r"$\lambda$ = 320 nm", 0.5 + 0.5 * np.cos(2 * np.pi * proj([0, 0, 1]) / 320), "xz"),
        ("B", "Cylindrical sinusoid", r"$\lambda$ = 64 nm, axis $z$", 0.5 + 0.5 * np.cos(2 * np.pi * Rc / 64), "xy"),
        ("B", "Spherical sinusoid", r"$\lambda$ = 64 nm", 0.5 + 0.5 * np.cos(2 * np.pi * R / 64), "xz"),
        ("C", "Crossing sinusoids", r"$\lambda$ = 64 nm, 60°",
         0.5 + 0.25 * np.cos(2 * np.pi * proj(inplane(0)) / 64) + 0.25 * np.cos(2 * np.pi * proj(inplane(60)) / 64), "xy"),
        ("D", "Planar membranes", r"$w$ = 8 nm, 30°", membrane, "xy"),
        ("D", "Cylindrical membrane", r"$w$ = 8 nm, $R$ = 64 nm", np.exp(-0.5 * ((Rc - 64) / 8) ** 2), "xy"),
        ("D", "Crossing membranes", r"$w$ = 8 nm, 60°",
         np.maximum(np.exp(-0.5 * (d1 / 8) ** 2), np.exp(-0.5 * (d2 / 8) ** 2)), "xy"),
        ("E", "Noisy membranes", "noise s.d. 0.02", membrane + 0.02 * noise, "xy"),
        ("E", "Noisy membranes", "noise s.d. 0.05", membrane + 0.05 * noise, "xy"),
        ("E", "Noisy membranes", "noise s.d. 0.1", membrane + 0.1 * noise, "xy"),
        ("D", "Planar membranes", r"$w$ = 16 nm, 0°", sheets(proj(inplane(0)), 16), "xy"),
    ]


def cut(v, view):
    if view == "xy":
        return v[ZC], (0, NX * DX, NY * DY, 0)
    half = int(round(NX * DX / DZ / 2))                       # slices covering the same physical length as x
    sl = v[ZC - half:ZC + half, NY // 2, :]
    return sl, (0, NX * DX, 2 * half * DZ, 0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "paper-cvpr2027/figures/fig_synthetic_slices.pdf")
    p.add_argument("--dpi", type=int, default=1000)
    a = p.parse_args()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "font.size": 7})
    panels = volumes()
    order = [0, 1, 2, 3, 4, 5, 11, 6, 7, 8, 9, 10]           # rows: sinusoids, sinusoids/membranes, membranes/noise
    fig, axes = plt.subplots(3, 4, figsize=(6.9, 5.6), constrained_layout=True)
    for ax, i in zip(axes.flat, order):
        fam, title, par, v, view = panels[i]
        img, ext = cut(v, view)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest")
        ax.set_title(f"({fam}) {title}\n{par}" + (" · $x$–$z$ cut" if view == "xz" else ""), fontsize=9, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
        x0, y0 = ext[1] - 50 - 200, ext[2] - 45               # 200 nm scale bar on a dark backing, bottom right
        ax.add_patch(plt.Rectangle((x0 - 30, y0 - 190), 260, 225, color="black", alpha=0.75, lw=0, zorder=3))
        ax.plot([x0, x0 + 200], [y0, y0], color="white", lw=1.4, solid_capstyle="butt", zorder=4)
        ax.text(x0 + 100, y0 - 40, "200 nm", color="white", ha="center", va="bottom", fontsize=7.5, zorder=4)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi)
    print(f"wrote {a.out} at {a.dpi} dpi")


if __name__ == "__main__":
    main()
