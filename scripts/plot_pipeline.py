"""Pipeline figure of the journal version (vector PDF; image tiles are embedded at >= 1000 dpi without interpolation).

python scripts/plot_pipeline.py --data <file.npz of pipeline_figure_data.py> --out Journal/figures/fig_pipeline.pdf
"""

import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

plt.rcParams.update(
    {"pdf.fonttype": 42, "font.family": "serif", "font.serif": ["Times New Roman", "TeX Gyre Termes"], "mathtext.fontset": "stix", "font.size": 7.5}
)
W, H = 7.16, 4.02
BLUE, ORANGE, GREEN, INK, GREY = "#1f5fa8", "#c2571a", "#2b7a4b", "#1a1a1a", "#6b6b6b"
TB, TO, TG = "#eef4fb", "#fdf2ea", "#ebf6ef"
AX = [BLUE, ORANGE, GREEN]  # principal axes u1, u2, u3


def up(img, size, dpi=1000):
    """Nearest-neighbour upsampling so that a tile printed `size` inches wide carries at least `dpi` pixels per inch."""
    return np.kron(img, np.ones((int(np.ceil(size * dpi / img.shape[1])),) * 2, img.dtype))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--preview", default="")
    a = p.parse_args()
    d = np.load(a.data)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def box(x, y, w, h, fc, ec, z=0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07", fc=fc, ec=ec, lw=0.9, zorder=z))

    def txt(x, y, s, size=7.5, c=INK, ha="center", va="center", w="normal", **k):
        ax.text(x, y, s, fontsize=size, color=c, ha=ha, va=va, fontweight=w, zorder=20, **k)

    def arrow(pts, c=INK, lw=1.1):
        pts = np.asarray(pts, float)
        if len(pts) > 2:
            ax.plot(pts[:-1, 0], pts[:-1, 1], c=c, lw=lw, solid_capstyle="butt", solid_joinstyle="miter", zorder=15)
        ax.add_patch(
            FancyArrowPatch(
                pts[-2], pts[-1], arrowstyle="-|>,head_length=3.2,head_width=1.8", color=c, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=1.0, zorder=15
            )
        )

    def tile(x, y, s, img, cmap="gray", vmin=0, vmax=1, ec=INK):
        ax.imshow(up(img, s), extent=(x, x + s, y, y + s), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none", zorder=5, aspect="auto")
        ax.add_patch(Rectangle((x, y), s, s, fc="none", ec=ec, lw=0.6, zorder=8))
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)

    r0, c0, n = 96, 96, 64  # crop (pixels) used wherever Gaussians are drawn
    crop = lambda v: v[r0 : r0 + n, c0 : c0 + n]
    g = d["slice"]
    vlo, vhi = np.percentile(g, [0.5, 99.5])

    def gaussians(x, y, s, m, C, wt, color, std=0.8, alpha=0.55, bg=0.0):
        """Cross-sections of the Gaussians with the slice, at `std` standard deviations, clipped to the tile."""
        clip = Rectangle((x, y), s, s, transform=ax.transData)
        sc = s / (n * 4.0)
        px = x + (m[:, 0] - c0 * 4) * sc
        py = y + s - (m[:, 1] - r0 * 4) * sc
        k = (px > x - 0.05) & (px < x + s + 0.05) & (py > y - 0.05) & (py < y + s + 0.05) & (wt > 0.6)
        ev, U = np.linalg.eigh(C[k])
        ev = np.sqrt(np.clip(ev, 0, None)) * 2 * std * sc
        ang = np.degrees(np.arctan2(-U[:, 1, 1], U[:, 0, 1]))
        ax.add_patch(Rectangle((x, y), s, s, fc="white", ec="none", zorder=4))
        e = EllipseCollection(
            ev[:, 1],
            np.maximum(ev[:, 0], 0.004),
            ang,
            units="x",
            offsets=np.c_[px[k], py[k]],
            transOffset=ax.transData,
            facecolors=color,
            edgecolors="white",
            linewidths=0.12,
            alpha=alpha,
            zorder=6,
        )
        ax.add_collection(e)
        e.set_clip_path(clip)
        ax.add_patch(Rectangle((x, y), s, s, fc="none", ec=INK, lw=0.6, zorder=8))

    # ---------------------------------------------------------------- top row
    top, bot = 2.16, 3.96
    S, ty = 1.0, 2.62  # tile size and lower edge, top row
    cap = 2.41  # caption line
    # input: sheared stack of real slices
    txt(0.60, 3.84, "Input volume", 8, w="bold")
    for i, im in enumerate(d["stack"]):
        x0, y0 = 0.07 + 0.055 * (4 - i), ty + 0.065 * (4 - i)
        tile(x0, y0, 0.80, im, vmin=vlo, vmax=vhi)
        for pch in ax.images[-1:]:
            pch.set_zorder(5 + i)
        ax.patches[-1].set_zorder(5 + i + 0.5)
    ax.add_patch(
        FancyArrowPatch(
            (0.91, 2.63),
            (1.12, 2.875),
            arrowstyle="<->,head_length=1.8,head_width=1.1",
            lw=0.6,
            color=GREY,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=1,
            zorder=12,
        )
    )
    txt(1.085, 2.70, "$z$", 7.5, GREY)
    txt(0.60, cap, "slices and spacing", 7)
    txt(0.60, cap - 0.13, r"$V,\ (h_x,h_y,h_z)$", 7)
    arrow([(1.14, ty + 0.5), (1.36, ty + 0.5)])

    box(1.28, top, 5.82, bot - top, TB, BLUE)
    txt(1.37, 3.84, "Phase I: estimate the population from the volume", 8.2, BLUE, ha="left", w="bold")
    txt(7.02, 3.84, "no Gaussian is optimised", 7.2, BLUE, ha="right", style="italic")
    X = [1.40, 2.56, 3.72, 4.88, 6.04]
    # 1 bands
    for i in range(3):
        b = d[f"band{i}"]
        v = np.percentile(abs(b), 99)
        s = 0.70
        x0, y0 = X[0] + 0.15 * i, ty + 0.30 - 0.15 * i
        tile(x0, y0, s, b, "RdBu_r", -v, v)
        ax.images[-1].set_zorder(5 + i)
        ax.patches[-1].set_zorder(5.5 + i)
    txt(X[0] + 0.5, cap, "(1) band-pass scale space", 7)
    txt(X[0] + 0.5, cap - 0.13, r"$D_s,\ E_s,\ w_s$", 7)
    # 2 orientation: tensor glyphs on the slice crop (thin along the strongest gradient direction)
    tile(X[1], ty, S, crop(g), vmin=vlo, vmax=vhi)
    J = d["J"]
    step = 8
    for r in range(r0 + step // 2, r0 + n, step):
        for c in range(c0 + step // 2, c0 + n, step):
            t = np.array([[J[0, r, c], J[1, r, c]], [J[1, r, c], J[2, r, c]]])
            ev, U = np.linalg.eigh(t)
            ratio = np.sqrt(max(ev[0], 1e-9) / max(ev[1], 1e-9))
            L = 0.105
            u = U[:, 0]  # u: along the structure
            ax.add_patch(
                Ellipse(
                    (X[1] + (c - c0 + 0.5) / n * S, ty + S - (r - r0 + 0.5) / n * S),
                    L,
                    L * max(ratio, 0.18),
                    angle=np.degrees(np.arctan2(-u[1], u[0])),
                    fc="#ffd24d",
                    ec="#5a3d00",
                    lw=0.35,
                    alpha=0.85,
                    zorder=7,
                )
            )
    txt(X[1] + 0.5, cap, "(2) structure tensor", 7)
    txt(X[1] + 0.5, cap - 0.13, r"$\overline{J}(x)\rightarrow u_{i1},u_{i2},u_{i3}$", 7)
    # 3 wavelength
    lam = d["lam"]
    lo, hi = np.percentile(lam, [1, 99])
    tile(X[2], ty, S, lam, "viridis_r", lo, hi)
    ax.imshow(np.linspace(0, 1, 256)[None], extent=(X[2] + 0.12, X[2] + 0.88, ty + 0.05, ty + 0.10), cmap="viridis_r", aspect="auto", zorder=9)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.add_patch(Rectangle((X[2] + 0.06, ty + 0.02), 0.88, 0.22, fc="white", alpha=0.82, ec="none", zorder=8.5))
    txt(X[2] + 0.12, ty + 0.165, f"{lo:.0f}", 6, ha="left")
    txt(X[2] + 0.88, ty + 0.165, f"{hi:.0f} nm", 6, ha="right")
    txt(X[2] + 0.5, cap, "(3) directional wavelength", 7)
    txt(X[2] + 0.5, cap - 0.13, r"$\lambda(x,u_{i1})\ \mathrm{shown}$", 7)
    # 4 candidate with its measured ratios: a real parent of this slice whose u1, u2 lie nearly in the slice
    mu, R, q, nn = d["mu"], d["R"], d["q"], d["n"]
    inside = (abs(mu[:, 2] - 2500) < 1) & (mu[:, :2] > 80).all(1) & (mu[:, :2] < 944).all(1)
    cand = np.where(inside & (nn == [3, 2, 1]).all(1) & (abs(R[:, 2, 2]) > 0.9))[0]
    i = cand[np.argmin(abs(q[cand, 0] - 0.25) + abs(q[cand, 1] - 0.15) + abs(q[cand, 2] - 0.05))]  # clearest of the real ones
    print("example parent", i, mu[i], q[i], nn[i], R[i], len(cand))
    cx, cy = mu[i, 0] / 4, mu[i, 1] / 4
    hw = 12
    rr, cc = int(round(cy - hw)), int(round(cx - hw))
    cr = g[rr : rr + 2 * hw, cc : cc + 2 * hw]
    s4 = 0.62
    x4, y4 = X[3], ty + 0.38
    sc = s4 / (2 * hw * 4)
    tile(x4, y4, s4, cr, vmin=vlo, vmax=vhi)
    ox, oy = x4 + (mu[i, 0] - cc * 4) * sc, y4 + s4 - (mu[i, 1] - rr * 4) * sc
    ax.add_patch(Ellipse((ox, oy), 40 * sc, 40 * sc, fc="#ffffff55", ec="white", lw=1.0, zorder=9))
    ax.add_patch(Ellipse((ox, oy), 80 * sc, 80 * sc, fc="none", ec="white", lw=0.7, ls=(0, (2, 1.5)), zorder=9))
    for kx in (0, 1):
        u = R[i][:, kx]
        e = np.array([u[0], -u[1]]) * 40 * sc
        arrow([(ox, oy), (ox + e[0], oy + e[1])], AX[kx], 1.2)
        ax.patches[-1].set_zorder(12)
        txt(
            ox + e[0] * 1.22,
            oy + e[1] * 1.22,
            f"$u_{{i{kx + 1}}}$",
            7.2,
            "white" if False else AX[kx],
            bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.8),
        )
    txt(x4 + s4 / 2, ty + S + 0.045 - 0.0, "", 6)
    bx, by, bh = X[3] + 0.68, ty + 0.42, 0.55  # measured ratios as bars
    for kx in range(3):
        hgt = q[i, kx] / 0.3 * bh
        ax.add_patch(Rectangle((bx + 0.11 * kx, by), 0.075, hgt, fc=AX[kx], ec="none", zorder=8))
        txt(bx + 0.0375 + 0.11 * kx, by - 0.06, f"${kx + 1}$", 6, AX[kx])
    ax.plot([bx - 0.03, bx + 0.33], [by, by], c=INK, lw=0.6, zorder=9)
    for m in (1, 2):
        ax.plot([bx - 0.03, bx + 0.31], [by + 0.1 * m / 0.3 * bh] * 2, c=INK, lw=0.6, ls=(0, (2, 1.2)), zorder=9)
    txt(bx + 0.31, by + 0.1 / 0.3 * bh + 0.045, r"$\tau_q$", 6, ha="right")
    txt(bx + 0.31, by + 0.2 / 0.3 * bh + 0.045, r"$2\tau_q$", 6, ha="right")
    txt(X[3] + 0.5, ty + 0.22, r"$\hat q_{ik}=P_{90}\{s_{ik}/\lambda(x,u_{ik})\}$", 6.9)
    txt(X[3] + 0.5, ty + 0.07, "over the support (dashed)", 6.4, GREY)
    txt(X[3] + 0.5, cap, "(4) directional violation", 7)
    txt(X[3] + 0.5, cap - 0.13, r"scale $s_{ik}$ against $\lambda(x,u_{ik})$", 7)
    # 5 subdivision in the u1-u2 frame
    ccx, ccy, rad = X[4] + 0.5, ty + 0.62, 0.31
    ax.add_patch(Ellipse((ccx, ccy), 2 * rad, 2 * rad, fc="none", ec=GREY, lw=0.8, ls=(0, (3, 2)), zorder=7))
    n1, n2 = nn[i, 0], nn[i, 1]
    for j1 in range(n1):
        for j2 in range(n2):
            t1, t2 = ((j1 + 0.5) / n1 - 0.5) * 2 * rad, ((j2 + 0.5) / n2 - 0.5) * 2 * rad
            ax.add_patch(Ellipse((ccx + t1, ccy + t2), 2 * rad / n1, 2 * rad / n2, fc=BLUE, alpha=0.35, ec=BLUE, lw=0.8, zorder=8))
    arrow([(ccx - 0.42, ccy - 0.42), (ccx - 0.14, ccy - 0.42)], AX[0], 1.0)
    txt(ccx - 0.09, ccy - 0.42, "$u_{i1}$", 7, AX[0])
    arrow([(ccx - 0.42, ccy - 0.42), (ccx - 0.42, ccy - 0.14)], AX[1], 1.0)
    txt(ccx - 0.42, ccy - 0.08, "$u_{i2}$", 7, AX[1])
    txt(ccx + 0.28, ccy - 0.42, f"$n_i=({n1},{n2},{nn[i, 2]})$", 7)
    txt(X[4] + 0.5, ty + 0.03, r"$n_{ik}=\lceil \hat q_{ik}/\tau_q\rceil\in[1,n_{\max}]$", 6.9)
    txt(X[4] + 0.5, cap, "(5) per-axis subdivision", 7)
    txt(X[4] + 0.5, cap - 0.13, r"children: scale $s_{ik}/n_{ik}$", 7)
    for q0 in range(4):
        arrow([(X[q0] + 1.025, ty + 0.56), (X[q0 + 1] - (-0.01 if q0 == 3 else 0.025), ty + 0.56)], BLUE, 1.0)

    # ---------------------------------------------------------------- connector and bottom row
    arrow([(X[4] + 0.5, top), (X[4] + 0.5, 2.06), (0.60, 2.06), (0.60, 1.83)], BLUE, 1.2)
    by0, S2, ty2, cap2 = 0.06, 1.0, 0.52, 0.33
    txt(0.60, 1.74, "Initial population", 8, w="bold")
    gaussians(0.10, ty2 + 0.06, S2 + 0.02, d["m0"], d["C0"], d["w0"], BLUE)
    txt(0.60, cap2 + 0.08, rf"$\Psi_0$, $M_0={int(d['count0']):,}$".replace(",", "{,}").replace("$\\Psi_0${,}", "$\\Psi_0$,"), 7)
    txt(0.60, cap2 - 0.05, "sections by one slice", 7, GREY)
    arrow([(1.14, 1.08), (1.36, 1.08)])

    box(1.28, by0, 3.10, 1.80, TO, ORANGE)
    txt(1.37, 1.74, "Phase II: optimise, population fixed", 8.2, ORANGE, ha="left", w="bold")
    s3 = 0.80
    yy = 0.76
    x1, x2 = 1.40, 2.50
    # field cut by the plane of a slice
    rng = np.random.default_rng(4)
    for _ in range(22):
        px, py = x1 + 0.08 + rng.random() * 0.66, yy + 0.04 + rng.random() * 0.72
        ax.add_patch(
            Ellipse(
                (px, py),
                0.07 + 0.10 * rng.random(),
                0.035 + 0.03 * rng.random(),
                angle=rng.random() * 180,
                fc=BLUE,
                alpha=0.28,
                ec=BLUE,
                lw=0.3,
                zorder=6,
            )
        )
    ax.add_patch(
        Polygon(
            [(x1 - 0.02, yy + 0.24), (x1 + 0.60, yy + 0.24), (x1 + 0.84, yy + 0.54), (x1 + 0.22, yy + 0.54)],
            fc=ORANGE,
            alpha=0.32,
            ec=ORANGE,
            lw=0.9,
            zorder=7,
        )
    )
    txt(x1 + 0.80, yy + 0.64, "one slice", 7.2, ORANGE)
    txt(x1 + 0.42, 0.63, "field evaluated", 7)
    txt(x1 + 0.42, 0.50, "on its plane", 7)
    arrow([(x1 + 0.87, yy + 0.4), (x2 - 0.04, yy + 0.4)], ORANGE)
    tile(x2, yy, s3, crop(d["pred0"]), vmin=vlo, vmax=vhi)
    txt(x2 + s3 / 2, 0.63, r"rendered $\widehat{V}$ on slice", 7)
    x3 = 3.49
    tile(x3, yy, s3, crop(g), vmin=vlo, vmax=vhi)
    txt(x3 + s3 / 2, 0.63, r"observed $V$ on slice", 7)
    mx = (x2 + s3 + x3) / 2
    txt(mx, yy + 0.4, r"$L$", 9, ORANGE, w="bold", bbox=dict(boxstyle="circle,pad=0.16", fc="white", ec=ORANGE, lw=0.9))
    # gradient loop
    arrow([(mx, yy + 0.27), (mx, 0.42), (x1 + 0.0, 0.42), (x1 + 0.0, 0.42), (x1 - 0.0, yy + 0.20)], ORANGE)
    txt((x1 + mx) / 2 + 0.12, 0.27, r"gradients update $\mu_i,\ R_i,\ s_i,\ \alpha_i$", 7, ORANGE)
    txt((x1 + mx) / 2 + 0.12, 0.14, "no splitting or pruning; one slice per iteration", 6.6, GREY)

    arrow([(4.38, 1.15), (4.62, 1.15)])
    box(4.62, by0, 1.16, 1.80, TG, GREEN)
    txt(5.20, 1.74, "Re-estimation", 8.2, GREEN, w="bold")
    txt(5.20, 1.60, "between cycles only", 6.8, GREEN, style="italic")
    # 1D profile of a parent and its two children with least-squares amplitudes (b = 1: centres +-s/2, scale s/2)
    t = np.linspace(-3, 3, 400)
    par = np.exp(-(t**2) / 2)
    ch = np.stack([np.exp(-((t - o) ** 2) / (2 * 0.25)) for o in (-0.5, 0.5)])
    co = np.linalg.lstsq(ch.T, par, rcond=None)[0]
    print("LS amplitudes", co)
    fx = lambda u: 5.20 + u / 3 * 0.46
    fy = lambda v: 0.92 + v * 0.50
    ax.plot(fx(t), fy(par), c=INK, lw=1.1, zorder=9)
    for a0, cc0 in zip(co, ch):
        ax.fill_between(fx(t), fy(0 * t), fy(a0 * cc0), color=GREEN, alpha=0.28, lw=0, zorder=7)
    ax.plot(fx(t), fy(co @ ch), c=GREEN, lw=1.1, ls=(0, (3, 1.5)), zorder=10)
    txt(5.63, 1.40, "parent", 6.4, INK, ha="right")
    txt(4.69, 1.40, "children", 6.4, GREEN, ha="left")
    txt(5.20, 0.77, r"split where $\hat q_{ik}>\tau_q$", 7)
    txt(5.20, 0.62, "child amplitudes by", 7)
    txt(5.20, 0.50, "least squares", 7)
    arrow([(4.62, 0.28), (4.38, 0.28)], GREEN, 1.2)
    txt(5.20, 0.28, "next cycle", 7.2, GREEN, w="bold")
    txt(5.20, 0.15, "(3 cycles in total)", 6.6, GREY)
    # output, after the last cycle
    txt(6.50, 1.74, "Output after last cycle", 8, w="bold")
    ax.plot([4.50, 4.50], [1.15, 1.95], c=INK, lw=1.1, zorder=14)
    ax.plot([4.50, 5.88], [1.95, 1.95], c=INK, lw=1.1, zorder=14)
    arrow([(5.88, 1.95), (5.88, 1.62 - 0.0), (5.88, 1.15), (5.96, 1.15)])
    st = 0.66
    xo = 5.97
    gaussians(xo, 0.92, st, d["m1"], d["C1"], d["w1"], BLUE, alpha=0.5)
    tile(xo, 0.14, st, crop(d["pred"]), vmin=vlo, vmax=vhi)
    xt = xo + st + 0.05
    txt(xt, 1.36, "optimised", 7, ha="left")
    txt(xt, 1.23, "Gaussians", 7, ha="left")
    txt(xt, 1.06, "$M=$", 7, ha="left")
    txt(xt + 0.0, 0.96, f"{int(d['count1']):,}", 7, ha="left")
    txt(xt, 0.56, "field", 7, ha="left")
    txt(xt, 0.41, r"$\widehat{V}(x)$", 7, ha="left")
    fig.savefig(a.out, dpi=1000)
    if a.preview:
        fig.savefig(a.preview, dpi=220)


if __name__ == "__main__":
    main()
