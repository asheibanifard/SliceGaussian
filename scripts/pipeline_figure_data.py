"""Data for the pipeline figure of the journal version: Phase I on CREMI A (settings of configs/paper.json) and a complete
trained model of the same volume. Writes one .npz used by scripts/plot_pipeline.py.

    python scripts/pipeline_figure_data.py --model runs/wallclock/cremi_a/slicegaussian/model.pt --out <file.npz>
"""

import argparse
import json
import numpy as np
import torch
from slicegaussian import phase1
from slicegaussian.field import GaussianField, covariance, render_planes, calibrate_amplitude
from slicegaussian.volume import Volume, blur3d


def sections(mu, cov, zc, box):
    """Cross-sections of Gaussians by the plane z = zc, those whose centre lies within 1.5 std of it and inside box."""
    d = zc - mu[:, 2]
    sz = cov[:, 2, 2]
    x0, x1, y0, y1 = box
    b = cov[:, :2, 2] / sz[:, None]
    m = mu[:, :2] + b * d[:, None]
    k = (d.abs() < 1.5 * sz.sqrt()) & (m[:, 0] > x0 - 8) & (m[:, 0] < x1 + 8) & (m[:, 1] > y0 - 8) & (m[:, 1] < y1 + 8)
    C = cov[:, :2, :2] - b[:, :, None] * cov[:, 2, :2][:, None, :]
    return m[k].cpu().numpy(), C[k].cpu().numpy(), torch.exp(-0.5 * d**2 / sz)[k].cpu().numpy(), k


def lam_u1(fields, k):
    """Directional wavelength along u_1 (strongest eigenvector of J_bar) on slice k, as used by the violation ratio."""
    J, G = (phase1.mat3(fields[x][:, k]).double() for x in ("J", "G"))
    u = torch.linalg.eigh(J)[1][..., :, 2]
    return (1 / torch.einsum("...i,...ij,...j->...", u, G, u).clamp(min=1e-18).sqrt()).float().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--volume", default="data/cremi/cremi_a.tif")
    p.add_argument("--model", required=True)
    p.add_argument("--config", default="configs/paper.json")
    p.add_argument("--slice", type=int, default=62)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    vol = Volume(a.volume, (40.0, 4.0, 4.0))
    cfg = phase1.P1Config(**json.load(open(a.config))["p1"])
    fields = phase1.analyse(vol, cfg)
    init, stats, _ = phase1.run(vol, cfg, fields=fields)
    mu, R, s, _ = phase1.parents(vol, fields, cfg)
    q = phase1.violation(mu, R, s, fields, vol, cfg, cfg.seed)
    n = init["n"]
    k = a.slice
    zc = (k + 0.5) * vol.dz
    o = {
        "slice": vol.gt[k].cpu().numpy(),
        "stack": vol.gt[[k - 16, k - 12, k - 8, k - 4, k]].cpu().numpy(),
        "lam": lam_u1(fields, k),
        "J": fields["J"][:, k].cpu().numpy(),
        "xz": vol.gt[:, 128].cpu().numpy(),
        "parents": len(mu),
        "count0": stats["gaussians"],
    }
    sig = fields["sigmas"]
    bl = lambda t: blur3d(vol.gt[None], (t / vol.dz, t / vol.dy, t / vol.dx))[0]
    for i, b in enumerate((2, 4, 6)):
        o[f"band{i}"] = (bl(sig[b]) - bl(sig[b + 1]))[k].cpu().numpy()
        o[f"band{i}_wl"] = 1 / phase1.band_frequency(sig[b], sig[b + 1])
    print("sigmas", sig, stats)
    o["mu"], o["R"], o["q"], o["n"] = (t.cpu().numpy() for t in (mu, R, q, n))
    full = (0, vol.W * 4, 0, vol.H * 4)
    c0 = covariance(init["R"], init["scales"])
    m, C, w, _ = sections(init["xyz"], c0, zc, full)
    o["m0"], o["C0"], o["w0"] = m, C, w
    f = GaussianField.load(a.model)
    m, C, w, kk = sections(f.xyz.detach(), f.covariance().detach(), zc, full)
    o["m1"], o["C1"], o["w1"], o["a1"] = m, C, w, f.amp.detach()[kk].cpu().numpy()
    o["count1"] = len(f)
    with torch.no_grad():
        o["pred"] = render_planes(f, vol, [k])[0].cpu().numpy()
        f0 = GaussianField(init["xyz"], init["scales"], init["R"], init["amp"])
        calibrate_amplitude(f0, vol)
        o["pred0"] = render_planes(f0, vol, [k])[0].cpu().numpy()
    for nm in ("pred0", "pred"):
        print(nm, -10 * np.log10(((o[nm] - o["slice"]) ** 2).mean()))
    np.savez_compressed(a.out, **o)
    sel = np.abs(o["mu"][:, 2] - zc) <= 20
    u, c = np.unique(o["n"][sel], axis=0, return_counts=True)
    for x, y in sorted(zip(c, u.tolist()), reverse=True)[:15]:
        print(x, y)
    print("parents on slice", sel.sum(), np.unique(o["mu"][:, 2])[-70:][:6], zc)


if __name__ == "__main__":
    main()
