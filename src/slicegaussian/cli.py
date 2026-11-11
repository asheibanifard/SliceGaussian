"""Command-line interface.

    slicegaussian fit    VOLUME --spacing DZ,DY,DX --out DIR [--config FILE] [--seed N] [...]
    slicegaussian render MODEL  --shape Z,Y,X --spacing DZ,DY,DX --out FILE.tif
    slicegaussian eval   MODEL  VOLUME --spacing DZ,DY,DX
"""
from __future__ import annotations

import argparse
import json
import sys


def _triple(s, cast=float):
    v = tuple(cast(x) for x in s.split(","))
    if len(v) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values (z,y,x)")
    return v


def cmd_fit(a):
    from .pipeline import Config, fit, load_volume
    cfg = Config.load(a.config) if a.config else Config()
    if a.seed is not None:
        cfg.seed = a.seed
    if a.cycles is not None:
        cfg.reestimation.cycles = a.cycles
    if a.cycle_iters is not None:
        cfg.reestimation.cycle_iters = a.cycle_iters
    if a.tau_q is not None:
        cfg.p1.tau_q = a.tau_q
    if a.target_psnr is not None:
        cfg.reestimation.target = a.target_psnr
    if a.eval_every is not None:
        cfg.p2.eval_every = a.eval_every
    if a.snapshot_every:
        cfg.p2.snapshot_every = a.snapshot_every
    if a.no_codec:
        cfg.codec = False
    vol = load_volume(a.volume, a.spacing)
    print(f"volume {vol.name}: {vol.Z} x {vol.H} x {vol.W}, spacing {vol.spacing()}, "
          f"invalid slices {vol.invalid_ids}", flush=True)
    _, final = fit(vol, a.out, cfg, snapshots=a.snapshot_every > 0, verbose=not a.quiet)
    keep = ("psnr", "ssim", "mae", "all_psnr", "gaussians", "init_seconds", "total_seconds", "encoded_bytes",
            "decoded_psnr")
    print(json.dumps({k: final[k] for k in keep if k in final}, indent=1))


def cmd_render(a):
    import numpy as np
    import tifffile
    import torch

    from .field import GaussianField, render_volume

    class _Grid:                                                 # the minimal geometry render_volume needs
        def __init__(self, shape, spacing):
            self.Z, self.H, self.W = shape
            self.dz, self.dy, self.dx = spacing

    field = GaussianField.load(a.model)
    with torch.no_grad():
        v = render_volume(field, _Grid(a.shape, a.spacing)).clamp(0, 1).cpu().numpy()
    tifffile.imwrite(a.out, (v * 255).round().astype(np.uint8) if a.uint8 else v.astype(np.float32))
    print(f"wrote {a.out}: {v.shape}")


def cmd_eval(a):
    from . import metrics
    from .field import GaussianField
    from .volume import Volume
    vol = Volume(a.volume, a.spacing)
    field = GaussianField.load(a.model)
    print(json.dumps(metrics.whole(field, vol), indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(prog="slicegaussian", description="Frequency-aware Gaussian representation of "
                                "slice-based volumes (SliceGaussian).")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit SliceGaussian to a volume")
    f.add_argument("volume", help="(Z, Y, X) stack: .tif, .npy or CREMI .hdf")
    f.add_argument("--spacing", type=_triple, required=True, help="voxel spacing dz,dy,dx in physical units")
    f.add_argument("--out", required=True, help="output folder")
    f.add_argument("--config", help="JSON settings (see configs/paper.json)")
    f.add_argument("--seed", type=int)
    f.add_argument("--cycles", type=int, help="Phase II cycles (default 3)")
    f.add_argument("--cycle-iters", type=int, help="iterations per cycle (default 10000)")
    f.add_argument("--tau-q", type=float, help="subdivision threshold (default 0.1)")
    f.add_argument("--target-psnr", type=float, help="stop when the 3D PSNR on valid slices reaches this value")
    f.add_argument("--eval-every", type=int, help="whole-volume evaluation interval (default 500)")
    f.add_argument("--snapshot-every", type=int, default=0, help="save GT | prediction | difference images")
    f.add_argument("--no-codec", action="store_true", help="skip the encoded-size measurement")
    f.add_argument("--quiet", action="store_true")
    f.set_defaults(fn=cmd_fit)

    r = sub.add_parser("render", help="render a fitted model to a volume")
    r.add_argument("model")
    r.add_argument("--shape", type=lambda s: _triple(s, int), required=True, help="Z,Y,X")
    r.add_argument("--spacing", type=_triple, required=True, help="dz,dy,dx")
    r.add_argument("--out", required=True, help="output .tif")
    r.add_argument("--uint8", action="store_true", help="write 8-bit instead of float32")
    r.set_defaults(fn=cmd_render)

    e = sub.add_parser("eval", help="score a fitted model against a volume")
    e.add_argument("model")
    e.add_argument("volume")
    e.add_argument("--spacing", type=_triple, required=True)
    e.set_defaults(fn=cmd_eval)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
