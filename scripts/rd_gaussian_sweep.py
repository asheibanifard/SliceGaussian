"""Train one additional native GaussianPile/3DGS population for the rate-distortion sweep."""
import argparse
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("volume")
    p.add_argument("--method", choices=["gaussianpile", "3dgs"], required=True)
    p.add_argument("--initial-n", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference-root", type=Path, default=Path(__file__).resolve().parents[2] / "freqpile")
    p.add_argument("--spacing", default="40,4,4", help="dz,dy,dx in physical units (default: CREMI)")
    a = p.parse_args()
    spacing = tuple(float(s) for s in a.spacing.split(","))
    sys.path[:0] = [str(a.reference_root / "experiments"), str(a.reference_root / "standalone"),
                    str(a.reference_root.parent / "gaussianpile")]
    import torch
    from slicegaussian import load_volume, metrics
    vol = load_volume(a.volume, spacing)
    mod = importlib.import_module("final_experiments_gaussianpile" if a.method == "gaussianpile"
                                  else "final_experiments_3dgs")
    (mod if a.method == "gaussianpile" else mod.fg).SPACING = spacing
    mod.fe.N0 = a.initial_n
    if a.method == "gaussianpile":
        mod.LOG_EVERY = 5000

        @torch.no_grad()
        def evaluate(model, cams, stack):
            pred = mod.render_stack(model.means, model.scales, model.quats, model.alphas, cams, mod.SETTINGS)
            return {"psnr3d": metrics.whole(None, vol, full=False, pred=pred)["psnr"]}
    else:
        mod.fg.LOG_EVERY = 5000

        @torch.no_grad()
        def evaluate(model, stack):
            pred = mod.render_stack(model, vol.Z, vol.dz, vol.dx, vol.H, vol.W)
            return {"psnr3d": metrics.whole(None, vol, full=False, pred=pred)["psnr"]}
    mod.evaluate = evaluate
    mod.train(a.volume, str(a.out), 0, SimpleNamespace(iterations=30000, save_every=0))
    (a.out / "rd_protocol.json").write_text(json.dumps({"method": a.method, "initial_n": a.initial_n,
        "spacing_zyx": spacing,
        "iterations": 30000, "seed": 0, "evaluation_every": 5000, "evaluation": "common valid-slice 3D PSNR",
        "training": "unchanged native initialisation, optimiser, density control and supervision",
        "count": "actual final count, not the initial setting", "reference_source": mod.__file__}, indent=2))


if __name__ == "__main__":
    main()
