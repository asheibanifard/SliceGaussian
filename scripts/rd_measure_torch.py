"""Encode/decode Gaussian and Torch-INR checkpoints and score every stored operating point."""
import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from slicegaussian import load_volume
from slicegaussian.field import GaussianField, render_volume
from slicegaussian.storage import encode_arrays, decode_arrays, score_psnr, PRECISIONS


VOLUMES = {"cremi_a": ("data/cremi/cremi_a.tif", (40., 4., 4.)),
           "cremi_b": ("data/cremi/cremi_b.tif", (40., 4., 4.)),
           "cremi_c": ("data/cremi/cremi_c.tif", (40., 4., 4.)),
           "epfl": ("data/epfl/epfl_ca1.tif", (5., 5., 5.)),
           "snemi3d": ("data/snemi3d/snemi3d.tif", (29., 6., 6.))}


def sources(reference, root, name):
    if name.startswith("cremi_"):
        yield from cremi_sources(reference, root, name[-1])
        return
    # EPFL / SNEMI3D: the main-comparison runs (runs/neural) plus the extra operating points in root/models.
    neural = Path("runs/neural") / name
    for tau in (.8, .4, .2, .1, .05):
        yield "slicegaussian", f"tau_{tau}", (neural / "slicegaussian/seed_0/model.pt" if tau == .1
                                              else root / f"models/{name}/slicegaussian/tau_{tau}/model.pt")
    for method in ("gaussianpile", "3dgs"):
        for n in (100000, 300000):
            yield method, f"initial_{n}", root / f"models/{name}/{method}/initial_{n}/model.pt"
        yield method, "initial_1000000", neural / method / "seed_0/model.pt"
    for method in ("neurcomp", "coordnet"):
        yield method, "reference_architecture", neural / method / "seed_0/net.pth"


def cremi_sources(reference, root, sample):
    old = reference / "results/paper"
    rd = old / "final-experiment-2/rate_distortion" / f"CREMI_{sample.upper()}"
    base = old / f"CREMI_{sample.upper()}"
    for tau in (.8, .4, .2, .1, .05):
        path = (old / f"final-experiment-2/CREMI_{sample.upper()}/slicegaussian/seed_0/model.pt"
                if tau == .1 else rd / f"tau_{tau}/seed_0/model.pt")
        yield "slicegaussian", f"tau_{tau}", path
    for method, folder in [("gaussianpile", "final_experiments_gaussianpile"),
                           ("3dgs", "final_experiments_3dgs")]:
        for n in (100000, 300000):
            yield method, f"initial_{n}", root / f"models/cremi_{sample}/{method}/initial_{n}/model.pt"
        yield method, "initial_1000000", base / folder / "seed_0/model.pt"
    for method in ("neurcomp", "coordnet"):
        yield method, "reference_architecture", base / method / "seed_0/net.pth"


def score(pred, vol):
    return score_psnr(pred.detach().cpu().numpy(), vol.volume, vol.valid_ids)


def build_evaluator(method, state, vol, reference):
    """Return a decoder renderer and the architecture information required by that decoder."""
    sys.path[:0] = [str(reference / "experiments"), str(reference / "standalone"),
                    str(reference.parent / "gaussianpile")]
    if method == "slicegaussian":
        architecture = {"field": "additive", "rotation": "wxyz quaternion", "scale": "log",
                        "radius_sigma": 3., "cull_epsilon": 1e-5}

        def predict(arrays):
            tensors = {k: torch.from_numpy(v.copy()).cuda() for k, v in arrays.items()}
            field = GaussianField(tensors["xyz"], tensors["log_scale"].exp(), tensors["quat"], tensors["amp"])
            field.p["log_scale"] = torch.nn.Parameter(tensors["log_scale"])
            return score(render_volume(field, vol), vol)
    elif method == "gaussianpile":
        from gaussianpile import GaussianModel, RenderSettings, render_stack, stack_cameras
        cams = stack_cameras(vol.Z, vol.H, vol.W, vol.spacing())
        settings = RenderSettings(sigma_z=0., norm_mode="paper")
        architecture = {"field": "GaussianPile", "sigma_z": 0., "normalisation": "paper",
                        "rotation": "wxyz quaternion", "scale": "log", "amplitude": "opacity logit",
                        "radius_sigma": settings.radius_sigma, "cull_epsilon": settings.cull_eps}

        def predict(arrays):
            model = GaussianModel.from_state({k: torch.from_numpy(v.copy()) for k, v in arrays.items()})
            pred = render_stack(model.means, model.scales, model.quats, model.alphas, cams, settings)
            return score(pred, vol)
    elif method == "3dgs":
        mod = importlib.import_module("final_experiments_3dgs")
        architecture = {"field": "slice-adapted grayscale 3DGS", "camera": "orthographic",
                        "slice_mask_std": 3., "rotation": "wxyz quaternion", "scale": "log",
                        "amplitude": "opacity logit", "colour_channels": 1}

        def predict(arrays):
            model = mod.Model3DGS.from_state({k: torch.from_numpy(v.copy()) for k, v in arrays.items()})
            pred = mod.render_stack(model, vol.Z, vol.dz, vol.dx, vol.H, vol.W)
            return score(pred, vol)
    else:
        dims = torch.tensor([vol.Z - 1, vol.H - 1, vol.W - 1], device="cuda", dtype=torch.float32)
        if method == "neurcomp":
            sys.path.insert(0, str(reference / "baselines/inr/external/neurcomp"))
            from siren import FieldNet
            width = state["net_layers.0.linear.weight"].shape[0]
            architecture = {"network": "FieldNet", "d_in": 3, "d_out": 1, "n_layers": 8,
                            "layers": [width] * 8, "w0": 30, "is_residual": True,
                            "coordinates": "aspect-scaled zyx in [-1,1]", "output": "(network+1)/2"}
            opt = SimpleNamespace(**{k: architecture[k] for k in
                                     ["d_in", "d_out", "n_layers", "layers", "w0", "is_residual"]})
            net = FieldNet(opt).cuda().eval()
            def coords(x):
                return (2 * x / dims - 1) * (dims / dims.max())
        else:
            sys.path.insert(0, str(reference / "baselines/inr/external/CoordNet/Code"))
            spec = importlib.util.spec_from_file_location("_rd_coordnet", reference / "baselines/inr/external/CoordNet/Code/model.py")
            coord_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(coord_module)
            CoordNet = coord_module.CoordNet
            architecture = {"network": "CoordNet", "d_in": 4, "d_out": 1, "initial_width": 64,
                            "residual_blocks": 10, "coordinates": "t=-1; zyx in [-1,1]",
                            "output": "(network+1)/2"}
            net = CoordNet(4, 1, 64, 10).cuda().eval()
            def coords(x):
                return torch.cat([-torch.ones_like(x[:, :1]), 2 * x / dims - 1], 1)
        yy, xx = torch.meshgrid(torch.arange(vol.H, device="cuda"), torch.arange(vol.W, device="cuda"), indexing="ij")

        def predict(arrays):
            net.load_state_dict({k: torch.from_numpy(v.copy()).cuda() for k, v in arrays.items()})
            pred = torch.empty_like(vol.gt)
            for z in range(vol.Z):
                xyz = torch.stack([torch.full_like(yy, z), yy, xx], -1).reshape(-1, 3).float()
                pred[z] = torch.cat([net(coords(x)).ravel() for x in xyz.split(1 << 16)]).reshape(vol.H, vol.W)
            pred = ((pred + 1) / 2).clamp(0, 1)
            return score(pred, vol)
    return predict, architecture


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", type=Path, default=Path(__file__).resolve().parents[2] / "freqpile")
    p.add_argument("--root", type=Path, default=Path("runs/rate_distortion"))
    p.add_argument("--samples", default="abc")
    p.add_argument("--methods", default="slicegaussian,gaussianpile,3dgs,neurcomp,coordnet")
    p.add_argument("--volumes", default=None,
                   help=f"comma-separated, from {', '.join(VOLUMES)}; overrides --samples (CREMI letters)")
    a = p.parse_args()
    names = a.volumes.split(",") if a.volumes else [f"cremi_{s}" for s in a.samples]
    for name in names:
        tif, spacing = VOLUMES[name]
        vol = load_volume(tif, spacing)
        for method, setting, path in sources(a.reference_root, a.root, name):
            if method not in a.methods.split(","):
                continue
            if not path.exists():
                raise FileNotFoundError(path)
            out = a.root / f"encoded/{name}/{method}/{setting}"
            out.mkdir(parents=True, exist_ok=True)
            if (out / "points.json").exists():
                print(f"Skipping {name}/{method}/{setting}", flush=True)
                continue
            state = torch.load(path, map_location="cpu", weights_only=False)
            if method == "slicegaussian":
                state = {"xyz": state["xyz"], "log_scale": state["scales"].log(),
                         "quat": state["quat"], "amp": state["amp"]}
            probe, architecture = build_evaluator(method, state, vol, a.reference_root)
            del probe
            arrays = {k: v.detach().numpy() for k, v in state.items()}
            raw = sum(v.nbytes for v in arrays.values())
            scalars = sum(v.size for v in arrays.values())
            count = len(arrays["xyz"]) if method in ("slicegaussian", "gaussianpile", "3dgs") else None
            if count is not None:
                assert scalars == count * (12 if method == "3dgs" else 11)
            metadata = {"method": method, "architecture": architecture, "shape": list(vol.shape),
                        "spacing": list(vol.spacing()), "input_min": vol.lo, "input_max": vol.hi,
                        "raw_fp32_bytes": raw, "state_scalars": scalars, "gaussians": count,
                        "inference_precision": "decoded parameters evaluated in FP32"}
            provenance = {"checkpoint": str(path.resolve()), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "checkpoint_file_bytes": path.stat().st_size}
            points = []
            for precision in PRECISIONS:
                target = out / (precision + ".rdz")
                try:
                    target.write_bytes(encode_arrays(arrays, metadata, precision))
                    decoded, got = decode_arrays(target.read_bytes())
                    assert got == metadata
                    if precision == "fp32":
                        assert all(np.array_equal(arrays[k], decoded[k]) for k in arrays)
                    # Rebuild from decoded tensors, rather than retaining the source model.
                    predict, decoded_architecture = build_evaluator(
                        got["method"], {k: torch.from_numpy(v.copy()) for k, v in decoded.items()}, vol, a.reference_root)
                    assert decoded_architecture == got["architecture"]
                    assert got["shape"] == list(vol.shape) and got["spacing"] == list(vol.spacing())
                    psnr = float(predict(decoded))
                    del predict
                    if precision == "fp32" and method in ("neurcomp", "coordnet"):
                        reference_prediction = path.parent / "prediction.npy"
                        expected = score_psnr(np.load(reference_prediction, mmap_mode="r"), vol.volume, vol.valid_ids)
                        if abs(psnr - expected) > .03:
                            raise RuntimeError(f"decoded FP32 {method} does not match its reference prediction: {psnr} vs {expected}")
                    row = {"volume": name, "method": method, "setting": setting,
                           "precision": precision, "gaussians": count, "state_scalars": scalars,
                           "raw_fp32_bytes": raw, "encoded_bytes": target.stat().st_size,
                           "decoded_psnr": psnr, "status": "ok", "archive": str(target), **provenance}
                except ValueError as error:
                    row = {"volume": name, "method": method, "setting": setting,
                           "precision": precision, "status": "unrepresentable", "error": str(error), **provenance}
                points.append(row)
                print(json.dumps(row), flush=True)
            (out / "points.json").write_text(json.dumps(points, indent=2))
            del state, arrays
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
