"""Score the same tensor archives with INIF's official JAX/Haiku decoder (inif environment)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", type=Path, default=Path(__file__).resolve().parents[2] / "freqpile")
    p.add_argument("--root", type=Path, default=Path("runs/rate_distortion"))
    a = p.parse_args()
    import jax
    if jax.default_backend() != "gpu":
        raise RuntimeError("INIF decoding requires the GPU JAX environment")
    import jax.numpy as jnp
    import numpy as np
    import tifffile
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/slicegaussian"))
    from storage import encode_arrays, decode_arrays, score_psnr, PRECISIONS
    sys.path.insert(0, str(a.reference_root / "baselines/inif/external/INIF"))
    from models import INR
    for sample in "abc":
        path = a.reference_root / f"results/paper/CREMI_{sample.upper()}/inif/seed_0/model.pkl"
        out = a.root / f"encoded/cremi_{sample}/inif/reference_architecture"
        out.mkdir(parents=True, exist_ok=True)
        if (out / "points.json").exists():
            continue
        with path.open("rb") as f:
            params = pickle.load(f)
        arrays = {layer + "::" + key: np.asarray(v, np.float32)
                  for layer, values in params.items() for key, v in values.items()}
        raw = tifffile.imread(f"data/cremi/cremi_{sample}.tif")
        lo, hi = float(raw.min()), float(raw.max())
        gt = (raw.astype(np.float32) - lo) / (hi - lo)
        valid = [z for z in range(len(raw)) if raw[z].min() != raw[z].max()]
        Z, H, W = gt.shape
        features = params["inif_model/~/sensory"]["w"].shape[1]
        model = INR.INIF(features=features, coords_channel=3)

        @jax.jit
        def apply(params, coords):
            with jax.ensure_compile_time_eval():
                return model.apply(params, None, coords)

        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        yx = np.stack([yy.ravel() / (H - 1) * 2 - 1, xx.ravel() / (W - 1) * 2 - 1], 1).astype(np.float32)
        metadata = {"method": "inif", "architecture": {"network": "INIF", "features": features,
                    "layers": 7, "frequency": 90., "coordinates": "zyx in [-1,1]",
                    "output_rule": "(network*(input_max-input_min)/100-input_min)/(input_max-input_min)"},
                    "shape": list(gt.shape), "spacing": [40., 4., 4.], "input_min": lo, "input_max": hi,
                    "raw_fp32_bytes": sum(v.nbytes for v in arrays.values()),
                    "state_scalars": sum(v.size for v in arrays.values()), "gaussians": None,
                    "inference_precision": "decoded parameters evaluated in FP32"}
        provenance = {"checkpoint": str(path.resolve()), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "checkpoint_file_bytes": path.stat().st_size}
        points = []
        for precision in PRECISIONS:
            target = out / (precision + ".rdz")
            target.write_bytes(encode_arrays(arrays, metadata, precision))
            decoded, decoded_metadata = decode_arrays(target.read_bytes())
            assert decoded_metadata == metadata
            if precision == "fp32":
                assert all(np.array_equal(arrays[k], decoded[k]) for k in arrays)
            tree = {}
            for key, value in decoded.items():
                layer, name = key.split("::")
                tree.setdefault(layer, {})[name] = jnp.asarray(value)
            pred = np.empty_like(gt)
            for z in range(Z):
                coords = np.concatenate([np.full((H * W, 1), z / (Z - 1) * 2 - 1, np.float32), yx], 1)
                value = np.asarray(apply(tree, coords)).reshape(H, W) * (hi - lo) / 100
                pred[z] = np.clip((value - lo) / (hi - lo), 0, 1)
            psnr = score_psnr(pred, gt, valid)
            if precision == "fp32":
                expected = score_psnr(np.load(path.parent / "prediction.npy", mmap_mode="r"), gt, valid)
                if abs(psnr - expected) > .03:
                    raise RuntimeError(f"decoded FP32 INIF does not match reference prediction: {psnr} vs {expected}")
            row = {"volume": f"cremi_{sample}", "method": "inif", "setting": "reference_architecture",
                   "precision": precision, "gaussians": None, "state_scalars": metadata["state_scalars"],
                   "raw_fp32_bytes": metadata["raw_fp32_bytes"], "encoded_bytes": target.stat().st_size,
                   "decoded_psnr": psnr, "status": "ok",
                   "archive": str(target), **provenance}
            points.append(row)
            print(json.dumps(row), flush=True)
        (out / "points.json").write_text(json.dumps(points, indent=2))


if __name__ == "__main__":
    main()
