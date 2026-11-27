"""Rate-distortion of every Gaussian model with the GaussianPile codec (Kong et al., Sec. 4.3; our implementation in
../gaussianpile/gaussianpile/compress.py), so that all Gaussian methods are stored as GaussianPile stores its own.

    python scripts/rd_measure_gpcodec.py [--volumes cremi_a,...]

The codec: Gaussians sorted in Morton (Z-)order of their quantized centres; centres normalised to the bounding box,
`pos_bits` per axis; log-scales and the (x, y, z) components of the quaternion on the positive half-sphere at
`attr_bits`; each stream delta-encoded, zig-zag mapped and compressed with LZMA. GaussianPile's opacity is quantized
on [0, 1], exactly as in compress.encode (checked: identical bytes). Quantities that GaussianPile does not have are
quantized at the same `attr_bits` on their own range [min, max], with the bounds in the header: the amplitude of
SliceGaussian (it may be negative or exceed 1) and the grey value of 3DGS. Settings (pos_bits, attr_bits): (12, 10),
(14, 12) = GaussianPile's default, (16, 14). Each file is decoded, rendered by the method's own renderer and scored.
Writes runs/rate_distortion/encoded_gpcodec/<volume>/<method>/<setting>/points.json.
"""

import argparse
import importlib.util
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "gaussianpile"))
from gaussianpile import compress  # noqa: E402

spec = importlib.util.spec_from_file_location("rd_torch", ROOT / "scripts/rd_measure_torch.py")
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)
SETTINGS = [(12, 10), (14, 12), (16, 14)]
MAGIC = b"GPCX1"


def encode(means, scales, quats, alpha, alpha_range, extra=None, pos_bits=14, attr_bits=12):
    """compress.encode, with the opacity range as a parameter and optional extra scalar streams (own range)."""
    means, scales, quats = (np.asarray(x, np.float64) for x in (means, scales, quats))
    alpha = np.asarray(alpha, np.float64).reshape(-1)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    quats = np.where(quats[:, :1] < 0, -quats, quats)
    lo, hi = means.min(0), means.max(0)
    qpos = compress._quant(means, lo, hi, pos_bits)
    order = np.argsort(compress.morton3(qpos[:, 0], qpos[:, 1], qpos[:, 2]), kind="stable")
    ls = np.log(scales)
    slo, shi = ls.min(0), ls.max(0)
    alo, ahi = alpha_range
    streams = {
        "pos": qpos[order],
        "alpha": compress._quant(alpha[:, None], alo, ahi, attr_bits)[order],
        "scale": compress._quant(ls, slo, shi, attr_bits)[order],
        "rot": compress._quant(quats[:, 1:], -1.0, 1.0, attr_bits)[order],
    }
    header = {
        "N": int(len(means)),
        "pos_bits": pos_bits,
        "attr_bits": attr_bits,
        "pos_lo": lo.tolist(),
        "pos_hi": hi.tolist(),
        "scale_lo": slo.tolist(),
        "scale_hi": shi.tolist(),
        "alpha_range": [alo, ahi],
        "extra": {},
        "streams": {},
    }
    for k, v in (extra or {}).items():
        v = np.asarray(v, np.float64).reshape(-1)
        header["extra"][k] = [float(v.min()), float(v.max())]
        streams[k] = compress._quant(v[:, None], v.min(), v.max(), attr_bits)[order]
    blobs = []
    for k, v in streams.items():
        b = compress._pack_stream(v)
        header["streams"][k] = {"shape": list(v.shape), "bytes": len(b)}
        blobs.append(b)
    hj = json.dumps(header).encode()
    out = io.BytesIO()
    out.write(MAGIC)
    out.write(struct.pack("<I", len(hj)))
    out.write(hj)
    for b in blobs:
        out.write(b)
    return out.getvalue()


def decode(buf):
    assert buf[:5] == MAGIC
    (hl,) = struct.unpack("<I", buf[5:9])
    h = json.loads(buf[9 : 9 + hl])
    off = 9 + hl
    st = {}
    for k, info in h["streams"].items():
        st[k] = compress._unpack_stream(buf[off : off + info["bytes"]], tuple(info["shape"]))
        off += info["bytes"]
    pb, ab = h["pos_bits"], h["attr_bits"]
    q = compress._dequant(st["rot"], -1.0, 1.0, ab).astype(np.float64)
    w = np.sqrt(np.clip(1.0 - (q**2).sum(1, keepdims=True), 0.0, None))
    q = np.concatenate([w, q], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    out = {
        "means": compress._dequant(st["pos"], np.array(h["pos_lo"]), np.array(h["pos_hi"]), pb),
        "scales": np.exp(compress._dequant(st["scale"], np.array(h["scale_lo"]), np.array(h["scale_hi"]), ab)),
        "quats": q.astype(np.float32),
        "alpha": compress._dequant(st["alpha"], *h["alpha_range"], ab)[:, 0],
    }
    for k, (lo, hi) in h["extra"].items():
        out[k] = compress._dequant(st[k], lo, hi, ab)[:, 0]
    return out


def check_identical_to_gaussianpile():
    rng = np.random.default_rng(0)
    m, s, q, a = rng.random((500, 3)) * 100, rng.random((500, 3)) + 0.1, rng.normal(size=(500, 4)), rng.random(500)
    ours, ref = encode(m, s, q, a, (0.0, 1.0)), compress.encode(m, s, q, a)
    strip = lambda b: b[9 + struct.unpack("<I", b[5:9])[0] :]  # the streams, without the header
    assert strip(ours) == strip(ref), "stream bytes differ from gaussianpile.compress.encode"


def to_arrays(method, state):
    """Returns (encode kwargs, function decoded -> state arrays for rd.build_evaluator)."""
    if method == "slicegaussian":
        amp = state["amp"].numpy()
        kw = dict(
            means=state["xyz"].numpy(),
            scales=state["log_scale"].exp().numpy(),
            quats=state["quat"].numpy(),
            alpha=amp,
            alpha_range=(float(amp.min()), float(amp.max())),
        )
        back = lambda d: {
            "xyz": d["means"],
            "log_scale": np.log(d["scales"]).astype(np.float32),
            "quat": d["quats"],
            "amp": d["alpha"].astype(np.float32),
        }
    else:
        alpha = torch.sigmoid(state["alpha_logit"]).numpy().reshape(-1)
        kw = dict(
            means=state["xyz"].numpy(), scales=state["log_scale"].exp().numpy(), quats=state["rot"].numpy(), alpha=alpha, alpha_range=(0.0, 1.0)
        )
        if method == "3dgs":
            kw["extra"] = {"color": state["color"].numpy().reshape(-1)}
        shape = tuple(state["alpha_logit"].shape)

        def back(d):
            a = np.clip(d["alpha"].astype(np.float64), 1e-7, 1 - 1e-7)
            arr = {
                "xyz": d["means"],
                "log_scale": np.log(d["scales"]).astype(np.float32),
                "rot": d["quats"],
                "alpha_logit": np.log(a / (1 - a)).astype(np.float32).reshape(shape),
            }
            if method == "3dgs":
                arr["color"] = d["color"].astype(np.float32).reshape(tuple(state["color"].shape))
            return arr

    return kw, back


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--volumes", default=",".join(rd.VOLUMES))
    p.add_argument("--reference-root", type=Path, default=ROOT.parent / "freqpile")
    p.add_argument("--root", type=Path, default=ROOT / "runs/rate_distortion")
    a = p.parse_args()
    check_identical_to_gaussianpile()
    from slicegaussian import load_volume

    for name in a.volumes.split(","):
        tif, spacing = rd.VOLUMES[name]
        vol = load_volume(str(ROOT / tif), spacing)
        Z = vol.Z * vol.H * vol.W
        for method, setting, path in rd.sources(a.reference_root, a.root, name):
            if method not in ("slicegaussian", "gaussianpile", "3dgs"):
                continue
            out = a.root / f"encoded_gpcodec/{name}/{method}/{setting}"
            if (out / "points.json").exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            state = torch.load(path, map_location="cpu", weights_only=False)
            if method == "slicegaussian":
                state = {"xyz": state["xyz"], "log_scale": state["scales"].log(), "quat": state["quat"], "amp": state["amp"]}
            kw, back = to_arrays(method, {k: v.float() for k, v in state.items()})
            predict, _ = rd.build_evaluator(method, state, vol, a.reference_root)
            rows = []
            for pb, ab in SETTINGS:
                blob = encode(**kw, pos_bits=pb, attr_bits=ab)
                (out / f"p{pb}_a{ab}.gpc").write_bytes(blob)
                psnr = float(predict(back(decode((out / f"p{pb}_a{ab}.gpc").read_bytes()))))
                rows.append(
                    {
                        "volume": name,
                        "method": method,
                        "setting": setting,
                        "codec": "gaussianpile",
                        "pos_bits": pb,
                        "attr_bits": ab,
                        "gaussians": int(len(kw["means"])),
                        "encoded_bytes": len(blob),
                        "bpv": 8 * len(blob) / Z,
                        "decoded_psnr": psnr,
                        "checkpoint": str(path),
                    }
                )
                print(json.dumps(rows[-1]), flush=True)
            (out / "points.json").write_text(json.dumps(rows, indent=1))
            del predict
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
