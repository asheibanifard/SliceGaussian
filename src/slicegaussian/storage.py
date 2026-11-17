"""Self-contained tensor archives for measured rate-distortion points.

FP32, FP16 or per-last-axis uniform 16/12/8-bit quantization, followed by LZMA
preset 6. The stored file includes a JSON decoder/architecture header, names,
shapes, dtypes and all quantization bounds. Optimizer state is not stored.
No framework imports: the same codec is usable by Torch and JAX inference.
"""
import json
import lzma
import struct

import numpy as np

MAGIC = b"SGRD1\0"
PRECISIONS = ("fp32", "fp16", "u16", "u12", "u8")


def _pack(values, bits):
    flat = values.ravel()
    if bits in (8, 16):
        return flat.astype("u1" if bits == 8 else "<u2").tobytes()
    flags = ((flat[:, None] >> np.arange(bits, dtype=np.uint32)) & 1).astype(np.uint8)
    return np.packbits(flags.ravel(), bitorder="little").tobytes()


def _unpack(data, count, bits):
    if bits in (8, 16):
        return np.frombuffer(data, "u1" if bits == 8 else "<u2", count=count).astype(np.uint32)
    flags = np.unpackbits(np.frombuffer(data, np.uint8), bitorder="little")[:count * bits].reshape(count, bits)
    return (flags.astype(np.uint32) << np.arange(bits, dtype=np.uint32)).sum(1, dtype=np.uint32)


def encode_arrays(arrays, metadata, precision="fp32"):
    if precision not in PRECISIONS:
        raise ValueError("unknown storage precision")
    descriptors, chunks = [], []
    for name in sorted(arrays):
        original = np.asarray(arrays[name])
        value = np.ascontiguousarray(original).reshape(original.shape)
        if value.dtype.kind not in "fiub":
            raise ValueError(f"unsupported dtype {value.dtype}")
        if value.dtype.kind == "f" and (value.dtype.itemsize != 4 or not np.isfinite(value).all()):
            raise ValueError("input floating tensors must be finite FP32")
        desc = {"name": name, "shape": list(value.shape), "dtype": value.dtype.newbyteorder("<").str}
        if value.dtype.kind != "f" or precision == "fp32" or value.size == 0:
            desc["mode"] = "raw"
            data = value.astype(value.dtype.newbyteorder("<"), copy=False).tobytes()
        elif precision == "fp16":
            with np.errstate(over="ignore"):
                half = value.astype("<f2")
            if not np.isfinite(half).all():
                raise ValueError("FP16 overflow; this operating point cannot be decoded faithfully")
            desc["mode"] = "fp16"
            data = half.tobytes()
        else:
            bits = int(precision[1:])
            channels = value.shape[-1] if value.ndim >= 2 else 1
            matrix = value.reshape(-1, channels).astype(np.float64)
            lo = matrix.min(0).astype("<f4")
            scale = ((matrix.max(0) - lo) / (2 ** bits - 1)).astype("<f4")
            q = np.rint((matrix - lo) / np.where(scale > 0, scale, 1))
            q = np.clip(q, 0, 2 ** bits - 1).astype(np.uint32)
            desc.update({"mode": "uniform", "bits": bits, "channels": channels})
            data = lo.tobytes() + scale.tobytes() + _pack(q, bits)
        desc["length"] = len(data)
        descriptors.append(desc)
        chunks.append(data)
    header = json.dumps({"version": 1, "precision": precision, "compression": "lzma-preset-6",
                         "metadata": metadata, "tensors": descriptors}, sort_keys=True,
                        separators=(",", ":"), allow_nan=False).encode("utf8")
    return MAGIC + struct.pack("<Q", len(header)) + header + lzma.compress(b"".join(chunks), preset=6)


def decode_arrays(blob):
    if not blob.startswith(MAGIC) or len(blob) < len(MAGIC) + 8:
        raise ValueError("invalid tensor archive")
    offset = len(MAGIC)
    size, = struct.unpack("<Q", blob[offset:offset + 8])
    offset += 8
    if offset + size > len(blob):
        raise ValueError("truncated tensor archive")
    head = json.loads(blob[offset:offset + size])
    body = lzma.decompress(blob[offset + size:])
    arrays, pos = {}, 0
    for desc in head["tensors"]:
        data = body[pos:pos + desc["length"]]
        if len(data) != desc["length"]:
            raise ValueError("truncated tensor payload")
        pos += len(data)
        shape, mode = tuple(desc["shape"]), desc["mode"]
        count = int(np.prod(shape))
        if mode == "raw":
            value = np.frombuffer(data, np.dtype(desc["dtype"])).copy()
        elif mode == "fp16":
            value = np.frombuffer(data, "<f2").astype(np.float32)
        elif mode == "uniform":
            channels, bits = desc["channels"], desc["bits"]
            lo = np.frombuffer(data, "<f4", count=channels).astype(np.float64)
            scale = np.frombuffer(data, "<f4", count=channels, offset=4 * channels).astype(np.float64)
            q = _unpack(data[8 * channels:], count, bits).reshape(-1, channels)
            value = (lo + scale * q).astype(np.float32).ravel()
        else:
            raise ValueError("unknown tensor coding mode")
        if len(value) != count:
            raise ValueError("tensor shape does not match payload")
        arrays[desc["name"]] = value.reshape(shape)
    if pos != len(body):
        raise ValueError("unclaimed bytes in tensor payload")
    return arrays, head["metadata"]


def score_psnr(prediction, target, valid_ids):
    """Common cross-framework whole-volume PSNR; FP64 MSE over valid slices."""
    pred = np.asarray(prediction)[valid_ids].astype(np.float64)
    gt = np.asarray(target)[valid_ids].astype(np.float64)
    if pred.shape != gt.shape or not np.isfinite(pred).all():
        raise ValueError("invalid decoded prediction")
    mse = np.square(np.clip(pred, 0, 1) - gt).mean()
    return float(-10 * np.log10(max(float(mse), 1e-20)))
