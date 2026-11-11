"""Representation size and a simple codec.

raw:     uncompressed float32 parameters, 11 per Gaussian (centre 3, log-scale 3, quaternion 4, amplitude 1): 44 B each.
encoded: a real codec, decoded before scoring.
         centres: uint16 per axis relative to the field's own bounding box (lo, hi stored in the header), so Gaussians
                  that drift slightly outside the volume box are not clamped (clamping them cost about 2.5 dB);
         log-scales, quaternions (normalized, w >= 0), amplitudes: float16;
         struct-of-arrays byte streams, compressed with lzma (preset 9 | extreme). Header: box lo, hi and count (52 B).
BPV = 8 bytes / voxels (voxels = Z x H x W of the volume).
"""
import lzma
import struct

import numpy as np
import torch
import torch.nn.functional as F

from . import metrics
from .field import GaussianField


def encode(field, vol):
    xyz = field.xyz.detach().cpu().numpy().astype(np.float64)
    lo, hi = xyz.min(0), xyz.max(0)
    span = np.maximum(hi - lo, 1e-12)
    qx = np.clip(np.round((xyz - lo) / span * 65535), 0, 65535).astype(np.uint16)
    q = F.normalize(field.p["quat"].detach(), dim=1)
    q = torch.where(q[:, :1] < 0, -q, q).cpu().numpy().astype(np.float16)
    ls = field.p["log_scale"].detach().cpu().numpy().astype(np.float16)
    amp = field.amp.detach().cpu().numpy().astype(np.float16)
    body = b"".join(np.ascontiguousarray(a.T).tobytes() for a in (qx, ls, q, amp[:, None]))
    head = struct.pack("<6dI", *lo, *span, len(xyz))
    return head + lzma.compress(body, preset=9 | lzma.PRESET_EXTREME)


def decode(blob, device="cuda"):
    *box, n = struct.unpack("<6dI", blob[:52])
    lo, span = np.array(box[:3], np.float64), np.array(box[3:], np.float64)
    body = lzma.decompress(blob[52:])
    off = 0

    def take(dtype, cols):
        nonlocal off
        nb = np.dtype(dtype).itemsize * n * cols
        a = np.frombuffer(body[off:off + nb], dtype).reshape(cols, n).T
        off += nb
        return a
    qx, ls, q, amp = take(np.uint16, 3), take(np.float16, 3), take(np.float16, 4), take(np.float16, 1)
    xyz = (lo + qx.astype(np.float64) / 65535 * span).astype(np.float32)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device)
    return GaussianField(t(xyz), t(ls).exp(), F.normalize(t(q), dim=1), t(amp[:, 0]), device=device)


@torch.no_grad()
def measure(field, vol):
    vox = vol.Z * vol.H * vol.W
    raw = len(field) * 11 * 4
    blob = encode(field, vol)
    dec = metrics.whole(decode(blob), vol)
    return {"raw_bytes": raw, "raw_bpv": 8 * raw / vox, "encoded_bytes": len(blob), "encoded_bpv": 8 * len(blob) / vox,
            "decoded_psnr": dec["psnr"], "decoded_ssim": dec["ssim"], "decoded_mae": dec["mae"],
            "raw_volume_bytes_uint8": vox}
