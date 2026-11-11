"""Slice snapshots during training: one PNG per checkpoint, rows = fixed slices, columns = GT | prediction | difference.

GT and prediction in grey on [0, 1]; difference = prediction - GT on a fixed diverging scale [-DIFF, +DIFF]
(red: too bright, blue: too dark, white: exact), the same for every checkpoint so images are comparable over time.
"""
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from .field import render_planes

DIFF = 0.25


def _diverging(d):
    """Map d in [-1, 1] to RGB: blue (#2a78d6) -> white -> red (#e34948)."""
    t = np.clip(d, -1, 1)[..., None]
    white = np.array([1.0, 1.0, 1.0])
    blue, red = np.array([0x2a, 0x78, 0xd6]) / 255, np.array([0xe3, 0x49, 0x48]) / 255
    return np.where(t < 0, white + (blue - white) * (-t), white + (red - white) * t)


def default_slices(vol, n=3):
    valid = vol.valid_ids
    return [valid[int(round(q * (len(valid) - 1)))] for q in np.linspace(0.25, 0.75, n)]


@torch.no_grad()
def save(field, vol, ks, it, psnr, folder):
    os.makedirs(folder, exist_ok=True)
    pred = render_planes(field, vol, ks).clamp(0, 1).cpu().numpy()
    gt = vol.gt[ks].cpu().numpy()
    rows = []
    for p, g in zip(pred, gt):
        grey = lambda a: np.repeat(a[..., None], 3, -1)
        rows.append(np.concatenate([grey(g), np.ones((g.shape[0], 4, 3)), grey(p), np.ones((g.shape[0], 4, 3)),
                                    _diverging((p - g) / DIFF)], 1))
        rows.append(np.ones((4, rows[-1].shape[1], 3)))
    img = (np.concatenate(rows[:-1], 0) * 255).astype(np.uint8)
    head = 18
    canvas = Image.new("RGB", (img.shape[1], img.shape[0] + head), "white")
    canvas.paste(Image.fromarray(img), (0, head))
    d = ImageDraw.Draw(canvas)
    W = vol.W + 4
    d.text((4, 3), f"GT   it {it}   PSNR {psnr:.2f} dB   slices {ks}", fill=(30, 30, 30))
    d.text((W + 4, 3), "prediction", fill=(30, 30, 30))
    d.text((2 * W + 4, 3), f"pred - GT  [-{DIFF}, +{DIFF}]", fill=(30, 30, 30))
    canvas.save(os.path.join(folder, f"it_{it:06d}.png"), optimize=False, compress_level=1)
