"""Measure actual x265 bitstreams and decoded PSNR for seven CRF operating points."""
import json
from pathlib import Path
import subprocess

import numpy as np
import tifffile

from slicegaussian.storage import score_psnr


def main():
    for sample in "abc":
        raw = tifffile.imread(f"data/cremi/cremi_{sample}.tif")
        gt = (raw.astype(np.float32) - float(raw.min())) / float(raw.max() - raw.min())
        quantized = np.round(gt * 255).astype(np.uint8)
        valid = [z for z in range(len(raw)) if raw[z].min() != raw[z].max()]
        Z, H, W = raw.shape
        out = Path(f"runs/rate_distortion/encoded/cremi_{sample}/hevc/crf_sweep")
        out.mkdir(parents=True, exist_ok=True)
        if (out / "points.json").exists():
            continue
        rows = []
        for crf in (10, 13, 16, 20, 24, 28, 34):
            target = out / f"crf_{crf}.mkv"
            metadata = {"shape": [Z, H, W], "spacing": [40., 4., 4.],
                        "input_min": float(raw.min()), "input_max": float(raw.max())}
            encode = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}",
                      "-i", "-", "-c:v", "libx265", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "gray",
                      "-x265-params", "log-level=error:pools=2:frame-threads=2",
                      "-metadata", "volume=" + json.dumps(metadata, separators=(",", ":")), str(target)]
            subprocess.run(encode, input=quantized.tobytes(), check=True, stderr=subprocess.DEVNULL)
            data = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(target), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                                  capture_output=True, check=True).stdout
            pred = np.frombuffer(data, np.uint8).reshape(Z, H, W).astype(np.float32) / 255
            row = {"volume": f"cremi_{sample}", "method": "hevc", "setting": f"crf_{crf}", "precision": "8bit_video",
                   "gaussians": None, "state_scalars": None, "raw_fp32_bytes": None,
                   "encoded_bytes": target.stat().st_size, "decoded_psnr": score_psnr(pred, gt, valid),
                   "status": "ok", "archive": str(target), "encoder_command": encode}
            rows.append(row)
            print(json.dumps(row), flush=True)
        (out / "points.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
