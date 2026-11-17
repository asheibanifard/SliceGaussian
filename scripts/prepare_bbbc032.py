"""Downloads BBBC032 (mouse blastocyst, spinning-disk confocal) and writes the two 125 x 256 x 256 crops used in the paper.

    python scripts/prepare_bbbc032.py --out data/bbbc032

Source: BBBC032v1, Rivron et al. 2018, Broad Bioimage Benchmark Collection (CC0 1.0). The archive (1.1 GB) holds one
12-bit stack per channel, 172 x 1344 x 1024 voxels at 0.5 x 0.101 x 0.101 um (spacing 500 x 101 x 101 nm, from the file
metadata). Channel 3 is Hoechst (nuclei) and channel 2 is WGA (membranes); channels 0 and 1 are sparse transcript puncta.

Both crops are raw[36:161, 419:675, 445:701] of their channel (the 125 x 256 x 256 window with the most nuclear and
membrane signal), clipped to their own 0.1 / 99.9 percentiles and rescaled to uint16, as bbbc032_{nuclei,membrane}.tif.
"""
import argparse
import hashlib
import json
import os
import urllib.request
import zipfile

URL = "https://data.broadinstitute.org/bbbc/BBBC032/BBBC032_v1_dataset.zip"
CHANNELS = {"nuclei": 3, "membrane": 2}
Z0, Y0, X0, DEPTH, SIZE = 36, 419, 445, 125, 256
SPACING_NM = (500.0, 101.0, 101.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/bbbc032")
    a = p.parse_args()
    import numpy as np
    import tifffile
    os.makedirs(os.path.join(a.out, "download"), exist_ok=True)
    archive = os.path.join(a.out, "download", "BBBC032_v1_dataset.zip")
    if not os.path.exists(archive):
        print(f"downloading {URL}", flush=True)
        urllib.request.urlretrieve(URL, archive + ".part")
        os.replace(archive + ".part", archive)
    provenance = {}
    with zipfile.ZipFile(archive) as z:
        for name, c in CHANNELS.items():
            member = f"BMP4blastocystC{c}.tif"
            src = os.path.join(a.out, "raw", member)
            if not os.path.exists(src):
                z.extract(member, os.path.join(a.out, "raw"))
            raw = tifffile.imread(src)[Z0:Z0 + DEPTH, Y0:Y0 + SIZE, X0:X0 + SIZE].astype(np.float64)
            lo, hi = np.percentile(raw, [0.1, 99.9])
            crop = np.round(np.clip((raw - lo) / (hi - lo), 0, 1) * 65535).astype(np.uint16)
            dst = os.path.join(a.out, f"bbbc032_{name}.tif")
            tifffile.imwrite(dst, crop)
            provenance[name] = {"source": f"{URL}#{member}", "channel": c, "shape": list(crop.shape), "dtype": "uint16",
                                "crop_zyx_origin": [Z0, Y0, X0], "spacing_zyx_nm": list(SPACING_NM),
                                "clip_raw_0.1_99.9": [float(lo), float(hi)],
                                "sha256": hashlib.sha256(crop.tobytes()).hexdigest(), "licence": "CC0 1.0"}
            print(f"wrote {dst}: {crop.shape} {crop.dtype}, clip [{lo:.0f}, {hi:.0f}]")
    with open(os.path.join(a.out, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)


if __name__ == "__main__":
    main()
