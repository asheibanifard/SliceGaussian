"""Downloads the CREMI samples A, B and C and writes the 125 x 256 x 256 crops used in the paper.

    python scripts/prepare_cremi.py --out data/cremi

Each crop is volumes/raw[:, 497:753, 497:753] of sample_{A,B,C}_20160501.hdf (uint8, spacing 40 x 4 x 4 nm), saved as
cremi_{a,b,c}.tif. The source files are about 1.5 GB each; existing files are not downloaded again.
"""
import argparse
import os
import urllib.request

URL = "https://cremi.org/static/data/sample_{s}_20160501.hdf"
ORIGIN, SIZE = 497, 256


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/cremi")
    p.add_argument("--samples", default="A,B,C")
    a = p.parse_args()
    import h5py
    import tifffile
    os.makedirs(a.out, exist_ok=True)
    for s in a.samples.split(","):
        src = os.path.join(a.out, f"sample_{s}_20160501.hdf")
        if not os.path.exists(src):
            print(f"downloading {URL.format(s=s)}", flush=True)
            urllib.request.urlretrieve(URL.format(s=s), src + ".part")
            os.replace(src + ".part", src)
        with h5py.File(src, "r") as f:
            raw = f["volumes/raw"][:, ORIGIN:ORIGIN + SIZE, ORIGIN:ORIGIN + SIZE]
        dst = os.path.join(a.out, f"cremi_{s.lower()}.tif")
        tifffile.imwrite(dst, raw)
        print(f"wrote {dst}: {raw.shape} {raw.dtype}")


if __name__ == "__main__":
    main()
