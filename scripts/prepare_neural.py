"""Central crops of two further neural EM volumes, as CREMI (256 x 256 in-plane, uint8, no intensity change).

    python scripts/prepare_neural.py

EPFL hippocampus (Lucchi et al.): mouse CA1, FIB-SEM, isotropic 5 nm; training.tif (165 x 768 x 1024) from
  https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/training.tif
  -> data/epfl/epfl_ca1.tif, 125 x 256 x 256 at 5 x 5 x 5 nm.
SNEMI3D (Kasthuri et al., ISBI 2013): mouse cortex, ATUM-SEM, 6 x 6 x 29 nm; training raw volume (100 x 1024 x 1024)
  from the torch-em mirror https://oc.embl.de/index.php/s/43iMotlXPyAB39z/download (SHA-256 checked)
  -> data/snemi3d/snemi3d.tif, 100 x 256 x 256 at 29 x 6 x 6 nm.
"""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
SNEMI_SHA = "5b130a24d9eb23d972fede0f1a403bc05f6808b361cfa22eff23b930b12f0615"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def centre(n, m):
    a = (n - m) // 2
    return slice(a, a + m)


def write(out, vol, meta):
    out.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(out, vol)
    meta.update({"shape": list(vol.shape), "dtype": str(vol.dtype), "min": int(vol.min()), "max": int(vol.max()),
                 "sha256": sha(out)})
    (out.parent / "provenance.json").write_text(json.dumps(meta, indent=1))
    print(out, vol.shape, vol.dtype, vol.min(), vol.max())


def main():
    src = ROOT / "data/epfl/download/training.tif"
    e = tifffile.imread(src)
    z, y, x = centre(e.shape[0], 125), centre(e.shape[1], 256), centre(e.shape[2], 256)
    write(ROOT / "data/epfl/epfl_ca1.tif", np.ascontiguousarray(e[z, y, x]), {
        "source": "EPFL CVLab, Electron Microscopy (CA1 hippocampus), training.tif", "source_sha256": sha(src),
        "crop_zyx": [[z.start, z.stop], [y.start, y.stop], [x.start, x.stop]], "spacing_zyx_nm": [5.0, 5.0, 5.0]})
    src = ROOT / "data/snemi3d/download/snemi_train.h5"
    assert sha(src) == SNEMI_SHA, "SNEMI3D checksum mismatch"
    with h5py.File(src, "r") as f:
        r = f["volumes/raw"]
        y, x = centre(r.shape[1], 256), centre(r.shape[2], 256)
        s = np.ascontiguousarray(r[:, y, x])
    write(ROOT / "data/snemi3d/snemi3d.tif", s, {
        "source": "SNEMI3D training raw volume (torch-em mirror, oc.embl.de)", "source_sha256": SNEMI_SHA,
        "crop_zyx": [[0, s.shape[0]], [y.start, y.stop], [x.start, x.stop]], "spacing_zyx_nm": [29.0, 6.0, 6.0]})


if __name__ == "__main__":
    main()
