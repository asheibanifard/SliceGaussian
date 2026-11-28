#!/usr/bin/env bash
# Remaining BBBC032 / baseline runs, strictly one after another. Each step skips finished runs; a failure does not stop
# the later steps.
cd "$(dirname "$0")/../.."
PY="micromamba run -n gaussian-volume python -u"
stamp() { echo "[$(date +%H:%M)] $*"; }

# 1. naive all-in-focus 3DGS (GaussianPile paper's 3DGS): BBBC032 nuclei, membrane; CREMI A, B, C
naive() {   # naive <out> <volume> <spacing>
    [ -f "$1/metrics.json" ] && { stamp "skip $1"; return; }
    mkdir -p "$(dirname "$1")"
    stamp "start $1"
    $PY scripts/gaussian_baseline.py "$2" --spacing "$3" --method 3dgs_naive --out "$1" > "$1.log" 2>&1
    stamp "exit $? $1"
}
for v in nuclei membrane; do naive runs/bbbc032/$v/3dgs_naive/seed_0 data/bbbc032/bbbc032_$v.tif 500,101,101; done
for v in a b c; do naive runs/baselines_cremi/cremi_$v/3dgs_naive/seed_0 data/cremi/cremi_$v.tif 40,4,4; done

# 2. INIF at CREMI's parameters per voxel (ratio 32 on uint16); the ratio-16 nuclei run is kept aside, not deleted
if [ -d runs/bbbc032/nuclei/inif ] && grep -q "ratio: 16.0" runs/bbbc032/nuclei/inif/seed_0.log 2>/dev/null; then
    mv runs/bbbc032/nuclei/inif runs/bbbc032/nuclei/inif_INVALID_uint16_ratio16
fi
stamp "start inif"; $PY scripts/reproduce_bbbc032.py --methods inif > runs/bbbc032/driver_inif.log 2>&1; stamp "exit $? inif"

# 3. membrane NeurComp and CoordNet (the driver was interrupted before them)
stamp "start membrane INRs"
$PY scripts/reproduce_bbbc032.py --volumes membrane --methods neurcomp coordnet > runs/bbbc032/driver_membrane_inr.log 2>&1
stamp "exit $? membrane INRs"

# 4. units check: nuclei with the in-plane pixel at 4 units, as on CREMI
out=runs/bbbc032/diagnostics/nuclei_cremi_units
if [ ! -f $out/final.json ]; then
    stamp "start units"
    micromamba run -n gaussian-volume slicegaussian fit data/bbbc032/bbbc032_nuclei.tif --spacing 19.80198,4,4 \
        --config configs/paper.json --seed 0 --out $out --quiet > $out.log 2>&1
    stamp "exit $? units"
fi
stamp "queue done"
