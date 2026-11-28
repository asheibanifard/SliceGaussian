#!/usr/bin/env bash
# GaussianPile with its recommended focus model, sigma_z = slice spacing (Kong et al., Sec. 4.1 / Tab. 4), seed 0.
# Same protocol as the sigma_z = 0 rows: 10^6 random Gaussians, 30,000 iterations, trained and scored on valid slices.
cd "$(dirname "$0")/../.."
R=runs/gaussianpile_sigmaz
run() {  # name volume spacing
  out=$R/$1/seed_0
  [ -f $out/metrics.json ] && return
  mkdir -p $out; echo "[$(date +%H:%M)] start $1"
  micromamba run -n gaussian-volume python scripts/gaussian_baseline.py $2 --spacing $3 --method gaussianpile \
    --sigma-z dz --valid-only --seed 0 --out $out > $R/$1.log 2>&1
  echo "[$(date +%H:%M)] exit $? $1"
}
run cremi_a data/cremi/cremi_a.tif 40,4,4
run cremi_b data/cremi/cremi_b.tif 40,4,4
run cremi_c data/cremi/cremi_c.tif 40,4,4
run epfl data/epfl/epfl_ca1.tif 5,5,5
run snemi3d data/snemi3d/snemi3d.tif 29,6,6
echo "[$(date +%H:%M)] all done"
