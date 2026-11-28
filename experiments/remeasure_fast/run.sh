#!/usr/bin/env bash
# Re-measurement after the raw-parameter CUDA kernels (render_planes builds covariances in CUDA and visits only the
# Gaussians that reach the slice). Method, loss and optimiser unchanged. Runs sequentially, nothing else on the GPU.
#  1. fresh-process wall-clock benchmark, SliceGaussian, CREMI A/B/C (seed 0)      -> runs/wallclock_fast/
#  2. complete method, CREMI A/B/C, seeds 0-2 (fitting time for the main table)  -> runs/remeasure_fast/cremi_<v>/seed_<s>/
#  3. complete method, EPFL CA1 and SNEMI3D, seed 0                                -> runs/remeasure_fast/<v>/seed_0/
cd "$(dirname "$0")/../.."
PY=~/micromamba/envs/gaussian-volume/bin/python
SG=~/micromamba/envs/gaussian-volume/bin/slicegaussian
$PY -c "from slicegaussian.cuda.fieldcut import _ext; e = _ext(); assert hasattr(e, 'raw_forward')"
for v in a b c; do
  out=runs/wallclock_fast/cremi_$v/slicegaussian
  [ -f $out/result.json ] && continue
  mkdir -p $out; echo "[$(date +%H:%M)] wallclock $v"
  $PY scripts/wallclock_benchmark.py data/cremi/cremi_$v.tif --method slicegaussian --out $out > $out/console.log 2>&1 \
    || echo "FAILED wallclock $v"
done
fit() {  # volume spacing out seed
  [ -f $3/final.json ] && return
  echo "[$(date +%H:%M)] fit $3"
  $SG fit $1 --spacing $2 --config configs/paper.json --seed $4 --out $3 --quiet > $3.log 2>&1 || echo "FAILED $3"
}
for v in a b c; do for s in 0 1 2; do
  mkdir -p runs/remeasure_fast/cremi_$v; fit data/cremi/cremi_$v.tif 40,4,4 runs/remeasure_fast/cremi_$v/seed_$s $s
done; done
mkdir -p runs/remeasure_fast/epfl runs/remeasure_fast/snemi3d
fit data/epfl/epfl_ca1.tif 5,5,5 runs/remeasure_fast/epfl/seed_0 0
fit data/snemi3d/snemi3d.tif 29,6,6 runs/remeasure_fast/snemi3d/seed_0 0
echo "[$(date +%H:%M)] all done"
