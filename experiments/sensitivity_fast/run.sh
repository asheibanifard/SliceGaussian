#!/usr/bin/env bash
# Parameter sensitivity with the raw-parameter CUDA kernels (seed 0, CREMI A/B/C), and matched-count controls for the
# 7-scale exception (three 10k cycles with optimizer restarts, no re-estimation). Default row: runs/remeasure_fast seed 0.
cd "$(dirname "$0")/../.."
R=runs/sensitivity_fast
SG=~/micromamba/envs/gaussian-volume/bin/slicegaussian
declare -A VOL=([cremi_a]=data/cremi/cremi_a.tif [cremi_b]=data/cremi/cremi_b.tif [cremi_c]=data/cremi/cremi_c.tif)
fit() {  # volume config outname
  out=$R/$1/$3/seed_0
  [ -f $out/final.json ] && return
  mkdir -p $(dirname $out); echo "[$(date +%H:%M)] $1 $3"
  $SG fit ${VOL[$1]} --spacing 40,4,4 --config $R/configs/$2.json --seed 0 --out $out --quiet > $out.log 2>&1 || echo "FAILED $out"
}
for v in cremi_a cremi_b cremi_c; do
  for m in m_s7 m_s11at7 m_s11 m_s7atdef; do fit $v ${m}_$v $m; done
done
for v in cremi_a cremi_b cremi_c; do
  for k in scales7 scales9 scales13 p75 p95 p100 tau0.8 tau0.4 tau0.2 tau0.05 reg0 reg0.01 reg1 ratio10 ratio100; do fit $v $k $k; done
done
echo "[$(date +%H:%M)] all done"
