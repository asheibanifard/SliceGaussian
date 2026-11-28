#!/usr/bin/env bash
# Re-estimation controls on CREMI A/B/C (seed 0): (1) Phase I bisected to the complete method's final count, one Phase II
# of 30,000 iterations, no re-estimation; (2) the default 3 x 10,000 cycles with fresh optimisers but no subdivision.
cd "$(dirname "$0")/../.."
for v in a b c; do
  for arm in matched_no_reest restarts_only; do
    cfg=runs/reest_ablation/${arm}_${v}.json; [ $arm = restarts_only ] && cfg=runs/reest_ablation/restarts_only.json
    out=runs/reest_ablation/cremi_$v/$arm
    [ -f $out/final.json ] && continue
    mkdir -p runs/reest_ablation/cremi_$v
    echo "[$(date +%H:%M)] start $out"
    micromamba run -n gaussian-volume slicegaussian fit data/cremi/cremi_$v.tif --spacing 40,4,4 --config $cfg --seed 0 \
      --out $out --quiet > $out.log 2>&1
    echo "[$(date +%H:%M)] exit $? $out"
  done
done
echo "[$(date +%H:%M)] reest done"
