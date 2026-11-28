#!/usr/bin/env bash
# Early-densification controls (reviewer round 2, weakness 1): same arms, count, 30,000 updates and shared
# Adam restarts as runs/allocation, but splits after 1k/2k/3k updates (B) or all after 1k updates (C).
cd "$(dirname "$0")/../.."
for sched in "B 1000,1000,1000,27000" "C 1000,0,0,29000"; do
  set -- $sched
  for v in a b c; do
    out=runs/allocation_early/sched_$1/cremi_$v
    [ -f $out/allocation.json ] && continue
    echo "[$(date +%H:%M)] start $out"
    micromamba run -n gaussian-volume python scripts/allocation_ablation.py data/cremi/cremi_$v.tif --spacing 40,4,4 \
      --segments $2 --out $out > $out.log 2>&1
    echo "[$(date +%H:%M)] exit $? $out"
  done
done
echo "[$(date +%H:%M)] early done"
