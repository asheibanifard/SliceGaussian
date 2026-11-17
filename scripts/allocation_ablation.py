"""Run the capacity-matched allocation controls on one volume and seed (requires CUDA)."""
import argparse
import json
from pathlib import Path

from slicegaussian import Config, load_volume
from slicegaussian.allocation_ablation import ARMS, run
from slicegaussian.pipeline import _save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("volume")
    parser.add_argument("--spacing", required=True, help="Z,Y,X physical spacing")
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--cycle-iters", type=int, default=7500)
    parser.add_argument("--segments", help="rounds+1 comma-separated optimisation lengths, e.g. 1000,1000,1000,27000")
    parser.add_argument("--arms", help="comma-separated subset of " + ",".join(ARMS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = Config.load(args.config)
    cfg.seed = args.seed
    vol = load_volume(args.volume, tuple(float(x) for x in args.spacing.split(",")))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def checkpoint(arm, result, field):
        _save_json(out / f"{arm}.json", result)
        field.save(out / f"{arm}.pt")
        print(f"FINISHED {arm}: {json.dumps(result['final'])}", flush=True)

    with (out / "progress.jsonl").open("w") as progress:
        def log(row):
            progress.write(json.dumps(row) + "\n")
            progress.flush()
            if "event" in row or row.get("iteration", 0) % 2500 == 0:
                print(json.dumps(row), flush=True)

        segments = [int(x) for x in args.segments.split(",")] if args.segments else None
        arms = tuple(args.arms.split(",")) if args.arms else ARMS
        result = run(vol, cfg, rounds=args.rounds, cycle_iters=args.cycle_iters, log=log, on_arm=checkpoint,
                     segments=segments, arms=arms)
    _save_json(out / "allocation.json", result)


if __name__ == "__main__":
    main()
