"""Fresh-process, synchronized end-to-end fitting timelines for SliceGaussian and GaussianPile.

Run through reproduce_wallclock.py, or pass --method and a single volume.
The clock begins before package imports and data loading. Evaluation is included.
Model export/plotting happen after the clock stops. Existing extension caches are retained.
"""
import time
PROCESS_START = time.perf_counter()

import argparse
import csv
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("volume")
    p.add_argument("--method", choices=["slicegaussian", "gaussianpile"], required=True)
    p.add_argument("--config", default="configs/paper.json")
    p.add_argument("--gaussianpile-root", type=Path,
                   default=Path(__file__).resolve().parents[2] / "gaussianpile")
    p.add_argument("--iterations", type=int, default=30000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.iterations < 1 or args.eval_every < 1:
        p.error("iterations and eval-every must be positive")

    import numpy as np
    import torch
    from slicegaussian import Config, load_volume
    from slicegaussian.pipeline import environment, seed_everything
    from slicegaussian import metrics, reestimate
    seed_everything(0)
    vol = load_volume(args.volume, (40, 4, 4))
    cfg = Config.load(args.config)
    cfg.seed = cfg.p1.seed = cfg.p2.seed = 0
    cfg.p2.eval_every = args.eval_every
    cfg.p2.eval_at = "0"  # common evaluation cadence for both methods
    cfg.p2.full_every = 5000
    cfg.p2.snapshot_every = 0
    cfg.codec = False
    args.out.mkdir(parents=True, exist_ok=True)
    log_file = (args.out / "progress.jsonl").open("w")

    def log(row):
        log_file.write(json.dumps(row) + "\n")
        log_file.flush()
        if "event" in row or row.get("iteration", 0) % 2500 == 0:
            print(json.dumps(row), flush=True)

    if args.method == "slicegaussian":
        if args.iterations % 3:
            raise ValueError("SliceGaussian iterations must divide into three equal cycles")
        re = cfg.reestimation
        field, info, trace = reestimate.run_cycles(
            vol, cfg.p1, cfg.p2, cycles=3, cycle_iters=args.iterations // 3,
            budget=re.budget, target=None, refit=re.refit, log=log, wall_start=PROCESS_START)
        endpoint = info["wall_events"][-1]
        final = {**trace[-1], "wall_seconds": info["wall_seconds"],
                 "accounted_fit_seconds": info["total_seconds"], "gaussians": len(field)}
        timeline = [{**r, "timing_event": "evaluation"} for r in trace] + info["wall_events"]
        timeline.sort(key=lambda r: r["wall_seconds"])
        final["psnr"] = endpoint["psnr"]
        # Export is outside the fitting clock for both methods.
        field.save(args.out / "model.pt")
        baseline_config = None
    else:
        sys.path.insert(0, str(args.gaussianpile_root.resolve()))
        from gaussianpile import GaussianModel, RenderSettings, render_slice, render_stack, stack_cameras
        from gaussianpile.losses import gaussianpile_loss
        cams = stack_cameras(vol.Z, vol.H, vol.W, vol.spacing())
        settings = RenderSettings(sigma_z=0., norm_mode="paper")
        hi = [(vol.W - 1) * vol.dx, (vol.H - 1) * vol.dy, (vol.Z - 1) * vol.dz]
        extent = float(np.linalg.norm(hi) / 2)
        model = GaussianModel.random_init(1_000_000, [0., 0., 0.], hi, device="cuda", seed=0)
        model.setup_optimizer(extent, max_iters=args.iterations, schedule="paper")
        torch.cuda.synchronize()
        init_seconds = time.perf_counter() - PROCESS_START
        train_start, evaluation_seconds = time.perf_counter(), 0.
        timeline = []

        @torch.no_grad()
        def evaluate(it):
            nonlocal evaluation_seconds
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            accounted = t0 - train_start - evaluation_seconds
            pred = render_stack(model.means, model.scales, model.quats, model.alphas, cams, settings)
            scores = metrics.scores(pred[vol.valid_ids], vol.gt[vol.valid_ids], vol,
                                    full=it == 0 or it == args.iterations or it % 5000 == 0)
            torch.cuda.synchronize()
            end = time.perf_counter()
            dt = end - t0
            evaluation_seconds += dt
            row = {"iteration": it, "gaussians": model.num, "timing_event": "evaluation", **scores,
                   "seconds": accounted, "wall_before_eval_seconds": t0 - PROCESS_START,
                   "wall_seconds": end - PROCESS_START, "evaluation_seconds": dt}
            timeline.append(row)
            log(row)

        evaluate(0)
        for it in range(1, args.iterations + 1):
            model.update_lr(it)
            # Preserve the original paper baseline's sampling of all planes, including blank C planes.
            k = int(torch.randint(vol.Z, (1,)))
            means2d = torch.zeros(model.num, 2, device="cuda", requires_grad=True)
            image, radii = render_slice(model.means, model.scales, model.quats, model.alphas,
                                        cams[k], settings, means2d)
            loss = gaussianpile_loss(image, vol.gt[k], .2)
            loss.backward()
            with torch.no_grad():
                if it <= 25000:
                    model.add_stats(means2d.grad, radii)
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                if 500 < it <= 25000 and it % 100 == 0:
                    model.densify_and_prune(5e-5, .02, extent)
            if it % args.eval_every == 0 or it == args.iterations:
                evaluate(it)
        torch.cuda.synchronize()
        end = time.perf_counter()
        final = {**timeline[-1], "wall_seconds": end - PROCESS_START,
                 "accounted_fit_seconds": init_seconds + end - train_start - evaluation_seconds}
        info = {"init_seconds": init_seconds, "evaluation_seconds": evaluation_seconds}
        torch.save(model.state(), args.out / "model.pt")
        baseline_config = {"source": str(args.gaussianpile_root.resolve()), "initial_gaussians": 1000000,
                           "sigma_z": 0., "normalisation": "paper", "lambda_ssim": .2,
                           "densify_from_exclusive": 500, "densify_until": 25000, "densify_every": 100,
                           "gradient_threshold": 5e-5, "min_alpha": .02,
                           "training_slice_policy": "all planes (original paper baseline)"}
    log_file.close()
    keys = sorted({k for r in timeline for k in r})
    with (args.out / "timeline.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(timeline)
    from dataclasses import asdict
    result = {"method": args.method, "volume": vol.describe(), "config": asdict(cfg),
              "iterations": args.iterations, "eval_every": args.eval_every, "seed": 0,
              "environment": environment(), "baseline_config": baseline_config, "info": info,
              "final": final, "clock": {"start": "before Python package imports and volume loading",
              "end": "after final synchronized evaluation/adequacy check; before export",
              "includes": ["imports", "volume loading/normalisation", "CUDA context/setup", "initialisation",
                           "Phase I", "training CUDA operations", "optimizer setup", "re-estimation/checks",
                           "PSNR evaluations", "scheduled full metric evaluations", "progress logging"],
              "extension_cache": "existing compiled extension cache retained for both methods"}}
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"finished": args.method, "final": final}), flush=True)


if __name__ == "__main__":
    main()
