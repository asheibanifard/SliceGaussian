"""Summarize completed single-seed allocation runs; create CSV, JSON and PNG curves."""
import argparse
import csv
import json
from pathlib import Path
from itertools import accumulate


def summarize(root):
    rows, pairs = [], []
    completed = []
    for path in sorted(root.glob("cremi_*/allocation.json")):
        result = json.loads(path.read_text())
        volume = path.parent.name
        completed.append((volume, result))
        protocol = result["protocol"]
        arms = result["arms"]
        expected_updates = (protocol["rounds"] + 1) * protocol["cycle_iters"]
        for arm, data in arms.items():
            if data["final"]["gaussians"] != protocol["target_count"]:
                raise ValueError(f"{volume}/{arm}: unmatched capacity")
            if data["trace"][-1]["iteration"] != expected_updates:
                raise ValueError(f"{volume}/{arm}: incomplete training")
        frozen = [[(e["indices"], e["axes"]) for e in arms[name]["splits"]]
                  for name in ("violation_upfront", "violation_replay")]
        if frozen[0] != frozen[1]:
            raise ValueError(f"{volume}: timing pair used different split decisions")
        for arm, data in arms.items():
            final = data["final"]
            trace = data["trace"]
            gaussian_updates = sum((r["iteration"] - prev["iteration"]) * r["gaussians"]
                                   for prev, r in zip(trace, trace[1:]))
            rows.append({"gaussian_updates": gaussian_updates, "volume": volume, "arm": arm, "seed": protocol["seed"],
                         **{key: final[key] for key in ("psnr", "ssim", "mae", "gaussians", "total_seconds")},
                         "optimisation_seconds": data["trace"][-1]["seconds"],
                         "shared_preparation_seconds": protocol["shared_analysis_seconds"],
                         "frozen_planning_seconds": protocol["planning_seconds"] if arm in
                         ("violation_upfront", "violation_replay") else 0.,
                         "allocation_seconds": sum(e["seconds"] for e in data["splits"]),
                         "peak_vram_gb": max(r["peak_vram_gb"] for r in data["trace"])})
        upfront, replay = (arms[name] for name in ("violation_upfront", "violation_replay"))
        budget = min(upfront["final"]["total_seconds"], replay["final"]["total_seconds"])
        at_budget = [max((r for r in data["trace"] if r["total_seconds"] <= budget),
                         key=lambda r: r["total_seconds"]) for data in (upfront, replay)]
        best_at_budget = [max((r for r in data["trace"] if r["total_seconds"] <= budget),
                              key=lambda r: r["psnr"]) for data in (upfront, replay)]
        gaussianpile_target = {"cremi_a": 32.33, "cremi_b": 31.11, "cremi_c": 31.57}[volume]
        time_to_gaussianpile = {}
        for arm, data in arms.items():
            reached = [r["total_seconds"] for r in data["trace"] if r["psnr"] >= gaussianpile_target]
            time_to_gaussianpile[arm] = min(reached) if reached else None
        target = arms["phase1_upfront"]["final"]["psnr"] - 1.0
        time_to_target = {}
        for arm, data in arms.items():
            reached = [r["total_seconds"] for r in data["trace"] if r["psnr"] >= target]
            time_to_target[arm] = min(reached) if reached else None
        pairs.append({"volume": volume, "gaussianpile_target_psnr": gaussianpile_target,
                      "time_to_gaussianpile_seconds": time_to_gaussianpile,
                      "target_psnr_reference_minus_1db": target,
                      "time_to_target_seconds": time_to_target,
                      "upfront_best_evaluation_at_budget": best_at_budget[0],
                      "replay_best_evaluation_at_budget": best_at_budget[1],
                      "best_so_far_time_matched_psnr_difference": best_at_budget[0]["psnr"] - best_at_budget[1]["psnr"],
                      "final_upfront_minus_replay": result["paired_upfront_minus_replay"],
                      "common_time_budget_seconds": budget,
                      "upfront_last_evaluation_at_budget": at_budget[0],
                      "replay_last_evaluation_at_budget": at_budget[1],
                      "time_matched_psnr_difference": at_budget[0]["psnr"] - at_budget[1]["psnr"]})
    if not rows:
        raise SystemExit("No completed allocation.json files")
    with (root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.json").write_text(json.dumps({"rows": rows, "paired": pairs}, indent=2))
    print("Volume  Arm                   PSNR     SSIM     MAE      Count    Total(s)")
    for row in rows:
        print(f"{row['volume']:8} {row['arm']:21} {row['psnr']:7.3f} {row['ssim']:8.5f} "
              f"{row['mae']:8.5f} {row['gaussians']:8} {row['total_seconds']:9.1f}")
    for pair in pairs:
        print(f"{pair['volume']} timing pair: final ΔPSNR "
              f"{pair['final_upfront_minus_replay']['psnr']:+.3f} dB; "
              f"last evaluated within {pair['common_time_budget_seconds']:.1f}s: "
              f"ΔPSNR {pair['time_matched_psnr_difference']:+.3f} dB; "
              f"best observed ΔPSNR {pair['best_so_far_time_matched_psnr_difference']:+.3f} dB")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(completed), 4, figsize=(20, 4 * len(completed)), squeeze=False)
    for i, (volume, result) in enumerate(completed):
        for arm, data in result["arms"].items():
            trace = data["trace"]
            axs[i, 0].plot([r["iteration"] for r in trace], [r["psnr"] for r in trace], label=arm)
            axs[i, 1].plot([r["total_seconds"] for r in trace], [r["psnr"] for r in trace], label=arm)
            axs[i, 2].plot([r["total_seconds"] for r in trace],
                           list(accumulate((r["psnr"] for r in trace), max)), label=arm)
            axs[i, 3].step([r["iteration"] for r in trace], [r["gaussians"] for r in trace],
                           label=arm, where="pre")
        for ax, xlabel, ylabel in zip(axs[i], ["Updates", "Total seconds (evaluation excluded)", "Total seconds (evaluation excluded)", "Updates"],
                                     ["PSNR (dB)", "PSNR (dB)", "Best observed PSNR (dB)", "Gaussians"]):
            ax.set(xlabel=xlabel, ylabel=ylabel, title=volume.upper())
            ax.grid(alpha=.25)
        axs[i, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "curves.png", dpi=180)
    plt.close(fig)
    labels = {"phase1_upfront": "Phase I (upfront)", "violation_upfront": "Violation (offline)",
              "violation_replay": "Violation (replay)", "violation_online": "Violation (adaptive)",
              "structure_online": "Structure (adaptive)"}
    with plt.rc_context({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7}):
        fig, axes = plt.subplots(len(completed), 2, figsize=(7, 2.3 * len(completed) + .4), squeeze=False)
        for i, (volume, result) in enumerate(completed):
            for arm, data in result["arms"].items():
                trace = data["trace"]
                axes[i, 0].plot([r["iteration"] / 1000 for r in trace], [r["psnr"] for r in trace],
                                label=labels[arm], linewidth=1.2)
                axes[i, 1].plot([r["total_seconds"] for r in trace],
                                list(accumulate((r["psnr"] for r in trace), max)),
                                label=labels[arm], linewidth=1.2)
            axes[i, 0].set(xlabel="Updates (thousands)", ylabel="PSNR (dB)", title=volume.upper().replace("_", " "))
            axes[i, 1].set(xlabel="Total seconds (evaluation excluded)", ylabel="Best observed PSNR (dB)",
                           title=volume.upper().replace("_", " "))
            for ax in axes[i]:
                ax.grid(alpha=.25)
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, names, loc="lower center", ncol=3, frameon=False)
        fig.tight_layout(rect=(0, .065, 1, 1))
        fig.savefig(root / "paper_curves.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("runs/allocation"))
    summarize(parser.parse_args().root)
