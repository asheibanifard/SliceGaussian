"""Plot measured PSNR against synchronized end-to-end fitting time and tabulate six quality targets."""
import argparse
import csv
import json
from pathlib import Path

from slicegaussian.timing import first_crossing

THRESHOLDS = (28., 30., 32., 34., 36., 38.)
METHODS = {"slicegaussian": ("SliceGaussian (3 cycles)", "#246eb9"),
           "gaussianpile": ("GaussianPile", "#d26936")}


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root):
    datasets, endpoints, crossings = {}, [], []
    for path in sorted(root.glob("cremi_*/*/result.json")):
        result = json.loads(path.read_text())
        with (path.parent / "timeline.csv").open() as f:
            trace = list(csv.DictReader(f))
        volume, method = path.parent.parent.name, result["method"]
        # Do not accept legacy 'seconds' as wall time or synthesize a missing timeline.
        for row in trace:
            row["wall_seconds"] = float(row["wall_seconds"])
            row["psnr"] = float(row["psnr"])
            row["iteration"] = int(row["iteration"])
        final = result["final"]
        if (volume, method) in datasets:
            raise ValueError(f"{volume}/{method}: duplicate runs; do not mix cache conditions")
        if not trace or trace[-1]["iteration"] != result["iterations"] or trace[-1]["wall_seconds"] > final["wall_seconds"]:
            raise ValueError(f"{volume}/{method}: incomplete or inconsistent wall timeline")
        if int(final["iteration"]) != result["iterations"] or result["iterations"] != 30000:
            raise ValueError(f"{volume}/{method}: incomplete run")
        for target in THRESHOLDS:
            crossings.append({"volume": volume, "method": method, "target_psnr": target,
                              **first_crossing(trace, target)})
        datasets[volume, method] = (result, trace)
        endpoints.append({"volume": volume, "method": method, "seed": result["seed"],
                          "iterations": final["iteration"], "final_psnr": final["psnr"],
                          "best_observed_psnr": max(r["psnr"] for r in trace),
                          "gaussians": final["gaussians"], "wall_seconds": final["wall_seconds"],
                          "accounted_fit_seconds": final["accounted_fit_seconds"],
                          "scheduled_evaluation_seconds": sum(float(r["evaluation_seconds"]) for r in trace
                                                    if r.get("evaluation_seconds")),
                          "setup_before_first_evaluation_seconds": float(trace[0]["wall_before_eval_seconds"]),
                          "checks_reestimation_seconds": sum(e["check_seconds"] for e in
                                                           result["info"].get("wall_events", []))})
    if not endpoints:
        raise SystemExit("No completed result.json files")
    write_csv(root / "thresholds.csv", crossings)
    write_csv(root / "endpoints.csv", endpoints)
    (root / "summary.json").write_text(json.dumps({"thresholds": crossings, "endpoints": endpoints}, indent=2))
    for e in endpoints:
        print(f"{e['volume']} {e['method']:14} final {e['final_psnr']:.3f} dB; "
              f"wall {e['wall_seconds']:.1f}s; accounted fit {e['accounted_fit_seconds']:.1f}s")
    for volume in sorted({v for v, m in datasets}):
        print(volume, "threshold dB | SliceGaussian seconds | GaussianPile seconds")
        for target in THRESHOLDS:
            vals = []
            for method in METHODS:
                found = next((r for r in crossings if r['volume'] == volume and r['method'] == method
                              and r['target_psnr'] == target), None)
                vals.append("pending" if found is None else
                            f"{found['seconds']:.1f}" if found['status'] == 'reached' else "NR")
            print(f" {target:.0f} | {' | '.join(vals)}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    volumes = sorted({v for v, m in datasets})
    with plt.rc_context({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7}):
        fig, axes = plt.subplots(1, len(volumes), figsize=(7.2, 2.7), squeeze=False)
        for ax, volume in zip(axes[0], volumes):
            for threshold in THRESHOLDS:
                ax.axhline(threshold, color=".7", linewidth=.45, linestyle="--", zorder=0)
            for method, (label, color) in METHODS.items():
                if (volume, method) not in datasets:
                    continue
                result, trace = datasets[volume, method]
                ax.plot([r["wall_seconds"] for r in trace], [r["psnr"] for r in trace],
                        color=color, linewidth=1.2, label=label)
                final = result["final"]
                ax.plot(final["wall_seconds"], final["psnr"], "o", color=color, markersize=4)
                if method == "slicegaussian":
                    for event in result["info"]["wall_events"]:
                        if event["timing_event"] == "reestimate":
                            ax.axvspan(event["check_start_wall_seconds"], event["wall_seconds"],
                                       color=color, alpha=.2)
                            ax.plot(event["wall_seconds"], event["psnr"], "v", color=color, markersize=4)
            ax.set(title=volume.upper().replace("_", " ") + (" (valid slices)" if volume.endswith("c") else ""),
                   xlabel="End-to-end wall time (s)", ylabel="3D PSNR (dB)", ylim=(5, 41), xlim=(0, None))
            ax.grid(axis="x", alpha=.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        handles += [Line2D([], [], marker="o", color=".3", linestyle="None", markersize=4),
                    Line2D([], [], marker="v", color="#246eb9", linestyle="None", markersize=4),
                    Line2D([], [], color=".7", linestyle="--", linewidth=.6)]
        labels += ["Complete run", "After re-estimation", "Quality thresholds (28:2:38 dB)"]
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig.tight_layout(rect=(0, .14, 1, 1))
        fig.savefig(root / "psnr_wallclock.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("runs/wallclock"))
    summarize(parser.parse_args().root)
