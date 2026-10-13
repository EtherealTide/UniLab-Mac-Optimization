#!/usr/bin/env python3
"""Extract TensorBoard curves and render one loss-vs-wall-clock figure."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

LOSS_TAGS = ("Loss/critic", "Loss/actor", "Loss/temperature")
EXTRA_TAGS = ("Train/mean_reward", "Perf/total_fps", "Perf/iteration_time")

ALGO_LABELS = {
    "sac_g1_walk_flat": "SAC · G1 walk flat",
    "flashsac_g1_motion_tracking": "FlashSAC · G1 motion tracking",
    "flashmimic_g1_motion_tracking": "FlashMimic task · g1_flash_mimic/mujoco · FlashSAC",
}
ALGO_COLORS = {
    "sac_g1_walk_flat": "#0072B2",
    "flashsac_g1_motion_tracking": "#D55E00",
    "flashmimic_g1_motion_tracking": "#009E73",
}
MACHINE_STYLES = {
    "mac-mini-m5-pro": "-",
    "mac-studio-m5-ultra": "--",
}
MACHINE_LABELS = {
    "mac-mini-m5-pro": "Mac mini M5 Pro",
    "mac-studio-m5-ultra": "Mac Studio M5 Ultra",
}


def _find_event_file(run_dir: Path) -> Path:
    files = sorted(run_dir.glob("events.out.tfevents.*"))
    if len(files) != 1:
        raise RuntimeError(
            f"expected one TensorBoard event file in {run_dir}, found {len(files)}"
        )
    return files[0]


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    result: list[float] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        result.append(running / min(index + 1, window))
    return result


def _load_protocol(path: Path) -> dict[str, int]:
    expected: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row or row[0].startswith("#"):
                continue
            if len(row) != 6:
                raise RuntimeError(f"invalid experiment row in {path}: {row}")
            expected[row[0]] = int(row[5])
    if set(expected) != set(ALGO_LABELS):
        raise RuntimeError(
            f"experiment matrix and plot labels differ: {sorted(expected)} vs "
            f"{sorted(ALGO_LABELS)}"
        )
    return expected


def _load_versions(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise RuntimeError(f"invalid version row in {path}: {raw_line}")
        versions[key] = value
    return versions


def _load_run(
    run_dir: Path,
    expected_iterations: dict[str, int],
    versions: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    summary_path = run_dir / "run_summary.json"
    benchmark_end_path = run_dir / "benchmark_end.json"
    benchmark_start_path = run_dir / "benchmark_start.json"
    for required in (summary_path, benchmark_start_path, benchmark_end_path):
        if not required.exists():
            raise RuntimeError(f"missing required result artifact: {required}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    benchmark_start = json.loads(benchmark_start_path.read_text(encoding="utf-8"))
    benchmark_end = json.loads(benchmark_end_path.read_text(encoding="utf-8"))
    machine = str(benchmark_start["machine_id"])
    experiment = str(benchmark_start["experiment_id"])
    if machine not in MACHINE_LABELS:
        raise RuntimeError(f"unexpected machine id in {run_dir}: {machine}")
    if experiment not in expected_iterations:
        raise RuntimeError(f"unexpected experiment id in {run_dir}: {experiment}")
    if run_dir.name != experiment or run_dir.parent.name != machine:
        raise RuntimeError(
            f"result path does not match metadata: {run_dir} vs {machine}/{experiment}"
        )
    expected_count = expected_iterations[experiment]
    expected_commit = (
        versions["FLASHMIMIC_COMMIT"]
        if experiment == "flashmimic_g1_motion_tracking"
        else versions["UNILAB_COMMIT"]
    )
    checks = {
        "run status": (summary.get("status"), "completed"),
        "completed iterations": (summary.get("completed_iterations"), expected_count),
        "requested iterations": (benchmark_start.get("iterations"), expected_count),
        "seed": (benchmark_start.get("seed"), 1),
        "wrapper exit code": (benchmark_end.get("exit_code"), 0),
        "end experiment id": (benchmark_end.get("experiment_id"), experiment),
        "end machine id": (benchmark_end.get("machine_id"), machine),
        "configured seed": (config["run"].get("configured_seed"), 1),
        "effective seed": (config["run"].get("effective_seed"), 1),
        "simulation backend": (config["run"].get("sim_backend"), "mujoco"),
        "learner device": (config["run"].get("device"), "mps"),
        "git commit": (config["run"]["git"].get("commit"), expected_commit),
        "git dirty": (config["run"]["git"].get("dirty"), False),
        "configured environments": (config["config"]["algo"].get("num_envs"), 2048),
        "configured max iterations": (
            config["config"]["algo"].get("max_iterations"),
            expected_count,
        ),
        "TensorBoard interval": (config["config"]["training"].get("log_interval"), 1),
        "no play": (config["config"]["training"].get("no_play"), True),
    }
    failures = {
        label: {"actual": actual, "expected": expected}
        for label, (actual, expected) in checks.items()
        if actual != expected
    }
    if failures:
        raise RuntimeError(f"invalid canonical run {run_dir}: {failures}")

    # Wrapper start precedes CLI imports and tracker construction. Its epoch
    # timestamp therefore gives the requested end-to-end wall-clock origin.
    run_start = float(benchmark_start["start_epoch_sec"])

    accumulator = event_accumulator.EventAccumulator(str(_find_event_file(run_dir)))
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    missing_loss_tags = set(LOSS_TAGS) - available
    if missing_loss_tags:
        raise RuntimeError(
            f"missing loss tags in {run_dir}: {sorted(missing_loss_tags)}"
        )
    rows: list[dict[str, Any]] = []
    for tag in (*LOSS_TAGS, *EXTRA_TAGS):
        if tag not in available:
            continue
        for scalar in accumulator.Scalars(tag):
            rows.append(
                {
                    "machine": machine,
                    "experiment": experiment,
                    "metric": tag,
                    "step": scalar.step,
                    "wall_time_sec": max(0.0, scalar.wall_time - run_start),
                    "value": scalar.value,
                }
            )

    by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_metric[row["metric"]].append(row)
    for values in by_metric.values():
        values.sort(key=lambda item: (item["step"], item["wall_time_sec"]))

    result_summary: dict[str, Any] = {
        "machine": machine,
        "experiment": experiment,
        "status": summary.get("status"),
        "completed_iterations": summary.get("completed_iterations"),
        "total_env_steps": summary.get("total_env_steps"),
        "training_wall_time_sec": summary.get("training_wall_time_sec"),
        "wall_time_sec": summary.get("wall_time_sec"),
        "benchmark_elapsed_sec": benchmark_end.get("elapsed_sec"),
        "final_mean_reward": summary.get("final_mean_reward"),
        "final_env_steps_per_sec": summary.get("final_env_steps_per_sec"),
        "tail_env_steps_per_sec": summary.get("tail_env_steps_per_sec"),
    }
    for tag in (*LOSS_TAGS, "Perf/total_fps"):
        values = by_metric.get(tag, [])
        result_summary[f"last_{tag.replace('/', '_').lower()}"] = (
            values[-1]["value"] if values else None
        )
    return rows, result_summary


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, Any]], output: Path, smoothing: int) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] in LOSS_TAGS:
            grouped[(row["metric"], row["experiment"], row["machine"])].append(row)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    for axis, tag in zip(axes, LOSS_TAGS):
        for (metric, experiment, machine), series in sorted(grouped.items()):
            if metric != tag:
                continue
            series.sort(key=lambda item: item["wall_time_sec"])
            x = [item["wall_time_sec"] / 60.0 for item in series]
            y = _rolling_mean([item["value"] for item in series], smoothing)
            label = f"{ALGO_LABELS.get(experiment, experiment)} · {MACHINE_LABELS.get(machine, machine)}"
            axis.plot(
                x,
                y,
                color=ALGO_COLORS.get(experiment),
                linestyle=MACHINE_STYLES.get(machine, "-"),
                linewidth=2.0,
                label=label,
            )
        axis.set_ylabel(tag.removeprefix("Loss/").title())
        axis.grid(True, alpha=0.25)
        if not axis.lines:
            axis.text(
                0.5,
                0.5,
                "No scalar data",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

    axes[-1].set_xlabel("End-to-end wall-clock time from benchmark start (minutes)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=2,
            frameon=False,
        )
    fig.suptitle(
        f"UniLab off-policy loss vs wall-clock time (rolling mean, n={smoothing})",
        fontsize=15,
        y=0.99,
    )
    fig.subplots_adjust(top=0.82, bottom=0.07, left=0.08, right=0.98, hspace=0.04)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/raw"))
    parser.add_argument(
        "--experiments", type=Path, default=Path("config/experiments.tsv")
    )
    parser.add_argument("--versions", type=Path, default=Path("config/versions.env"))
    parser.add_argument("--processed", type=Path, default=Path("results/processed"))
    parser.add_argument(
        "--figure", type=Path, default=Path("figures/loss_wall_clock.png")
    )
    parser.add_argument("--smoothing", type=int, default=10)
    args = parser.parse_args()

    expected_iterations = _load_protocol(args.experiments)
    versions = _load_versions(args.versions)
    run_dirs = sorted(path.parent for path in args.input.rglob("run_config.json"))
    if not run_dirs:
        raise SystemExit(f"no run_config.json files found under {args.input}")

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        rows, summary = _load_run(run_dir, expected_iterations, versions)
        all_rows.extend(rows)
        summaries.append(summary)

    observed_pairs = [(row["experiment"], row["machine"]) for row in summaries]
    pair_counts = Counter(observed_pairs)
    duplicate_pairs = {pair: count for pair, count in pair_counts.items() if count != 1}
    expected_pairs = {
        (experiment, machine)
        for experiment in expected_iterations
        for machine in MACHINE_LABELS
    }
    observed_set = set(observed_pairs)
    if duplicate_pairs or observed_set != expected_pairs:
        raise RuntimeError(
            "canonical result matrix mismatch: "
            f"missing={sorted(expected_pairs - observed_set)}, "
            f"unexpected={sorted(observed_set - expected_pairs)}, "
            f"duplicates={duplicate_pairs}"
        )

    all_rows.sort(
        key=lambda row: (row["experiment"], row["machine"], row["metric"], row["step"])
    )
    summaries.sort(key=lambda row: (row["experiment"], row["machine"]))
    _write_csv(
        args.processed / "curves.csv",
        all_rows,
        ["machine", "experiment", "metric", "step", "wall_time_sec", "value"],
    )
    summary_fields = sorted({key for row in summaries for key in row})
    _write_csv(args.processed / "summary.csv", summaries, summary_fields)
    _plot(all_rows, args.figure, args.smoothing)

    print(f"Processed and validated {len(observed_set)}/{len(expected_pairs)} runs")
    print(args.processed / "curves.csv")
    print(args.processed / "summary.csv")
    print(args.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
