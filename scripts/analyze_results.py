#!/usr/bin/env python3
"""Validate benchmark artifacts and render reward-vs-wall-clock results."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

REWARD_TAG = "Train/mean_reward"
EXTRA_TAGS = ("Perf/total_fps", "Perf/iteration_time")

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
    if not files:
        raise RuntimeError(f"no TensorBoard event file in {run_dir}")
    if len(files) == 1:
        return files[0]
    # A collection can retain a previous attempt beside a newly copied run.
    # Event filenames contain the writer epoch; use the newest writer rather
    # than silently concatenating independent attempts.
    return max(files, key=lambda path: (path.name, path.stat().st_mtime_ns))


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


def _load_protocol(path: Path) -> set[str]:
    experiments: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row or row[0].startswith("#"):
                continue
            if len(row) != 6:
                raise RuntimeError(f"invalid experiment row in {path}: {row}")
            int(row[5])
            experiments.add(row[0])
    if experiments != set(ALGO_LABELS):
        raise RuntimeError(
            f"experiment matrix and plot labels differ: {sorted(experiments)} vs {sorted(ALGO_LABELS)}"
        )
    return experiments


def _tail_half_throughput(
    iteration_events: list[dict[str, Any]],
    *,
    run_dir: Path | None = None,
) -> dict[str, int | float]:
    """Aggregate environment-step throughput over the latter half of a run."""
    if len(iteration_events) < 2:
        raise RuntimeError("tail-half throughput requires at least two iterations")

    start = len(iteration_events) // 2
    selected = iteration_events[start:]
    previous_step = int(iteration_events[start - 1]["step"])
    total_steps = 0
    total_time = 0.0
    for selected_index, event in enumerate(selected, start=start):
        step = int(event["step"])
        step_delta = step - previous_step
        iteration_time = float(event["value"])
        if step_delta < 0:
            context = f" in {run_dir}" if run_dir is not None else ""
            raise RuntimeError(
                f"negative TensorBoard step delta: {step_delta}{context}; "
                f"previous_step={previous_step}, step={step}, "
                f"event_index={selected_index}. The event log "
                "contains a non-monotonic Perf/iteration_time step; "
                "recollect this run into an empty result directory rather than "
                "combining/resuming event files."
            )
        if iteration_time <= 0:
            raise RuntimeError(f"non-positive iteration time: {iteration_time}")
        total_steps += step_delta
        total_time += iteration_time
        previous_step = step

    return {
        "tail_half_iteration_count": len(selected),
        "tail_half_env_steps": total_steps,
        "tail_half_iteration_time_sec": total_time,
        "tail_half_env_steps_per_sec": total_steps / total_time,
    }


def _load_run(
    run_dir: Path,
    experiments: set[str],
    *,
    require_video: bool,
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
    if experiment not in experiments:
        raise RuntimeError(f"unexpected experiment id in {run_dir}: {experiment}")
    # The local collection destination is user-selectable. Use the machine ID
    # recorded by the benchmark rather than requiring the parent directory to
    # have a particular spelling (for example mac-ultra-m5-ultra).
    if run_dir.name != experiment:
        raise RuntimeError(
            f"result path does not match experiment metadata: {run_dir} vs {experiment}"
        )
    completed_iterations = summary.get("completed_iterations")
    if not isinstance(completed_iterations, int) or completed_iterations < 2:
        raise RuntimeError(
            f"invalid completed iteration count in {run_dir}: {completed_iterations!r}"
        )
    checks = {
        "run status": (summary.get("status"), "completed"),
        "requested iterations": (benchmark_start.get("iterations"), completed_iterations),
        "seed": (benchmark_start.get("seed"), 1),
        "wrapper exit code": (benchmark_end.get("exit_code"), 0),
        "end experiment id": (benchmark_end.get("experiment_id"), experiment),
        "end machine id": (benchmark_end.get("machine_id"), machine),
        "configured seed": (config["run"].get("configured_seed"), 1),
        "effective seed": (config["run"].get("effective_seed"), 1),
        "simulation backend": (config["run"].get("sim_backend"), "mujoco"),
        "learner device": (config["run"].get("device"), "mps"),
        "configured environments": (config["config"]["algo"].get("num_envs"), 2048),
        "configured max iterations": (
            config["config"]["algo"].get("max_iterations"),
            completed_iterations,
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
    video_path = run_dir / "play_video.mp4"
    if require_video and (not video_path.is_file() or video_path.stat().st_size == 0):
        raise RuntimeError(f"missing or empty final policy video: {video_path}")

    # Wrapper start precedes CLI imports and tracker construction. Its epoch
    # timestamp therefore gives the requested end-to-end wall-clock origin.
    run_start = float(benchmark_start["start_epoch_sec"])

    # EventAccumulator otherwise retains only the latest 10,000 scalar events.
    # Load the complete stream so long default runs are analyzed in full.
    accumulator = event_accumulator.EventAccumulator(
        str(_find_event_file(run_dir)),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    if REWARD_TAG not in available:
        raise RuntimeError(f"missing reward tag {REWARD_TAG!r} in {run_dir}")
    rows: list[dict[str, Any]] = []
    for tag in (REWARD_TAG, *EXTRA_TAGS):
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
    raw_iteration_events = by_metric.get("Perf/iteration_time", [])
    if len(raw_iteration_events) < 2:
        raise RuntimeError(
            f"tail-half throughput requires at least two Perf/iteration_time "
            f"samples in {run_dir}, found {len(raw_iteration_events)}"
        )
    reward_events = by_metric.get(REWARD_TAG, [])
    if not reward_events:
        raise RuntimeError(
            f"missing {REWARD_TAG!r} samples in {run_dir}"
        )
    duplicate_steps: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    previous_step: int | None = None
    previous_wall_time: float | None = None
    for event_index, event in enumerate(raw_iteration_events):
        step = int(event["step"])
        wall_time = float(event["wall_time_sec"])
        value = float(event["value"])
        duplicate_steps[step].append((event_index, wall_time, value))
        if value <= 0:
            raise RuntimeError(
                f"non-positive iteration time in {run_dir}: event_index="
                f"{event_index}, step={step}, value={value}"
            )
        if previous_step is not None and step < previous_step:
            detail = duplicate_steps[step]
            raise RuntimeError(
                f"Perf/iteration_time steps moved backwards in {run_dir}: "
                f"event_index={event_index}, previous_step={previous_step}, "
                f"step={step}, records_for_step={detail}. This usually means the "
                "run was resumed or logs from multiple attempts were combined; "
                "recollect into a fresh result directory."
            )
        if previous_wall_time is not None and wall_time < previous_wall_time:
            raise RuntimeError(
                f"Perf/iteration_time wall times are not monotonic in {run_dir}: "
                f"event_index={event_index}, previous_wall_time={previous_wall_time}, "
                f"wall_time={wall_time}, step={step}"
            )
        previous_step = step
        previous_wall_time = wall_time
    for values in by_metric.values():
        values.sort(key=lambda item: (item["step"], item["wall_time_sec"]))

    iteration_events = by_metric.get("Perf/iteration_time", [])
    tail_half = _tail_half_throughput(iteration_events, run_dir=run_dir)

    result_summary: dict[str, Any] = {
        "machine": machine,
        "experiment": experiment,
        "status": summary.get("status"),
        "completed_iterations": summary.get("completed_iterations"),
        "iteration_time_samples": len(iteration_events),
        "total_env_steps": summary.get("total_env_steps"),
        "training_wall_time_sec": summary.get("training_wall_time_sec"),
        "wall_time_sec": summary.get("wall_time_sec"),
        "benchmark_elapsed_sec": benchmark_end.get("elapsed_sec"),
        "final_mean_reward": summary.get("final_mean_reward"),
        "final_env_steps_per_sec": summary.get("final_env_steps_per_sec"),
        # Preserve the actual local collection directory (which may be an
        # alias) so report links point to the file that was analyzed.
        "video_path": f"{run_dir.parent.name}/{experiment}/play_video.mp4",
        "video_available": video_path.is_file() and video_path.stat().st_size > 0,
        **tail_half,
    }
    for tag in (REWARD_TAG, "Perf/total_fps"):
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
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] == REWARD_TAG:
            grouped[(row["experiment"], row["machine"])].append(row)

    # Each recipe has a different default iteration budget and therefore a
    # different wall-clock duration. Keep one x scale per panel; sharing the
    # FlashMimic range makes the shorter SAC curves visually unreadable.
    fig, axes = plt.subplots(len(ALGO_LABELS), 1, figsize=(14, 13), sharex=False)
    for axis, experiment in zip(axes, ALGO_LABELS, strict=True):
        panel_max_minutes = 0.0
        for machine in MACHINE_LABELS:
            series = grouped.get((experiment, machine), [])
            series.sort(key=lambda item: item["wall_time_sec"])
            x = [item["wall_time_sec"] / 60.0 for item in series]
            y = _rolling_mean([item["value"] for item in series], smoothing)
            if x:
                panel_max_minutes = max(panel_max_minutes, max(x))
            axis.plot(
                x,
                y,
                color=ALGO_COLORS[experiment],
                linestyle=MACHINE_STYLES[machine],
                linewidth=2.0,
                label=MACHINE_LABELS[machine],
            )
        axis.set_title(ALGO_LABELS[experiment], loc="left")
        axis.set_ylabel("Mean reward")
        # Start at zero and add only a small margin after the last recorded
        # point, independently for this recipe's panel.
        axis.set_xlim(0.0, max(1.0, panel_max_minutes * 1.05))
        axis.set_xlabel("Wall-clock time from benchmark start (minutes)")
        axis.grid(True, alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="upper right", frameon=False, ncol=2)
    fig.suptitle(
        f"UniLab mean reward vs wall-clock time (rolling mean, n={smoothing})",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _format_float(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):,.{digits}f}"


def _write_report(
    path: Path,
    figure: Path,
    raw_root: Path,
    summaries: list[dict[str, Any]],
    *,
    smoothing: int,
) -> None:
    """Write the two-table report directly from validated run summaries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure_link = os.path.relpath(figure, path.parent)
    protocol_note = "Each row reports the completed iteration count recorded by that run."
    lines = [
        "# UniLab Apple Silicon performance report",
        "",
        "The figure and two tables below are generated from the validated "
        f"TensorBoard and run metadata artifacts. {protocol_note}",
        "",
        f"![Mean reward versus wall-clock time]({figure_link})",
        "",
        "| Machine | Experiment | Status | Iterations | Environment steps | "
        "Training wall time (s) | End-to-end wall time (s) | Tail-half "
        "throughput (steps/s) | Final mean reward | Final video |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        video_path = raw_root / str(row["video_path"])
        video = (
            f"[MP4]({os.path.relpath(video_path, path.parent)})"
            if row["video_available"]
            else "not collected"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    MACHINE_LABELS[row["machine"]],
                    ALGO_LABELS[row["experiment"]],
                    str(row["status"]),
                    f"{int(row['completed_iterations']):,}",
                    f"{int(row['total_env_steps']):,}",
                    _format_float(row["training_wall_time_sec"]),
                    _format_float(row["benchmark_elapsed_sec"], 0),
                    _format_float(row["tail_half_env_steps_per_sec"]),
                    _format_float(row["final_mean_reward"], 4),
                    video,
                )
            )
            + " |"
        )

    lines.extend(
        (
            "",
            "| Matched recipe | Studio / mini tail-half throughput | Studio "
            "training-time reduction | Studio end-to-end reduction |",
            "|---|---:|---:|---:|",
        )
    )
    by_pair = {(row["experiment"], row["machine"]): row for row in summaries}
    for experiment in ALGO_LABELS:
        mini = by_pair[(experiment, "mac-mini-m5-pro")]
        studio = by_pair[(experiment, "mac-studio-m5-ultra")]
        ratio = float(studio["tail_half_env_steps_per_sec"]) / float(
            mini["tail_half_env_steps_per_sec"]
        )
        training_reduction = 100.0 * (
            1.0 - float(studio["training_wall_time_sec"]) / float(mini["training_wall_time_sec"])
        )
        elapsed_reduction = 100.0 * (
            1.0 - float(studio["benchmark_elapsed_sec"]) / float(mini["benchmark_elapsed_sec"])
        )
        lines.append(
            f"| {ALGO_LABELS[experiment]} | {ratio:.2f}× | "
            f"{training_reduction:.1f}% | {elapsed_reduction:.1f}% |"
        )

    lines.extend(
        (
            "",
            "Throughput is the time-weighted environment-step rate over the latter "
            f"half of each run. Reward curves use a {smoothing}-event rolling mean; "
            "the CSV retains raw values. FlashMimic is the "
            "`g1_flash_mimic/mujoco` task package using the FlashSAC learner. "
            "Training timing excludes final policy playback; each MP4 is rendered "
            "after training from that run's final checkpoint.",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/raw"))
    parser.add_argument("--experiments", type=Path, default=Path("config/experiments.tsv"))
    parser.add_argument("--processed", type=Path, default=Path("results/processed"))
    parser.add_argument("--figure", type=Path, default=Path("figures/reward_wall_clock.png"))
    parser.add_argument("--report", type=Path, default=Path("REPORT.md"))
    parser.add_argument("--smoothing", type=int, default=10)
    parser.add_argument(
        "--allow-missing-videos",
        action="store_true",
        help="allow intermediate analysis before the six final MP4 files are collected",
    )
    args = parser.parse_args()

    if args.smoothing < 1:
        parser.error("--smoothing must be at least 1")
    experiments = _load_protocol(args.experiments)
    run_dirs = sorted(path.parent for path in args.input.rglob("run_config.json"))
    if not run_dirs:
        raise SystemExit(f"no run_config.json files found under {args.input}")

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        rows, summary = _load_run(
            run_dir,
            experiments,
            require_video=not args.allow_missing_videos,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    observed_pairs = [(row["experiment"], row["machine"]) for row in summaries]
    pair_counts = Counter(observed_pairs)
    duplicate_pairs = {pair: count for pair, count in pair_counts.items() if count != 1}
    expected_pairs = {
        (experiment, machine) for experiment in experiments for machine in MACHINE_LABELS
    }
    observed_set = set(observed_pairs)
    if duplicate_pairs or observed_set != expected_pairs:
        raise RuntimeError(
            "canonical result matrix mismatch: "
            f"missing={sorted(expected_pairs - observed_set)}, "
            f"unexpected={sorted(observed_set - expected_pairs)}, "
            f"duplicates={duplicate_pairs}"
        )

    all_rows.sort(key=lambda row: (row["experiment"], row["machine"], row["metric"], row["step"]))
    experiment_order = {name: index for index, name in enumerate(ALGO_LABELS)}
    machine_order = {name: index for index, name in enumerate(MACHINE_LABELS)}
    summaries.sort(
        key=lambda row: (
            experiment_order[row["experiment"]],
            machine_order[row["machine"]],
        )
    )
    _write_csv(
        args.processed / "curves.csv",
        all_rows,
        ["machine", "experiment", "metric", "step", "wall_time_sec", "value"],
    )
    summary_fields = sorted({key for row in summaries for key in row})
    _write_csv(args.processed / "summary.csv", summaries, summary_fields)
    _plot(all_rows, args.figure, args.smoothing)
    _write_report(
        args.report,
        args.figure,
        args.input,
        summaries,
        smoothing=args.smoothing,
    )

    print(f"Processed and validated {len(observed_set)}/{len(expected_pairs)} runs")
    print(args.processed / "curves.csv")
    print(args.processed / "summary.csv")
    print(args.figure)
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
