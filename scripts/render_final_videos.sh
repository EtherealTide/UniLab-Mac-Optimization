#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_REPO="$(cd -- "$SCRIPT_DIR/.." && pwd)"

export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
DEPLOY_ROOT="${DEPLOY_ROOT:-$HOME/unilab-mac-benchmark}"
UNILAB_DIR="${UNILAB_DIR:-$DEPLOY_ROOT/UniLab}"
FLASHMIMIC_DIR="${FLASHMIMIC_DIR:-$DEPLOY_ROOT/FlashMimic}"
RESULTS_ROOT="${RESULTS_ROOT:-$DEPLOY_ROOT/results}"
MACHINE_ID="${MACHINE_ID:-$(scutil --get LocalHostName 2>/dev/null || hostname -s)}"
ONLY_EXPERIMENT="${ONLY_EXPERIMENT:-}"
VIDEO_STEPS="${VIDEO_STEPS:-800}"
OVERWRITE_VIDEO="${OVERWRITE_VIDEO:-0}"

if [[ ! "$MACHINE_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid MACHINE_ID: $MACHINE_ID" >&2
  exit 2
fi
if [[ -n "$ONLY_EXPERIMENT" && ! "$ONLY_EXPERIMENT" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid ONLY_EXPERIMENT: $ONLY_EXPERIMENT" >&2
  exit 2
fi
if [[ ! "$VIDEO_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "VIDEO_STEPS must be a positive integer: $VIDEO_STEPS" >&2
  exit 2
fi
if [[ "$OVERWRITE_VIDEO" != "0" && "$OVERWRITE_VIDEO" != "1" ]]; then
  echo "OVERWRITE_VIDEO must be 0 or 1: $OVERWRITE_VIDEO" >&2
  exit 2
fi

matched=0
while IFS=$'\t' read -r experiment_id entrypoint algo task sim _default_iterations <&3; do
  [[ -z "$experiment_id" || "$experiment_id" == \#* ]] && continue
  if [[ -n "$ONLY_EXPERIMENT" && "$experiment_id" != "$ONLY_EXPERIMENT" ]]; then
    continue
  fi
  matched=$((matched + 1))
  run_dir="$RESULTS_ROOT/$MACHINE_ID/$experiment_id"
  end_json="$run_dir/benchmark_end.json"
  video_path="$run_dir/play_video.mp4"
  if [[ ! -f "$end_json" ]] || ! grep -Eq '"exit_code"[[:space:]]*:[[:space:]]*0' "$end_json"; then
    echo "Training did not complete successfully: $run_dir" >&2
    exit 1
  fi
  shopt -s nullglob
  checkpoints=("$run_dir"/model_*.pt)
  shopt -u nullglob
  if [[ "${#checkpoints[@]}" -eq 0 ]]; then
    echo "No model_*.pt checkpoint found in $run_dir" >&2
    exit 1
  fi
  checkpoint_path=""
  checkpoint_iteration=-1
  for candidate in "${checkpoints[@]}"; do
    checkpoint_name="$(basename -- "$candidate")"
    if [[ "$checkpoint_name" =~ ^model_([0-9]+)\.pt$ ]] && \
      ((10#${BASH_REMATCH[1]} > checkpoint_iteration)); then
      checkpoint_iteration=$((10#${BASH_REMATCH[1]}))
      checkpoint_path="$candidate"
    fi
  done
  if [[ -z "$checkpoint_path" ]]; then
    echo "No numerically named model_<iteration>.pt checkpoint found in $run_dir" >&2
    exit 1
  fi
  if [[ -e "$video_path" && "$OVERWRITE_VIDEO" != "1" ]]; then
    echo "Refusing to overwrite existing video: $video_path" >&2
    exit 1
  fi
  if [[ -e "$video_path" ]]; then
    rm -f -- "$video_path"
  fi

  start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\n' \
    "{" \
    "  \"experiment_id\": \"$experiment_id\"," \
    "  \"machine_id\": \"$MACHINE_ID\"," \
    "  \"start_time_utc\": \"$start_utc\"," \
    "  \"play_steps\": $VIDEO_STEPS" \
    "}" >"$run_dir/video_start.json"

  echo "Rendering final policy for $experiment_id on $MACHINE_ID"
  if [[ "$entrypoint" == "unilab" ]]; then
    play_command=(
      uv run --directory "$UNILAB_DIR" --no-sync eval
      --algo "$algo" --task "$task" --sim "$sim" --render-mode record
      "algo.load_run=$checkpoint_path"
      training.play_env_num=1
      "training.play_steps=$VIDEO_STEPS"
      training.no_play=false
      training.export_onnx=false
    )
  elif [[ "$entrypoint" == "flashmimic" ]]; then
    play_command=(
      uv run --directory "$FLASHMIMIC_DIR" --no-sync flashmimic-train
      "task=$task/$sim"
      training.play_only=true
      training.play_render_mode=record
      training.play_env_num=1
      "training.play_steps=$VIDEO_STEPS"
      training.no_play=false
      training.export_onnx=false
      "algo.load_run=$checkpoint_path"
    )
  else
    echo "Unknown entrypoint '$entrypoint' for $experiment_id" >&2
    exit 2
  fi

  set +e
  "${play_command[@]}" 2>&1 | tee "$run_dir/video_console.log"
  play_status="${PIPESTATUS[0]}"
  set -e
  if [[ "$play_status" -eq 0 && ( ! -f "$video_path" || ! -s "$video_path" ) ]]; then
    echo "Playback exited successfully but did not create a non-empty MP4: $video_path" >&2
    play_status=1
  fi
  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\n' \
    "{" \
    "  \"experiment_id\": \"$experiment_id\"," \
    "  \"machine_id\": \"$MACHINE_ID\"," \
    "  \"end_time_utc\": \"$end_utc\"," \
    "  \"exit_code\": $play_status," \
    "  \"video\": \"play_video.mp4\"" \
    "}" >"$run_dir/video_end.json"
  if [[ "$play_status" -ne 0 ]]; then
    exit "$play_status"
  fi
done 3<"$BENCHMARK_REPO/config/experiments.tsv"

if [[ "$matched" -eq 0 ]]; then
  echo "No experiment matched ONLY_EXPERIMENT=$ONLY_EXPERIMENT" >&2
  exit 2
fi
echo "Rendered $matched final policy video(s) for $MACHINE_ID"
