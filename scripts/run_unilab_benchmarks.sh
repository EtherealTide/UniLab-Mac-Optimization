#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_REPO="$(cd -- "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=../config/versions.env
source "$BENCHMARK_REPO/config/versions.env"

export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
DEPLOY_ROOT="${DEPLOY_ROOT:-$HOME/unilab-mac-benchmark}"
UNILAB_DIR="${UNILAB_DIR:-$DEPLOY_ROOT/UniLab}"
FLASHMIMIC_DIR="${FLASHMIMIC_DIR:-$DEPLOY_ROOT/FlashMimic}"
RESULTS_ROOT="${RESULTS_ROOT:-$DEPLOY_ROOT/results}"
MACHINE_ID="${MACHINE_ID:-$(scutil --get LocalHostName 2>/dev/null || hostname -s)}"
SEED="${SEED:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-}"
ITERATIONS_OVERRIDE="${ITERATIONS_OVERRIDE:-}"
ONLY_EXPERIMENT="${ONLY_EXPERIMENT:-}"

if [[ -n "$MAX_ITERATIONS" && -n "$ITERATIONS_OVERRIDE" && "$MAX_ITERATIONS" != "$ITERATIONS_OVERRIDE" ]]; then
  echo "MAX_ITERATIONS and legacy ITERATIONS_OVERRIDE disagree." >&2
  exit 2
fi
MAX_ITERATIONS="${MAX_ITERATIONS:-$ITERATIONS_OVERRIDE}"

if [[ ! "$MACHINE_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid MACHINE_ID: $MACHINE_ID" >&2
  exit 2
fi
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer: $SEED" >&2
  exit 2
fi
if [[ -n "$MAX_ITERATIONS" && ! "$MAX_ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_ITERATIONS must be a positive integer: $MAX_ITERATIONS" >&2
  exit 2
fi
if [[ -n "$ONLY_EXPERIMENT" && ! "$ONLY_EXPERIMENT" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid ONLY_EXPERIMENT: $ONLY_EXPERIMENT" >&2
  exit 2
fi

if [[ ! -d "$UNILAB_DIR/.git" ]]; then
  echo "Missing deployment at $UNILAB_DIR; run scripts/setup_macos.sh first." >&2
  exit 1
fi
if [[ ! -x /usr/bin/script ]]; then
  echo "Missing /usr/bin/script; a pseudo-terminal is required for live training output." >&2
  exit 1
fi

uv run --directory "$UNILAB_DIR" --frozen --no-sync python \
  "$BENCHMARK_REPO/scripts/verify_environment.py" \
  --repo "$UNILAB_DIR" \
  --python-version "$PYTHON_VERSION" \
  --unilab-commit "$UNILAB_COMMIT" \
  --unilab-version "$UNILAB_VERSION" \
  --unilab-rl-version "$UNILAB_RL_VERSION" \
  --unisim-version "$UNISIM_VERSION" \
  --torch-version "$TORCH_VERSION" \
  --mujoco-version "$MUJOCO_VERSION" \
  --mjbatch-version "$MJBATCH_VERSION" \
  --flashmimic-repo "$FLASHMIMIC_DIR" \
  --flashmimic-commit "$FLASHMIMIC_COMMIT"

uv run --directory "$FLASHMIMIC_DIR" --frozen --no-sync python \
  "$BENCHMARK_REPO/scripts/verify_flashmimic_environment.py" \
  --repo "$FLASHMIMIC_DIR" \
  --commit "$FLASHMIMIC_COMMIT" \
  --python-version "$PYTHON_VERSION" \
  --flashmimic-version "$FLASHMIMIC_VERSION" \
  --unilab-version "$UNILAB_VERSION" \
  --unilab-rl-version "$UNILAB_RL_VERSION" \
  --unisim-version "$UNISIM_VERSION" \
  --torch-version "$FLASHMIMIC_TORCH_VERSION" \
  --mujoco-version "$MUJOCO_VERSION"

mkdir -p "$RESULTS_ROOT/$MACHINE_ID"
while IFS=$'\t' read -r experiment_id entrypoint algo task sim default_iterations <&3; do
  [[ -z "$experiment_id" || "$experiment_id" == \#* ]] && continue
  if [[ -n "$ONLY_EXPERIMENT" && "$experiment_id" != "$ONLY_EXPERIMENT" ]]; then
    continue
  fi
  iterations="${MAX_ITERATIONS:-$default_iterations}"
  run_dir="$RESULTS_ROOT/$MACHINE_ID/$experiment_id"
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite existing result: $run_dir" >&2
    exit 1
  fi
  mkdir -p "$run_dir"
  start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  start_epoch="$(date +%s)"
  printf '%s\n' \
    "{" \
    "  \"experiment_id\": \"$experiment_id\"," \
    "  \"machine_id\": \"$MACHINE_ID\"," \
    "  \"start_time_utc\": \"$start_utc\"," \
    "  \"start_epoch_sec\": $start_epoch," \
    "  \"seed\": $SEED," \
    "  \"iterations\": $iterations" \
    "}" >"$run_dir/benchmark_start.json"

  echo "Starting $experiment_id on $MACHINE_ID at $start_utc"
  set +e
  if [[ "$entrypoint" == "unilab" ]]; then
    train_command=(
      uv run --directory "$UNILAB_DIR" --frozen --no-sync train
      --algo "$algo" --task "$task" --sim "$sim"
    )
  elif [[ "$entrypoint" == "flashmimic" ]]; then
    train_command=(
      uv run --directory "$FLASHMIMIC_DIR" --frozen --no-sync flashmimic-train
      "task=$task/$sim"
    )
  else
    echo "Unknown entrypoint '$entrypoint' for $experiment_id" >&2
    exit 2
  fi
  /usr/bin/script -qF "$run_dir/console.log" "${train_command[@]}" \
      "algo.seed=$SEED" \
      "algo.max_iterations=$iterations" \
      "algo.save_interval=$iterations" \
      training.log_interval=1 \
      training.no_play=true \
      "training.log_dir=$run_dir"
  train_status="$?"
  set -e

  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  end_epoch="$(date +%s)"
  printf '%s\n' \
    "{" \
    "  \"experiment_id\": \"$experiment_id\"," \
    "  \"machine_id\": \"$MACHINE_ID\"," \
    "  \"end_time_utc\": \"$end_utc\"," \
    "  \"end_epoch_sec\": $end_epoch," \
    "  \"elapsed_sec\": $((end_epoch - start_epoch))," \
    "  \"exit_code\": $train_status" \
    "}" >"$run_dir/benchmark_end.json"
  if [[ "$train_status" -ne 0 ]]; then
    echo "$experiment_id failed with exit code $train_status" >&2
    exit "$train_status"
  fi
done 3<"$BENCHMARK_REPO/config/experiments.tsv"
