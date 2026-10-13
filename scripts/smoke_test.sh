#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_REPO="$(cd -- "$SCRIPT_DIR/.." && pwd)"

export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
DEPLOY_ROOT="${DEPLOY_ROOT:-$HOME/unilab-mac-benchmark}"
UNILAB_DIR="${UNILAB_DIR:-$DEPLOY_ROOT/UniLab}"
FLASHMIMIC_DIR="${FLASHMIMIC_DIR:-$DEPLOY_ROOT/FlashMimic}"
MACHINE_ID="${MACHINE_ID:-$(scutil --get LocalHostName 2>/dev/null || hostname -s)}"
SMOKE_ROOT="${SMOKE_ROOT:-$DEPLOY_ROOT/smoke/$MACHINE_ID}"

if [[ -e "$SMOKE_ROOT" ]]; then
  echo "Refusing to overwrite existing smoke output: $SMOKE_ROOT" >&2
  exit 1
fi
mkdir -p "$SMOKE_ROOT"

common_overrides=(
  algo.num_envs=16
  algo.batch_size=32
  algo.replay_buffer_n=16
  algo.learning_starts=2
  algo.max_iterations=3
  algo.save_interval=3
  algo.updates_per_step=1
  training.log_interval=1
  training.no_play=true
)

env FORCE_COLOR=1 TTY_INTERACTIVE=1 \
  uv run --directory "$UNILAB_DIR" --no-sync train \
  --algo sac --task g1_walk_flat --sim mujoco \
  "${common_overrides[@]}" \
  "training.log_dir=$SMOKE_ROOT/sac_g1_walk_flat" \
  2>&1 | tee "$SMOKE_ROOT/sac_g1_walk_flat.log"

env FORCE_COLOR=1 TTY_INTERACTIVE=1 \
  uv run --directory "$UNILAB_DIR" --no-sync train \
  --algo flashsac --task g1_motion_tracking --sim mujoco \
  "${common_overrides[@]}" \
  "training.log_dir=$SMOKE_ROOT/flashsac_g1_motion_tracking" \
  2>&1 | tee "$SMOKE_ROOT/flashsac_g1_motion_tracking.log"

env FORCE_COLOR=1 TTY_INTERACTIVE=1 \
  uv run --directory "$FLASHMIMIC_DIR" --no-sync flashmimic-train \
  task=g1_flash_mimic/mujoco \
  "${common_overrides[@]}" \
  "training.log_dir=$SMOKE_ROOT/flashmimic_g1_motion_tracking" \
  2>&1 | tee "$SMOKE_ROOT/flashmimic_g1_motion_tracking.log"

echo "All three smoke tests completed on $MACHINE_ID"
