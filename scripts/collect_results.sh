#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 SSH_TARGET REMOTE_RESULTS_DIR LOCAL_DEST" >&2
  exit 2
fi

SSH_TARGET="$1"
REMOTE_RESULTS_DIR="$2"
LOCAL_DEST="$3"
if [[ -d "$LOCAL_DEST" && -n "$(find "$LOCAL_DEST" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty destination: $LOCAL_DEST" >&2
  exit 1
fi
mkdir -p "$LOCAL_DEST"

# Checkpoints stay on the training host; final policy videos are publication artifacts.
rsync -az \
  --exclude='*_cold_asset_excluded/***' \
  --include='*/' \
  --include='events.out.tfevents.*' \
  --include='run_config.json' \
  --include='run_summary.json' \
  --include='benchmark_start.json' \
  --include='benchmark_end.json' \
  --include='video_start.json' \
  --include='video_end.json' \
  --include='console.log' \
  --include='video_console.log' \
  --include='play_video.mp4' \
  --exclude='*' \
  -e 'ssh -F /dev/null' \
  "$SSH_TARGET:$REMOTE_RESULTS_DIR/" "$LOCAL_DEST/"

(
  cd "$LOCAL_DEST"
  find . -type f ! -name SHA256SUMS -print \
    | LC_ALL=C sort \
    | xargs shasum -a 256 >SHA256SUMS
)
