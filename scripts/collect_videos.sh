#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 SSH_TARGET REMOTE_RESULTS_DIR LOCAL_DEST" >&2
  exit 2
fi

SSH_TARGET="$1"
REMOTE_RESULTS_DIR="$2"
LOCAL_DEST="$3"

mkdir -p "$LOCAL_DEST"

# Videos are generated beside each run's final checkpoint as play_video.mp4.
# Keep the experiment directory names so the six files remain unambiguous.
rsync -az --ignore-existing --prune-empty-dirs \
  --include='*/' \
  --include='play_video.mp4' \
  --exclude='*' \
  -e 'ssh -F /dev/null' \
  "$SSH_TARGET:$REMOTE_RESULTS_DIR/" "$LOCAL_DEST/"

if ! find "$LOCAL_DEST" -type f -name 'play_video.mp4' -print -quit | grep -q .; then
  echo "No play_video.mp4 found under $LOCAL_DEST" >&2
  exit 1
fi

(
  cd "$LOCAL_DEST"
  find . -type f -name 'play_video.mp4' -print \
    | LC_ALL=C sort \
    | xargs shasum -a 256 >VIDEO_SHA256SUMS
)

find "$LOCAL_DEST" -type f -name 'play_video.mp4' -print | LC_ALL=C sort
