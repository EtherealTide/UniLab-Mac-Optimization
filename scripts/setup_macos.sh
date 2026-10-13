#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_REPO="$(cd -- "$SCRIPT_DIR/.." && pwd)"

export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
DEPLOY_ROOT="${1:-$HOME/unilab-mac-benchmark}"
UNILAB_DIR="$DEPLOY_ROOT/UniLab"
FLASHMIMIC_DIR="$DEPLOY_ROOT/FlashMimic"
UNILAB_REPO="${UNILAB_REPO:-https://github.com/Motphys/UniLab.git}"
FLASHMIMIC_REPO="${FLASHMIMIC_REPO:-git@github.com:Motphys/FlashMimic.git}"
MACHINE_ID="${MACHINE_ID:-$(scutil --get LocalHostName 2>/dev/null || hostname -s)}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This benchmark requires Apple Silicon macOS (Darwin arm64)." >&2
  exit 1
fi
if ! xcode-select -p >/dev/null 2>&1; then
  echo "Apple Command Line Tools are missing. Run: xcode-select --install" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://astral.sh/uv/install.sh" >&2
  exit 1
fi
if [[ ! "$MACHINE_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid MACHINE_ID: $MACHINE_ID" >&2
  exit 2
fi

update_checkout() {
  local repo_url="$1"
  local repo_dir="$2"
  local bundle="${3:-}"
  if [[ ! -d "$repo_dir/.git" ]]; then
    if [[ -n "$bundle" ]]; then
      git clone "$bundle" "$repo_dir"
      git -C "$repo_dir" remote set-url origin "$repo_url"
    else
      git clone --filter=blob:none "$repo_url" "$repo_dir"
    fi
  fi
  if [[ -n "$(git -C "$repo_dir" status --porcelain)" ]]; then
    echo "Refusing to update a dirty benchmark checkout: $repo_dir" >&2
    exit 1
  fi
  git -C "$repo_dir" fetch origin
  local remote_head
  local branch
  remote_head="$(git -C "$repo_dir" symbolic-ref --quiet --short refs/remotes/origin/HEAD || true)"
  branch="${remote_head#origin/}"
  if [[ -z "$branch" ]]; then
    branch="main"
  fi
  if git -C "$repo_dir" show-ref --verify --quiet "refs/heads/$branch"; then
    git -C "$repo_dir" switch "$branch"
  else
    git -C "$repo_dir" switch --track -c "$branch" "origin/$branch"
  fi
  git -C "$repo_dir" pull --ff-only origin "$branch"
  if [[ "$(git -C "$repo_dir" rev-parse HEAD)" != \
    "$(git -C "$repo_dir" rev-parse "origin/$branch")" ]]; then
    echo "Checkout contains local commits not present on origin/$branch: $repo_dir" >&2
    exit 1
  fi
}

mkdir -p "$DEPLOY_ROOT"
update_checkout "$UNILAB_REPO" "$UNILAB_DIR"
update_checkout "$FLASHMIMIC_REPO" "$FLASHMIMIC_DIR" "${FLASHMIMIC_BUNDLE:-}"

# This benchmark does not enforce repository commits or package versions. Each
# upstream checkout resolves its own dependency graph; the actual Git/package
# provenance is captured below alongside the result bundle.
uv sync --directory "$UNILAB_DIR" --locked --extra mujoco --extra uni_rl
uv sync --directory "$FLASHMIMIC_DIR" --locked

uv run --directory "$UNILAB_DIR" --no-sync python -c \
  'import torch; assert torch.backends.mps.is_available(), "Torch MPS is unavailable"'
uv run --directory "$FLASHMIMIC_DIR" --no-sync python -c \
  'import torch; assert torch.backends.mps.is_available(), "Torch MPS is unavailable"'

uv run --directory "$UNILAB_DIR" --no-sync unilab-pull-assets --robot g1
uv run --directory "$UNILAB_DIR" --no-sync python -c \
  'from unilab.assets.hub import resolve_motion_files; print(resolve_motion_files("motions/g1/dance1_subject2_part.npz"))'
uv run --directory "$FLASHMIMIC_DIR" --no-sync unilab-pull-assets --robot g1

uv run --directory "$UNILAB_DIR" --no-sync python \
  "$BENCHMARK_REPO/scripts/collect_machine_info.py" \
  --repo "$UNILAB_DIR" \
  --output "$DEPLOY_ROOT/machine-${MACHINE_ID}.json"

uv run --directory "$FLASHMIMIC_DIR" --no-sync python \
  "$BENCHMARK_REPO/scripts/collect_machine_info.py" \
  --repo "$FLASHMIMIC_DIR" \
  --output "$DEPLOY_ROOT/machine-${MACHINE_ID}-flashmimic.json"

echo "Current UniLab deployment is ready at $UNILAB_DIR"
