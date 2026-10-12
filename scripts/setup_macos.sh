#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_REPO="$(cd -- "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=../config/versions.env
source "$BENCHMARK_REPO/config/versions.env"

export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
DEPLOY_ROOT="${1:-$HOME/unilab-mac-benchmark}"
UNILAB_DIR="$DEPLOY_ROOT/UniLab"
FLASHMIMIC_DIR="$DEPLOY_ROOT/FlashMimic"
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

mkdir -p "$DEPLOY_ROOT"
if [[ ! -d "$UNILAB_DIR/.git" ]]; then
  git clone --filter=blob:none "$UNILAB_REPO" "$UNILAB_DIR"
fi
if [[ -n "$(git -C "$UNILAB_DIR" status --porcelain)" ]]; then
  echo "Refusing to change a dirty benchmark checkout: $UNILAB_DIR" >&2
  exit 1
fi
if ! git -C "$UNILAB_DIR" cat-file -e "${UNILAB_COMMIT}^{commit}" 2>/dev/null; then
  git -C "$UNILAB_DIR" fetch origin "$UNILAB_COMMIT"
fi
git -C "$UNILAB_DIR" switch --detach "$UNILAB_COMMIT"

if [[ ! -d "$FLASHMIMIC_DIR/.git" ]]; then
  if [[ -n "${FLASHMIMIC_BUNDLE:-}" ]]; then
    git clone "$FLASHMIMIC_BUNDLE" "$FLASHMIMIC_DIR"
    git -C "$FLASHMIMIC_DIR" remote set-url origin "$FLASHMIMIC_REPO"
  else
    git clone --filter=blob:none "$FLASHMIMIC_REPO" "$FLASHMIMIC_DIR"
  fi
fi
if [[ -n "$(git -C "$FLASHMIMIC_DIR" status --porcelain)" ]]; then
  echo "Refusing to change a dirty benchmark checkout: $FLASHMIMIC_DIR" >&2
  exit 1
fi
if ! git -C "$FLASHMIMIC_DIR" cat-file -e "${FLASHMIMIC_COMMIT}^{commit}" 2>/dev/null; then
  git -C "$FLASHMIMIC_DIR" fetch origin "$FLASHMIMIC_COMMIT"
fi
git -C "$FLASHMIMIC_DIR" switch --detach "$FLASHMIMIC_COMMIT"

actual_unilab_lock="$(shasum -a 256 "$UNILAB_DIR/uv.lock" | awk '{print $1}')"
actual_flashmimic_lock="$(shasum -a 256 "$FLASHMIMIC_DIR/uv.lock" | awk '{print $1}')"
if [[ "$actual_unilab_lock" != "$UNILAB_LOCK_SHA256" ]]; then
  echo "UniLab uv.lock checksum mismatch: $actual_unilab_lock" >&2
  exit 1
fi
if [[ "$actual_flashmimic_lock" != "$FLASHMIMIC_LOCK_SHA256" ]]; then
  echo "FlashMimic uv.lock checksum mismatch: $actual_flashmimic_lock" >&2
  exit 1
fi

uv python install "$PYTHON_VERSION"
uv sync --directory "$UNILAB_DIR" --frozen --python "$PYTHON_VERSION" --extra mujoco --extra uni_rl
# FlashMimic currently imports a compatibility symbol removed after UniLab's
# 1.3.4 release. Its own frozen lock is therefore a separate, auditable runtime.
uv sync --directory "$FLASHMIMIC_DIR" --frozen --python "$PYTHON_VERSION"

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
  --flashmimic-commit "$FLASHMIMIC_COMMIT" \
  | tee "$DEPLOY_ROOT/verify-${MACHINE_ID}-unilab.json"

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
  --mujoco-version "$MUJOCO_VERSION" \
  | tee "$DEPLOY_ROOT/verify-${MACHINE_ID}-flashmimic.json"

uv run --directory "$UNILAB_DIR" --frozen --no-sync unilab-pull-assets --robot g1
uv run --directory "$UNILAB_DIR" --frozen --no-sync python -c \
  'from unilab.assets.hub import resolve_motion_files; print(resolve_motion_files("motions/g1/dance1_subject2_part.npz"))'
# FlashMimic has an independent frozen environment. Materialize the same robot
# assets there before any timed run so its first benchmark does not include a
# network download or asset extraction in the wall-clock measurement.
uv run --directory "$FLASHMIMIC_DIR" --frozen --no-sync unilab-pull-assets --robot g1

uv run --directory "$UNILAB_DIR" --frozen --no-sync python \
  "$BENCHMARK_REPO/scripts/collect_machine_info.py" \
  --repo "$UNILAB_DIR" \
  --output "$DEPLOY_ROOT/machine-${MACHINE_ID}.json"

uv run --directory "$FLASHMIMIC_DIR" --frozen --no-sync python \
  "$BENCHMARK_REPO/scripts/collect_machine_info.py" \
  --repo "$FLASHMIMIC_DIR" \
  --output "$DEPLOY_ROOT/machine-${MACHINE_ID}-flashmimic.json"

echo "Pinned UniLab deployment is ready at $UNILAB_DIR"
