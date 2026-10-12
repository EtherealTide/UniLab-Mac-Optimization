#!/usr/bin/env python3
"""Fail closed unless the benchmark runtime matches the published pins."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--unilab-commit", required=True)
    parser.add_argument("--unilab-version", required=True)
    parser.add_argument("--unilab-rl-version", required=True)
    parser.add_argument("--unisim-version", required=True)
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--mujoco-version", required=True)
    parser.add_argument("--mjbatch-version", required=True)
    parser.add_argument("--flashmimic-repo", type=Path, required=True)
    parser.add_argument("--flashmimic-commit", required=True)
    args = parser.parse_args()

    import torch

    expected = {
        "unilab": args.unilab_version,
        "unilab-rl": args.unilab_rl_version,
        "unisim-core": args.unisim_version,
        "torch": args.torch_version,
        "mujoco": args.mujoco_version,
        "mjbatch-uni": args.mjbatch_version,
    }
    actual = {name: importlib.metadata.version(name) for name in expected}
    mismatches = {
        name: {"expected": expected[name], "actual": actual[name]}
        for name in expected
        if actual[name] != expected[name]
    }
    head = _git(args.repo, "rev-parse", "HEAD")
    dirty = bool(_git(args.repo, "status", "--porcelain"))
    if head != args.unilab_commit:
        mismatches["unilab_commit"] = {"expected": args.unilab_commit, "actual": head}
    if dirty:
        mismatches["unilab_worktree"] = {"expected": "clean", "actual": "dirty"}
    flashmimic_head = _git(args.flashmimic_repo, "rev-parse", "HEAD")
    flashmimic_dirty = bool(_git(args.flashmimic_repo, "status", "--porcelain"))
    if flashmimic_head != args.flashmimic_commit:
        mismatches["flashmimic_commit"] = {
            "expected": args.flashmimic_commit,
            "actual": flashmimic_head,
        }
    if flashmimic_dirty:
        mismatches["flashmimic_worktree"] = {"expected": "clean", "actual": "dirty"}
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        mismatches["platform"] = {
            "expected": "Darwin arm64",
            "actual": f"{platform.system()} {platform.machine()}",
        }
    if not torch.backends.mps.is_available():
        mismatches["mps"] = {"expected": True, "actual": False}
    if platform.python_version() != args.python_version:
        mismatches["python"] = {
            "expected": args.python_version,
            "actual": platform.python_version(),
        }

    payload = {
        "git_head": head,
        "git_dirty": dirty,
        "flashmimic_git_head": flashmimic_head,
        "flashmimic_git_dirty": flashmimic_dirty,
        "packages": actual,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "mismatches": mismatches,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
