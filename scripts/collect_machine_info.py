#!/usr/bin/env python3
"""Collect machine and resolved runtime provenance used by the report."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _command(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sysctl(name: str) -> str | None:
    return _command("sysctl", "-n", name)


def _package(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _system_profiler(data_type: str) -> Any:
    output = _command("system_profiler", data_type, "-json")
    if output is None:
        return None
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return output


def _redact_hardware_identifiers(value: Any) -> Any:
    """Remove stable device identifiers that are irrelevant to benchmarks."""
    if isinstance(value, dict):
        return {
            key: _redact_hardware_identifiers(item)
            for key, item in value.items()
            if not any(token in key.lower() for token in ("serial", "uuid", "udid"))
        }
    if isinstance(value, list):
        return [_redact_hardware_identifiers(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    git_head = _command("git", "-C", str(args.repo), "rev-parse", "HEAD")
    git_status = _command("git", "-C", str(args.repo), "status", "--porcelain")
    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "macos": _command("sw_vers"),
        "hardware": _redact_hardware_identifiers(_system_profiler("SPHardwareDataType")),
        "displays": _redact_hardware_identifiers(_system_profiler("SPDisplaysDataType")),
        "cpu": {
            "brand": _sysctl("machdep.cpu.brand_string"),
            "physical_cores": _sysctl("hw.physicalcpu"),
            "logical_cores": _sysctl("hw.logicalcpu"),
            "memory_bytes": _sysctl("hw.memsize"),
        },
        "toolchain": {
            "clang": _command("clang", "--version"),
            "git": _command("git", "--version"),
            "uv": _command("uv", "--version"),
        },
        "git": {"head": git_head, "dirty": bool(git_status)},
        "python": platform.python_version(),
        "packages": {
            name: _package(name)
            for name in (
                "unilab",
                "unilab-rl",
                "unisim-core",
                "torch",
                "mujoco",
                "mjbatch-uni",
                "numpy",
                "hydra-core",
                "tensorboard",
                "flashmimic",
            )
        },
        "torch": {
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "environment": {
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "HF_ENDPOINT": os.environ.get("HF_ENDPOINT"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
