#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Path helpers for Together-hosted judge runs.

New runs are archived under a timestamped root:

    ~/together_batch/{run_id}/{model_slug}/{target}/

The latest *full* run for each model/target is additionally exposed through
the stable alias:

    ~/together_batch/latest/{model_slug}/{target}/

Older runs that predate the timestamped layout may still live at:

    ~/together_batch/{model_slug}/{target}/
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

DATA_ROOT = Path.home() / "together_batch"
LATEST_ROOT = DATA_ROOT / "latest"
RUN_ID_FORMAT = "%Y%m%d_%H%M%SZ"


def model_slug(model: str) -> str:
    """Filesystem-safe slug for a Together model ID."""
    return model.replace("/", "_")


def default_run_id() -> str:
    """Return the default UTC run identifier used for archived runs."""
    return datetime.now(timezone.utc).strftime(RUN_ID_FORMAT)


def run_target_dir(run_id: str, model: str, target: str) -> Path:
    """Return the archived output directory for one Together run."""
    return DATA_ROOT / run_id / model_slug(model) / target


def latest_target_dir(model: str, target: str) -> Path:
    """Return the stable alias for the latest full run of one target."""
    return LATEST_ROOT / model_slug(model) / target


def legacy_target_dir(model: str, target: str) -> Path:
    """Return the pre-archive output directory used by older runs."""
    return DATA_ROOT / model_slug(model) / target


def resolve_target_dir(model: str, target: str, run_id: str | None = None) -> Path:
    """Resolve a Together target directory.

    Priority:
    1. Explicit archived run id.
    2. The stable `latest/` alias.
    3. The legacy pre-archive location.
    """
    if run_id is not None:
        return run_target_dir(run_id, model, target)

    latest_dir = latest_target_dir(model, target)
    if latest_dir.exists():
        return latest_dir

    legacy_dir = legacy_target_dir(model, target)
    if legacy_dir.exists():
        return legacy_dir

    return latest_dir


def publish_latest_run(run_id: str, model: str, target: str) -> Path:
    """Point the stable latest alias at an archived run directory."""
    archived_dir = run_target_dir(run_id, model, target)
    if not archived_dir.exists():
        raise FileNotFoundError(f"missing archived Together run at {archived_dir}")

    alias_dir = latest_target_dir(model, target)
    alias_dir.parent.mkdir(parents=True, exist_ok=True)

    if alias_dir.is_symlink() or alias_dir.is_file():
        alias_dir.unlink()
    elif alias_dir.exists():
        if any(alias_dir.iterdir()):
            raise RuntimeError(f"refusing to replace non-empty latest alias directory at {alias_dir}")
        alias_dir.rmdir()

    alias_dir.symlink_to(archived_dir, target_is_directory=True)
    return alias_dir
