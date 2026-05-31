# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for downstream-scaling evals."""

from __future__ import annotations

import os
import re

from marin.execution.executor import InputName, MirroredValue, versioned
from marin.utils import fsspec_exists, fsspec_glob

_STEP_CHECKPOINT_RE = re.compile(r"(?:^|/)step-(\d+)/?$")


def version_path(path: str | InputName | MirroredValue) -> str | InputName | MirroredValue:
    if isinstance(path, str):
        return versioned(path)  # type: ignore[return-value]
    return path


def _is_hf_checkpoint_dir(path: str) -> bool:
    return fsspec_exists(os.path.join(path, "config.json")) and fsspec_exists(
        os.path.join(path, "tokenizer_config.json")
    )


def _step_number(path: str) -> int:
    match = _STEP_CHECKPOINT_RE.search(path.rstrip("/"))
    if match is None:
        raise ValueError(f"Cannot order checkpoint path without step-N suffix: {path}")
    return int(match.group(1))


def discover_hf_checkpoints(base_path: str) -> list[str]:
    """Discover HF checkpoints without relying on filesystem mtimes."""

    base_path = base_path.rstrip("/")
    if _is_hf_checkpoint_dir(base_path):
        return [base_path]

    checkpoints = sorted(
        {
            os.path.dirname(config_path)
            for config_path in fsspec_glob(os.path.join(base_path, "**/config.json"))
            if _is_hf_checkpoint_dir(os.path.dirname(config_path))
        },
        key=_step_number,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No HF checkpoints found under {base_path}")
    return checkpoints
