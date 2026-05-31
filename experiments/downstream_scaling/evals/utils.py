# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for downstream-scaling evals."""

from __future__ import annotations

import os
import re

import fsspec
from marin.execution.executor import InputName, MirroredValue
from marin.execution.types import versioned
from marin.utils import fsspec_exists, fsspec_glob
from rigging.filesystem import REGION_TO_DATA_BUCKET, marin_region, mirror_budget

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


# Generous ceiling for a single-checkpoint cross-region copy. The default
# TransferBudget (10 GB) would otherwise abort on any model larger than ~10 GB
# (llama-3.1-8B in bf16 is ~16 GB).
_MIRROR_LOCALIZE_BUDGET_GB = 1024


def localize_mirror_path(path: str) -> str:
    """Resolve a mirror:// checkpoint path to a concrete local-region gs:// URL.

    Copies the checkpoint into the local-region bucket if it isn't there yet
    (per-file copy-on-read via MirrorFileSystem, locked). Needed because vLLM's
    runai-streamer loader reads object storage directly and does not understand
    the mirror:// scheme. Non-mirror:// paths pass through unchanged.

    The worker region comes from marin_region() — not marin_prefix(), which is
    mirror:// itself when the mirror is on.
    """
    if not path.startswith("mirror://"):
        return path
    region = marin_region()
    if region is None or region not in REGION_TO_DATA_BUCKET:
        raise RuntimeError(f"cannot localize {path!r}: region={region!r} has no marin data bucket")
    rel = path[len("mirror://") :]
    mirror_fs = fsspec.filesystem("mirror")
    with mirror_budget(_MIRROR_LOCALIZE_BUDGET_GB):
        for f in mirror_fs.find(rel):
            mirror_fs.info(f)  # per-file copy-on-read into the local-region bucket
    return f"gs://{REGION_TO_DATA_BUCKET[region]}/{rel}"
