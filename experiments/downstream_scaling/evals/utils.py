# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for downstream-scaling evals."""

from __future__ import annotations

import os
import re

import fsspec
from thalas.execution.executor import InputName, MirroredValue
from thalas.execution.types import versioned
from rigging.filesystem import data_config, marin_region, mirror_budget

_STEP_CHECKPOINT_RE = re.compile(r"(?:^|/)step-(\d+)/?$")


def fsspec_exists(path: str) -> bool:
    fs, fs_path = fsspec.core.url_to_fs(path)
    return fs.exists(fs_path)


def fsspec_glob(path: str) -> list[str]:
    protocol, _ = fsspec.core.split_protocol(path)
    fs, fs_path = fsspec.core.url_to_fs(path)
    matches = fs.glob(fs_path)
    if protocol is None:
        return sorted(matches)

    paths = []
    for match in matches:
        if fsspec.core.split_protocol(match)[0] is not None:
            paths.append(match)
        else:
            paths.append(f"{protocol}://{match}")
    return sorted(paths)


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


# Default to pre-seeded-only model localization: resolve local mirror files, but
# fail before copying any non-local checkpoint bytes. Callers that explicitly
# want model copy-on-read can pass a positive budget_gb.
_MIRROR_LOCALIZE_BUDGET_GB = 0.0


def localize_mirror_path(path: str, budget_gb: float = _MIRROR_LOCALIZE_BUDGET_GB) -> str:
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
    if region is None or region not in data_config().region_buckets:
        raise RuntimeError(f"cannot localize {path!r}: region={region!r} has no marin data bucket")
    rel = path[len("mirror://") :]
    mirror_fs = fsspec.filesystem("mirror")
    with mirror_budget(budget_gb):
        for f in mirror_fs.find(rel):
            mirror_fs.info(f)  # per-file copy-on-read into the local-region bucket
    return f"gs://{data_config().region_buckets[region].name}/{rel}"
