# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for next-gen mixture loop artifacts and IDs."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from collections.abc import Mapping
from typing import Any

import fsspec


_NON_ALNUM = re.compile(r"[^a-zA-Z0-9._-]+")


def slugify_name(value: str) -> str:
    """Create a filesystem-safe slug while preserving readability."""
    value = value.strip().replace("/", "_")
    value = _NON_ALNUM.sub("_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "loop"


def loop_root_path(state_root: str, loop_name: str) -> str:
    """Return the persistent root path for a loop."""
    return os.path.join(state_root, slugify_name(loop_name))


def stable_hash(payload: Any, *, prefix: str = "") -> str:
    """Generate a deterministic short hash for JSON-like payloads."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    if prefix:
        return f"{prefix}-{digest}"
    return digest


def _json_default(obj: Any):
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def read_json(path: str, default: Any = None) -> Any:
    """Read JSON via fsspec; return *default* if missing."""
    fs, _, _ = fsspec.get_fs_token_paths(path)
    if not fs.exists(path):
        return default
    with fsspec.open(path, "r") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    """Write JSON via fsspec with parent directory creation."""
    fs, _, _ = fsspec.get_fs_token_paths(path)
    parent = os.path.dirname(path)
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fsspec.open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=_json_default)


def normalize_phase_weights(phase_weights: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Normalize nested phase/domain weights to plain dicts with float values."""
    return {
        str(phase): {str(domain): float(weight) for domain, weight in domain_weights.items()}
        for phase, domain_weights in phase_weights.items()
    }
