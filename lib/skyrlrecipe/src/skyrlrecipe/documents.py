# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paths in sparse recipe and dense launch documents."""

from collections.abc import Mapping
from typing import Any

from .model import thaw

MISSING = object()
_ROLLOUT_BUFFER_PATH = "trainer.rollout_buffer"
_LEGACY_LEASE_LIMIT = "max_in_flight"
_LEASE_LIMIT = "max_outstanding_leases"


def normalize_setting_path(path: str) -> str:
    """Translate the old dotted lease limit at the settings boundary."""
    if path == f"{_ROLLOUT_BUFFER_PATH}.{_LEGACY_LEASE_LIMIT}":
        return f"{_ROLLOUT_BUFFER_PATH}.{_LEASE_LIMIT}"
    return path


def normalize_rollout_buffer_config(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """Read the old lease limit without carrying it into the composed config."""
    buffer = get_path(document, _ROLLOUT_BUFFER_PATH)
    if not isinstance(buffer, Mapping) or _LEGACY_LEASE_LIMIT not in buffer:
        return document
    updated = dict(buffer)
    legacy_limit = updated.pop(_LEGACY_LEASE_LIMIT)
    updated.setdefault(_LEASE_LIMIT, legacy_limit)
    return {**document, "trainer": {**document["trainer"], "rollout_buffer": updated}}


def get_path(document: Mapping[str, Any], path: str, default: Any = MISSING) -> Any:
    node: Any = document
    for key in path.split("."):
        if not isinstance(node, Mapping) or key not in node:
            return default
        node = node[key]
    return node


def set_path(document: dict[str, Any], path: str, value: Any) -> None:
    *parents, leaf = path.split(".")
    for key in parents:
        current = document.get(key)
        if not isinstance(current, dict):
            current = {}
            document[key] = current
        document = current
    document[leaf] = value


def merge_mappings(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """Merge mappings recursively; lists replace and explicit empty mappings are preserved."""
    merged = thaw(base)
    for key, value in patch.items():
        if isinstance(value, Mapping):
            current = merged.get(key)
            merged[key] = merge_mappings(current if isinstance(current, Mapping) else {}, value)
        else:
            merged[key] = thaw(value)
    return merged
