# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paths in sparse recipe and dense launch documents."""

from collections.abc import Iterator, Mapping
from typing import Any

from .model import thaw

MISSING = object()


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


def leaves(document: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Yield indivisible values; empty mappings contribute a leaf."""
    for key, value in document.items():
        path = (*prefix, key)
        if isinstance(value, Mapping) and value:
            yield from leaves(value, path)
        else:
            yield path, value


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
