# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paths in sparse recipe and dense launch documents."""

from collections.abc import Iterator, Mapping
from typing import Any

from .model import Section, thaw

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


def combine_parts(base: Mapping[str, Any], parts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Combine independent contributions, detecting leaf and ancestor conflicts."""
    contributions: dict[tuple[str, ...], tuple[str, Any]] = {}
    combined = dict(base)
    for name, document in sorted(parts.items()):
        for path, value in leaves(document):
            for previous, (source, existing) in contributions.items():
                same_path = path == previous
                parent_path = path[: len(previous)] == previous or previous[: len(path)] == path
                if same_path and document_key(value) == document_key(existing):
                    continue
                if same_path or parent_path:
                    location = ".".join(min(path, previous, key=len))
                    raise ValueError(
                        f"{location}: parts {source!r} and {name!r} set conflicting values; use merge for an override"
                    )
            contributions[path] = name, value
        combined = merge_mappings(combined, document)
    return combined


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


def document_key(value: Any) -> tuple:
    """Compare JSON values with numeric equality and ordered mapping keys."""
    if isinstance(value, Mapping):
        return "mapping", tuple((key, document_key(item)) for key, item in value.items())
    if isinstance(value, tuple | list):
        return "list", tuple(document_key(item) for item in value)
    if isinstance(value, Section):
        return document_key(value.to_skyrl())
    if isinstance(value, bool):
        return "bool", value
    if isinstance(value, int | float):
        return "number", value
    return type(value).__name__, value
