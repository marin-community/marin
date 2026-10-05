"""Paths in sparse recipe and dense launch documents."""

from collections.abc import Iterator, Mapping
from typing import Any

from .model import document_key, merge_mappings

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
    """Yield indivisible values; empty mappings contribute no override."""
    for key, value in document.items():
        path = (*prefix, key)
        if isinstance(value, Mapping):
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
        combined = merge_mappings(combined, _without_empty_mappings(document))
    return combined


def _without_empty_mappings(document: Mapping[str, Any]) -> dict[str, Any]:
    contribution = {}
    for key, value in document.items():
        if isinstance(value, Mapping):
            nested = _without_empty_mappings(value)
            if nested:
                contribution[key] = nested
        else:
            contribution[key] = value
    return contribution
