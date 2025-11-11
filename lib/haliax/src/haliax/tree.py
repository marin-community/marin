# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Convenience wrappers for :mod:`haliax.tree_util` that mirror :mod:`jax.tree`."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence, TypeVar

from . import tree_util

T = TypeVar("T")


def map(fn: Callable[..., T], tree: Any, *rest: Any, is_leaf: Callable[[Any], bool] | None = None) -> Any:
    """Alias for :func:`haliax.tree_util.tree_map` matching :func:`jax.tree.map`."""

    return tree_util.tree_map(fn, tree, *rest, is_leaf=is_leaf)


def scan_aware_map(fn: Callable[..., T], tree: Any, *rest: Any, is_leaf: Callable[[Any], bool] | None = None) -> Any:
    """Alias for :func:`haliax.tree_util.scan_aware_tree_map` with :mod:`jax.tree` style naming."""

    return tree_util.scan_aware_tree_map(fn, tree, *rest, is_leaf=is_leaf)


def flatten(tree: Any, *, is_leaf: Callable[[Any], bool] | None = None) -> tuple[Sequence[Any], Any]:
    """Alias for :func:`haliax.tree_util.tree_flatten` matching :func:`jax.tree.flatten`."""

    return tree_util.tree_flatten(tree, is_leaf=is_leaf)


def unflatten(treedef: Any, leaves: Iterable[Any]) -> Any:
    """Alias for :func:`haliax.tree_util.tree_unflatten` matching :func:`jax.tree.unflatten`."""

    return tree_util.tree_unflatten(treedef, leaves)


def leaves(tree: Any, *, is_leaf: Callable[[Any], bool] | None = None) -> Sequence[Any]:
    """Alias for :func:`haliax.tree_util.tree_leaves` matching :func:`jax.tree.leaves`."""

    return tree_util.tree_leaves(tree, is_leaf=is_leaf)


def structure(tree: Any, *, is_leaf: Callable[[Any], bool] | None = None) -> Any:
    """Alias for :func:`haliax.tree_util.tree_structure` matching :func:`jax.tree.structure`."""

    return tree_util.tree_structure(tree, is_leaf=is_leaf)


__all__ = [
    "map",
    "scan_aware_map",
    "flatten",
    "unflatten",
    "leaves",
    "structure",
]
