# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
from typing import Sequence, TypeVar, cast

import jax.numpy as jnp

import equinox as eqx
import jax
from jax._src.tree_util import DictKey, FlattenedIndexKey, GetAttrKey, KeyEntry, PyTreeDef, SequenceKey
from jaxtyping import PyTree
from enum import Enum


class NonePolicy(str, Enum):
    PRESERVE = "preserve"
    REPLACE = "replace"
    ERROR = "error"


T = TypeVar("T", bound=PyTree)


def inference_mode(tree: T, value: bool, none_policy: str = NonePolicy.REPLACE) -> T:
    """
    Analogous to [equinox.nn.inference_mode][] (neé [equinox.tree_inference][]), but
    it works in the presence of nones for the `inference` argument.
    """

    has_inference = lambda leaf: hasattr(leaf, "inference")  # noqa: E731

    def replace_fn(node):
        if not has_inference(node):
            return node

        if node.inference is None:
            if none_policy == NonePolicy.PRESERVE:
                return node
            elif none_policy == NonePolicy.ERROR:
                raise ValueError(f"None found in {tree}.inference with none_policy={none_policy}")
            else:
                assert none_policy == NonePolicy.REPLACE, f"Unknown none_policy {none_policy}"

        if dataclasses.is_dataclass(node):
            return dataclasses.replace(node, inference=value)
        else:
            return eqx.tree_at(lambda x: x.inference, node, value, is_leaf=lambda x: x is node)

    def rec_set(tree):
        if has_inference(tree):
            tree = replace_fn(tree)

        if jax.tree_util.tree_leaves(tree) == [tree]:
            return tree

        return jax.tree_util.tree_map(rec_set, tree, is_leaf=lambda x: has_inference(x) and tree is not x)

    return rec_set(tree)


def tree_flatten_one_level_with_keys(
    pytree: PyTree,
) -> tuple[list[tuple[KeyEntry | None, PyTree]], PyTreeDef]:
    """
    Adapted form equinox.tree_flatten_one_level to return keys

    If the passed in PyTree is a leaf, it will return a single-element list with None as the key and the PyTree as the value.
    """
    seen_pytree = False

    def is_leaf(node):
        nonlocal seen_pytree
        if node is pytree:
            if seen_pytree:
                # We expect to only see it once as the root.
                # This catches for example
                # ```python
                # x = []
                # x.append(x)
                # tree_subnodes(x)
                # ```
                # Note that it intentionally does *not* catch
                # ```python
                # x = []
                # y = []
                # x.append(y)
                # y.append(x)
                # tree_subnodes(x)
                # ```
                # as `x` is not an immediate subnode of itself.
                # If you want to check for that then use `tree_check`.
                try:
                    type_string = type(pytree).__name__
                except AttributeError:
                    type_string = "<unknown>"
                raise ValueError(
                    f"PyTree node of type `{type_string}` is immediately "
                    "self-referential; that is to say it appears within its own PyTree "
                    "structure as an immediate subnode. (For example "
                    "`x = []; x.append(x)`.) This is not allowed."
                )
            else:
                seen_pytree = True
            return False
        else:
            return True

    out_paths, out_treedef = jax.tree_util.tree_flatten_with_path(pytree, is_leaf=is_leaf)

    out = []
    for path, value in out_paths:
        if not path:
            return [(None, value)], out_treedef

        assert len(path) == 1, "Only one level of flattening is supported"
        out.append((path[0], value))

    return out, out_treedef


def key_path_to_str(path: Sequence) -> str:
    """Helper method to format optimizer state keys."""
    if not path:
        return ""
    path_elem = path[-1]
    match path_elem:
        case SequenceKey(idx):  # type: ignore
            out = f"{idx}"
        case DictKey(key):  # type: ignore
            out = f"{key}"
        case GetAttrKey():  # type: ignore
            out = str(path_elem)
        case FlattenedIndexKey(idx):  # type: ignore
            out = f"{idx}"
        case _:
            path_elem = str(path_elem)
            out = f"{path_elem}"

    if out.startswith("."):
        out = out[1:]

    return out


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class PackedLeaf:
    """Metadata describing the location and shape of a packed leaf."""

    offset: int = dataclasses.field(metadata={"static": True})
    shape: tuple[int, ...] = dataclasses.field(metadata={"static": True})


def pack_pytree(tree: PyTree, dtype=jnp.float32) -> tuple[PyTree, jnp.ndarray]:
    """Pack all leaves of ``tree`` into a single 1-D array.

    Args:
        tree: Pytree of array-like objects.
        dtype: Desired dtype of the packed array.

    Returns:
        A pair ``(offset_tree, flat_array)`` where ``offset_tree`` mirrors the
        structure of ``tree`` but each leaf contains a :class:`PackedLeaf`
        indicating where that leaf's data is stored in ``flat_array``.
    """

    leaves, treedef = jax.tree_util.tree_flatten(tree)

    flat_leaves = []
    offset_leaves = []
    current = 0
    for leaf in leaves:
        arr = jnp.asarray(leaf, dtype=dtype)
        flat = arr.reshape(-1)
        flat_leaves.append(flat)
        offset_leaves.append(PackedLeaf(offset=current, shape=arr.shape))  # type: ignore[call-arg]
        current += flat.size

    if flat_leaves:
        packed = jnp.concatenate(flat_leaves)
    else:
        packed = jnp.array([], dtype=dtype)

    offset_tree = jax.tree_util.tree_unflatten(treedef, offset_leaves)
    return offset_tree, packed


def unpack_pytree(offset_tree: PyTree, packed: jnp.ndarray) -> PyTree:
    """Reconstruct a pytree packed with :func:`pack_pytree`."""

    offset_leaves, treedef = jax.tree_util.tree_flatten(offset_tree)
    offset_leaves = [cast(PackedLeaf, x) for x in offset_leaves]

    leaves = []
    for off in offset_leaves:
        size = functools.reduce(int.__mul__, off.shape, 1)
        leaf = packed[off.offset : off.offset + size].reshape(off.shape)
        leaves.append(leaf)

    return jax.tree_util.tree_unflatten(treedef, leaves)
