# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from typing import Generic, List, Sequence, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PyTree

from haliax.jax_utils import is_jax_array_like

from .jagged_array import JaggedArrayStore, PreparedBatch


T = TypeVar("T", bound=PyTree)


# TODO at some point if we turn this into a real library, it would be nice to store the schema
# TODO: some data is probably best not stored as a jagged array, but as a flat array?
# TODO: also sometimes we might want a rowstore actually


def heuristic_is_leaf(x):
    if isinstance(x, list):
        return jnp.isscalar(x[0])
    else:
        return False


def heuristic_is_leaf_batched(x):
    if isinstance(x, list):
        return jnp.isscalar(x[0]) or is_jax_array_like(x[0])
    else:
        return False


class TreeStore(Generic[T]):
    """
    A TreeStore stores batched data as a tree of ragged arrays.
    """

    path: str
    mode: str
    tree: PyTree[JaggedArrayStore]

    def __init__(self, tree, path: str, mode: str):
        self.path = path
        self.mode = mode
        self.tree = tree

    @property
    def batch_preparer(self):
        return TreeBatchPreparer(jtu.tree_map(lambda writer: 9, self.tree, is_leaf=heuristic_is_leaf))

    @staticmethod
    def open(exemplar: T, path: str, *, mode="a", cache_metadata: bool = False) -> "TreeStore":
        """
        Open a TreeStoreBuilder from a file.
        """
        tree = _construct_builder_tree(exemplar, path, mode, cache_metadata)
        return TreeStore(tree, path, mode)

    def append(self, ex: T):
        return self.extend([ex])

    def extend(self, batch: Sequence[T]):
        """
        Append a batch of data to the store.
        """
        jtu.tree_map(
            lambda writer, *xs: writer.extend([np.asarray(x) for x in xs]),
            self.tree,
            *batch,
            is_leaf=heuristic_is_leaf,
        )

    def extend_with_batch(self, batch: T):
        """
        Append a batch of data (as a pytree with batched leaves) to the store.

        This method works only when the "leaves" are lists of numpy arrays or scalars.
        For instance, HF's BatchEncoding is a dict of lists of numpy arrays.
        """
        jtu.tree_map(
            lambda writer, xs: writer.extend(xs if isinstance(xs, PreparedBatch) else [np.asarray(x) for x in xs]),
            self.tree,
            batch,
            is_leaf=heuristic_is_leaf_batched,
        )

    async def extend_with_batch_async(self, batch: T):
        """
        Append a batch of data (as a pytree with batched leaves) to the store.

        This method works only when the "leaves" are lists of numpy arrays or scalars.
        For instance, HF's BatchEncoding is a dict of lists of numpy arrays.
        """
        futures = jtu.tree_map(
            lambda writer, xs: writer.extend_async(
                xs if isinstance(xs, PreparedBatch) else [np.asarray(x) for x in xs]
            ),
            self.tree,
            batch,
            is_leaf=heuristic_is_leaf_batched,
        )

        await asyncio.gather(*jax.tree.leaves(futures))

    def trim_to_size(self, size: int):
        """
        Trim the store to a given size.
        """
        # TODO These all return ts Futures so in theory we could await them all at once
        jtu.tree_map(lambda writer: writer.trim_to_size(size), self.tree, is_leaf=heuristic_is_leaf)

    async def trim_to_size_async(self, size: int):
        """
        Trim the store to a given size.
        """
        futures = jtu.tree_map(lambda writer: writer.trim_to_size_async(size), self.tree, is_leaf=heuristic_is_leaf)
        leaves, structure = jax.tree.flatten(futures)

        await asyncio.gather(*leaves)

    def reload(self) -> "TreeStore":
        """
        Close the builder and return a TreeStore.
        """
        tree = jtu.tree_map(lambda builder: builder.reload(), self.tree, is_leaf=heuristic_is_leaf)
        return TreeStore(tree, self.path, self.mode)

    def __len__(self):
        if self.tree is None:
            return 0
        else:
            return len(jax.tree.leaves(self.tree)[0])

    async def get_batch(self, indices) -> List[T]:
        grouped = jtu.tree_map(lambda reader: reader.get_batch(indices), self.tree, is_leaf=heuristic_is_leaf)

        leaves, structure = jtu.tree_flatten(grouped, is_leaf=heuristic_is_leaf)

        awaited_leaves = await asyncio.gather(*leaves)
        return [jtu.tree_unflatten(structure, [leaf[i] for leaf in awaited_leaves]) for i in range(len(indices))]

    def __getitem__(self, item):
        if self.tree is None:
            raise IndexError("No data in store")
        elif isinstance(item, slice):
            # debatch
            leaves, structure = jax.tree.flatten(self.tree, is_leaf=heuristic_is_leaf)
            # batched_items = jtu.tree_map(lambda reader: reader[item], self.tree, is_leaf=heuristic_is_leaf)
            batched_item_leaves = [leaf[item] for leaf in leaves]
            num_items = len(leaves[0])
            return [jtu.tree_unflatten(structure, [leaf[i] for leaf in batched_item_leaves]) for i in range(num_items)]
        else:
            return jtu.tree_map(lambda reader: reader[item], self.tree, is_leaf=heuristic_is_leaf)

    def __iter__(self):
        if self.tree is None:
            return
        else:
            for i in range(len(self)):
                yield self[i]

    def get_batch_sync(self, indices) -> List[T]:
        # TODO: would be better to batch these up
        grouped = jtu.tree_map(lambda reader: reader.get_batch_sync(indices), self.tree, is_leaf=heuristic_is_leaf)

        out = [jtu.tree_map(lambda _, leaf: leaf[i], self.tree, grouped) for i in range(len(indices))]

        return out

    async def async_len(self) -> int:
        return await jax.tree.leaves(self.tree)[0].num_rows_async()


def _construct_builder_tree(exemplar, path, mode, cache_metadata):
    def open_builder(tree_path, item):
        item = np.asarray(item)
        rank = item.ndim
        render_tree_path = "/".join(_render_path_elem(x) for x in tree_path)
        return JaggedArrayStore.open(
            os.path.join(path, render_tree_path),
            mode=mode,
            item_rank=rank,
            dtype=item.dtype,
            cache_metadata=cache_metadata,
        )

    return jtu.tree_map_with_path(open_builder, exemplar, is_leaf=heuristic_is_leaf)


def _render_path_elem(x):
    match x:
        case jtu.DictKey(key):
            return f"{key}"
        case jtu.GetAttrKey(key):
            return f"{key}"
        case jtu.SequenceKey(i):
            return f"{i}"
        case jtu.FlattenedIndexKey(i):
            return f"{i}"
        case _:
            return str(x)


class TreeBatchPreparer(Generic[T]):
    def __init__(self, exemplar: T):
        self.exemplar = exemplar

    def __call__(self, batch: List[T]) -> PyTree:
        return jtu.tree_map(
            lambda _, *xs: PreparedBatch.from_batch([np.asarray(x) for x in xs]),
            self.exemplar,
            *batch,
            is_leaf=heuristic_is_leaf,
        )
