# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import abc
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Generic, Optional, Sequence, TypeAlias, TypeVar

import jax.random
import numpy as np
from jaxtyping import PRNGKeyArray

from levanter.data._prp import PermType, Permutation
from levanter.utils import thread_utils
from levanter.utils.jax_utils import local_cpu_mesh

logger = logging.getLogger(__name__)


T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")
U = TypeVar("U")

# When we decide to standardize on 3.12, we can use fancier things
# P = ParamSpec("P")

MapFunction: TypeAlias = Callable[..., U]


class DatasetBase(abc.ABC, Generic[T_co]):
    """
    Base class for sync and async datasets. This class is not meant to be used directly.
    """

    @abc.abstractmethod
    def as_async_dataset(self) -> "AsyncDataset[T_co]":
        raise NotImplementedError("...")

    @abc.abstractmethod
    def as_sync_dataset(self) -> "SyncDataset[T_co]":
        raise NotImplementedError("...")


class AsyncDataset(DatasetBase[T_co]):
    """
    An asynchronous dataset that can be used with async/await syntax.

    The core methods in this class are:
    * `async_len`: Returns the final length of the dataset.
    * `get_batch`: Returns a batch of items from the dataset.
    """

    @abc.abstractmethod
    async def async_len(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def is_finite(self) -> bool:
        """
        Returns whether the dataset has a known finite length.
        If this returns False, the dataset is infinite.
        """
        raise NotImplementedError

    async def getitem_async(self, index: int) -> T_co:
        """
        Returns the item at the given index. Typically implemented as a wrapper around `get_batch`.

        In general, it is better to call (and override) `get_batch` instead of this method.
        """
        return (await self.get_batch([index]))[0]

    @abc.abstractmethod
    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        raise NotImplementedError

    def as_sync_dataset(self):
        return SyncifiedDataset(self)

    def as_async_dataset(self) -> "AsyncDataset[T_co]":
        return self

    def map(self, fn: MapFunction[U], *extra_args, **extra_kwargs) -> "MappedAsyncDataset[T_co, U]":
        return MappedAsyncDataset(self, fn, *extra_args, **extra_kwargs)

    def map_batches(self, fn: MapFunction[Sequence[U]], *extra_args, **extra_kwargs) -> "BatchMappedAsyncDataset[U]":
        return BatchMappedAsyncDataset(self, fn, *extra_args, **extra_kwargs)

    def slice_dataset(self, start_index: Optional[int] = None, end_index: Optional[int] = None):
        """
        Slices the dataset from `start_index` to `end_index`.
        """
        return SlicedAsyncDataset(self, start_index, end_index)

    def take(self, n: int):
        """
        Alias for `slice_dataset(end_index=n)`.
        """
        return self.slice_dataset(end_index=n)

    def shuffle(self, key: PRNGKeyArray, *, perm_type: PermType = "feistel"):
        return PermutationDataset(self, key, perm_type=perm_type)

    def block_shuffle(
        self,
        *,
        io_block_size: int,
        window_blocks: int,
        key: PRNGKeyArray,
        perm_type: PermType = "feistel",
    ):
        return BlockShufflingDataset(
            self,
            io_block_size,
            window_blocks=window_blocks,
            key=key,
            perm_type=perm_type,
        )


class SyncDataset(DatasetBase[T_co]):
    """
    A synchronous dataset that can be used with regular Python syntax. In Levanter, we mainly do not use this class.
    You can use this class if it's easier, then convert it to an AsyncDataset using `as_async_dataset`. This
    is not as efficient as using an AsyncDataset directly, but it can be useful for testing or for simpler code.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Returns the final length of the data store.
        May raise if the length is not known.
        """

    @abc.abstractmethod
    def is_finite(self) -> bool:
        """
        Whether the dataset has a known finite length.
        """
        pass

    def __getitem__(self, index: int) -> T_co:
        return self.get_batch([index])[0]

    @abc.abstractmethod
    def get_batch(self, indices: Sequence[int] | np.ndarray) -> Sequence[T_co]:
        pass

    def as_async_dataset(self) -> "AsyncDataset[T_co]":
        return AsyncifiedDataset(self)

    def as_sync_dataset(self) -> "SyncDataset[T_co]":
        return self


class SyncifiedDataset(SyncDataset[T_co]):
    def __init__(self, dataset: AsyncDataset[T_co]):
        self.dataset = dataset

    def _run_coroutine(self, coro):
        return thread_utils.blocking_wait(coro)

    def __len__(self) -> int:
        return self._run_coroutine(self.dataset.async_len())

    def is_finite(self) -> bool:
        return self.dataset.is_finite()

    def get_batch(self, indices: Sequence[int] | np.ndarray) -> Sequence[T_co]:
        return self._run_coroutine(self.dataset.get_batch(indices))

    def __getitem__(self, index: int) -> T_co:
        return self._run_coroutine(self.dataset.getitem_async(index))


class AsyncifiedDataset(AsyncDataset[T_co]):
    def __init__(self, dataset: SyncDataset[T_co]):
        self.dataset = dataset

    async def async_len(self) -> int:
        return len(self.dataset)

    def is_finite(self) -> bool:
        return self.dataset.is_finite()

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        return self.dataset.get_batch(indices)

    async def getitem_async(self, index: int) -> T_co:
        return self.dataset[index]

    def __repr__(self):
        return f"WrappedAsyncDataset({repr(self.dataset)})"

    def __str__(self):
        return f"WrappedAsyncDataset({str(self.dataset)})"


class ListAsyncDataset(AsyncDataset[T]):
    """
    A simple dataset that wraps a list. Mostly for testing.
    """

    def __init__(self, data: list[T]):
        self.data = data

    async def async_len(self) -> int:
        return len(self.data)

    def is_finite(self) -> bool:
        return True

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T]:
        if not indices:
            return []
        return [self.data[i] for i in indices]


class MappedAsyncDataset(AsyncDataset[U], Generic[T, U]):
    """
    A dataset that applies a function to each item in the dataset.
    You can pass extra arguments to the function using `*extra_args` and `**extra_kwargs`.
    If a kwarg called `key` is passed, it will be treated as a PRNGKey and folded in with the index of the item
    for each call to the function.
    """

    def __init__(
        self,
        dataset: AsyncDataset[T],
        fn: MapFunction[U],
        *extra_args,
        **extra_kwargs,
    ):
        self.dataset = dataset
        self.fn = fn
        self._extra_args = extra_args
        self._extra_kwargs = extra_kwargs

    async def async_len(self) -> int:
        return await self.dataset.async_len()

    def is_finite(self) -> bool:
        return self.dataset.is_finite()

    def _maybe_fold_in_key(self, key, index):
        if key is not None:
            key = jax.random.fold_in(key, index)
        return key

    async def get_batch(self, indices: Sequence[int]) -> Sequence[U]:
        items = await self.dataset.get_batch(indices)
        return [self._call_fn(i, item) for i, item in zip(indices, items)]

    async def getitem_async(self, index: int) -> U:
        return self._call_fn(index, await self.dataset.getitem_async(index))

    def _call_fn(self, index, item):
        if "key" in self._extra_kwargs:
            key = self._maybe_fold_in_key(self._extra_kwargs["key"], index)
            kwargs = {**self._extra_kwargs, "key": key}
        else:
            kwargs = self._extra_kwargs
        return self.fn(item, *self._extra_args, **kwargs)


class SlicedAsyncDataset(AsyncDataset[U]):
    def __init__(
        self,
        dataset: AsyncDataset[U],
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
    ):
        if start_index is None:
            start_index = 0
        if end_index is not None and start_index > end_index:
            raise ValueError("End index must come after start index.")

        self.start_index: int = start_index
        self.end_index: int | None = end_index
        self.dataset = dataset

    async def get_batch(self, indices: Sequence[int]) -> Sequence[U]:
        if not indices:
            return []

        shifted_indices = [(index + self.start_index) for index in indices]
        max_index = max(shifted_indices)

        if self.end_index is not None and max_index >= self.end_index:
            raise ValueError("Requested indices beyond the end of the dataset")

        return await self.dataset.get_batch(shifted_indices)

    async def async_len(self) -> int:
        if self.end_index is not None and not self.dataset.is_finite():
            return self.end_index - self.start_index

        underlying_length = await self.dataset.async_len()

        if self.end_index is None:
            return max(underlying_length - self.start_index, 0)
        else:
            return max(min(self.end_index, underlying_length) - self.start_index, 0)

    def is_finite(self) -> bool:
        return self.end_index is not None or self.dataset.is_finite()


class BatchMappedAsyncDataset(AsyncDataset[U]):
    """
    A dataset that applies a function to each batch of items in the dataset.
    You can pass extra arguments to the function using `*extra_args` and `**extra_kwargs`.
    If a kwarg called `key` is passed, it will be treated as a PRNGKey and folded in with the index of the item
    for each call to the function. The key will be split into a key for each item in the batch.
    """

    def __init__(
        self,
        dataset: AsyncDataset[T],
        fn: MapFunction[Sequence[U]],
        *extra_args,
        **extra_kwargs,
    ):
        self.dataset: AsyncDataset = dataset
        self.fn = fn
        self._extra_args = extra_args
        self._extra_kwargs = extra_kwargs

    async def async_len(self) -> int:
        return await self.dataset.async_len()

    def is_finite(self) -> bool:
        return self.dataset.is_finite()

    def _maybe_fold_in_key(self, key, indices: Sequence[int]):
        if key is not None:
            key = _fold_in_key_vmap(key, np.array(indices))
        return key

    async def get_batch(self, indices: Sequence[int]) -> Sequence[U]:
        items = await self.dataset.get_batch(indices)
        return self._call_fn(indices, items)

    async def getitem_async(self, index: int) -> U:
        return self._call_fn([index], [await self.dataset.getitem_async(index)])[0]

    def _call_fn(self, indices: Sequence[int], items):
        if "key" in self._extra_kwargs:
            key = self._maybe_fold_in_key(self._extra_kwargs["key"], indices)
            kwargs = {**self._extra_kwargs, "key": key}
        else:
            kwargs = self._extra_kwargs
        return self.fn(items, *self._extra_args, **kwargs)


@jax.jit
def _fold_in_key_vmap(key, indices):
    return jax.vmap(lambda i: jax.random.fold_in(key, i))(indices)


def _key_on_local_cpu(key: PRNGKeyArray) -> PRNGKeyArray:
    """Canonicalize a PRNG key onto the local CPU device."""
    with local_cpu_mesh():
        return jax.device_put(jax.device_get(key))


def _fold_in_on_local_cpu(key: PRNGKeyArray, data: int) -> PRNGKeyArray:
    """Fold in an integer into a PRNG key while pinned to local CPU mesh."""
    with local_cpu_mesh():
        key = jax.device_put(jax.device_get(key))
        return jax.device_put(jax.device_get(jax.random.fold_in(key, data)))


class PermutationDataset(AsyncDataset[T_co]):
    """A permutation dataset that wraps another dataset and applies a permutation to the indices."""

    # TODO: add epoch reshuffling

    def __init__(self, dataset: AsyncDataset[T_co], key: PRNGKeyArray, perm_type: PermType = "feistel"):
        self.dataset = dataset
        self.key = _key_on_local_cpu(key)
        self._permutation: Optional[Permutation] = None
        self._perm_type = perm_type

    async def async_len(self) -> int:
        return await self.dataset.async_len()

    def is_finite(self) -> bool:
        return self.dataset.is_finite()

    async def getitem_async(self, index: int) -> T_co:
        permutation = await self._get_permutation()
        return await self.dataset.getitem_async(permutation(index))

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        permutation = await self._get_permutation()
        return await self.dataset.get_batch(
            [int(permutation(i)) for i in indices]
        )  # cast to int to be sure it's python int

    async def _get_permutation(self):
        if self._permutation is None:
            self._permutation = Permutation.make(self._perm_type, await self.async_len(), self.key)
        return self._permutation


@dataclass(frozen=True)
class _BlockShuffleState:
    dataset_len: int
    num_full_blocks: int
    tail_size: int
    total_blocks: int
    num_windows: int
    last_window_id: int
    length_before_last_window: int
    full_window_size: int


@dataclass(frozen=True)
class _WindowLayout:
    """
    Layout of one logical shuffle window.

    `full_blocks` are permuted blocks of size `io_block_size`.
    `tail_size` is the optional tiny block at the end of the final window.
    """

    full_blocks: tuple[int, ...]
    full_region_size: int
    tail_size: int

    @property
    def window_size(self) -> int:
        return self.full_region_size + self.tail_size


class BlockShufflingDataset(AsyncDataset[T_co]):
    """
    A dataset that applies hierarchical block shuffling for better I/O locality.

    It works in two stages:
    1. Permute full blocks of size `io_block_size`.
    2. Within each window of `window_blocks`, permute examples from full blocks.

    If the dataset has a final partial (tiny) block, that tail block is always kept at
    the very end and is not mixed into earlier positions. This keeps the edge case
    simple and deterministic.
    """

    def __init__(
        self,
        dataset: AsyncDataset[T_co],
        io_block_size: int,
        *,
        window_blocks: int,
        key: PRNGKeyArray | int,
        perm_type: PermType = "feistel",
    ):
        if io_block_size <= 0:
            raise ValueError(f"io_block_size must be positive, got {io_block_size}")
        if window_blocks <= 0:
            raise ValueError(f"window_blocks must be positive, got {window_blocks}")

        self.dataset = dataset
        self.io_block_size = io_block_size
        self.window_blocks = window_blocks
        self._perm_type = perm_type

        if isinstance(key, int):
            key = jax.random.PRNGKey(key)
        key = _key_on_local_cpu(key)
        with local_cpu_mesh():
            block_key, window_full_key, window_tail_key = jax.random.split(key, 3)
        self.key = key
        self._block_key = block_key
        self._window_full_key = window_full_key
        self._window_tail_key = window_tail_key

        self._state: _BlockShuffleState | None = None
        self._full_block_permutation: Optional[Permutation] = None

    def is_finite(self) -> bool:
        return self.dataset.is_finite()

    async def async_len(self) -> int:
        state = await self._ensure_initialized()
        return state.dataset_len

    async def getitem_async(self, index: int) -> T_co:
        mapped = await self._map_index(index)
        return await self.dataset.getitem_async(mapped)

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        if not indices:
            return []
        mapped = [await self._map_index(i) for i in indices]
        return await self.dataset.get_batch(mapped)

    async def _ensure_initialized(self) -> _BlockShuffleState:
        if self._state is not None:
            return self._state

        if not self.dataset.is_finite():
            raise ValueError("BlockShufflingDataset only supports finite datasets")

        dataset_len = await self.dataset.async_len()
        if dataset_len < 0:
            raise ValueError(f"Dataset length must be non-negative, got {dataset_len}")

        num_full_blocks = dataset_len // self.io_block_size
        tail_size = dataset_len % self.io_block_size
        total_blocks = num_full_blocks + int(tail_size > 0)
        full_window_size = self.io_block_size * self.window_blocks
        num_windows = (total_blocks + self.window_blocks - 1) // self.window_blocks if total_blocks > 0 else 0
        last_window_id = max(0, num_windows - 1)
        length_before_last_window = 0 if num_windows <= 1 else (num_windows - 1) * full_window_size

        if num_full_blocks > 1:
            self._full_block_permutation = Permutation.make(self._perm_type, num_full_blocks, self._block_key)

        self._state = _BlockShuffleState(
            dataset_len=dataset_len,
            num_full_blocks=num_full_blocks,
            tail_size=tail_size,
            total_blocks=total_blocks,
            num_windows=num_windows,
            last_window_id=last_window_id,
            length_before_last_window=length_before_last_window,
            full_window_size=full_window_size,
        )
        return self._state

    def _state_or_error(self) -> _BlockShuffleState:
        if self._state is None:
            raise RuntimeError("BlockShufflingDataset is not initialized")
        return self._state

    @lru_cache(maxsize=4)
    def _window_layout(self, window_id: int) -> _WindowLayout:
        state = self._state_or_error()
        if window_id < 0 or window_id >= state.num_windows:
            raise IndexError(f"Window id {window_id} out of bounds for {state.num_windows} windows")

        block_start = window_id * self.window_blocks
        blocks_remaining = state.total_blocks - block_start
        blocks_in_window = min(self.window_blocks, blocks_remaining)

        has_tail_in_window = state.tail_size > 0 and (block_start + blocks_in_window == state.total_blocks)
        full_blocks_in_window = blocks_in_window - int(has_tail_in_window)

        physical_full_blocks: list[int] = []
        for block_position in range(block_start, block_start + full_blocks_in_window):
            if self._full_block_permutation is None:
                # Identity for 0 or 1 full blocks.
                physical_full_blocks.append(block_position)
            else:
                physical_full_blocks.append(int(self._full_block_permutation(block_position)))

        tail_size = state.tail_size if has_tail_in_window else 0
        full_region_size = len(physical_full_blocks) * self.io_block_size
        return _WindowLayout(
            full_blocks=tuple(physical_full_blocks), full_region_size=full_region_size, tail_size=tail_size
        )

    @lru_cache(maxsize=4)
    def _window_full_region_permutation(self, window_id: int) -> Optional[Permutation]:
        layout = self._window_layout(window_id)
        if layout.full_region_size <= 1:
            return None
        key = _fold_in_on_local_cpu(self._window_full_key, window_id)
        return Permutation.make(self._perm_type, layout.full_region_size, key)

    @lru_cache(maxsize=4)
    def _window_tail_region_permutation(self, window_id: int) -> Optional[Permutation]:
        layout = self._window_layout(window_id)
        if layout.tail_size <= 1:
            return None
        key = _fold_in_on_local_cpu(self._window_tail_key, window_id)
        return Permutation.make(self._perm_type, layout.tail_size, key)

    async def _map_index(self, index: int) -> int:
        if index < 0:
            raise ValueError("Negative indices are not supported")

        state = await self._ensure_initialized()

        if index >= state.dataset_len:
            raise IndexError(f"Index {index} out of bounds for dataset length {state.dataset_len}")

        if state.num_windows == 0:
            raise IndexError(f"Index {index} out of bounds for empty dataset")

        if state.num_windows == 1:
            window_id = 0
            offset_in_window = index
        elif index < state.length_before_last_window:
            window_id = index // state.full_window_size
            offset_in_window = index % state.full_window_size
        else:
            window_id = state.last_window_id
            offset_in_window = index - state.length_before_last_window

        layout = self._window_layout(window_id)

        if offset_in_window >= layout.window_size:
            raise IndexError(
                f"Offset {offset_in_window} out of bounds for window {window_id} with length {layout.window_size}"
            )

        if offset_in_window < layout.full_region_size:
            perm = self._window_full_region_permutation(window_id)
            permuted_offset = offset_in_window if perm is None else int(perm(offset_in_window))
            block_offset, offset_in_block = divmod(permuted_offset, self.io_block_size)
            physical_full_block = layout.full_blocks[block_offset]
            return physical_full_block * self.io_block_size + offset_in_block

        # Final tiny tail block stays at the very end, but can be locally permuted.
        tail_offset = offset_in_window - layout.full_region_size
        tail_perm = self._window_tail_region_permutation(window_id)
        permuted_tail_offset = tail_offset if tail_perm is None else int(tail_perm(tail_offset))
        return state.num_full_blocks * self.io_block_size + permuted_tail_offset

    def __repr__(self):
        return (
            "BlockShufflingDataset("
            f"{repr(self.dataset)}, io_block_size={self.io_block_size}, "
            f"window_blocks={self.window_blocks})"
        )

    def __str__(self):
        return (
            "BlockShufflingDataset("
            f"{str(self.dataset)}, io_block_size={self.io_block_size}, "
            f"window_blocks={self.window_blocks})"
        )
