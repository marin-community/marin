# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import abc
import logging
from typing import Callable, Generic, Optional, Sequence, TypeAlias, TypeVar

import jax.random
import numpy as np
from jaxtyping import PRNGKeyArray

from levanter.data._prp import PermType
from levanter.utils import thread_utils


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
        import levanter.data.permutation as permutation

        return permutation.PermutationDataset(self, key, perm_type=perm_type)

    def era_shuffle(self, era_length: int, key: PRNGKeyArray, *, perm_type: PermType = "feistel"):
        import levanter.data.permutation as permutation

        return permutation.EraShufflingDataset(self, era_length, key=key, perm_type=perm_type)


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
        self.dataset = dataset
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
