# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import abc
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
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


_executor = ThreadPoolExecutor(max_workers=10)


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
    An asynchronous dataset that can be used with async/await syntax. In Levanter, we use AsyncDataset for two purposes:
    * To represent datasets that are inherently asynchronous (e.g. reading from disk, network, etc.).
    * To represent datasets that are still being constructed.

    The core methods in this class are:
    * `async_len`: Returns the final length of the dataset.
    * `get_batch`: Returns a batch of items from the dataset.
    * `current_len`: Returns the current length of the dataset. This may be None if no current length is known.
    """

    def __init__(self):
        self._min_known_len = 0

    @abc.abstractmethod
    async def async_len(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    async def final_length_is_known(self) -> bool:
        """Returns whether the final length of the dataset is known.
        If this returns False, the current_len of the dataset may change in the future."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_finite(self) -> bool:
        """
        Returns whether the dataset will have a known length in the future (e.g. if it's being constructed).
        If this returns False, the length of the dataset is infinite or unknowable.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def current_len(self) -> Optional[int]:
        """
        Returns the current length of the dataset that won't require (expensive) waiting.

        If the current length is not known, returns None. This might block temporarily for a short time to get the
        current length.
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

    async def wait_until_len_at_least(self, length: int) -> int:
        """
        Returns the length of the dataset once it is at least `length` or if the dataset has a known (finished) length.

        The default implementation is a naive busy-wait loop. You should override this method for more efficient
        implementations.
        """
        if length <= self._min_known_len:
            return self._min_known_len

        res_len = await naive_busy_wait_until_len_at_least(self, length)
        self._min_known_len = max(self._min_known_len, res_len)
        return res_len

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


async def naive_busy_wait_until_len_at_least(dataset: AsyncDataset[T_co], length: int) -> int:
    """
    Runs a busy-wait loop until the dataset has at least `length` items or the final length is known.

    Returns the current length of the dataset when either the dataset has at least `length` items or the final length is
    known.

    You should probably implement this in a more efficient way. This is just a naive implementation.
    """
    while not await dataset.final_length_is_known():
        current_len = await dataset.current_len()
        if current_len is None:
            raise ValueError("Dataset has unknown length")
        if current_len <= length:
            await asyncio.sleep(0.1)
        else:
            return current_len

    return await dataset.async_len()


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
    def has_len(self) -> bool:
        """
        Whether the data store currently has a known length. If this returns False, then the length of the data store
        may change in the future.
        """
        pass

    @abc.abstractmethod
    def current_len(self) -> Optional[int]:
        """
        Returns the current length of the data store. If the length is infinite or not known, returns None.
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

    def has_len(self) -> bool:
        return self.dataset.is_finite()

    def current_len(self) -> Optional[int]:
        return self._run_coroutine(self.dataset.current_len())

    def get_batch(self, indices: Sequence[int] | np.ndarray) -> Sequence[T_co]:
        return self._run_coroutine(self.dataset.get_batch(indices))

    def __getitem__(self, index: int) -> T_co:
        return self._run_coroutine(self.dataset.getitem_async(index))


class AsyncifiedDataset(AsyncDataset[T_co]):
    def __init__(self, dataset: SyncDataset[T_co]):
        super().__init__()
        self.dataset = dataset

    async def async_len(self) -> int:
        return len(self.dataset)

    async def final_length_is_known(self) -> bool:
        return self.dataset.has_len()

    def is_finite(self) -> bool:
        return self.dataset.has_len()

    async def current_len(self) -> Optional[int]:
        return self.dataset.current_len()

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

    def __init__(self, data: list[T], is_complete: bool = False):
        super().__init__()
        self.data = data
        self.is_complete = is_complete
        if not is_complete:
            self.complete_promise: Optional[asyncio.Future[None]] = asyncio.Future()
            self.length_updated: Optional[asyncio.Condition] = asyncio.Condition()
        else:
            self.complete_promise = None
            self.length_updated = None

    async def async_len(self) -> int:
        # this is the final length
        if not self.is_complete:
            assert self.complete_promise is not None
            await self.complete_promise
        return len(self.data)

    async def final_length_is_known(self) -> bool:
        return self.is_complete

    def is_finite(self) -> bool:
        return True

    async def current_len(self) -> Optional[int]:
        return len(self.data)

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T]:
        await self.wait_until_len_at_least(max(indices) + 1)
        return [self.data[i] for i in indices]

    def append(self, item: T):
        if self.is_complete:
            raise ValueError("Cannot append to a finalized dataset")
        self.data.append(item)
        asyncio.create_task(self.notify_length_update())

    def finalize(self):
        self.is_complete = True
        if self.complete_promise is not None:
            self.complete_promise.set_result(None)
            if not asyncio.get_event_loop().is_running():
                _executor.submit(lambda: asyncio.run(self.notify_length_update()))
            else:
                asyncio.create_task(self.notify_length_update())

    async def notify_length_update(self):
        async with self.length_updated:
            self.length_updated.notify_all()

    async def wait_until_len_at_least(self, length: int) -> int:
        if self.is_complete:
            return len(self.data)

        assert self.length_updated is not None

        async with self.length_updated:
            while len(self.data) < length and not self.is_complete:
                await self.length_updated.wait()

        return len(self.data)


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
        super().__init__()
        self.dataset = dataset
        self.fn = fn
        self._extra_args = extra_args
        self._extra_kwargs = extra_kwargs

    async def async_len(self) -> int:
        return await self.dataset.async_len()

    async def final_length_is_known(self) -> bool:
        return await self.dataset.final_length_is_known()

    def is_finite(self) -> bool:
        return self.dataset.is_finite()

    async def current_len(self) -> Optional[int]:
        return await self.dataset.current_len()

    def _maybe_fold_in_key(self, key, index):
        if key is not None:
            key = jax.random.fold_in(key, index)
        return key

    async def get_batch(self, indices: Sequence[int]) -> Sequence[U]:
        items = await self.dataset.get_batch(indices)
        return [self._call_fn(i, item) for i, item in zip(indices, items)]

    async def getitem_async(self, index: int) -> U:
        return self._call_fn(index, await self.dataset.getitem_async(index))

    async def wait_until_len_at_least(self, length: int) -> int:
        return await self.dataset.wait_until_len_at_least(length)

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
        super().__init__()
        if start_index is None:
            start_index = 0
        if end_index is not None and start_index > end_index:
            raise ValueError("End index must come after start index.")

        self.start_index: int = start_index
        self.end_index: int | None = end_index
        self.dataset = dataset
        self._min_known_len = dataset._min_known_len if end_index is None else (end_index - start_index)

    async def get_batch(self, indices: Sequence[int]) -> Sequence[U]:
        shifted_indices = [(index + self.start_index) for index in indices]
        max_index = max(shifted_indices)

        if self.end_index is not None and max_index > self.end_index:
            raise ValueError("Requested indices beyond the end of the dataset")

        return await self.dataset.get_batch(shifted_indices)

    async def async_len(self) -> int:
        underlying_length = await self.dataset.async_len()

        if self.end_index is None:
            return underlying_length - self.start_index
        else:
            return min(self.end_index, underlying_length) - self.start_index

    async def final_length_is_known(self) -> bool:
        underlying_is_known = await self.dataset.final_length_is_known()
        return underlying_is_known and self.end_index is not None

    def is_finite(self) -> bool:
        return self.dataset.is_finite() and self.end_index is not None

    async def current_len(self) -> Optional[int]:
        underlying_length = await self.dataset.current_len()
        if self.end_index is not None:
            if underlying_length is None:
                return self.end_index - self.start_index
            return min(self.end_index, underlying_length) - self.start_index
        elif underlying_length is not None:
            return underlying_length - self.start_index
        else:
            return underlying_length


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
        super().__init__()
        self.dataset = dataset
        self.fn = fn
        self._extra_args = extra_args
        self._extra_kwargs = extra_kwargs

    async def async_len(self) -> int:
        return await self.dataset.async_len()

    async def final_length_is_known(self) -> bool:
        return await self.dataset.final_length_is_known()

    def is_finite(self) -> bool:
        return self.dataset.is_finite()

    async def current_len(self) -> Optional[int]:
        return await self.dataset.current_len()

    def _maybe_fold_in_key(self, key, indices: Sequence[int]):
        if key is not None:
            key = _fold_in_key_vmap(key, np.array(indices))
        return key

    async def get_batch(self, indices: Sequence[int]) -> Sequence[U]:
        items = await self.dataset.get_batch(indices)
        return self._call_fn(indices, items)

    async def getitem_async(self, index: int) -> U:
        return self._call_fn([index], [await self.dataset.getitem_async(index)])[0]

    async def wait_until_len_at_least(self, length: int) -> int:
        return await self.dataset.wait_until_len_at_least(length)

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


class EpochDataset(AsyncDataset[T_co]):
    """
    A dataset that wraps another dataset, providing infinite epochs by recycling indices.
    If `max_epochs` is specified, it limits the number of cycles before raising StopIteration.

    :param dataset: The dataset to wrap.
    :param max_epochs: The maximum number of epochs to cycle through. If None, cycle indefinitely.
    """

    def __init__(self, dataset: AsyncDataset[T_co], max_epochs: Optional[int] = None):
        super().__init__()
        self.dataset = dataset
        self.max_epochs = max_epochs

    async def async_len(self) -> int:
        if self.max_epochs is None:
            raise ValueError("Cannot determine length of an infinite dataset without max_epochs.")
        # Return the total number of samples: max_epochs * length of the dataset
        return self.max_epochs * await self.dataset.async_len()

    async def final_length_is_known(self) -> bool:
        return await self.dataset.final_length_is_known()

    def is_finite(self) -> bool:
        # EpochDataset can be finite if max_epochs is set.
        return self.max_epochs is not None

    async def current_len(self) -> Optional[int]:
        # If max_epochs is None, the dataset is effectively infinite.
        if self.max_epochs is None:
            return None

        # If the final length of the dataset is not known, return the current length of the underlying dataset.
        if not await self.dataset.final_length_is_known():
            return await self.dataset.current_len()

        # If the final length is known, return the max_epochs * async_len of the dataset.
        return self.max_epochs * await self.dataset.async_len()

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        # Use self.wait_until_len_at_least to ensure we have enough data for the batch.
        max_index = max(indices)
        ds_len = await self.dataset.wait_until_len_at_least(max_index + 1)

        # Determine the epoch based on the largest index
        epoch = max_index // ds_len

        # If max_epochs is specified, raise an error if the epoch exceeds the allowed number of epochs
        if self.max_epochs is not None and epoch >= self.max_epochs:
            raise StopIteration(
                f"Reached maximum number of epochs: epoch {epoch} exceeds the maximum allowed {self.max_epochs}"
            )

        # Wrap the indices within the bounds of the dataset length
        wrapped_indices = [idx % ds_len for idx in indices]

        # Delegate to the underlying dataset's get_batch
        return await self.dataset.get_batch(wrapped_indices)

    async def wait_until_len_at_least(self, length: int) -> int:
        """
        Returns the length of the dataset once it is at least `length` or if the dataset has a known (finished) length.
        If the dataset's actual length is less than `length`, it returns the minimum of async_len and the current length.
        """
        # Wait until the underlying dataset's length is at least `length`
        if not self.is_finite():
            return length

        if await self.dataset.final_length_is_known():
            base_length = await self.dataset.async_len()
        else:
            base_length = await self.dataset.wait_until_len_at_least(length)

        if base_length < length:
            # hit epoch boundary
            assert self.max_epochs is not None
            return self.max_epochs * base_length

        return base_length
