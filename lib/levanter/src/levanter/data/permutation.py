# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import Optional, Sequence

import jax.random
from async_lru import alru_cache
from jaxtyping import PRNGKeyArray

from levanter.data import AsyncDataset
from levanter.data._prp import PermType, Permutation
from levanter.data.dataset import T_co
from levanter.utils.jax_utils import local_cpu_mesh


class PermutationDataset(AsyncDataset[T_co]):
    """A dataset that wraps another dataset and applies a per-epoch permutation to the indices.

    The dataset is logically infinite by default: indices beyond the underlying dataset
    length wrap into new epochs, each with a distinct deterministic permutation derived
    via ``jax.random.fold_in(key, epoch)``.  Every window of ``dataset_len`` consecutive
    indices visits each element exactly once, so after ``k`` epochs each element has
    been seen exactly ``k`` times.

    Args:
        dataset: The underlying dataset to permute.
        key: PRNG key for deterministic shuffling.
        perm_type: Permutation algorithm (default ``"feistel"``).
        max_epochs: If set, the dataset terminates after this many epochs.
            ``None`` (default) means infinite.
    """

    def __init__(
        self,
        dataset: AsyncDataset[T_co],
        key: PRNGKeyArray,
        perm_type: PermType = "feistel",
        max_epochs: Optional[int] = None,
    ):
        self.dataset = dataset
        self.key = key
        self._perm_type = perm_type
        self._max_epochs = max_epochs
        self._cached_len: Optional[int] = None

        @alru_cache(maxsize=4)
        async def _get_epoch_permutation(epoch: int) -> Permutation:
            dataset_len = await self._get_dataset_len()
            epoch_key = jax.random.fold_in(self.key, epoch)
            return Permutation.make(self._perm_type, dataset_len, epoch_key)

        self._get_epoch_permutation = _get_epoch_permutation

    async def _get_dataset_len(self) -> int:
        if self._cached_len is None:
            self._cached_len = await self.dataset.async_len()
        return self._cached_len

    async def async_len(self) -> int:
        if self._max_epochs is not None:
            return self._max_epochs * await self._get_dataset_len()
        return sys.maxsize

    def is_finite(self) -> bool:
        return self._max_epochs is not None

    def metrics_for_global_index(self, global_index: int) -> dict[str, float]:
        if self._cached_len is None or self._cached_len == 0:
            return {}
        epoch = global_index // self._cached_len
        progress = (global_index % self._cached_len) / self._cached_len
        return {"data/epoch": float(epoch), "data/epoch_progress": progress}

    async def getitem_async(self, index: int) -> T_co:
        dataset_len = await self._get_dataset_len()
        epoch = index // dataset_len
        position = index % dataset_len
        perm = await self._get_epoch_permutation(epoch)
        return await self.dataset.getitem_async(int(perm(position)))

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        dataset_len = await self._get_dataset_len()
        mapped = []
        for i in indices:
            epoch = i // dataset_len
            position = i % dataset_len
            perm = await self._get_epoch_permutation(epoch)
            mapped.append(int(perm(position)))
        return await self.dataset.get_batch(mapped)


class EraShufflingDataset(AsyncDataset[T_co]):
    r"""
    A dataset that shuffles the data in "eras" of fixed length. Era shuffling is somewhere in between a shuffle buffer
    and a permutation. It's a "local" permutation where pi(i) \in [ (i//L) * L, (i//L + 1) * L ) for some era length L.

    The dataset is logically infinite by default: indices beyond the underlying dataset
    length wrap into new epochs, each with a fresh set of era permutations derived via
    ``jax.random.fold_in(key, era)``.  Every window of ``dataset_len`` consecutive
    indices visits each element exactly once.

    The advantages of era shuffling are:
    - It's stateless, so resumes are easy
    - Like shuffle buffers, it's a decent compromise between full shuffling and no shuffling
    - Like a shuffle buffer, it's streaming: we don't need to know the length of the data in advance

    The disadvantages are:
    - It's not as good as full shuffling
    - It distributes less well than a shuffle buffer does. It's more like a "local" shuffle buffer.
    - You have to wait for an era to fill before you can start shuffling it. With prefetching, this is less of an issue.

    Args:
        dataset: The underlying dataset to shuffle.
        era_length: Number of elements per era.
        key: PRNG key for deterministic shuffling.
        perm_type: Permutation algorithm (default ``"feistel"``).
        max_epochs: If set, the dataset terminates after this many epochs.
            ``None`` (default) means infinite.

    # TODO: given the way tokenization works (where it runs way ahead of training), we can probably increase the era
    length # over time. This would be a nice feature to have.
    """

    def __init__(
        self,
        dataset: AsyncDataset[T_co],
        era_length: int,
        *,
        key: jax.random.PRNGKey,
        perm_type: PermType = "feistel",
        max_epochs: Optional[int] = None,
    ):
        self.dataset = dataset
        self.era_length = era_length

        # Force key to CPU (like MixtureDataset does) to prevent JAX device placement errors
        with local_cpu_mesh():
            if isinstance(key, int):
                key = jax.random.PRNGKey(key)
            else:
                key = jax.device_put(jax.device_get(key))

        self.key = key
        self._perm_type = perm_type
        self._max_epochs = max_epochs
        self._cached_len: Optional[int] = None

        @alru_cache(maxsize=4)  # we're mostly going to be going sequentially
        async def gen_era_permutation(era: int) -> Permutation:
            mix_key = jax.random.fold_in(key, era)
            return Permutation.make(self._perm_type, self.era_length, mix_key)

        self.gen_era_permutation = gen_era_permutation

    async def _get_dataset_len(self) -> int:
        if self._cached_len is None:
            self._cached_len = await self.dataset.async_len()
        return self._cached_len

    async def _get_index(self, idx: int) -> int:
        """Map a logical index to a dataset index.

        The logical index space is infinite. Era-local shuffling is applied within
        each era, and the result is mapped back to the original dataset range via
        modulo so that each epoch-window visits every element exactly once.
        """
        if idx < 0:
            raise ValueError("Negative indices are not supported")
        era = idx // self.era_length
        permutation = await self.gen_era_permutation(era)
        shuffled = permutation(idx - era * self.era_length) + era * self.era_length

        if self.dataset.is_finite():
            dataset_len = await self._get_dataset_len()
            return shuffled % dataset_len
        return shuffled

    async def async_len(self) -> int:
        if self._max_epochs is not None:
            return self._max_epochs * await self._get_dataset_len()
        return sys.maxsize

    def is_finite(self) -> bool:
        return self._max_epochs is not None

    def metrics_for_global_index(self, global_index: int) -> dict[str, float]:
        if self._cached_len is None or self._cached_len == 0:
            return {}
        epoch = global_index // self._cached_len
        progress = (global_index % self._cached_len) / self._cached_len
        return {"data/epoch": float(epoch), "data/epoch_progress": progress}

    async def getitem_async(self, index: int) -> T_co:
        return await self.dataset.getitem_async(await self._get_index(index))

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        return await self.dataset.get_batch([await self._get_index(i) for i in indices])

    def __repr__(self):
        return f"EraShufflingDataset({repr(self.dataset)}, era_length={self.era_length})"

    def __str__(self):
        return f"EraShufflingDataset({str(self.dataset)})"
