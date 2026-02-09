# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Sequence

import jax.random
from async_lru import alru_cache
from jaxtyping import PRNGKeyArray

from levanter.data import AsyncDataset
from levanter.data._prp import PermType, Permutation
from levanter.data.dataset import T_co


class PermutationDataset(AsyncDataset[T_co]):
    """A permutation dataset that wraps another dataset and applies a permutation to the indices."""

    # TODO: add epoch reshuffling

    def __init__(self, dataset: AsyncDataset[T_co], key: PRNGKeyArray, perm_type: PermType = "feistel"):
        self.dataset = dataset
        self.key = key
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


class EraShufflingDataset(AsyncDataset[T_co]):
    r"""
    A dataset that shuffles the data in "eras" of fixed length. Era shuffling is somewhere in between a shuffle buffer
    and a permutation. It's a "local" permutation where pi(i) \in [ (i//L) * L, (i//L + 1) * L ) for some era length L.

    The advantages of era shuffling are:
    - It's stateless, so resumes are easy
    - Like shuffle buffers, it's a decent compromise between full shuffling and no shuffling
    - Like a shuffle buffer, it's streaming: we don't need to know the length of the data in advance

    The disadvantages are:
    - It's not as good as full shuffling
    - It distributes less well than a shuffle buffer does. It's more like a "local" shuffle buffer.
    - You have to wait for an era to fill before you can start shuffling it. With prefetching, this is less of an issue.


    # TODO: given the way tokenization works (where it runs way ahead of training), we can probably increase the era
    length # over time. This would be a nice feature to have.
    """

    def __init__(
        self, dataset: AsyncDataset[T_co], era_length: int, *, key: jax.random.PRNGKey, perm_type: PermType = "feistel"
    ):
        self.dataset = dataset
        self.era_length = era_length
        self.key = key
        self._perm_type = perm_type

        @alru_cache(maxsize=4)  # we're mostly going to be going sequentially
        async def gen_era_permutation(era: int) -> Permutation:
            # TODO: support epochs
            if self.dataset.is_finite():
                # edge case: final era may be shorter than era_length
                dataset_len = await self.dataset.async_len()
                era_length_val = min(self.era_length, dataset_len - era * self.era_length)
            else:
                era_length_val = self.era_length

            mix_key = jax.random.fold_in(key, era)
            return Permutation.make(self._perm_type, era_length_val, mix_key)

        self.gen_era_permutation = gen_era_permutation

    async def _get_index(self, idx: int) -> int:
        if idx < 0:
            raise ValueError("Negative indices are not supported")
        era = idx // self.era_length
        permutation = await self.gen_era_permutation(era)
        out = permutation(idx - era * self.era_length) + era * self.era_length

        return out

    async def async_len(self) -> int:
        return await self.dataset.async_len()

    def is_finite(self) -> bool:
        return self.dataset.is_finite()

    async def getitem_async(self, index: int) -> T_co:
        return await self.dataset.getitem_async(await self._get_index(index))

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        return await self.dataset.get_batch([await self._get_index(i) for i in indices])

    def __repr__(self):
        return f"EraShufflingDataset({repr(self.dataset)}, era_length={self.era_length})"

    def __str__(self):
        return f"EraShufflingDataset({str(self.dataset)})"
