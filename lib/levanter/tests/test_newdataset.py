# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import jax.random
import pytest

from levanter.data import BlockShufflingDataset, EraShufflingDataset, PermutationDataset
from levanter.data.dataset import ListAsyncDataset


@pytest.mark.asyncio
async def test_length_of_sequence_dataset_is_accurate():
    data = [1, 2, 3]
    dataset = ListAsyncDataset(data)
    assert dataset.is_finite()
    assert await dataset.async_len() == 3


@pytest.mark.asyncio
async def test_list_dataset_get_item_returns_correct_item():
    data = ["a", "b", "c"]
    dataset = ListAsyncDataset(data)
    assert await dataset.getitem_async(1) == "b"


@pytest.mark.asyncio
async def test_list_async_dataset_single_item():
    dataset = ListAsyncDataset(["a"])
    assert await dataset.async_len() == 1
    assert await dataset.get_batch([0]) == ["a"]


@pytest.mark.asyncio
async def test_permutation_dataset_is_at_least_sometimes_permuted():
    ok = 0
    for seed in range(10):
        data = [1, 2, 3, 4]
        dataset = ListAsyncDataset(data)
        permuted_dataset = PermutationDataset(dataset, jax.random.PRNGKey(seed))
        batch = await permuted_dataset.get_batch([0, 1, 2, 3])
        if batch != [1, 2, 3, 4]:
            ok += 1

    assert ok > 5, "Permutation dataset is not actually permuting"


@pytest.mark.asyncio
async def test_era_shuffling_dataset_returns_correct_length():
    data = list(range(100))
    dataset = ListAsyncDataset(data)
    era_length = 10
    key = jax.random.PRNGKey(0)
    shuffling_dataset = EraShufflingDataset(dataset, era_length, key=key)
    assert shuffling_dataset.is_finite()
    assert await shuffling_dataset.async_len() == 100


@pytest.mark.asyncio
async def test_era_shuffling_dataset_get_batch_returns_shuffled_batch():
    data = list(range(20))
    dataset = ListAsyncDataset(data)
    era_length = 5
    key = jax.random.PRNGKey(0)
    shuffling_dataset = EraShufflingDataset(dataset, era_length, key=key)
    batch_indices = [0, 1, 2, 3, 4]
    batch = await shuffling_dataset.get_batch(batch_indices)
    assert set(batch) == set([0, 1, 2, 3, 4])  # Ensures all elements are from the first era but does not assume order
    assert batch != [0, 1, 2, 3, 4]  # Ensures the batch is shuffled


@pytest.mark.asyncio
async def test_era_shuffling_returns_full_finite_length():
    data = list(range(16))
    dataset = ListAsyncDataset(data)
    era_length = 5
    key = jax.random.PRNGKey(0)
    shuffling_dataset = EraShufflingDataset(dataset, era_length, key=key)
    assert await shuffling_dataset.async_len() == 16
    batch = await shuffling_dataset.get_batch(list(range(16)))
    assert set(batch) == set(range(16))


@pytest.mark.asyncio
async def test_era_shuffling_raises_on_out_of_bounds_index():
    dataset = ListAsyncDataset(list(range(16)))
    shuffling_dataset = EraShufflingDataset(dataset, era_length=5, key=jax.random.PRNGKey(0))
    with pytest.raises(IndexError, match="out of bounds"):
        await shuffling_dataset.getitem_async(16)


@pytest.mark.asyncio
async def test_block_shuffling_dataset_is_deterministic_and_a_permutation():
    data = list(range(37))
    dataset = ListAsyncDataset(data)
    key = jax.random.PRNGKey(0)

    ds1 = BlockShufflingDataset(dataset, io_block_size=4, window_blocks=3, key=key)
    ds2 = BlockShufflingDataset(dataset, io_block_size=4, window_blocks=3, key=key)

    indices = list(range(len(data)))
    batch1 = await ds1.get_batch(indices)
    batch2 = await ds2.get_batch(indices)

    assert batch1 == batch2
    assert sorted(batch1) == data


@pytest.mark.asyncio
async def test_block_shuffling_dataset_is_often_nontrivial():
    data = list(range(37))
    dataset = ListAsyncDataset(data)

    nontrivial = 0
    for seed in range(10):
        block_shuffled = BlockShufflingDataset(
            dataset,
            io_block_size=4,
            window_blocks=3,
            key=jax.random.PRNGKey(seed),
        )
        batch = await block_shuffled.get_batch(list(range(len(data))))
        if batch != data:
            nontrivial += 1

    assert nontrivial >= 7, f"Expected non-trivial permutation for most seeds, got {nontrivial}/10"


@pytest.mark.asyncio
async def test_block_shuffling_keeps_tiny_tail_block_at_end():
    data = list(range(10))  # tiny tail block has 2 examples when io_block_size=4
    dataset = ListAsyncDataset(data)
    block_shuffled = BlockShufflingDataset(dataset, io_block_size=4, window_blocks=2, key=jax.random.PRNGKey(0))

    batch = await block_shuffled.get_batch(list(range(len(data))))

    # By design, the final tiny block stays at the very end (possibly permuted within itself).
    assert set(batch[-2:]) == {8, 9}
    assert all(x < 8 for x in batch[:-2])
    assert sorted(batch) == data


@pytest.mark.asyncio
async def test_block_shuffling_handles_dataset_smaller_than_block():
    data = list(range(3))
    dataset = ListAsyncDataset(data)
    block_shuffled = BlockShufflingDataset(dataset, io_block_size=8, window_blocks=4, key=jax.random.PRNGKey(0))

    batch = await block_shuffled.get_batch([0, 1, 2])
    assert sorted(batch) == data


@pytest.mark.asyncio
async def test_block_shuffle_convenience_api():
    data = list(range(12))
    dataset = ListAsyncDataset(data)
    block_shuffled = dataset.block_shuffle(io_block_size=4, window_blocks=2, key=jax.random.PRNGKey(0))

    batch = await block_shuffled.get_batch(list(range(12)))
    assert sorted(batch) == data


class _CountingListAsyncDataset(ListAsyncDataset[int]):
    def __init__(self, data: list[int]):
        super().__init__(data)
        self.batch_calls = 0
        self.indices_requested = 0

    async def get_batch(self, indices: Sequence[int]) -> Sequence[int]:
        self.batch_calls += 1
        self.indices_requested += len(indices)
        return await super().get_batch(indices)


@pytest.mark.asyncio
async def test_block_shuffle_cache_reuses_materialized_blocks():
    data = list(range(32))
    backing = _CountingListAsyncDataset(data)
    block_shuffled = BlockShufflingDataset(
        backing,
        io_block_size=4,
        window_blocks=2,
        cache_windows=2,  # enough cache for two windows (4 blocks)
        key=jax.random.PRNGKey(0),
    )

    indices = list(range(16))
    first = await block_shuffled.get_batch(indices)
    calls_after_first = backing.batch_calls
    requested_after_first = backing.indices_requested

    second = await block_shuffled.get_batch(indices)

    assert second == first
    assert backing.batch_calls == calls_after_first
    assert backing.indices_requested == requested_after_first


@pytest.mark.asyncio
async def test_block_shuffle_cache_evicts_when_capacity_is_small():
    data = list(range(64))
    backing = _CountingListAsyncDataset(data)
    block_shuffled = BlockShufflingDataset(
        backing,
        io_block_size=4,
        window_blocks=2,
        cache_windows=1,  # 2 blocks max
        key=jax.random.PRNGKey(0),
    )

    first_chunk = list(range(16))
    second_chunk = list(range(16, 32))

    await block_shuffled.get_batch(first_chunk)
    requested_after_first = backing.indices_requested
    await block_shuffled.get_batch(second_chunk)
    requested_after_second = backing.indices_requested
    await block_shuffled.get_batch(first_chunk)
    requested_after_third = backing.indices_requested

    assert requested_after_second > requested_after_first
    assert requested_after_third > requested_after_second
