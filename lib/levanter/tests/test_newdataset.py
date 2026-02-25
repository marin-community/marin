# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

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
    unique_permutations: set[tuple[int, ...]] = set()
    for seed in range(10):
        block_shuffled = BlockShufflingDataset(
            dataset,
            io_block_size=4,
            window_blocks=3,
            key=jax.random.PRNGKey(seed),
        )
        batch = await block_shuffled.get_batch(list(range(len(data))))
        unique_permutations.add(tuple(batch))
        if batch != data:
            nontrivial += 1

    assert nontrivial >= 7, f"Expected non-trivial permutation for most seeds, got {nontrivial}/10"
    assert len(unique_permutations) > 1


@pytest.mark.asyncio
async def test_block_shuffling_handles_dataset_smaller_than_block():
    data = list(range(3))
    dataset = ListAsyncDataset(data)
    block_shuffled = BlockShufflingDataset(dataset, io_block_size=8, window_blocks=4, key=jax.random.PRNGKey(0))

    batch = await block_shuffled.get_batch([0, 1, 2])
    assert sorted(batch) == data
