# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from collections import Counter

import jax.random
import pytest

from levanter.data import EraShufflingDataset, PermutationDataset
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
async def test_permutation_dataset_is_infinite():
    """PermutationDataset reports infinite length and is_finite() == False."""
    data = list(range(10))
    dataset = ListAsyncDataset(data)
    perm_ds = PermutationDataset(dataset, jax.random.PRNGKey(0))
    assert not perm_ds.is_finite()
    assert await perm_ds.async_len() == sys.maxsize


@pytest.mark.asyncio
async def test_permutation_dataset_single_epoch_is_permutation():
    """The first epoch (indices 0..N-1) visits every element exactly once."""
    data = list(range(10))
    dataset = ListAsyncDataset(data)
    perm_ds = PermutationDataset(dataset, jax.random.PRNGKey(0))
    batch = await perm_ds.get_batch(list(range(10)))
    assert set(batch) == set(data)


@pytest.mark.asyncio
async def test_permutation_dataset_multi_epoch_exact_coverage():
    """Each epoch-window of N indices visits every element exactly once."""
    data = list(range(10))
    dataset = ListAsyncDataset(data)
    n = len(data)
    num_epochs = 3
    perm_ds = PermutationDataset(dataset, jax.random.PRNGKey(42))

    for epoch in range(num_epochs):
        batch = await perm_ds.get_batch(list(range(epoch * n, (epoch + 1) * n)))
        assert set(batch) == set(data), f"Epoch {epoch} did not cover all elements"

    # Over all epochs, every element appears exactly num_epochs times
    all_items = await perm_ds.get_batch(list(range(num_epochs * n)))
    counts = Counter(all_items)
    for item, count in counts.items():
        assert count == num_epochs, f"Item {item} appeared {count} times, expected {num_epochs}"


@pytest.mark.asyncio
async def test_permutation_dataset_reshuffles_across_epochs():
    """Different epochs produce different orderings."""
    data = list(range(20))
    dataset = ListAsyncDataset(data)
    n = len(data)
    num_epochs = 3
    perm_ds = PermutationDataset(dataset, jax.random.PRNGKey(7))

    epochs = []
    for epoch in range(num_epochs):
        batch = await perm_ds.get_batch(list(range(epoch * n, (epoch + 1) * n)))
        epochs.append(batch)

    # At least two of the three epochs should differ in ordering
    num_different = sum(1 for i in range(num_epochs) for j in range(i + 1, num_epochs) if epochs[i] != epochs[j])
    assert num_different >= 1, "Multi-epoch permutation should produce different orderings across epochs"


@pytest.mark.asyncio
async def test_permutation_dataset_deterministic_resume():
    """Reading the same index twice gives the same result (deterministic & resumable)."""
    data = list(range(50))
    dataset = ListAsyncDataset(data)
    perm_ds = PermutationDataset(dataset, jax.random.PRNGKey(99))

    batch1 = await perm_ds.get_batch([0, 25, 50, 75, 100])
    batch2 = await perm_ds.get_batch([0, 25, 50, 75, 100])
    assert batch1 == batch2


@pytest.mark.asyncio
async def test_permutation_dataset_take_bounds_iteration():
    """Using .take() with PermutationDataset produces a finite dataset of the right length."""
    data = list(range(10))
    dataset = ListAsyncDataset(data)
    perm_ds = PermutationDataset(dataset, jax.random.PRNGKey(0))
    bounded = perm_ds.take(30)  # 3 epochs worth
    assert bounded.is_finite()
    assert await bounded.async_len() == 30


# --- EraShufflingDataset tests ---


@pytest.mark.asyncio
async def test_era_shuffling_dataset_is_infinite():
    """EraShufflingDataset reports infinite length for finite underlying datasets."""
    data = list(range(100))
    dataset = ListAsyncDataset(data)
    era_ds = EraShufflingDataset(dataset, era_length=10, key=jax.random.PRNGKey(0))
    assert not era_ds.is_finite()
    assert await era_ds.async_len() == sys.maxsize


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
async def test_era_shuffling_first_epoch_covers_all():
    """The first epoch of era-shuffled data covers every element when era_length divides dataset_len."""
    data = list(range(20))
    dataset = ListAsyncDataset(data)
    era_ds = EraShufflingDataset(dataset, era_length=5, key=jax.random.PRNGKey(0))
    batch = await era_ds.get_batch(list(range(20)))
    assert set(batch) == set(range(20))


@pytest.mark.asyncio
async def test_era_shuffling_multi_epoch_coverage():
    """Era shuffling covers all elements when iterated over multiple epochs.

    Era shuffling provides local shuffling, not exact per-epoch coverage (use
    ``PermutationDataset`` for that).  When ``era_length`` divides ``dataset_len``
    evenly, coverage is exact; otherwise it's approximate.
    """
    # Use era_length that divides dataset_len evenly for exact coverage test
    data = list(range(20))
    dataset = ListAsyncDataset(data)
    n = len(data)
    num_epochs = 3
    era_ds = EraShufflingDataset(dataset, era_length=5, key=jax.random.PRNGKey(0))

    all_items = await era_ds.get_batch(list(range(num_epochs * n)))
    counts = Counter(all_items)
    assert set(counts.keys()) == set(data)
    for item, count in counts.items():
        assert count == num_epochs, f"Item {item} appeared {count} times, expected {num_epochs}"


@pytest.mark.asyncio
async def test_era_shuffling_take_bounds_iteration():
    """Using .take() with EraShufflingDataset produces a finite dataset."""
    data = list(range(16))
    dataset = ListAsyncDataset(data)
    era_ds = EraShufflingDataset(dataset, era_length=5, key=jax.random.PRNGKey(0))
    bounded = era_ds.take(32)
    assert bounded.is_finite()
    assert await bounded.async_len() == 32


# --- Convenience method tests ---


@pytest.mark.asyncio
async def test_shuffle_method():
    """The .shuffle() convenience method produces an infinite PermutationDataset."""
    data = list(range(10))
    dataset = ListAsyncDataset(data)
    perm_ds = dataset.shuffle(jax.random.PRNGKey(0))
    assert not perm_ds.is_finite()
    batch = await perm_ds.get_batch(list(range(10)))
    assert set(batch) == set(data)


@pytest.mark.asyncio
async def test_era_shuffle_method():
    """The .era_shuffle() convenience method produces an infinite EraShufflingDataset."""
    data = list(range(20))
    dataset = ListAsyncDataset(data)
    era_ds = dataset.era_shuffle(5, jax.random.PRNGKey(0))
    assert not era_ds.is_finite()
    batch = await era_ds.get_batch(list(range(20)))
    assert set(batch) == set(data)
