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


# --- max_epochs tests ---


@pytest.mark.asyncio
async def test_permutation_dataset_max_epochs_finite():
    """With max_epochs set, PermutationDataset is finite and has correct length."""
    data = list(range(10))
    dataset = ListAsyncDataset(data)
    perm_ds = PermutationDataset(dataset, jax.random.PRNGKey(0), max_epochs=3)
    assert perm_ds.is_finite()
    assert await perm_ds.async_len() == 30  # 3 * 10


@pytest.mark.asyncio
async def test_permutation_dataset_max_epochs_single():
    """max_epochs=1 gives a single-pass dataset that terminates."""
    data = list(range(10))
    dataset = ListAsyncDataset(data)
    perm_ds = PermutationDataset(dataset, jax.random.PRNGKey(0), max_epochs=1)
    assert perm_ds.is_finite()
    assert await perm_ds.async_len() == 10
    batch = await perm_ds.get_batch(list(range(10)))
    assert set(batch) == set(data)


@pytest.mark.asyncio
async def test_permutation_dataset_max_epochs_none_is_infinite():
    """max_epochs=None (default) gives an infinite dataset."""
    data = list(range(10))
    dataset = ListAsyncDataset(data)
    perm_ds = PermutationDataset(dataset, jax.random.PRNGKey(0), max_epochs=None)
    assert not perm_ds.is_finite()
    assert await perm_ds.async_len() == sys.maxsize


@pytest.mark.asyncio
async def test_era_shuffling_max_epochs_finite():
    """With max_epochs set, EraShufflingDataset is finite."""
    data = list(range(20))
    dataset = ListAsyncDataset(data)
    era_ds = EraShufflingDataset(dataset, era_length=5, key=jax.random.PRNGKey(0), max_epochs=2)
    assert era_ds.is_finite()
    assert await era_ds.async_len() == 40  # 2 * 20


@pytest.mark.asyncio
async def test_era_shuffling_max_epochs_none_is_infinite():
    """max_epochs=None (default) gives an infinite EraShufflingDataset."""
    data = list(range(20))
    dataset = ListAsyncDataset(data)
    era_ds = EraShufflingDataset(dataset, era_length=5, key=jax.random.PRNGKey(0), max_epochs=None)
    assert not era_ds.is_finite()
    assert await era_ds.async_len() == sys.maxsize


# --- metrics_for_global_index tests ---


@pytest.mark.asyncio
async def test_permutation_dataset_metrics():
    """metrics_for_global_index reports epoch and progress."""
    data = list(range(10))
    dataset = ListAsyncDataset(data)
    perm_ds = PermutationDataset(dataset, jax.random.PRNGKey(0))

    # Force _cached_len to be populated
    await perm_ds.getitem_async(0)

    metrics_0 = perm_ds.metrics_for_global_index(0)
    assert metrics_0["data/epoch"] == 0.0
    assert metrics_0["data/epoch_progress"] == 0.0

    metrics_5 = perm_ds.metrics_for_global_index(5)
    assert metrics_5["data/epoch"] == 0.0
    assert metrics_5["data/epoch_progress"] == 0.5

    metrics_10 = perm_ds.metrics_for_global_index(10)
    assert metrics_10["data/epoch"] == 1.0
    assert metrics_10["data/epoch_progress"] == 0.0

    metrics_25 = perm_ds.metrics_for_global_index(25)
    assert metrics_25["data/epoch"] == 2.0
    assert metrics_25["data/epoch_progress"] == 0.5


@pytest.mark.asyncio
async def test_era_shuffling_dataset_metrics():
    """metrics_for_global_index on EraShufflingDataset reports epoch and progress."""
    data = list(range(20))
    dataset = ListAsyncDataset(data)
    era_ds = EraShufflingDataset(dataset, era_length=5, key=jax.random.PRNGKey(0))

    # Force _cached_len to be populated
    await era_ds.getitem_async(0)

    metrics_0 = era_ds.metrics_for_global_index(0)
    assert metrics_0["data/epoch"] == 0.0
    assert metrics_0["data/epoch_progress"] == 0.0

    metrics_10 = era_ds.metrics_for_global_index(10)
    assert metrics_10["data/epoch"] == 0.0
    assert metrics_10["data/epoch_progress"] == 0.5

    metrics_20 = era_ds.metrics_for_global_index(20)
    assert metrics_20["data/epoch"] == 1.0
    assert metrics_20["data/epoch_progress"] == 0.0


@pytest.mark.asyncio
async def test_base_dataset_metrics_empty():
    """Base AsyncDataset returns empty metrics by default."""
    data = list(range(10))
    dataset = ListAsyncDataset(data)
    assert dataset.metrics_for_global_index(0) == {}
    assert dataset.metrics_for_global_index(5) == {}


@pytest.mark.asyncio
async def test_mapped_dataset_delegates_metrics():
    """MappedAsyncDataset delegates metrics_for_global_index to the inner dataset."""
    data = list(range(10))
    dataset = ListAsyncDataset(data)
    perm_ds = PermutationDataset(dataset, jax.random.PRNGKey(0))

    # Force _cached_len to be populated
    await perm_ds.getitem_async(0)

    mapped = perm_ds.map(lambda x: x * 2)
    metrics = mapped.metrics_for_global_index(10)
    assert metrics["data/epoch"] == 1.0


@pytest.mark.asyncio
async def test_sliced_dataset_delegates_metrics():
    """SlicedAsyncDataset delegates metrics with shifted index."""
    data = list(range(10))
    dataset = ListAsyncDataset(data)
    perm_ds = PermutationDataset(dataset, jax.random.PRNGKey(0))

    # Force _cached_len to be populated
    await perm_ds.getitem_async(0)

    sliced = perm_ds.slice_dataset(start_index=10)
    # Index 0 in the sliced dataset maps to index 10 in the inner dataset (epoch 1)
    metrics = sliced.metrics_for_global_index(0)
    assert metrics["data/epoch"] == 1.0


# --- Convenience method max_epochs tests ---


@pytest.mark.asyncio
async def test_shuffle_method_with_max_epochs():
    """The .shuffle() convenience method accepts max_epochs."""
    data = list(range(10))
    dataset = ListAsyncDataset(data)
    perm_ds = dataset.shuffle(jax.random.PRNGKey(0), max_epochs=2)
    assert perm_ds.is_finite()
    assert await perm_ds.async_len() == 20


@pytest.mark.asyncio
async def test_era_shuffle_method_with_max_epochs():
    """The .era_shuffle() convenience method accepts max_epochs."""
    data = list(range(20))
    dataset = ListAsyncDataset(data)
    era_ds = dataset.era_shuffle(5, jax.random.PRNGKey(0), max_epochs=3)
    assert era_ds.is_finite()
    assert await era_ds.async_len() == 60


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
