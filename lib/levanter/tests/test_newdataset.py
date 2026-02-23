# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

import jax.random
import pytest

from levanter.data import EraShufflingDataset, PermutationDataset
from levanter.data.dataset import ListAsyncDataset


@pytest.mark.asyncio
async def test_permutation_multi_epoch_coverage_and_reshuffling():
    """Each window of N consecutive indices is a complete permutation of the dataset,
    and different epochs produce different orderings."""
    data = list(range(50))
    n = len(data)
    num_epochs = 5
    perm_ds = PermutationDataset(ListAsyncDataset(data), jax.random.PRNGKey(42))

    epoch_orderings = []
    for epoch in range(num_epochs):
        batch = await perm_ds.get_batch(list(range(epoch * n, (epoch + 1) * n)))
        # Every element appears exactly once per epoch
        assert sorted(batch) == data, f"Epoch {epoch} missing elements"
        epoch_orderings.append(batch)

    # Epochs are reshuffled — not all orderings are the same
    distinct_orderings = len({tuple(o) for o in epoch_orderings})
    assert distinct_orderings > 1, "Expected different orderings across epochs"

    # Total over all epochs: every element appears exactly num_epochs times
    all_items = [item for ordering in epoch_orderings for item in ordering]
    counts = Counter(all_items)
    assert all(c == num_epochs for c in counts.values())


@pytest.mark.asyncio
async def test_permutation_deterministic_resumption():
    """Reading the same indices across separate calls returns identical results,
    enabling deterministic resume from any checkpoint."""
    data = list(range(100))
    perm_ds = PermutationDataset(ListAsyncDataset(data), jax.random.PRNGKey(7))

    # Simulate training that reads some indices, then resumes from the middle
    indices = [0, 50, 100, 150, 200, 99]
    first_read = await perm_ds.get_batch(indices)
    second_read = await perm_ds.get_batch(indices)
    assert first_read == second_read


@pytest.mark.asyncio
async def test_permutation_max_epochs_terminates_dataset():
    """max_epochs bounds the dataset: elements are accessible within the epoch window
    and the dataset reports finite with the correct length."""
    data = list(range(20))
    n = len(data)
    perm_ds = PermutationDataset(ListAsyncDataset(data), jax.random.PRNGKey(0), max_epochs=3)

    # The dataset is finite and correctly sized
    assert perm_ds.is_finite()
    assert await perm_ds.async_len() == 3 * n

    # All 3 epochs worth of data are accessible and form complete permutations
    all_items = await perm_ds.get_batch(list(range(3 * n)))
    counts = Counter(all_items)
    assert all(c == 3 for c in counts.values())

    # Without max_epochs, dataset is infinite
    infinite_ds = PermutationDataset(ListAsyncDataset(data), jax.random.PRNGKey(0))
    assert not infinite_ds.is_finite()


@pytest.mark.asyncio
async def test_era_shuffling_local_shuffle_with_coverage():
    """Era shuffling shuffles within eras while maintaining full dataset coverage
    when era_length evenly divides dataset_len."""
    data = list(range(40))
    n = len(data)
    era_ds = EraShufflingDataset(ListAsyncDataset(data), era_length=10, key=jax.random.PRNGKey(0))

    # First epoch covers all elements
    epoch_0 = await era_ds.get_batch(list(range(n)))
    assert sorted(epoch_0) == data

    # Data within each era is shuffled (not identity ordering)
    first_era = await era_ds.get_batch(list(range(10)))
    assert set(first_era) == set(range(10))  # correct elements
    # At least one era across multiple seeds should be shuffled
    any_shuffled = first_era != list(range(10))
    if not any_shuffled:
        # Try with a different seed to avoid flakiness
        era_ds2 = EraShufflingDataset(ListAsyncDataset(data), era_length=10, key=jax.random.PRNGKey(99))
        first_era2 = await era_ds2.get_batch(list(range(10)))
        any_shuffled = first_era2 != list(range(10))
    assert any_shuffled, "Era shuffling should reorder elements within eras"

    # Multi-epoch: 3 epochs gives 3x coverage
    num_epochs = 3
    all_items = await era_ds.get_batch(list(range(num_epochs * n)))
    counts = Counter(all_items)
    assert all(c == num_epochs for c in counts.values())


@pytest.mark.asyncio
async def test_era_shuffling_max_epochs():
    """EraShufflingDataset respects max_epochs for termination."""
    data = list(range(20))
    era_ds = EraShufflingDataset(ListAsyncDataset(data), era_length=5, key=jax.random.PRNGKey(0), max_epochs=2)

    assert era_ds.is_finite()
    assert await era_ds.async_len() == 40

    all_items = await era_ds.get_batch(list(range(40)))
    counts = Counter(all_items)
    assert all(c == 2 for c in counts.values())


@pytest.mark.asyncio
async def test_metrics_propagate_through_composition():
    """Metrics from a PermutationDataset propagate correctly through map and slice wrappers."""
    data = list(range(10))
    perm_ds = PermutationDataset(ListAsyncDataset(data), jax.random.PRNGKey(0))

    # Prime the cache so metrics work
    await perm_ds.getitem_async(0)

    # map() preserves metrics
    mapped = perm_ds.map(lambda x: x * 2)
    metrics = mapped.metrics_for_global_index(15)
    assert metrics["data/epoch"] == 1.0
    assert 0.0 < metrics["data/epoch_progress"] < 1.0

    # slice() shifts the index correctly
    sliced = perm_ds.slice_dataset(start_index=10)
    metrics = sliced.metrics_for_global_index(0)
    # Index 0 in the slice -> index 10 in inner -> epoch 1
    assert metrics["data/epoch"] == 1.0
    assert metrics["data/epoch_progress"] == 0.0

    # Chained: map(slice(perm))
    chained = perm_ds.slice_dataset(start_index=20).map(lambda x: x + 1)
    metrics = chained.metrics_for_global_index(5)
    # Index 5 -> slice shifts to 25 -> epoch 2, progress 0.5
    assert metrics["data/epoch"] == 2.0
    assert metrics["data/epoch_progress"] == 0.5


@pytest.mark.asyncio
async def test_convenience_methods_produce_working_datasets():
    """The .shuffle() and .era_shuffle() convenience methods produce functional datasets."""
    data = list(range(20))
    ds = ListAsyncDataset(data)

    # shuffle() with max_epochs
    shuffled = ds.shuffle(jax.random.PRNGKey(0), max_epochs=2)
    assert shuffled.is_finite()
    items = await shuffled.get_batch(list(range(40)))
    assert sorted(items) == sorted(data * 2)

    # era_shuffle() infinite
    era = ds.era_shuffle(5, jax.random.PRNGKey(0))
    assert not era.is_finite()
    epoch_items = await era.get_batch(list(range(20)))
    assert sorted(epoch_items) == data
