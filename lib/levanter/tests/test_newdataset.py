# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

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


# --- Multi-epoch tests ---


@pytest.mark.asyncio
async def test_permutation_dataset_multi_epoch_length():
    """num_epochs multiplies the reported length."""
    data = list(range(10))
    dataset = ListAsyncDataset(data)
    perm_ds = PermutationDataset(dataset, jax.random.PRNGKey(0), num_epochs=3)
    assert await perm_ds.async_len() == 30


@pytest.mark.asyncio
async def test_permutation_dataset_multi_epoch_covers_all_items():
    """Every original item appears exactly num_epochs times across the full index range."""
    from collections import Counter

    data = list(range(10))
    dataset = ListAsyncDataset(data)
    num_epochs = 3
    perm_ds = PermutationDataset(dataset, jax.random.PRNGKey(42), num_epochs=num_epochs)
    total_len = await perm_ds.async_len()
    batch = await perm_ds.get_batch(list(range(total_len)))
    counts = Counter(batch)
    assert set(counts.keys()) == set(data)
    for item, count in counts.items():
        assert count == num_epochs, f"Item {item} appeared {count} times, expected {num_epochs}"


@pytest.mark.asyncio
async def test_permutation_dataset_multi_epoch_reshuffles():
    """Different epochs produce different orderings."""
    data = list(range(20))
    dataset = ListAsyncDataset(data)
    num_epochs = 3
    n = len(data)
    perm_ds = PermutationDataset(dataset, jax.random.PRNGKey(7), num_epochs=num_epochs)

    epochs = []
    for epoch in range(num_epochs):
        batch = await perm_ds.get_batch(list(range(epoch * n, (epoch + 1) * n)))
        epochs.append(batch)

    # At least two of the three epochs should differ in ordering
    num_different = sum(1 for i in range(num_epochs) for j in range(i + 1, num_epochs) if epochs[i] != epochs[j])
    assert num_different >= 1, "Multi-epoch permutation should produce different orderings across epochs"


@pytest.mark.asyncio
async def test_permutation_dataset_single_epoch_unchanged():
    """With num_epochs=1, behavior is identical to the original PermutationDataset."""
    data = list(range(10))
    dataset = ListAsyncDataset(data)
    perm_ds = PermutationDataset(dataset, jax.random.PRNGKey(0), num_epochs=1)
    assert await perm_ds.async_len() == 10
    batch = await perm_ds.get_batch(list(range(10)))
    assert set(batch) == set(data)


@pytest.mark.asyncio
async def test_permutation_dataset_num_epochs_validation():
    """num_epochs < 1 raises ValueError."""
    data = list(range(10))
    dataset = ListAsyncDataset(data)
    with pytest.raises(ValueError, match="num_epochs"):
        PermutationDataset(dataset, jax.random.PRNGKey(0), num_epochs=0)


@pytest.mark.asyncio
async def test_era_shuffling_multi_epoch_length():
    """num_epochs multiplies the reported length for EraShufflingDataset."""
    data = list(range(16))
    dataset = ListAsyncDataset(data)
    era_ds = EraShufflingDataset(dataset, era_length=5, key=jax.random.PRNGKey(0), num_epochs=3)
    assert await era_ds.async_len() == 48


@pytest.mark.asyncio
async def test_era_shuffling_multi_epoch_covers_all_items():
    """Every original item appears num_epochs times across all epochs."""
    from collections import Counter

    data = list(range(16))
    dataset = ListAsyncDataset(data)
    num_epochs = 3
    era_ds = EraShufflingDataset(dataset, era_length=5, key=jax.random.PRNGKey(0), num_epochs=num_epochs)
    total_len = await era_ds.async_len()
    batch = await era_ds.get_batch(list(range(total_len)))
    counts = Counter(batch)
    assert set(counts.keys()) == set(data)
    for item, count in counts.items():
        assert count == num_epochs, f"Item {item} appeared {count} times, expected {num_epochs}"


@pytest.mark.asyncio
async def test_era_shuffling_multi_epoch_raises_on_out_of_bounds():
    """Index beyond num_epochs * dataset_len raises IndexError."""
    data = list(range(16))
    dataset = ListAsyncDataset(data)
    era_ds = EraShufflingDataset(dataset, era_length=5, key=jax.random.PRNGKey(0), num_epochs=2)
    # Total length is 32; index 32 is out of bounds
    with pytest.raises(IndexError, match="out of bounds"):
        await era_ds.getitem_async(32)


@pytest.mark.asyncio
async def test_era_shuffling_num_epochs_validation():
    """num_epochs < 1 raises ValueError."""
    data = list(range(10))
    dataset = ListAsyncDataset(data)
    with pytest.raises(ValueError, match="num_epochs"):
        EraShufflingDataset(dataset, era_length=5, key=jax.random.PRNGKey(0), num_epochs=0)


@pytest.mark.asyncio
async def test_shuffle_method_passes_num_epochs():
    """The .shuffle() convenience method forwards num_epochs."""
    data = list(range(10))
    dataset = ListAsyncDataset(data)
    perm_ds = dataset.shuffle(jax.random.PRNGKey(0), num_epochs=3)
    assert await perm_ds.async_len() == 30


@pytest.mark.asyncio
async def test_era_shuffle_method_passes_num_epochs():
    """The .era_shuffle() convenience method forwards num_epochs."""
    data = list(range(16))
    dataset = ListAsyncDataset(data)
    era_ds = dataset.era_shuffle(5, jax.random.PRNGKey(0), num_epochs=2)
    assert await era_ds.async_len() == 32
