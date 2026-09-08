# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
from collections.abc import Sequence

import jax.random
import numpy as np
import pytest

from levanter.data.dataset import AsyncDataset, BlockShufflingDataset, ListAsyncDataset, PermutationDataset


class _IndexAsyncDataset(AsyncDataset[int]):
    def __init__(self, length: int):
        self.length = length

    async def async_len(self) -> int:
        return self.length

    def is_finite(self) -> bool:
        return True

    async def get_batch(self, indices: Sequence[int]) -> Sequence[int]:
        return [int(index) for index in indices]


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
async def test_nested_permutation_dataset_batch_matches_scalar_access():
    dataset = ListAsyncDataset(list(range(257)))
    for seed in range(3):
        dataset = PermutationDataset(dataset, jax.random.PRNGKey(seed))

    indices = [256, 3, 41, 3, 128, 0, 255]
    batch = await dataset.get_batch(indices)
    scalar_items = [await dataset.getitem_async(index) for index in indices]

    assert batch == scalar_items


@pytest.mark.asyncio
async def test_random_holdout_split_is_disjoint_complete_and_deterministic():
    dataset = ListAsyncDataset(list(range(1_003)))
    retained, holdout = dataset.random_holdout_split(127, key=jax.random.PRNGKey(17))
    retained_again, holdout_again = dataset.random_holdout_split(127, key=jax.random.PRNGKey(17))
    _, different_holdout = dataset.random_holdout_split(127, key=jax.random.PRNGKey(18))

    retained_items = await retained.get_batch(range(await retained.async_len()))
    holdout_items = await holdout.get_batch(range(await holdout.async_len()))
    retained_again_items = await retained_again.get_batch(range(await retained_again.async_len()))
    holdout_again_items = await holdout_again.get_batch(range(await holdout_again.async_len()))
    different_holdout_items = await different_holdout.get_batch(range(await different_holdout.async_len()))

    assert len(retained_items) == 876
    assert len(holdout_items) == 127
    assert set(retained_items).isdisjoint(holdout_items)
    assert set(retained_items) | set(holdout_items) == set(range(1_003))
    assert retained_items == retained_again_items
    assert holdout_items == holdout_again_items
    assert set(holdout_items) != set(different_holdout_items)

    state = await retained.partition.state()
    replacement_map = dict(
        zip(
            state.retained_replacement_indices.tolist(),
            state.retained_replacement_sources.tolist(),
            strict=True,
        )
    )
    assert retained_items == [replacement_map.get(index, index) for index in range(876)]


@pytest.mark.asyncio
@pytest.mark.parametrize(("dataset_len", "num_holdout"), [(2, 1), (17, 16), (29, 7), (31, 11)])
async def test_random_holdout_split_edge_cases(dataset_len: int, num_holdout: int):
    dataset = _IndexAsyncDataset(dataset_len)
    retained, holdout = dataset.random_holdout_split(num_holdout, key=jax.random.PRNGKey(9))
    retained_items = await retained.get_batch(range(await retained.async_len()))
    holdout_items = await holdout.get_batch(range(await holdout.async_len()))

    assert len(retained_items) == dataset_len - num_holdout
    assert len(holdout_items) == num_holdout
    assert sorted((*retained_items, *holdout_items)) == list(range(dataset_len))


@pytest.mark.asyncio
async def test_random_holdout_split_batch_order_duplicates_and_errors():
    dataset = _IndexAsyncDataset(101)
    retained, holdout = dataset.random_holdout_split(17, key=jax.random.PRNGKey(13))
    retained_indices = [83, 0, 19, 19, 4, 72]
    holdout_indices = [16, 0, 8, 8, 3]

    assert await retained.get_batch(retained_indices) == [
        await retained.getitem_async(index) for index in retained_indices
    ]
    assert await holdout.get_batch(holdout_indices) == [
        await holdout.getitem_async(index) for index in holdout_indices
    ]
    assert await retained.get_batch([]) == []
    assert await holdout.get_batch([]) == []
    with pytest.raises(IndexError):
        await retained.get_batch([84])
    with pytest.raises(IndexError):
        await holdout.get_batch([17])
    with pytest.raises(IndexError):
        await retained.get_batch([-1])
    with pytest.raises(IndexError):
        await holdout.get_batch([-1])
    with pytest.raises(ValueError):
        dataset.random_holdout_split(0, key=jax.random.PRNGKey(13))
    too_large, _ = dataset.random_holdout_split(101, key=jax.random.PRNGKey(13))
    with pytest.raises(ValueError):
        await too_large.async_len()


@pytest.mark.asyncio
async def test_random_holdout_retained_view_preserves_nested_shuffle_locality():
    base = _IndexAsyncDataset(2_000_000_123)
    retained, holdout = base.random_holdout_split(4_096, key=jax.random.PRNGKey(101))
    support = retained.block_shuffle(
        io_block_size=256,
        window_blocks=512,
        key=jax.random.PRNGKey(202),
    ).slice_dataset(end_index=136_704)
    training = support.block_shuffle(
        io_block_size=256,
        window_blocks=512,
        key=jax.random.PRNGKey(303),
    )

    full_training = retained.block_shuffle(
        io_block_size=256,
        window_blocks=512,
        key=jax.random.PRNGKey(303),
    )

    mapped = await training.get_batch(range(4_096))
    full_mapped = await full_training.get_batch(range(4_096))
    holdout_items = await holdout.get_batch(range(4_096))
    mapped_digest = hashlib.sha256(np.asarray(mapped, dtype=np.uint64).tobytes()).hexdigest()
    full_mapping_digest = hashlib.sha256(np.asarray(full_mapped, dtype=np.uint64).tobytes()).hexdigest()
    holdout_digest = hashlib.sha256(np.asarray(holdout_items, dtype=np.uint64).tobytes()).hexdigest()
    state = await retained.partition.state()
    sorted_holdout_digest = hashlib.sha256(state.sorted_holdout_indices.astype(np.uint64).tobytes()).hexdigest()

    # Finite rows have support and run-order shuffles; full rows have only the
    # run-order shuffle. These exact counts pin both production I/O paths.
    assert len({index // 256 for index in mapped}) == 645
    assert len({index // 256 for index in full_mapped}) == 512
    assert mapped_digest == "52f328ec32d8cebeb9cc76cee84feaa9b0a878d2368a4b946626a4e5b94fe8cf"
    assert full_mapping_digest == "37669a52ac207b31553ed412a2c77efe80b701769869f9b734bfece9aa8b1530"
    assert holdout_digest == "948561662a2cc24721f84db7bfedd59750fd9e9184fcc974fbbf60917b2a2b07"
    assert sorted_holdout_digest == "b53937803471d35e94b19a6333c5091a6f8f4c9c1177b0e6c5632b1d9dac0fab"
    assert state.sorted_holdout_indices[:5].tolist() == [618_976, 1_001_860, 1_778_064, 2_356_120, 2_591_228]
    assert state.sorted_holdout_indices[-5:].tolist() == [
        1_997_169_208,
        1_997_965_163,
        1_998_239_919,
        1_999_495_040,
        1_999_639_701,
    ]
    replacement_blocks = {index // 256 for index in state.retained_replacement_sources}
    assert len(replacement_blocks) == 17
    assert len(replacement_blocks) <= (4_096 + 255) // 256 + 1
    assert set(mapped).isdisjoint(holdout_items)
    assert set(full_mapped).isdisjoint(holdout_items)


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
async def test_nested_block_shuffling_batch_matches_scalar_access_across_windows():
    data = list(range(4_173))
    dataset = ListAsyncDataset(data)
    for seed in range(3):
        dataset = BlockShufflingDataset(
            dataset,
            io_block_size=8,
            window_blocks=16,
            key=jax.random.PRNGKey(seed),
        )

    indices = [0, 1, 127, 128, 129, 2_047, 2_048, 3_999, 4_172, 128]
    batch = await dataset.get_batch(indices)
    scalar_items = [await dataset.getitem_async(index) for index in indices]

    assert batch == scalar_items


@pytest.mark.asyncio
async def test_block_shuffling_large_batch_preserves_frozen_mapping():
    dataset = BlockShufflingDataset(
        _IndexAsyncDataset(2_000_000_123),
        io_block_size=256,
        window_blocks=512,
        key=2_026_081_102,
    )

    mapped = await dataset.get_batch(range(400_000))
    digest = hashlib.sha256(np.asarray(mapped, dtype=np.uint64).tobytes()).hexdigest()

    assert digest == "4c5190c0140340865bee9086a1a0a79363fa46078a9aee895d8a3c275778fbd4"
    assert mapped[:5] == [1_977_381_195, 1_787_786_142, 996_279, 465_086_885, 939_567_600]
    assert mapped[-5:] == [1_509_175_575, 476_033_428, 168_653_708, 881_187_860, 1_969_152_398]


@pytest.mark.asyncio
async def test_block_shuffling_window_caches_are_bounded_per_instance():
    first = BlockShufflingDataset(
        _IndexAsyncDataset(8_192),
        io_block_size=8,
        window_blocks=16,
        key=jax.random.PRNGKey(1),
    )
    second = BlockShufflingDataset(
        _IndexAsyncDataset(8_192),
        io_block_size=8,
        window_blocks=16,
        key=jax.random.PRNGKey(2),
    )
    await first.async_len()
    await second.async_len()

    for window_id in range(6):
        first._window_layout(window_id)
        first._window_full_region_permutation(window_id)
    second._window_layout(0)
    second._window_full_region_permutation(0)

    assert list(first._window_layout_cache) == [2, 3, 4, 5]
    assert list(first._window_full_region_permutation_cache) == [2, 3, 4, 5]
    assert list(second._window_layout_cache) == [0]
    assert list(second._window_full_region_permutation_cache) == [0]


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
