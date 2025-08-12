import jax
import pytest

from levanter.data import ListAsyncDataset, MixtureDataset


def create_datasets():
    ds1 = ListAsyncDataset([1, 2, 3, 4, 5])
    ds2 = ListAsyncDataset([10, 20, 30, 40, 50])
    ds3 = ListAsyncDataset([100, 200, 300, 400, 500])
    ds1.finalize()
    ds2.finalize()
    ds3.finalize()
    return {"ds1": ds1, "ds2": ds2, "ds3": ds3}


@pytest.mark.asyncio
async def test_mixture_dataset_stage_transitions():
    datasets = create_datasets()
    # Define three stages with different weights
    stages = [
        (0, {"ds1": 1.0, "ds2": 0.0, "ds3": 0.0}),  # Stage 1: only ds1
        (20, {"ds1": 0.0, "ds2": 1.0, "ds3": 0.0}),  # Stage 2: only ds2
        (40, {"ds1": 0.0, "ds2": 0.0, "ds3": 1.0}),  # Stage 3: only ds3
    ]

    mixture_ds = MixtureDataset(datasets, stages, block_size=10, key=jax.random.PRNGKey(42), randomize_blocks=False)

    # Test first stage (should only get values from ds1)
    batch1 = await mixture_ds.get_batch(list(range(10)))
    assert all(x in [1, 2, 3, 4, 5] for x in batch1), f"Unexpected values in first stage: {batch1}"

    # Test second stage (should only get values from ds2)
    batch2 = await mixture_ds.get_batch(list(range(20, 30)))
    assert all(x in [10, 20, 30, 40, 50] for x in batch2), f"Unexpected values in second stage: {batch2}"

    # Test third stage (should only get values from ds3)
    batch3 = await mixture_ds.get_batch(list(range(40, 50)))
    assert all(x in [100, 200, 300, 400, 500] for x in batch3), f"Unexpected values in third stage: {batch3}"


@pytest.mark.asyncio
async def test_mixture_dataset_gradual_transition():
    datasets = create_datasets()
    # Define stages with gradual transitions
    stages = [
        (0, {"ds1": 0.8, "ds2": 0.2, "ds3": 0.0}),  # Mostly ds1
        (20, {"ds1": 0.2, "ds2": 0.6, "ds3": 0.2}),  # Mostly ds2
        (40, {"ds1": 0.0, "ds2": 0.2, "ds3": 0.8}),  # Mostly ds3
    ]

    mixture_ds = MixtureDataset(datasets, stages, block_size=10, key=jax.random.PRNGKey(42))

    # Sample a large batch from each stage and verify proportions
    def count_sources(batch):
        counts = {"ds1": 0, "ds2": 0, "ds3": 0}
        for x in batch:
            if x < 10:
                counts["ds1"] += 1
            elif x < 100:
                counts["ds2"] += 1
            else:
                counts["ds3"] += 1
        return counts

    # Test first stage
    batch1 = await mixture_ds.get_batch(list(range(20)))
    counts1 = count_sources(batch1)
    assert counts1["ds1"] > counts1["ds2"] > counts1["ds3"], f"Unexpected distribution in first stage: {counts1}"

    # Test second stage
    batch2 = await mixture_ds.get_batch(list(range(20, 40)))
    counts2 = count_sources(batch2)
    assert (
        counts2["ds2"] > counts2["ds1"] and counts2["ds2"] > counts2["ds3"]
    ), f"Unexpected distribution in second stage: {counts2}"

    # Test third stage
    batch3 = await mixture_ds.get_batch(list(range(40, 60)))
    counts3 = count_sources(batch3)
    assert counts3["ds3"] > counts3["ds2"] > counts3["ds1"], f"Unexpected distribution in third stage: {counts3}"


@pytest.mark.asyncio
async def test_mixture_dataset_invalid_stage_configurations():
    datasets = create_datasets()

    # Test stages that don't start at 0
    with pytest.raises(AssertionError):
        MixtureDataset(datasets, [(10, {"ds1": 1.0})], block_size=10, key=jax.random.PRNGKey(42))

    # Test stages with start indices not multiple of block_size
    with pytest.raises(AssertionError):
        MixtureDataset(datasets, [(0, {"ds1": 1.0}), (15, {"ds2": 1.0})], block_size=10, key=jax.random.PRNGKey(42))

    # Test stages in wrong order
    with pytest.raises(AssertionError):
        MixtureDataset(
            datasets,
            [(0, {"ds1": 1.0}), (20, {"ds2": 1.0}), (10, {"ds3": 1.0})],
            block_size=10,
            key=jax.random.PRNGKey(42),
        )


@pytest.mark.asyncio
async def test_mixture_dataset_zero_weight_handling():
    datasets = create_datasets()
    # Define stages where some datasets have zero weight
    stages = [
        (0, {"ds1": 1.0, "ds2": 0.0, "ds3": 0.0}),
        (20, {"ds1": 0.0, "ds2": 1.0, "ds3": 0.0}),
    ]

    mixture_ds = MixtureDataset(datasets, stages, block_size=10, key=jax.random.PRNGKey(42), randomize_blocks=False)

    # Verify that zero-weight datasets are not sampled
    batch1 = await mixture_ds.get_batch(list(range(10)))
    assert all(x < 10 for x in batch1), f"Found samples from zero-weight datasets in first stage: {batch1}"

    batch2 = await mixture_ds.get_batch(list(range(20, 30)))
    assert all(10 <= x < 100 for x in batch2), f"Found samples from zero-weight datasets in second stage: {batch2}"


@pytest.mark.asyncio
async def test_mixture_dataset_block_boundaries():
    datasets = create_datasets()
    # Define stages with transition at block boundary
    stages = [
        (0, {"ds1": 1.0, "ds2": 0.0, "ds3": 0.0}),
        (10, {"ds1": 0.0, "ds2": 1.0, "ds3": 0.0}),
    ]

    mixture_ds = MixtureDataset(datasets, stages, block_size=10, key=jax.random.PRNGKey(42), randomize_blocks=False)

    # Test the boundary between stages
    batch = await mixture_ds.get_batch(list(range(5, 15)))  # Should span both stages
    first_half = batch[:5]
    second_half = batch[5:]

    assert all(x < 10 for x in first_half), f"Unexpected values at end of first stage: {first_half}"
    assert all(10 <= x < 100 for x in second_half), f"Unexpected values at start of second stage: {second_half}"
