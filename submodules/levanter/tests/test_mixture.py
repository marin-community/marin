import jax
import numpy as np
import pytest

from levanter.data import ListAsyncDataset, MixtureDataset
from levanter.data.mixture import StopStrategy, rescale_mixture_schedule_for_batch_schedule
from levanter.schedule import BatchSchedule, ScheduleStep


def datasets():
    ds1 = ListAsyncDataset([1, 2, 3, 4, 5])
    ds2 = ListAsyncDataset([10, 20, 30, 40, 50])
    ds3 = ListAsyncDataset([100, 200, 300, 400, 500])
    ds1.finalize()
    ds2.finalize()
    ds3.finalize()
    return {"ds1": ds1, "ds2": ds2, "ds3": ds3}


def weights():
    return {"ds1": 0.5, "ds2": 0.3, "ds3": 0.2}


def block_size():
    return 10


def key():
    return jax.random.PRNGKey(42)


@pytest.mark.asyncio
async def test_mixture_dataset_getitem():
    mixture_ds = MixtureDataset(datasets(), weights(), 10, key=key, randomize_blocks=False)

    item = await mixture_ds.getitem_async(0)
    assert item in [1, 10, 100], f"Unexpected item: {item}"


@pytest.mark.asyncio
async def test_mixture_dataset_get_batch():
    mixture_ds = MixtureDataset(datasets(), weights(), 10, key=key(), randomize_blocks=False)

    batch = await mixture_ds.get_batch([0, 1, 2])
    assert len(batch) == 3
    assert all(item in [1, 2, 3, 10, 20, 30, 100, 200, 300] for item in batch)


@pytest.mark.asyncio
async def test_mixture_dataset_block_assignments():
    mixture_ds = MixtureDataset(datasets(), weights(), 10, key=key())

    block_assignment = await mixture_ds._get_block(0)
    assert block_assignment is not None
    assert len(block_assignment) == 10


@pytest.mark.skip
@pytest.mark.asyncio
async def test_mixture_dataset_stop_strategy_first():
    mixture_ds = MixtureDataset(datasets(), weights(), 10, key=key, stop_strategy=StopStrategy.FIRST_STOP_STRATEGY)

    with pytest.raises(NotImplementedError):
        await mixture_ds.async_len()


@pytest.mark.asyncio
async def test_mixture_dataset_stop_strategy_restart():
    mixture_ds = MixtureDataset(
        datasets(), weights(), block_size=10, key=key(), stop_strategy=StopStrategy.RESTART_STRATEGY
    )

    with pytest.raises(ValueError):
        await mixture_ds.async_len()


@pytest.mark.asyncio
async def test_mixture_dataset_simulated_data_size():
    weights = {"ds1": 1 / 3, "ds2": 1 / 3, "ds3": 1 / 3}
    mixture_ds = MixtureDataset(
        {name: dataset.slice_dataset(end_index=1) for name, dataset in datasets().items()},
        weights,
        block_size=10,
        key=key(),
        randomize_blocks=False,
        stop_strategy=StopStrategy.RESTART_STRATEGY,
    )
    for _ in range(10):
        batch = await mixture_ds.get_batch([0, 1, 2])
        assert len(batch) == 3
        assert all(item in [1, 10, 100] for item in batch)

    mixture_ds = MixtureDataset(
        {name: dataset.slice_dataset(end_index=2) for name, dataset in datasets().items()},
        weights,
        block_size=10,
        key=key(),
        randomize_blocks=False,
        stop_strategy=StopStrategy.RESTART_STRATEGY,
    )
    for _ in range(10):
        batch = await mixture_ds.get_batch([0, 1, 2])
        assert len(batch) == 3
        assert all(item in [1, 2, 10, 20, 100, 200] for item in batch)


@pytest.mark.asyncio
async def test_mixture_dataset_normalized_weights():
    weights = {"ds1": 0, "ds2": 0.5, "ds3": 0.5}
    mixture_ds = MixtureDataset(datasets(), weights, block_size=10, key=key(), randomize_blocks=False)

    batch = await mixture_ds.get_batch([0, 1, 2])
    assert len(batch) == 3
    assert all(item in [10, 20, 30, 100, 200, 300] for item in batch)


@pytest.mark.asyncio
async def test_mixture_dataset_unpermuted_ids():
    mixture_ds = MixtureDataset(datasets(), weights(), block_size=10, key=key())

    unpermuted_ids = mixture_ds._compute_unpermuted_ids(
        mixture_ds._compute_expected_counts_per_block(weights(), block_size())
    )
    assert len(unpermuted_ids) == 10
    assert unpermuted_ids[0] >> 32 in range(3)  # Ensure the dataset ID is valid


@pytest.mark.asyncio
async def test_mixture_dataset_remap_indices():
    dses = datasets()
    mixture_ds = MixtureDataset(dses, weights(), block_size=10, key=key())

    remapped_indices = await mixture_ds._remap_indices(dses["ds1"], [0, 1, 2])
    assert len(remapped_indices) == 3
    assert remapped_indices == [0, 1, 2]

    # check wrap around
    len_ds1 = await dses["ds1"].async_len()
    remapped_indices = await mixture_ds._remap_indices(dses["ds1"], [len_ds1 - 1, len_ds1, len_ds1 + 1])
    assert len(remapped_indices) == 3

    assert remapped_indices == [len_ds1 - 1, 0, 1]


@pytest.mark.asyncio
async def test_mixture_dataset_respects_weights():
    w = weights()
    mixture_ds = MixtureDataset(datasets(), w, block_size(), key=key())

    # Check that the dataset respects the weights
    num_samples = 1000
    samples = await mixture_ds.get_batch(list(range(num_samples)))

    counts = {"ds1": 0, "ds2": 0, "ds3": 0}
    for sample in samples:
        if sample < 10:
            counts["ds1"] += 1
        elif sample < 100:
            counts["ds2"] += 1
        else:
            counts["ds3"] += 1

    for dataset, count in counts.items():
        assert abs(count / num_samples - w[dataset]) < 0.1, f"Dataset {dataset} has unexpected weight"


@pytest.mark.asyncio
async def test_mixture_dataset_randomizes_blocks():
    mixture_ds = MixtureDataset(datasets(), weights(), block_size=10, key=key())

    block_assignment_1 = await mixture_ds._get_block(0)
    block_assignment_2 = await mixture_ds._get_block(0)

    assert np.all(block_assignment_1 == block_assignment_2), "Block assignments should be randomized"

    block_assignment_3 = await mixture_ds._get_block(1)
    assert not np.all(block_assignment_1 == block_assignment_3), "Block assignments should be randomized"


@pytest.mark.asyncio
async def test_mixture_dataset_samples_all_elements():
    mixture_ds = MixtureDataset(datasets(), weights(), block_size=10, key=key())

    num_samples = 1000
    samples = await mixture_ds.get_batch(list(range(num_samples)))

    assert len(samples) == num_samples
    assert set(samples) == {1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500}


def test_rescale_mixture_schedule_for_batch_schedule():
    mixture_schedule = [(0, {"ds1": 0.5, "ds2": 0.5}), (10, {"ds1": 0.2, "ds2": 0.8})]
    batch_schedule = BatchSchedule([ScheduleStep(start=0, value=10), ScheduleStep(start=5, value=20)])

    rescaled_schedule = rescale_mixture_schedule_for_batch_schedule(mixture_schedule, batch_schedule)

    expected_schedule = [(0, {"ds1": 0.5, "ds2": 0.5}), (150, {"ds1": 0.2, "ds2": 0.8})]
    assert rescaled_schedule == expected_schedule

    # double check changing on the cusp
    batch_schedule = BatchSchedule([ScheduleStep(start=0, value=10), ScheduleStep(start=10, value=20)])

    rescaled_schedule = rescale_mixture_schedule_for_batch_schedule(mixture_schedule, batch_schedule)

    expected_schedule = [(0, {"ds1": 0.5, "ds2": 0.5}), (100, {"ds1": 0.2, "ds2": 0.8})]

    assert rescaled_schedule == expected_schedule
