# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from levanter.data.dataset import ListAsyncDataset
from levanter.data.mixture import MixtureDataset
from levanter.data.mixture_epoch import epoch_weight_stages, plan_mixture_epoch


@pytest.mark.asyncio
async def test_small_positive_source_is_not_lost_to_block_quantization():
    datasets = {"large": ListAsyncDataset(["large"]), "tiny": ListAsyncDataset(["tiny"])}
    mixture = MixtureDataset(datasets, {"large": 1.0, "tiny": 1e-9}, block_size=64, key=0)
    samples = await mixture.get_batch(list(range(128)))
    assert samples.count("tiny") == 2
    assert len(samples) == 128


@pytest.mark.asyncio
@pytest.mark.parametrize("large_count", [47, 50])
@pytest.mark.parametrize("first_index", [0, 14])
async def test_native_epoch_reads_every_sft_item_once_including_store_smaller_than_block_count(
    large_count, first_index
):
    counts = {"sft/large": large_count, "sft/tiny": 1}
    plan = plan_mixture_epoch(counts, batch_size=2, replay_fraction=0.25, target_block_size=16)
    stages = epoch_weight_stages(counts, {"replay/large": 1.0, "replay/tiny": 1e-10}, plan)
    datasets = {name: ListAsyncDataset([(name, index) for index in range(count)]) for name, count in counts.items()}
    datasets.update({name: ListAsyncDataset([(name, 0)]) for name in ("replay/large", "replay/tiny")})
    mixture = MixtureDataset(
        datasets,
        [(step * 2, weights) for step, weights in stages],
        plan.block_size,
        key=17,
        sequence_count=plan.run_sequences,
        first_index=first_index,
    )
    samples = await mixture.get_batch(list(range(first_index, first_index + plan.run_sequences)))
    assert sorted(sample for sample in samples if sample[0].startswith("sft/")) == sorted(
        (name, index) for name, count in counts.items() for index in range(count)
    )
    assert sum(sample[0] == "replay/tiny" for sample in samples) == 1
    assert await mixture.async_len() == first_index + plan.run_sequences
    with pytest.raises(IndexError):
        await mixture.getitem_async(first_index + plan.run_sequences)


@pytest.mark.asyncio
async def test_partial_final_block_covers_many_single_item_sources_without_exhausting_replay_early():
    counts = {"sft/large": 7, **{f"sft/tiny/{index}": 1 for index in range(9)}}
    plan = plan_mixture_epoch(counts, batch_size=2, replay_fraction=0.2, target_block_size=16)
    stages = epoch_weight_stages(counts, {"replay": 1.0}, plan)
    datasets = {name: ListAsyncDataset([(name, i) for i in range(count)]) for name, count in counts.items()}
    datasets["replay"] = ListAsyncDataset([("replay", 0)])
    mixture = MixtureDataset(
        datasets,
        [(step * 2, weights) for step, weights in stages],
        plan.block_size,
        key=17,
        sequence_count=plan.run_sequences,
    )
    actual = await mixture.get_batch(list(range(plan.run_sequences)))
    assert sorted(row for row in actual if row[0] != "replay") == sorted(
        (name, i) for name, count in counts.items() for i in range(count)
    )
    assert len(actual) == 20


def test_impossible_block_refuses_to_silently_drop_positive_sources():
    with pytest.raises(ValueError, match="cannot cover 3 positive-weight sources"):
        MixtureDataset(
            {name: ListAsyncDataset([name]) for name in ("a", "b", "c")}, {"a": 100, "b": 1, "c": 1}, 2, key=0
        )
