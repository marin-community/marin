# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

import pytest
from levanter.data.dataset import ListAsyncDataset
from levanter.data.mixture import MixtureDataset

from experiments.grug.snowball_sft.mixture import ReplaySource, SftSource, build_mixture


@pytest.mark.asyncio
async def test_scheduled_epoch_consumes_each_nonheldout_sft_example_once():
    result = build_mixture(
        {name: SftSource(f"/unused/{name}", count) for name, count in {"a": 30, "b": 9, "c": 2, "d": 2}.items()},
        {"cell": ReplaySource("/unused/short", "/unused/long", 1.0)},
        tokenizer="unused",
        cache_catalog=None,
        steps=48,
        batch_size=4,
        block_size=64,
        heldout_sft_sources=("a",),
        holdout_sequences=1,
    )
    assert result.holdout_counts == {"sft/source/a": 1, "sft/pooled": 1}
    assert result.training_counts == {"sft/source/a": 29, "sft/source/b": 9, "sft/pooled": 3}
    sources = {
        name: ListAsyncDataset([(name, index) for index in range(count)])
        for name, count in result.training_counts.items()
    }
    sources["pretrain/cell"] = ListAsyncDataset([("replay", 0)])
    schedule = [(step * 4, weights) for step, weights in result.data.train_weights]
    mixture = MixtureDataset(sources, schedule, 64, key=42, randomize_blocks=False)
    samples = await mixture.get_batch(list(range(48 * 4)))
    expected = Counter({(name, index): 1 for name, count in result.training_counts.items() for index in range(count)})
    expected[("replay", 0)] = 48 * 4 - sum(result.training_counts.values())
    assert Counter(samples) == expected
