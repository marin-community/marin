# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax
import pytest
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import LmDataConfig
from levanter.schedule import BatchSchedule
from marin.execution.lazy import StepContext

from experiments.grug.moe_hero_ep.harrier_mix_2026_08_18 import (
    HARRIER_MIX_2026_08_18_STORE,
)
from experiments.grug.moe_hero_ep.train import build_train_dataset
from experiments.grug.moe_hero_pipeline.data import raw_hero_data_config


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_steps,batch_size,max_seq_len", [(390251, 32, 4096), (1_000_000, 384, 65536)])
async def test_short_hero_trial_keeps_raw_components_for_long_schedules(
    monkeypatch, tmp_path, schedule_steps, batch_size, max_seq_len
):
    ctx = StepContext.for_run(output_path=str(tmp_path), prefix=str(tmp_path), deps=(HARRIER_MIX_2026_08_18_STORE,))
    raw_config = raw_hero_data_config(
        ctx=ctx, schedule_steps=schedule_steps, batch_size=batch_size, max_seq_len=max_seq_len
    )
    # Substitute only cache I/O. Actual budget slicing, shuffle, mixture sampling,
    # and retrieval run on finite in-memory datasets, including a rare component.
    datasets = {name: ListAsyncDataset(list(range(838))) for name in raw_config.components}
    monkeypatch.setattr(LmDataConfig, "build_caches", lambda self, split: {})
    monkeypatch.setattr(LmDataConfig, "build_token_datasets", lambda self, caches, pos, split: datasets.copy())
    short_config = dataclasses.replace(
        raw_config, target_budget=18_750_000_000_000, experiment_budget=20 * batch_size * max_seq_len
    )
    short_dataset = build_train_dataset(
        short_config, max_seq_len=max_seq_len, batch_schedule=BatchSchedule(batch_size), key=jax.random.key(0)
    )
    with pytest.raises(ValueError, match="empty finite dataset"):
        await short_dataset.get_batch(list(range(32)))

    raw_dataset = build_train_dataset(
        raw_config, max_seq_len=max_seq_len, batch_schedule=BatchSchedule(batch_size), key=jax.random.key(0)
    )
    assert {name: await ds.async_len() for name, ds in raw_dataset.datasets.items()} == {name: 838 for name in datasets}
    samples = await raw_dataset.get_batch(list(range(32)))
    assert len(samples) == 32
    assert all(0 <= sample < 838 for sample in samples)
