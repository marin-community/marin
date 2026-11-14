# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

import jax
import numpy as np
import pytest
from haliax import Axis
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.schedule import BatchSchedule
from levanter.store.cache import CacheMetadata, SerialCacheWriter

from experiments.post_training.long_loop.runs.e12.mixed_data import E12DataConfig


@pytest.mark.parametrize("repair_once,expected_repairs", [(False, 4), (True, 2)])
def test_aligned_replacement_keeps_general_positions_and_shifted_repair_weights(
    tmp_path, monkeypatch, repair_once, expected_repairs
):
    Pos = Axis("position", 16)
    general = ListAsyncDataset(
        [
            GrugLmExample.causal(tokens=np.full(16, 100 + i, np.int32), loss_weight=np.ones(16, np.float32))
            for i in range(30)
        ]
    )
    observed_keys = []

    def original(self, pos, batch_schedule, *, key):
        observed_keys.append(np.asarray(key))
        return general

    monkeypatch.setattr(LmDataConfig, "train_mixture", original)
    ids = np.asarray([1, 2, 3, 128001] + [128001] * 12, np.int32)
    weights = np.asarray([0, 1, 1, 0] + [0] * 12, np.float32)
    with SerialCacheWriter(
        str(tmp_path),
        {"input_ids": ids[:0], "loss_weights": weights[:0]},
        metadata=CacheMetadata({"input_ids_key": "input_ids", "loss_weights_key": "loss_weights"}),
    ) as writer:
        writer.write_batch([{"input_ids": ids, "loss_weights": weights}])
    config = E12DataConfig(
        repair_cache=str(tmp_path), repair_row_weight=0.25, outer_block_size=8, repair_once=repair_once
    )
    key = jax.random.PRNGKey(0)
    mixture = config.train_mixture(Pos, BatchSchedule(4), key=key)
    rows = asyncio.run(mixture.get_batch(list(range(16))))
    expected = 0
    repairs = 0
    for position, row in enumerate(rows):
        if int(row.tokens[0]) == 1:
            np.testing.assert_array_equal(np.asarray(row.loss_weight), weights)
            repairs += 1
        else:
            assert int(row.tokens[0]) == 100 + position
            expected += 1
    assert repairs == expected_repairs and expected == 16 - expected_repairs
    control = E12DataConfig(repair_cache=str(tmp_path), repair_row_weight=0.25, outer_block_size=8, control_only=True)
    assert control.train_mixture(Pos, BatchSchedule(4), key=key) is general
    np.testing.assert_array_equal(observed_keys[0], key)
