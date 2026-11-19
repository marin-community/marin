# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import replace

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
from experiments.post_training.long_loop.sft_runtime.teacher_targets import TeacherLmExample


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


@pytest.mark.parametrize("repair_once", [False, True])
def test_teacher_rows_keep_mixture_order_and_mask_general_rows(tmp_path, monkeypatch, repair_once):
    Pos = Axis("position", 16)
    topk = 2
    general = ListAsyncDataset(
        [
            GrugLmExample.causal(tokens=np.full(16, 100 + i, np.int32), loss_weight=np.ones(16, np.float32))
            for i in range(30)
        ]
    )
    monkeypatch.setattr(LmDataConfig, "train_mixture", lambda self, pos, batch_schedule, *, key: general)
    # Distinct repair rows, so a reordering among them would show.
    repair_rows = []
    for j in range(3):
        ids = np.asarray([j + 1, 2, 3, 128001] + [128001] * 12, np.int32)
        weights = np.asarray([0, 1, 1, 0] + [0] * 12, np.float32) * (j + 1)
        repair_rows.append(
            {
                "input_ids": ids,
                "loss_weights": weights,
                "teacher_ids": np.arange(16 * topk, dtype=np.int32) + 1000 * j,
                "teacher_logprobs": -np.arange(16 * topk, dtype=np.float32) / 10 - j,
            }
        )
    with SerialCacheWriter(
        str(tmp_path),
        {name: value[:0] for name, value in repair_rows[0].items()},
        metadata=CacheMetadata({"input_ids_key": "input_ids", "loss_weights_key": "loss_weights"}),
    ) as writer:
        writer.write_batch(repair_rows)
    key = jax.random.PRNGKey(0)
    config = E12DataConfig(
        repair_cache=str(tmp_path),
        repair_row_weight=0.25,
        outer_block_size=8,
        repair_once=repair_once,
        teacher_topk=topk,
    )
    # A hard-only arm trains on the same config with teacher_topk=0.
    plain_config = replace(config, teacher_topk=0)
    plain = asyncio.run(plain_config.train_mixture(Pos, BatchSchedule(4), key=key).get_batch(range(24)))
    rows = asyncio.run(config.train_mixture(Pos, BatchSchedule(4), key=key).get_batch(range(24)))
    repairs = []
    for plain_row, row in zip(plain, rows, strict=True):
        assert not isinstance(plain_row, TeacherLmExample)
        assert isinstance(row, TeacherLmExample)
        np.testing.assert_array_equal(np.asarray(row.tokens), np.asarray(plain_row.tokens))
        np.testing.assert_array_equal(np.asarray(row.loss_weight), np.asarray(plain_row.loss_weight))
        first = int(row.tokens[0])
        if first < 100:
            source = repair_rows[first - 1]
            repairs.append(first)
            np.testing.assert_array_equal(row.teacher.ids, source["teacher_ids"].reshape(16, topk))
            np.testing.assert_array_equal(row.teacher.logprobs, source["teacher_logprobs"].reshape(16, topk))
            np.testing.assert_array_equal(row.teacher.mask, (source["loss_weights"] > 0).astype(np.float32))
        else:
            assert row.teacher.ids.shape == (16, topk)
            assert not np.any(row.teacher.mask)
    assert len(repairs) == (2 if repair_once else 6)
    assert len(set(repairs)) > 1
