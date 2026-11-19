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


def test_teacher_rows_keep_mixture_order_and_mask_general_rows(tmp_path, monkeypatch):
    Pos = Axis("position", 16)
    topk = 2
    general = ListAsyncDataset(
        [
            GrugLmExample.causal(tokens=np.full(16, 100 + i, np.int32), loss_weight=np.ones(16, np.float32))
            for i in range(30)
        ]
    )
    monkeypatch.setattr(LmDataConfig, "train_mixture", lambda self, pos, batch_schedule, *, key: general)
    ids = np.asarray([1, 2, 3, 128001] + [128001] * 12, np.int32)
    weights = np.asarray([0, 1, 1, 0] + [0] * 12, np.float32)
    teacher_ids = np.arange(16 * topk, dtype=np.int32)
    teacher_logprobs = -np.arange(16 * topk, dtype=np.float32) / 10
    row = {"input_ids": ids, "loss_weights": weights, "teacher_ids": teacher_ids, "teacher_logprobs": teacher_logprobs}
    with SerialCacheWriter(
        str(tmp_path),
        {name: value[:0] for name, value in row.items()},
        metadata=CacheMetadata({"input_ids_key": "input_ids", "loss_weights_key": "loss_weights"}),
    ) as writer:
        writer.write_batch([row])
    key = jax.random.PRNGKey(0)
    common = dict(repair_cache=str(tmp_path), repair_row_weight=0.25, outer_block_size=8, repair_once=False)
    plain = asyncio.run(E12DataConfig(**common).train_mixture(Pos, BatchSchedule(4), key=key).get_batch(range(16)))
    rows = asyncio.run(
        E12DataConfig(**common, teacher_topk=topk).train_mixture(Pos, BatchSchedule(4), key=key).get_batch(range(16))
    )
    repairs = 0
    for plain_row, row in zip(plain, rows, strict=True):
        assert isinstance(row, TeacherLmExample)
        np.testing.assert_array_equal(np.asarray(row.tokens), np.asarray(plain_row.tokens))
        np.testing.assert_array_equal(np.asarray(row.loss_weight), np.asarray(plain_row.loss_weight))
        if int(row.tokens[0]) == 1:
            repairs += 1
            np.testing.assert_array_equal(row.teacher.ids, teacher_ids.reshape(16, topk))
            np.testing.assert_array_equal(row.teacher.logprobs, teacher_logprobs.reshape(16, topk))
            np.testing.assert_array_equal(row.teacher.mask, (weights > 0).astype(np.float32))
        else:
            assert row.teacher.ids.shape == (16, topk)
            assert not np.any(row.teacher.mask)
    assert repairs == 4
