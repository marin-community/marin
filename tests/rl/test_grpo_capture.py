# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import numpy as np
import pytest
from marin.rl.grpo_capture import CapturedRollout, read_captured_rollout, write_captured_rollout


def _capture() -> CapturedRollout:
    mask = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.int32)
    old_logprobs = np.array([[-2, -3, 0], [-4, 0, 0]], dtype=np.float32)
    return CapturedRollout(
        sequences=np.array([[8, 9, 1, 2, 0], [7, 6, 3, 0, 0]]),
        attention_mask=np.array([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]),
        response_mask=mask,
        loss_mask=mask,
        rewards=np.array([[0, 1, 0], [0, 0, 0]], dtype=np.float32),
        old_logprobs=old_logprobs,
        group_ids=np.array([0, 0]),
        objective_partition_ids=np.array([0, 1]),
        reference_logprobs=old_logprobs - 0.1 * mask,
    )


def test_captured_rollout_round_trips_learner_inputs(tmp_path):
    capture = _capture()
    path = str(tmp_path / "batch.npz")
    manifest = {"tokenizer": "test-tokenizer", "provenance": {"policy_version": 3}}

    write_captured_rollout(path, capture, manifest)
    restored, metadata = read_captured_rollout(path)

    for field in dataclasses.fields(capture):
        np.testing.assert_array_equal(getattr(restored, field.name), getattr(capture, field.name))
    assert metadata == {"schema_version": 1, **manifest}


def test_captured_rollout_rejects_loss_outside_the_response(tmp_path):
    path = str(tmp_path / "batch.npz")
    capture = _capture()

    with pytest.raises(ValueError, match="contained in response_mask"):
        write_captured_rollout(path, dataclasses.replace(capture, loss_mask=np.ones_like(capture.loss_mask)), {})
