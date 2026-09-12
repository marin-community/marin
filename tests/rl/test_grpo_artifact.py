# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json

import numpy as np
import pytest
from marin.rl.grpo_artifact import GoldenRollout, read_golden_rollout, write_golden_rollout

from scripts.rl.compare_skyrl_golden_updates import compare_updates


@pytest.fixture
def golden_rollout():
    mask = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.int32)
    old = np.array([[-2, -3, 0], [-4, 0, 0]], dtype=np.float32)
    batch = GoldenRollout(
        sequences=np.array([[8, 9, 1, 2, 0], [7, 6, 3, 0, 0]]),
        attention_mask=np.array([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]),
        response_mask=mask,
        loss_mask=mask,
        rewards=np.array([[0, 1, 0], [0, 0, 0]], dtype=np.float32),
        advantages=np.array([[0.7, 0.7, 0], [-0.7, 0, 0]], dtype=np.float32),
        old_logprobs=old,
        behavior_logprobs=old - 0.2 * mask,
        group_ids=np.array([0, 0]),
        objective_partition_ids=np.array([0, 1]),
        reference_logprobs=old - 0.1 * mask,
        current_logprobs=old + 0.01 * mask,
        logprob_gradients=np.array([[-0.175, -0.175, 0], [0.35, 0, 0]], dtype=np.float32),
    )
    return batch


def test_golden_rollout_preserves_replay_inputs_and_optional_reference(tmp_path, golden_rollout):
    batch = golden_rollout
    old = batch.old_logprobs
    mask = batch.loss_mask
    path = str(tmp_path / "batch.npz")
    manifest = {"provenance": {"policy_version": 3}, "oracle": {"loss": 0.25}}
    write_golden_rollout(path, batch, manifest)
    restored, metadata = read_golden_rollout(path)
    for field in dataclasses.fields(batch):
        np.testing.assert_array_equal(getattr(restored, field.name), getattr(batch, field.name))
    assert metadata == {"schema_version": 1, **manifest}
    write_golden_rollout(path, dataclasses.replace(batch, reference_logprobs=None), manifest)
    assert read_golden_rollout(path)[0].reference_logprobs is None

    with pytest.raises(ValueError, match="contained in response_mask"):
        write_golden_rollout(path, dataclasses.replace(batch, loss_mask=np.ones_like(mask)), manifest)
    # A rejected write must leave the previously usable artifact intact.
    np.testing.assert_array_equal(read_golden_rollout(path)[0].old_logprobs, old)


@pytest.mark.parametrize("rows,width", [(0, 3), (2, 0), (2, 4)])
def test_golden_rollout_rejects_non_replayable_dimensions(rows, width):
    tokens = np.zeros((rows, 3), dtype=np.int32)
    values = np.zeros((rows, width), dtype=np.float32)
    batch = GoldenRollout(
        sequences=tokens,
        attention_mask=tokens,
        response_mask=values,
        loss_mask=values,
        rewards=values,
        advantages=values,
        old_logprobs=values,
        behavior_logprobs=values,
        group_ids=np.zeros(rows, dtype=np.int32),
        objective_partition_ids=np.zeros(rows, dtype=np.int32),
    )
    with pytest.raises(ValueError, match=r"nonempty|response width"):
        batch.validate()


def test_golden_update_comparison_rejects_changed_membership_and_checkpoint_overwrite(tmp_path, golden_rollout):
    manifest = {
        "config": {"trainer": {"algorithm": {}, "policy": {}}},
        "oracle": {"loss": 0.25},
        "provenance": {
            "source_sha256": {},
            "initial_policy": {"uri": "s3://model/weights", "identity": "step-0"},
            "initial_optimizer": {"step": 0},
            "rng_seed": 17,
            "torch_version": "2.11.0",
        },
    }
    objects = {
        "source_objects": [
            {
                "uri": "s3://model/weights/shard-0",
                "size": 1024,
                "etag": "original-content",
                "version_id": None,
                "checksum_sha256": None,
            }
        ]
    }
    probes = {"names": ["weight"], "values": [0.25], "optimizer_parameter_values": [0.25001]}
    for name in ("first", "replay"):
        directory = tmp_path / name
        directory.mkdir()
        write_golden_rollout(str(directory / "golden.npz"), golden_rollout, manifest)
        (directory / "source-identity.json").write_text(json.dumps(objects))
        (directory / "golden.npz.rank-0.json").write_text(
            json.dumps({"before": probes, "after": probes, "grad_norm": 0.5})
        )
    first = str(tmp_path / "first/golden.npz")
    replay = str(tmp_path / "replay/golden.npz")
    assert compare_updates(first, replay, 1)["passed"]

    relabeled = dataclasses.replace(golden_rollout, group_ids=np.array([37, 37]))
    write_golden_rollout(replay, relabeled, manifest)
    assert compare_updates(first, replay, 1)["passed"]
    write_golden_rollout(replay, dataclasses.replace(golden_rollout, group_ids=np.array([37, 38])), manifest)
    membership_result = compare_updates(first, replay, 1)
    assert not membership_result["passed"]
    assert not membership_result["checks"]["group_ids"]["passed"]
    write_golden_rollout(replay, relabeled, manifest)

    # The logical model URI and even bounded probes can survive a shard overwrite.
    objects["source_objects"][0]["etag"] = "different-content"
    (tmp_path / "replay/source-identity.json").write_text(json.dumps(objects))
    result = compare_updates(first, replay, 1)
    assert not result["passed"]
    assert not result["checks"]["provenance/source_objects"]["passed"]
    assert result["checks"]["provenance/initial_policy"]["passed"]
