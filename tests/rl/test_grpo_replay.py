# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from marin.rl.grpo_artifact import GoldenRollout, write_golden_rollout
from marin.rl.grpo_replay import replay_golden_rollout


def test_replay_preserves_original_partitions_for_unequal_response_lengths(tmp_path):
    # Two responses with scores 0 and 2 have sample std sqrt(2). Each was a
    # separate oracle microbatch: both responses have half the objective weight.
    advantage = 1 / (np.sqrt(2) + 1e-6)
    advantages = np.asarray([[-advantage, -advantage], [advantage, 0]], dtype=np.float32)
    gradients = np.asarray([[advantage / 4, advantage / 4], [-advantage / 2, 0]], dtype=np.float32)
    mask = np.asarray([[1, 1], [1, 0]], dtype=np.float32)
    batch = GoldenRollout(
        sequences=np.asarray([[5, 7, 8], [5, 9, 0]], dtype=np.int32),
        attention_mask=np.asarray([[1, 1, 1], [1, 1, 0]]),
        response_mask=mask,
        loss_mask=mask,
        rewards=np.asarray([[0, 0], [2, 0]], dtype=np.float32),
        advantages=advantages,
        old_logprobs=np.full((2, 2), -2, dtype=np.float32),
        behavior_logprobs=np.full((2, 2), -3, dtype=np.float32),
        group_ids=np.asarray([91, 91]),
        objective_partition_ids=np.asarray([10, 20]),
        logprob_gradients=gradients,
    )
    algorithm = {
        "advantage_estimator": "grpo",
        "policy_loss_type": "regular",
        "loss_reduction": "token_mean",
        "advantage_batch_normalize": False,
        "use_kl_in_reward": False,
        "use_entropy_loss": False,
        "use_tis": False,
        "think_token_weight": 1.0,
        "use_kl_loss": False,
        "eps_clip_low": 0.2,
        "eps_clip_high": 0.2,
        "grpo_norm_by_std": True,
    }
    manifest = {
        "config": {"trainer": {"algorithm": algorithm}},
        "oracle": {
            "loss": 0.0,
            "metrics": {
                "ppo_clip_ratio": 0.0,
                "ppo_clip_ratio_low": 0.0,
                "ppo_clip_ratio_high": 0.0,
                "ppo_clip_pressure_low": 0.0,
                "ppo_clip_pressure_high": 0.0,
                "ppo_ratio_exact_unit_fraction": 1.0,
            },
        },
        "provenance": {"kl_gradient": "detached", "oracle_base_commit": "8e33e01707b7225ecde1d6b8ad172a3dd4dc8661"},
    }
    uri = str(tmp_path / "capture.npz")
    write_golden_rollout(uri, batch, manifest)
    result = replay_golden_rollout(uri, atol=1e-5, rtol=1e-5)
    assert result.logprob_gradients.max_absolute_error < 1e-5
    assert result.advantages.max_absolute_error < 1e-5
    assert result.response_tokens == 3
    assert result.loss == pytest.approx(0, abs=1e-5)

    # A capture whose recorded oracle loss differs must fail replay, even if
    # its arrays and schema remain valid.
    manifest["oracle"]["loss"] = 0.5
    write_golden_rollout(uri, batch, manifest)
    with pytest.raises(AssertionError, match="loss"):
        replay_golden_rollout(uri, atol=1e-5, rtol=1e-5)

    manifest["oracle"]["loss"] = 0.0
    manifest["oracle"]["metrics"]["ppo_ratio_exact_unit_fraction"] = 0.5
    write_golden_rollout(uri, batch, manifest)
    with pytest.raises(AssertionError, match="ppo_ratio_exact_unit_fraction"):
        replay_golden_rollout(uri, atol=1e-5, rtol=1e-5)
