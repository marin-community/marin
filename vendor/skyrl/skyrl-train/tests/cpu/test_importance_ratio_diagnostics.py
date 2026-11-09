"""Policy-age classification from real training-batch construction."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from skyrl_train.config.utils import get_default_config
from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.importance_ratio_diagnostics import mismatch_ratio_metrics


@pytest.mark.parametrize(
    "lease_ages,policy_steps,expected_ages,fresh,unknown",
    [
        (None, None, [-1, -1, -1, -1], 0, 4),
        ([0, 0], None, [-1, -1, -1, -1], 0, 4),
        ([3, 3], [[7, -1, 4], [-1]], [0, -1, 3, -1], 1, 2),
    ],
)
def test_batch_construction_keeps_unstamped_tokens_out_of_fresh_metrics(
    lease_ages, policy_steps, expected_ages, fresh, unknown
):
    config = get_default_config()
    config.trainer.algorithm.off_policy_correction = "tis"
    config.trainer.algorithm.resolved_group_advantage = GroupAdvantageInvariant.exact_physical(
        physical_group_size=2
    ).to_config()
    config.trainer.train_batch_size = 1
    config.generator.trajectory_retention.enabled = False
    trainer = RayPPOTrainer(
        config,
        tracker=None,
        tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=2),
        train_dataset=[None],
        inference_engine_client=None,
        trajectory_runner=SimpleNamespace(set_trajectory_sink=lambda sink: None),
        context=None,
        callbacks=[],
    )
    trainer.policy_model = SimpleNamespace(actor_infos=[SimpleNamespace(rank=SimpleNamespace(dp_size=1))])
    trainer.global_step = 7
    trajectory = {
        "prompt_token_ids": [[1], [1]],
        "response_ids": [[3, 4, 5], [6]],
        "rewards": [[0.0, 0.0, 0.0], [1.0]],
        "loss_masks": [[1, 1, 1], [1]],
        "rollout_logprobs": [np.zeros(3), np.zeros(1)],
    }
    if policy_steps is not None:
        trajectory["rollout_policy_steps"] = [np.asarray(row) for row in policy_steps]
    batch = trainer.convert_to_training_input(trajectory, ["prompt", "prompt"], rollout_staleness=lease_ages)
    torch.testing.assert_close(
        batch["rollout_staleness"][batch["response_mask"].bool()], torch.tensor(expected_ages, dtype=torch.int32)
    )
    metrics = mismatch_ratio_metrics(
        batch["rollout_logprobs"], batch["rollout_logprobs"], batch["loss_mask"], batch["rollout_staleness"]
    )
    assert metrics["policy/mismatch/staleness0/selected_tokens"] == fresh
    assert metrics["policy/mismatch/unknown/selected_tokens"] == unknown
    assert metrics["policy/mismatch/pooled/selected_tokens"] == 4
