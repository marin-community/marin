# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml
from marin.execution.lazy import StepContext

from experiments.post_training.curriculum_rl import launch


@pytest.mark.parametrize("label", sorted(launch.SCALES))
def test_curriculum_presets_build_valid_megatron_launches(label: str) -> None:
    policy = launch.SNOWBALL_POLICY if label.startswith("snowball") else launch.QWEN_POLICY
    arm = launch.build_arms(specs=(launch.ARMS["naive"],), scale=label, policy=policy, version="2026.09.26")[
        "naive"
    ]
    run_config = arm.rl.build_config(StepContext.for_fingerprint(arm.rl.runtime_args, arm.rl.deps))
    trainer = yaml.safe_load(run_config.launch_config_yaml)["skyrl"]["trainer"]
    assert trainer["strategy"] == "megatron", label
    geometry = trainer["policy"]["megatron_config"]
    assert trainer["ref"]["megatron_config"] == {
        key: value for key, value in geometry.items() if key != "optimizer_checkpoint_sharding_type"
    }, label
    if policy is launch.SNOWBALL_POLICY:
        assert trainer["flash_attn"] is False
        assert trainer["gradient_checkpointing"] is True
        assert trainer["offload_optimizer_during_rollouts"] is True
        assert geometry["optimizer_checkpoint_sharding_type"] == "dp_reshardable"
    plan = launch.SCALES[label].role_plan
    policy_world_size = plan.policy_num_nodes * plan.policy_num_gpus_per_node
    model_parallel_size = (
        geometry["tensor_model_parallel_size"]
        * geometry["pipeline_model_parallel_size"]
        * geometry["context_parallel_size"]
    )
    expert_size = geometry["expert_model_parallel_size"] * geometry["expert_tensor_parallel_size"]
    assert model_parallel_size > 0 and expert_size > 0, label
    assert policy_world_size % model_parallel_size == 0, label
    assert (policy_world_size // model_parallel_size) % expert_size == 0, label


@pytest.mark.parametrize(
    ("scale", "policy"),
    [
        ("smoke", launch.SNOWBALL_POLICY),
        ("full", launch.SNOWBALL_POLICY),
        ("snowball-smoke", launch.QWEN_POLICY),
    ],
)
def test_curriculum_rejects_policy_scale_mismatch(scale, policy) -> None:
    with pytest.raises(ValueError, match="incompatible"):
        launch.build_arms(specs=(launch.ARMS["naive"],), scale=scale, policy=policy, version="test")
