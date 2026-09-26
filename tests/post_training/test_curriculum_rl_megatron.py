# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml

from experiments.post_training.curriculum_rl import launch


@pytest.mark.parametrize("preset", tuple(launch.SCALES.values()), ids=lambda preset: preset.label)
def test_curriculum_megatron_policy_and_reference_use_same_geometry(preset: launch.ScalePreset) -> None:
    policy = launch.SNOWBALL_POLICY if preset.label.startswith("snowball") else launch.QWEN_POLICY
    config = yaml.safe_load(launch.rl_config_yaml(preset, launch.ARMS["naive"], policy))
    trainer = config["trainer"]

    assert trainer["strategy"] == "megatron"
    assert trainer["policy"]["megatron_config"] == trainer["ref"]["megatron_config"]
    assert trainer["policy"]["megatron_config"] == {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 2 if policy is launch.SNOWBALL_POLICY else 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": launch.GPUS_PER_NODE if policy is launch.SNOWBALL_POLICY else 1,
        "expert_tensor_parallel_size": 1,
    }
