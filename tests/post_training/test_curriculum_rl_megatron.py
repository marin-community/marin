# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import yaml

from experiments.post_training.curriculum_rl import launch


def test_curriculum_presets_use_the_approved_megatron_geometry() -> None:
    expected_parallelism = {
        "smoke": (1, 1),
        "full": (1, 1),
        "snowball-smoke": (2, 8),
        "snowball-full": (2, 8),
        "snowball-smoke-r4": (2, 8),
        "snowball-full-r4": (2, 8),
        "snowball-smoke-r5": (2, 8),
        "snowball-full-r5": (2, 8),
    }
    assert set(expected_parallelism) == set(launch.SCALES)

    for label, (pipeline_size, expert_size) in expected_parallelism.items():
        policy = launch.SNOWBALL_POLICY if label.startswith("snowball") else launch.QWEN_POLICY
        config = yaml.safe_load(launch.rl_config_yaml(launch.SCALES[label], launch.ARMS["naive"], policy))
        trainer = config["trainer"]
        expected = {
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": pipeline_size,
            "context_parallel_size": 1,
            "expert_model_parallel_size": expert_size,
            "expert_tensor_parallel_size": 1,
        }

        assert trainer["strategy"] == "megatron", label
        assert trainer["policy"]["megatron_config"] == expected, label
        assert trainer["ref"]["megatron_config"] == expected, label
