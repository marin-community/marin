# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from experiments import llama_3_8b_hybrid_opd_math500, llama_3_8b_opd_math500


def _opd_args(**overrides):
    defaults = {
        "teacher_checkpoint": None,
        "experiment_name_suffix": "opd-test",
        "project_name": "test-project",
        "num_train_steps": 7,
        "checkpointer_save_interval": 600,
        "keep_last_temporary_checkpoints": 5,
        "debug_checkpointer": False,
        "debug_checkpointer_log_interval": 60.0,
        "debug_checkpointer_dump_stacks_after": 60.0,
        "train_batch_size": 16,
        "n_prompts": 16,
        "train_tpu_type": "v5p-8",
        "inference_tpu_type": "v5p-8",
        "num_train_slices": 1,
        "train_ram": "400g",
        "inference_ram": "400g",
        "zone": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _hybrid_args(**overrides):
    defaults = {
        "teacher_checkpoint": "teacher-checkpoint",
        "opd_coef": 0.125,
        "experiment_name_suffix": "hybrid-test",
        "project_name": "test-project",
        "num_train_steps": 7,
        "checkpointer_save_interval": 600,
        "keep_last_temporary_checkpoints": 5,
        "debug_checkpointer": False,
        "debug_checkpointer_log_interval": 60.0,
        "debug_checkpointer_dump_stacks_after": 60.0,
        "train_batch_size": 64,
        "n_prompts": 16,
        "n_generations_per_prompt": 4,
        "train_tpu_type": "v5p-8",
        "inference_tpu_type": "v5p-8",
        "num_train_slices": 1,
        "train_ram": "400g",
        "inference_ram": "400g",
        "zone": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_opd_math500_config_rejects_batch_larger_than_rollout_batch():
    with pytest.raises(ValueError, match="train_batch_size"):
        llama_3_8b_opd_math500.build_experiment_config(_opd_args(train_batch_size=17))


def test_hybrid_opd_math500_config_rejects_group_size_one():
    with pytest.raises(ValueError, match="must be > 1"):
        llama_3_8b_hybrid_opd_math500.build_experiment_config(_hybrid_args(n_generations_per_prompt=1))


def test_hybrid_opd_math500_config_rejects_batch_larger_than_rollout_batch():
    with pytest.raises(ValueError, match="train_batch_size"):
        llama_3_8b_hybrid_opd_math500.build_experiment_config(_hybrid_args(train_batch_size=65))
