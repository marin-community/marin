# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from marin.rl.opd_losses import HybridRLOOOPDSampledTokenReverseKLLoss, OPDSampledTokenReverseKLLoss
from marin.rl.teacher import INITIAL_POLICY_TEACHER_CHECKPOINT

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


def test_opd_math500_config_defaults_teacher_to_initial_policy_marker():
    config = llama_3_8b_opd_math500.build_experiment_config(_opd_args())

    assert isinstance(config.rl_loss, OPDSampledTokenReverseKLLoss)
    assert config.teacher is not None
    assert config.teacher.checkpoint == INITIAL_POLICY_TEACHER_CHECKPOINT
    assert config.n_generations_per_prompt == 1
    assert config.inference_n == 1
    assert config.train_batch_size == 16


def test_opd_math500_config_allows_explicit_teacher_checkpoint():
    config = llama_3_8b_opd_math500.build_experiment_config(_opd_args(teacher_checkpoint="teacher-checkpoint"))

    assert config.teacher is not None
    assert config.teacher.checkpoint == "teacher-checkpoint"


def test_opd_math500_config_rejects_batch_larger_than_rollout_batch():
    with pytest.raises(ValueError, match="train_batch_size"):
        llama_3_8b_opd_math500.build_experiment_config(_opd_args(train_batch_size=17))


def test_hybrid_opd_math500_config_defaults_teacher_to_initial_policy_marker():
    config = llama_3_8b_hybrid_opd_math500.build_experiment_config(_hybrid_args(teacher_checkpoint=None))

    assert config.teacher is not None
    assert config.teacher.checkpoint == INITIAL_POLICY_TEACHER_CHECKPOINT


def test_hybrid_opd_math500_config_uses_grouped_rollouts_and_matching_inference_n():
    config = llama_3_8b_hybrid_opd_math500.build_experiment_config(_hybrid_args())

    assert isinstance(config.rl_loss, HybridRLOOOPDSampledTokenReverseKLLoss)
    assert config.rl_loss.opd_coef == pytest.approx(0.125)
    assert config.teacher is not None
    assert config.teacher.checkpoint == "teacher-checkpoint"
    assert config.n_generations_per_prompt == 4
    assert config.inference_n == config.n_generations_per_prompt
    assert config.train_batch_size == 64
    assert not config.inflight_weight_updates
    assert config.max_rollout_step_delay == 0
    assert config.replay_buffer_max_samples == 1


def test_hybrid_opd_math500_config_rejects_group_size_one():
    with pytest.raises(ValueError, match="must be > 1"):
        llama_3_8b_hybrid_opd_math500.build_experiment_config(_hybrid_args(n_generations_per_prompt=1))


def test_hybrid_opd_math500_config_rejects_batch_larger_than_rollout_batch():
    with pytest.raises(ValueError, match="train_batch_size"):
        llama_3_8b_hybrid_opd_math500.build_experiment_config(_hybrid_args(train_batch_size=65))
