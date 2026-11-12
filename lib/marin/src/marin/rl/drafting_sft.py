# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reusable configuration for online draft-model SFT during SkyRL rollouts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MegatronDraftPolicy:
    """Target-policy training and rollout geometry for online draft SFT."""

    policy_num_nodes: int
    gpus_per_node: int
    inference_engine_expert_parallel_size: int
    train_batch_size: int
    policy_mini_batch_size: int
    micro_train_batch_size_per_gpu: int
    n_samples_per_prompt: int
    eval_batch_size: int
    micro_forward_batch_size_per_gpu: int
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    context_parallel_size: int
    expert_model_parallel_size: int
    expert_tensor_parallel_size: int
    learning_rate: float


@dataclass(frozen=True)
class OnlineEagleTraining:
    """Bounded update and promotion policy for the online EAGLE trainer."""

    num_speculative_tokens: int
    interval_steps: int
    max_tokens_per_update: int
    max_window_tokens: int
    max_tokens_per_micro_batch: int
    max_sequences_per_prompt_group: int
    min_train_sequences: int
    holdout_fraction: float
    min_holdout_sequences: int
    epochs_per_update: int
    learning_rate: float
    max_validation_loss_increase: float
    max_validation_agreement_decrease: float
    reserved_gpu_memory_gib: float
