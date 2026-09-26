# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reusable configuration for online draft-model SFT during SkyRL rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import yaml
from rigging.filesystem.storage_path import StoragePath

from marin.rl.skyrl import SkyRLRolePlan


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


@dataclass(frozen=True)
class DraftSftPlan:
    """Concrete inputs consumed by experiment-level SkyRL artifact wiring."""

    role_plan: SkyRLRolePlan
    num_nodes: int
    config_yaml: str


def draft_sft_plan(
    initial_draft: StoragePath,
    initial_draft_identity: str,
    *,
    policy: MegatronDraftPolicy,
    training: OnlineEagleTraining,
    environment: str,
    project_name: str,
    request_window_tokens: int,
    max_new_tokens_per_turn: int,
    max_steps: int,
    checkpoint_interval: int,
) -> DraftSftPlan:
    """Build a Megatron policy plus online-EAGLE draft-training plan.

    The returned value contains no artifact handles. An experiment binds its
    storage paths and recipe into the appropriate artifact graph.
    """
    role_plan = SkyRLRolePlan(
        colocate_all=False,
        policy_num_nodes=policy.policy_num_nodes,
        policy_num_gpus_per_node=policy.gpus_per_node,
        num_inference_engines=1,
        inference_engine_tensor_parallel_size=1,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=policy.gpus_per_node,
        inference_engine_expert_parallel_size=policy.inference_engine_expert_parallel_size,
        train_batch_size=policy.train_batch_size,
        policy_mini_batch_size=policy.policy_mini_batch_size,
        micro_train_batch_size_per_gpu=policy.micro_train_batch_size_per_gpu,
        n_samples_per_prompt=policy.n_samples_per_prompt,
    )
    config = {
        "entrypoint": "standard",
        "context_budget": {
            "request_window_tokens": request_window_tokens,
            "max_new_tokens_per_turn": max_new_tokens_per_turn,
            "max_turns": 1,
        },
        "environment": {"env_class": environment},
        "trainer": {
            "strategy": "megatron",
            "flash_attn": False,
            "use_sample_packing": False,
            "offload_optimizer_during_rollouts": True,
            "gradient_checkpointing": True,
            "algorithm": {"advantage_estimator": "grpo", "use_kl_loss": False},
            "epochs": 1,
            "max_steps": max_steps,
            "update_epochs_per_batch": 1,
            "eval_batch_size": policy.eval_batch_size,
            "eval_before_train": False,
            "eval_interval": -1,
            "ckpt_interval": checkpoint_interval,
            "resume_mode": "latest",
            "logger": "wandb",
            "project_name": project_name,
            "policy": {
                "optimizer_config": {"lr": policy.learning_rate, "max_grad_norm": 1.0},
                "megatron_config": {
                    "tensor_model_parallel_size": policy.tensor_model_parallel_size,
                    "pipeline_model_parallel_size": policy.pipeline_model_parallel_size,
                    "context_parallel_size": policy.context_parallel_size,
                    "expert_model_parallel_size": policy.expert_model_parallel_size,
                    "expert_tensor_parallel_size": policy.expert_tensor_parallel_size,
                },
            },
        },
        "generator": {
            "backend": "vllm",
            "model_dtype": "bfloat16",
            "vllm_attention_backend": "FLASH_ATTN",
            "gpu_memory_utilization": 0.75,
            "enforce_eager": False,
            "run_engines_locally": True,
            "weight_sync_backend": "nccl",
            "async_engine": True,
            "batched": False,
            "engine_init_kwargs": {"async_scheduling": False},
            "speculative_decoding": {
                "method": "eagle3",
                "model": {
                    "source_uri": str(initial_draft),
                    "source_identity": initial_draft_identity,
                },
                "num_speculative_tokens": training.num_speculative_tokens,
                "training": {key: value for key, value in asdict(training).items() if key != "num_speculative_tokens"},
            },
            "sampling_params": {"temperature": 1.0, "top_p": 1.0},
        },
        "data": {"kind": "parquet", "train_data": [], "val_data": []},
        "extra_env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    }
    # One whole-node rollout engine and one dedicated DraftTrainer node sit
    # alongside the policy nodes.
    return DraftSftPlan(
        role_plan=role_plan,
        num_nodes=policy.policy_num_nodes + 2,
        config_yaml=yaml.safe_dump(config, sort_keys=False),
    )
