# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Joint closure-loss recipe for the protocol-FTPO initializer."""

from copy import deepcopy
from enum import StrEnum

from marin.rl.skyrl import SkyRLRolePlan

ROLE_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=4,
    policy_num_gpus_per_node=8,
    num_inference_engines=1,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=8,
    inference_engine_expert_parallel_size=8,
    train_batch_size=64,
    policy_mini_batch_size=64,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=1,
)


class ClosureSupervision(StrEnum):
    CLOSURE = "verified_closure"
    CLOSURE_AND_CONTINUATION = "verified_closure_and_continuation"


def training_config(
    *,
    role_plan: SkyRLRolePlan,
    max_steps: int,
    eval_batch_size: int,
    learning_rate: float,
    supervision: ClosureSupervision,
    candidate_logprobs: int,
) -> dict:
    """Render the full-model smoke recipe; artifact inputs supply model and data paths."""
    plan = role_plan
    geometry: dict[str, object] = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 2,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 16,
        "expert_tensor_parallel_size": 1,
        "logprob_chunk_size": 256,
        "transformer_config_kwargs": {},
    }
    policy_geometry = deepcopy(geometry)
    policy_geometry["ddp_config"] = {
        "overlap_grad_reduce": False,
        "overlap_param_gather": False,
        "grad_reduce_in_fp32": False,
    }
    policy_geometry["optimizer_checkpoint_sharding_type"] = "dp_reshardable"
    return {
        "entrypoint": "taskcompendium",
        "context_budget": {
            "request_window_tokens": 32768,
            "max_new_tokens_per_turn": 1024,
            # Closure sessions stop after one turn; BFCL needs multiple shell calls.
            "max_turns": 64,
        },
        "trainer": {
            "strategy": "megatron",
            "seed": 42,
            "flash_attn": False,
            "gradient_checkpointing": True,
            "gradient_checkpointing_use_reentrant": False,
            "offload_optimizer_during_rollouts": False,
            "use_sample_packing": False,
            "epochs": 2,
            "max_steps": max_steps,
            "update_epochs_per_batch": 1,
            "micro_forward_batch_size_per_gpu": 1,
            "eval_batch_size": eval_batch_size,
            "eval_before_train": True,
            "evaluation_runner": "harbor",
            "eval_interval": 1,
            "ckpt_interval": 1,
            "hf_save_interval": max_steps,
            "max_ckpts_to_keep": 2,
            "resume_mode": "none",
            "logger": "wandb",
            "project_name": "structure-closure",
            "tracker_commit_each_step": True,
            "algorithm": {
                "advantage_estimator": "uniform",
                "advantage_batch_normalize": False,
                "policy_loss_type": "ftpo",
                "loss_reduction": "token_mean",
                "think_token_weight": 1.0,
                "use_entropy_loss": False,
                "use_kl_loss": False,
                "use_kl_in_reward": False,
                "off_policy_correction": "none",
                "dynamic_sampling": {"type": None},
                "ftpo": {
                    "boundary": supervision.value,
                    "margin": 2.0,
                    "lambda_mse": 0.4,
                    "lambda_mse_target": 0.05,
                    "tau_mse_target": 0.5,
                    "min_p": 0.01,
                    "max_chosen_tokens": 20,
                    "rejected_balance_strength": 0.3,
                    "chosen_balance_strength": 0.5,
                    "repair_sft_weight": 1.0,
                    "early_stopping_chosen_win": None,
                },
            },
            "policy": {
                "optimizer_config": {
                    "optimizer": "AdamW",
                    "lr": learning_rate,
                    "weight_decay": 0.0,
                    "max_grad_norm": 0.5,
                },
                "megatron_config": policy_geometry,
            },
            "ref": {"megatron_config": geometry},
            "placement": {
                "policy_strict_spread_pg": True,
            },
            "rollout_buffer": {
                "max_staleness_steps": 0,
                "batch_policy": "rolling",
                "max_in_flight": plan.train_batch_size,
                "object_store_root": None,
            },
        },
        "generator": {
            "backend": "vllm",
            "model_dtype": "bfloat16",
            "vllm_attention_backend": "FLASH_ATTN",
            "run_engines_locally": True,
            "weight_sync_backend": "nccl",
            "weight_sync_transport": "expert_block",
            "weight_sync_pause": {"mode": "keep", "clear_cache": True},
            "expert_block_sync": {"timeout_seconds": 600, "verify": True},
            "eval_n_samples_per_prompt": 1,
            "gpu_memory_utilization": 0.75,
            "max_num_seqs": 64,
            "max_num_batched_tokens": 8192,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "enforce_eager": False,
            "enable_http_endpoint": False,
            "chat_template": None,
            "sampling_params": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "logprobs": candidate_logprobs},
            "engine_init_kwargs": {
                "moe_backend": "triton",
                "logprobs_mode": "raw_logprobs",
                "enable_auto_tools": True,
                "tool_parser": "hermes",
            },
        },
        "terminal_bench": {
            "harbor": {
                "environment_type": "daytona",
                # Only evaluation uses machines; closure training is a pure token task.
                "daytona_api_key_env": "DAYTONA_API_KEY",
                "eval_timeout_override_sec": 1800,
                "n_concurrent_trials": 8,
                "ttl_minutes": 60,
            },
        },
        "data": {"kind": "parquet", "train_data": [], "val_data": [], "shuffle": True},
        "extra_env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "NCCL_DEBUG": "WARN"},
    }
