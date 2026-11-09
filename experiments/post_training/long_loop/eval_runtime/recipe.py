# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate released Micro210 rollouts through native Harbor environments without training."""


import click
from marin.rl.skyrl import (
    SkyRLRolePlan,
    _materialize_role_plan_config,
)

from experiments.post_training.long_loop.eval_runtime.profile import (
    AGENT_TIMEOUT,
    ATTEMPT_TIMEOUT,
    CLEANUP_TIMEOUT,
    MAX_OUTPUT_TOKENS,
    SEED,
    SETUP_TIMEOUT,
    TOTAL_CONTEXT_LENGTH,
    VERIFIER_TIMEOUT,
    counterpart_environment,
    pi_agent_config,
)


def recipe(judge_base_url: str, credential_env: str) -> tuple[dict, SkyRLRolePlan]:
    plan = SkyRLRolePlan(
        colocate_all=True,
        policy_num_nodes=1,
        policy_num_gpus_per_node=2,
        num_inference_engines=1,
        inference_engine_tensor_parallel_size=2,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=1,
        inference_engine_expert_parallel_size=1,
        train_batch_size=1,
        policy_mini_batch_size=1,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=1,
    )
    geometry = {
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
    }
    config = {
        "entrypoint": "generate",
        "config_groups": {"terminal_bench_config": "terminal_bench"},
        "context_budget": {
            "request_window_tokens": TOTAL_CONTEXT_LENGTH,
            "max_new_tokens_per_turn": MAX_OUTPUT_TOKENS,
            "max_turns": 1024,
        },
        "environment": {
            "task_sessions": {
                "local": {},
                "remote_judge": {"base_url": judge_base_url, "api_key_env": "JUDGE_API_KEY"},
            }
        },
        "trainer": {
            "strategy": "megatron",
            "evaluation_runner": "task",
            "eval_interval": 1,
            "eval_batch_size": 32,
            "resume_mode": "none",
            "seed": SEED,
            "algorithm": {"use_kl_loss": False},
            "policy": {"megatron_config": geometry},
            "ref": {"megatron_config": geometry},
            "logger": "console",
        },
        "generator": {
            "backend": "vllm",
            "model_dtype": "bfloat16",
            "max_num_seqs": 8,
            "engine_init_kwargs": {"enable_auto_tool_choice": True, "tool_call_parser": "qwen3_coder"},
            "run_engines_locally": True,
            "enable_http_endpoint": False,
            "gpu_memory_utilization": 0.85,
            "enforce_eager": False,
            "chat_template": {"source": "name", "name_or_path": None},
            "chat_template_kwargs": {"enable_thinking": True},
            "sampling_params": {"temperature": 0.6, "top_p": 0.95, "logprobs": 0},
            "eval_sampling_params": {"temperature": 0.6, "top_p": 0.95},
            "eval_n_samples_per_prompt": 1,
        },
        "terminal_bench": {
            "harbor": {
                "environment_type": "daytona",
                "daytona_api_key_env": credential_env,
                "eval_daytona_api_key_env": credential_env,
                "verifier_override_timeout_sec": VERIFIER_TIMEOUT,
                "override_timeout_sec": AGENT_TIMEOUT,
                "eval_timeout_override_sec": AGENT_TIMEOUT,
                "trial_attempt_timeout_sec": ATTEMPT_TIMEOUT,
                "n_concurrent_trials": 8,
                "max_retries": 0,
            },
            "installed_agent": {
                "agent": pi_agent_config(),
                "setup_timeout": SETUP_TIMEOUT,
                "run_timeout": AGENT_TIMEOUT,
                "cleanup_timeout": CLEANUP_TIMEOUT,
                "service_startup_timeout": 60,
                "service_shutdown_timeout": 60,
                "judge_model": "MiniMaxAI/MiniMax-M3-MXFP8",
                "trial_reward_key": "reward",
            },
        },
        "trajectory_runner": {"max_concurrent_tasks": 8},
        "data": {"kind": "parquet", "shuffle": False, "train_data": [], "val_data": []},
        "extra_env": {**counterpart_environment(), "JUDGE_BASE_URL": judge_base_url},
    }
    _materialize_role_plan_config(config, plan)
    return config, plan


def validate_environment_preflight(manifest: dict, data_uri: str) -> None:
    """Reject incomplete or mismatched environment checks before constructing a GPU artifact."""
    if manifest.get("scope") == "diagnostic":
        raise click.ClickException("A diagnostic preflight does not cover an evaluation dataset.")
    required = set(manifest["required_source_keys"])
    missing = required - set(manifest["validated_source_keys"])
    if manifest["failures"] or missing or len(required) != manifest["expected_source_keys"]:
        raise click.ClickException(f"Environment manifest is incomplete ({len(missing)} missing keys).")
    if data_uri != manifest["data_uri"]:
        raise click.ClickException("Environment manifest belongs to a different dataset scope.")
