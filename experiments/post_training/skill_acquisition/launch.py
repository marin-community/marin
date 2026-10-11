# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run localized tool-protocol FTPO with Pi and held-out native validation."""

from dataclasses import replace
from pathlib import Path

import click
import yaml
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint

from experiments.post_training.bfcl_rl.collect import COLLECTION_EXECUTION
from experiments.post_training.bfcl_rl.final_dpo import (
    POLICY_PARALLELISM,
    REFERENCE_PARALLELISM,
    megatron_config,
)

MODEL = "open-athena/Grug-67B-A2B-BFCL-Recovery"
MODEL_REVISION = "850a32627724d274004241dd10f6cb46fd3f6f83"
MODEL_SOURCE = (
    "s3://marin-us-east-02a/marin/users/benfeuer/models/" "bfcl-rl-recovery-dpo-native/2026.10.08.304/hf/step-1"
)
ROLE_PLAN = SkyRLRolePlan(
    colocate_all=False,
    colocate_policy_ref=False,
    policy_num_nodes=15,
    policy_num_gpus_per_node=8,
    reference_num_nodes=8,
    reference_num_gpus_per_node=8,
    num_inference_engines=1,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=8,
    inference_engine_expert_parallel_size=8,
    train_batch_size=16,
    policy_mini_batch_size=16,
    micro_train_batch_size_per_gpu=2,
    n_samples_per_prompt=2,
)


def protocol_recipe(updates: int) -> str:
    """Retain source-v125 optimizer and serving settings under the current FTPO contract."""
    source = Path(__file__).parents[1] / "bfcl_rl/v125_async.yaml"
    baseline = yaml.safe_load(source.read_text())["skyrl"]
    optimizer = baseline["trainer"]["policy"]["optimizer_config"]
    generator = baseline["generator"]
    serving_fields = (
        "backend",
        "model_dtype",
        "vllm_attention_backend",
        "gpu_memory_utilization",
        "max_num_seqs",
        "max_num_batched_tokens",
        "enable_prefix_caching",
        "enable_chunked_prefill",
        "enforce_eager",
        "run_engines_locally",
        "weight_sync_backend",
    )
    recipe = {
        "entrypoint": "terminal_bench",
        "config_groups": {"algorithm_recipe": "ftpo", "terminal_bench_config": "terminal_bench"},
        "context_budget": {
            "request_window_tokens": 40960,
            "serving_window_tokens": 73728,
            "max_new_tokens_per_turn": 8192,
            "max_turns": 128,
        },
        "data": {"kind": "tasks", "train_data": [], "val_data": [], "shuffle": True},
        "trainer": {
            "strategy": "megatron",
            "epochs": 1,
            "max_steps": updates,
            "update_epochs_per_batch": 1,
            "use_sample_packing": False,
            "micro_forward_batch_size_per_gpu": ROLE_PLAN.micro_train_batch_size_per_gpu,
            "gradient_checkpointing": True,
            "flash_attn": True,
            "offload_optimizer_during_rollouts": True,
            "eval_before_train": True,
            "eval_interval": 2,
            "eval_batch_size": 125,
            "resume_mode": "none",
            "ckpt_interval": 2,
            "hf_save_interval": -1,
            "enable_db_registration": False,
            "logger": "wandb",
            "project_name": "online-rl-for-skill-acquisition",
            "policy": {
                "sequence_parallel_size": 1,
                "megatron_config": megatron_config(POLICY_PARALLELISM),
                "optimizer_config": optimizer,
            },
            "ref": {"sequence_parallel_size": 1, "megatron_config": megatron_config(REFERENCE_PARALLELISM)},
            "algorithm": {"use_kl_loss": False, "ftpo": {"boundary": "repeated_tool_opener"}},
            "rollout_buffer": {"max_staleness_steps": 0, "max_in_flight": 32, "batch_policy": "full_batch"},
            "step_phase_budgets": {"evaluation": 21600, "policy_training": 7200, "checkpoint_work": 7200},
            "callbacks": [
                {"type": "logging"},
                {"type": "checkpoint", "save_steps": 2, "save_on_train_end": True},
                {"type": "evaluation", "eval_steps": 2, "eval_before_train": True, "eval_on_train_end": True},
            ],
        },
        "generator": {
            **{key: generator[key] for key in serving_fields},
            "model_loading": "stage_local",
            "eval_n_samples_per_prompt": 1,
            "sampling_params": {"temperature": 1.0, "top_p": 1.0, "logprobs": 20},
            "eval_sampling_params": {"temperature": 1.0, "top_p": 1.0, "logprobs": 0},
            "engine_init_kwargs": {
                "enable_auto_tool_choice": True,
                "tool_call_parser": "hermes",
                "logprobs_mode": "raw_logprobs",
                "model_loader_extra_config": {"distributed": True},
            },
            "trajectory_retention": {
                "enabled": True,
                "required": True,
                "phases": ["train", "eval"],
                "sample_fraction": 1.0,
                "sample_count_per_step": 0,
                "max_bytes_per_step": 16 * 1024**3,
                "max_bytes_per_run": 512 * 1024**3,
            },
        },
        "terminal_bench": {
            "harbor": {
                "environment_type": "daytona",
                "daytona_api_key_env": "DAYTONA_RL_API_KEY",
                "eval_daytona_api_key_env": "DAYTONA_EVAL_API_KEY",
                "override_cpus": 2,
                "override_memory_mb": 4096,
                "override_storage_mb": 10240,
                "override_timeout_sec": 1800,
                "eval_timeout_override_sec": 1800,
                "verifier_override_timeout_sec": 300,
                "n_concurrent_trials": 32,
                "max_retries": 0,
                "enable_reward_shaping": False,
            },
            "installed_agent": {
                "agent": {"name": "pi", "kwargs": {"version": "0.87.0", "collect_rollout_details": True}},
                "setup_timeout": 360,
                "run_timeout": 1800,
                "cleanup_timeout": 60,
                "service_startup_timeout": 60,
                "service_shutdown_timeout": 60,
            },
        },
        "trajectory_runner": {"rollout_workers": {"num_workers": 1, "cpus_per_worker": 4}},
    }
    return yaml.safe_dump(recipe, sort_keys=False)


def protocol_spec(input_uri: str, input_version: str, updates: int) -> SkyRLSpec:
    inputs = ArtifactStep.adopt(
        user_owned_name("data/online-skill-acquisition-inputs"), input_version, input_uri, kind=Artifact
    )
    checkpoint = ArtifactStep.adopt(
        user_owned_name("models/bfcl-recovery-initializer"),
        "2026.10.11",
        MODEL_SOURCE,
        kind=LevanterCheckpoint,
        config={"model": MODEL, "revision": MODEL_REVISION},
    )
    name = user_owned_name("models/online-skill-acquisition-protocol-ftpo")
    return SkyRLSpec(
        name=name,
        version=resolve_version(name, None),
        config_yaml=protocol_recipe(updates),
        runtime=SkyRLRuntime(SkyRLRuntimeProfile.MEGATRON),
        model=ArtifactHfModel(checkpoint, MODEL, MODEL_REVISION, relative_path=""),
        train_data=(ArtifactDataSource(inputs, relative_path="bfcl-training.parquet"),),
        validation_data=(
            ArtifactDataSource(inputs, relative_path="bfcl-parity.parquet"),
            ArtifactDataSource(inputs, relative_path="diagnostic.parquet"),
        ),
        topology=SkyRLTopology(num_nodes=24, gpus_per_node=8, gpu_variant="H100", role_plan=ROLE_PLAN),
        retention=SkyRLRetentionPolicy(resume_checkpoint_count=2, temporary_storage_ttl_days=14),
        seed=42,
    )


@click.command(help=__doc__)
@click.option("--input-uri", required=True)
@click.option("--input-version", required=True)
@click.option("--updates", type=click.IntRange(min=1), required=True)
@rl_build_options
def main(input_uri: str, input_version: str, updates: int) -> ArtifactStep:
    execution = replace(
        COLLECTION_EXECUTION,
        wandb_entity="nyu-dice-lab",
        coordinator_timeout_hours=72,
        job_timeout_seconds=72 * 3600,
    )
    return skyrl_step(protocol_spec(input_uri, input_version, updates), execution, export_hf=True)


if __name__ == "__main__":
    main()
