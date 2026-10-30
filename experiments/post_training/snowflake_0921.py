# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the September 21 Snowball RL lineage on the d512 Snowflake SFT."""

from __future__ import annotations

from dataclasses import asdict
from enum import StrEnum

import click
import yaml
from fray.types import ResourceConfig
from iris.rpc import job_pb2
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    MegatronGeometry,
    SkyRLPolicyModel,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
    skyrl_temporary_run_path,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.curriculum_rl.launch import HfSnapshotConfig, mirror_hf_model

MODEL_ID = "open-athena/Snowflake-d512-Datakit-SFT-262K-2026.10.22"
MODEL_REVISION = "cbc90512edcb5e38bad11e82d46bb69c3ef98afa"
MODEL_VERSION = "2026.10.22"
FTPO_DATA = "s3://marin-us-east-02a/marin/users/benfeuer/datasets/antidoom/dapo-65877096-split-v1"
RLVR_DATA = (
    "s3://marin-us-east-02a/marin/users/benfeuer/datasets/snowball-ultra-rlvr/"
    "20260920-9e7a35d6c979/ordinary-only-v2-agentic57t-compatible/rlvr1"
)
CLUSTER = "cw-us-east-02a"
SEED = 42


class Stage(StrEnum):
    FTPO_MATH = "ftpo-math"
    FTPO_FULL_WINDOW = "ftpo-full-window"
    RLVR = "rlvr"
    REFERENCE_KL = "reference-kl"


STEPS = {
    Stage.FTPO_MATH: 35,
    Stage.FTPO_FULL_WINDOW: 45,  # Cumulative: resume the optimizer from step 35.
    Stage.RLVR: 12,
    Stage.REFERENCE_KL: 20,
}

FTPO_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=1,
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
RLVR_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=1,
    policy_num_gpus_per_node=8,
    num_inference_engines=1,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=8,
    inference_engine_expert_parallel_size=8,
    train_batch_size=512,
    policy_mini_batch_size=512,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=16,
)
GEOMETRY = MegatronGeometry(1, 1, 1, 8, 1)


def _model_step() -> ArtifactStep[LevanterCheckpoint]:
    return ArtifactStep(
        name=user_owned_name("models/snowflake-d512-datakit-sft-262k"),
        version=MODEL_VERSION,
        artifact_type=LevanterCheckpoint,
        run=remote(
            mirror_hf_model,
            resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="32g", target_cluster=CLUSTER),
        ),
        build_config=lambda ctx: HfSnapshotConfig(ctx.output_path, MODEL_ID, MODEL_REVISION),
    )


def _data_step(name: str, uri: str) -> ArtifactStep[Artifact]:
    return ArtifactStep.adopt(name, "2026.10.30", uri, kind=Artifact)


def _config(stage: Stage, plan: SkyRLRolePlan, resume_path: str | None) -> dict:
    ftpo = stage in (Stage.FTPO_MATH, Stage.FTPO_FULL_WINDOW)
    algorithm = (
        {
            "policy_loss_type": "ftpo",
            "advantage_estimator": "uniform",
            "advantage_batch_normalize": False,
            "loss_reduction": "token_mean",
            "think_token_weight": 1.0,
            "use_kl_in_reward": False,
            "use_kl_loss": False,
            "use_entropy_loss": False,
            "off_policy_correction": "none",
            "dynamic_sampling": {"type": None},
            "ftpo": {},
        }
        if ftpo
        else {
            "policy_loss_type": "regular",
            "advantage_estimator": "rloo_n",
            "advantage_batch_normalize": False,
            "loss_reduction": "sequence_mean" if stage == Stage.REFERENCE_KL else "token_mean",
            "group_advantage_min_size": 4,
            "use_kl_in_reward": False,
            "use_kl_loss": stage == Stage.REFERENCE_KL,
            "kl_loss_coef": 0.01,
            "use_entropy_loss": stage == Stage.REFERENCE_KL,
            "entropy_loss_coef": 0.001,
            "off_policy_correction": "tis",
            "tito_full": True,
            "dynamic_sampling": {"type": None},
        }
    )
    config = {
        "entrypoint": "standard",
        "context_budget": {
            "request_window_tokens": 18432 if ftpo else 102400,
            "max_new_tokens_per_turn": 16384 if ftpo else 49152,
            "max_turns": 1 if ftpo else 999999,
        },
        "data": {"kind": "parquet", "shuffle": ftpo},
        "trainer": {
            "strategy": "megatron",
            "seed": SEED,
            "flash_attn": not ftpo,
            "gradient_checkpointing": True,
            "offload_optimizer_during_rollouts": True,
            "use_sample_packing": not ftpo,
            "epochs": 10 if ftpo else 2,
            "max_steps": STEPS[stage],
            "update_epochs_per_batch": 1,
            "micro_forward_batch_size_per_gpu": plan.micro_train_batch_size_per_gpu,
            "eval_batch_size": 100,
            "eval_before_train": True,
            "eval_interval": 5 if ftpo else 2,
            "ckpt_interval": 5 if ftpo else 2,
            "resume_mode": "from_path" if resume_path else "none",
            "logger": "wandb",
            "project_name": "snowflake-d512-0921",
            "algorithm": algorithm,
            "policy": {
                "sequence_parallel_size": 1,
                "optimizer_config": {
                    "optimizer": "AdamW",
                    "lr": 1e-6 if ftpo else 4e-6,
                    "weight_decay": 0.0,
                    "max_grad_norm": 0.5,
                },
                "megatron_config": {**asdict(GEOMETRY), "optimizer_checkpoint_sharding_type": "dp_reshardable"},
            },
            "ref": {"sequence_parallel_size": 1, "megatron_config": asdict(GEOMETRY)},
            "rollout_buffer": {
                "max_staleness_steps": 0 if ftpo else 2,
                "batch_policy": "full_batch" if ftpo else "rolling",
                "max_in_flight": plan.train_batch_size if ftpo else 128,
            },
        },
        "generator": {
            "backend": "vllm",
            "model_dtype": "bfloat16",
            "vllm_attention_backend": "FLASH_ATTN",
            "run_engines_locally": True,
            "weight_sync_backend": "nccl",
            "gpu_memory_utilization": 0.8,
            "max_num_seqs": 32 if ftpo else 16,
            "max_num_batched_tokens": 8192 if ftpo else 16384,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "enforce_eager": False,
            "sampling_params": {"temperature": 0.0 if ftpo else 1.0, "top_p": 1.0, "logprobs": 32 if ftpo else 0},
            "engine_init_kwargs": {"logprobs_mode": "raw_logprobs"} if ftpo else {},
            "trajectory_reward_shaping": {"enabled": False},
        },
        "environment": {"skyrl_gym": {"aime": {"length_penalty_weight": 0.0}}},
        "extra_env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    }
    if not ftpo:
        judge = {
            "base_url": "https://api.together.xyz/v1",
            "model": "openai/gpt-oss-120b",
            "api_key_env": "TOGETHER_API_KEY",
            "reasoning_effort": "low",
            "timeout_seconds": 600.0,
        }
        config["environment"]["skyrl_gym"]["nemotron_ultra"] = {
            "sandbox": {"host": "snowball-nemo-skills-sandbox.iris.svc.cluster.local", "port": 6000},
            "judges": {"general": dict(judge), "safety": dict(judge)},
            "genrm": {
                "judge": {**judge, "response_transport": "chat_completions"},
                "num_rollouts_per_prompt": plan.n_samples_per_prompt,
                "group_answer_length_penalty_coeff": 0.0,
                "group_reasoning_length_penalty_coeff": 0.0,
                "max_output_tokens": 512,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_concurrent_comparisons": 2,
            },
        }
        config["environment"]["skyrl_gym"]["max_env_workers"] = 128
        config["generator"]["engine_init_kwargs"] = {
            "enable_auto_tool_choice": True,
            "tool_call_parser": "hermes",
        }
    if resume_path:
        config["trainer"].update(resume_path=resume_path, restore_dataloader_state=True)
    return config


def _run(
    stage: Stage,
    version: str,
    model: ArtifactHfModel | SkyRLPolicyModel,
    data: ArtifactStep[Artifact],
    resume_path: str | None = None,
) -> ArtifactStep[SkyRLRun]:
    plan = FTPO_PLAN if stage in (Stage.FTPO_MATH, Stage.FTPO_FULL_WINDOW) else RLVR_PLAN
    name = user_owned_name(f"checkpoints/snowflake-d512-0921-{stage.value}")
    return skyrl_step(
        SkyRLSpec(
            name=name,
            version=version,
            config_yaml=yaml.safe_dump(_config(stage, plan, resume_path), sort_keys=False),
            model=model,
            train_data=(ArtifactDataSource(data, "train.parquet"),),
            validation_data=(ArtifactDataSource(data, "validation.parquet"),),
            topology=SkyRLTopology(plan.num_nodes, plan.policy_num_gpus_per_node, "H100", plan),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=5),
            seed=SEED,
        ),
        IrisSkyRLExecution(
            cluster=CLUSTER,
            cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
            cpu=32,
            memory="512Gi",
            disk="2TB",
            priority="interactive",
            max_retries=0,
            target_cluster=CLUSTER,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=72,
            job_timeout_seconds=72 * 3600,
            wandb_entity="dogml",
            required_env_vars=("WANDB_API_KEY", "TOGETHER_API_KEY"),
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
        ),
        export_hf=True,
    )


def build_workflow(version: str) -> dict[Stage, ArtifactStep[SkyRLRun]]:
    model = ArtifactHfModel(_model_step(), MODEL_ID, MODEL_REVISION, relative_path="hf")
    ftpo_data = _data_step("documents/antidoom-dapo-65877096", FTPO_DATA)
    rlvr_data = _data_step("documents/snowball-ultra-rlvr1-ordinary", RLVR_DATA)
    math = _run(Stage.FTPO_MATH, version, model, ftpo_data)
    resume_path = prefix_join(skyrl_temporary_run_path(math.path(), ttl_days=14), "checkpoints/global_step_35")
    window = _run(Stage.FTPO_FULL_WINDOW, version, SkyRLPolicyModel(math), ftpo_data, resume_path)
    rlvr = _run(Stage.RLVR, version, SkyRLPolicyModel(window), rlvr_data)
    reference_kl = _run(Stage.REFERENCE_KL, version, SkyRLPolicyModel(rlvr), rlvr_data)
    return {
        Stage.FTPO_MATH: math,
        Stage.FTPO_FULL_WINDOW: window,
        Stage.RLVR: rlvr,
        Stage.REFERENCE_KL: reference_kl,
    }


@click.command()
@click.option(
    "--stage", type=click.Choice([stage.value for stage in Stage]), default=Stage.REFERENCE_KL.value, show_default=True
)
@rl_build_options
def main(stage: str) -> ArtifactStep[SkyRLRun]:
    """Build the selected terminal stage and its upstream dependencies."""
    return build_workflow(resolve_version("snowflake-d512-0921", None))[Stage(stage)]


if __name__ == "__main__":
    main()
