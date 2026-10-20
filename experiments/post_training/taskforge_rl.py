# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate and train Qwen3-30B-A3B-Thinking-2507 on the gen1000 Taskforge environments.

``--stage baseline`` evaluates the base policy over the whole task pool, n samples per task, and
writes per-rollout rewards under the run's ``exports/`` directory. ``--stage rl`` trains on a
train parquet and evaluates a held-out parquet every few steps. Both read ``task_rollout_spec``
parquets through MarinSkyRL's ``taskcompendium`` entrypoints, and both run on cw-rno2a, where
rollout workers run bubblewrap task machines in-pod under the privileged container profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from pathlib import PurePosixPath

import click
import yaml
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.fingerprint import canonical_json, fingerprint_hash
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    _materialize_role_plan_config,
    _role_plan_config_values,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint

from experiments.models import qwen3_30b_a3b_thinking_2507

EXPERIMENT_NAME = "taskforge-gen1000-rl"
CLUSTER = "cw-rno2a"
GPU_VARIANT = "H100"
GPUS_PER_NODE = 8
# An rno2a H100 node allocates 127.96 CPUs and 2,014 GiB. Rollout workers, bubblewrap task and
# verifier machines, and Ray all share the driver node, so the gang takes nearly the whole host.
CPUS_PER_NODE = 120
MEMORY_PER_NODE = "1800GB"
DISK_PER_NODE = "2TB"
CONTAINER_PROFILE = "CONTAINER_PROFILE_PRIVILEGED"
WANDB_ENTITY = "marin-community"
WANDB_PROJECT = "marin-taskforge-rl"
SEED = 17

# The CoreWeave mirror staged on 2026-10-18 from the pinned Hugging Face revision.
MODEL_SOURCE = (
    "s3://marin-us-east-02a/marin/models/" "Qwen--Qwen3-30B-A3B-Thinking-2507--144afc2f379b542fdd4e85a1fcd5e1f79112d95d"
)
MODEL_MIRROR_VERSION = "2026.10.18"
DATA_ROOT = "s3://marin-us-east-02a/marin/taskforge/gen1000/rl"
DATA_VERSION = "2026.10.20"
DEFAULT_POOL_URI = f"{DATA_ROOT}/pool-v1/pool.parquet"
# The 2026-10-20 split of the pool: 493 training tasks, and 417 held-out tasks whose source label is
# heldout_incap or heldout_outcap, so evaluation reports the two held-out kinds separately.
DEFAULT_TRAIN_URI = f"{DATA_ROOT}/split-v1/train.parquet"
DEFAULT_HELDOUT_URI = f"{DATA_ROOT}/split-v1/heldout.parquet"
# TODO(G5): the MarinSkyRL template that opens <think> after the assistant header, like the official
# Thinking-2507 template, while keeping past reasoning so multi-turn token prefixes stay exact.
DEFAULT_CHAT_TEMPLATE = "chat_templates/qwen3_thinking_2507_acc.jinja2"

# Policy: Megatron TP4/PP2/EP4 (DP2 over 16 GPUs) with the reference colocated, as in MarinSkyRL's
# 32GPU_qwen3_coder_30b_a3b_ep4 recipe for the same architecture.
POLICY_GEOMETRY = {
    "tensor_model_parallel_size": 4,
    "pipeline_model_parallel_size": 2,
    "context_parallel_size": 1,
    "expert_model_parallel_size": 4,
    "expert_tensor_parallel_size": 1,
}
POLICY_NODES = 2
# Rollout: four TP2 engines per node, each with two of the model's four KV heads per rank. The
# 2026-10-20 baseline measured about 90 rollouts/min per node at this geometry with no queueing.
ENGINE_TENSOR_PARALLEL = 2
ENGINES_PER_NODE = GPUS_PER_NODE // ENGINE_TENSOR_PARALLEL
RL_INFERENCE_NODES = 4

# 32 prompts x 8 samples per step: about 15 steps per epoch over the 493 training tasks.
PROMPTS_PER_STEP = 32
SAMPLES_PER_PROMPT = 8
BASELINE_EVAL_SAMPLES = 8
RL_EVAL_SAMPLES = 4
LEARNING_RATE = 1.0e-6
RL_MAX_STEPS = 40
EVAL_INTERVAL = 5
CHECKPOINT_INTERVAL = 10
BASELINE_NODES = 1
BASELINE_EVAL_INTERVAL = 1000
# A thinking turn on these tasks runs 5-15k tokens (baseline median 7.6k, p95 17k per rollout) and
# can reach the 32k cap; the window holds a few such turns plus shell output.
CONTEXT_BUDGET = {"request_window_tokens": 65536, "max_new_tokens_per_turn": 32768, "max_turns": 16}
# Qwen's recommended Thinking-2507 decoding. Training samples at temperature one to stay on-policy.
EVAL_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
TRAIN_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1}
# Every rollout of a step or an evaluation starts at once: 256 per training step, and up to 1,668 for
# a 417-task held-out evaluation at n=4 (or 592 for a baseline shard at n=8). vLLM queues what its KV
# cannot hold; a TP2 engine keeps about 780k-900k tokens of KV.
MAX_CONCURRENT_TASKS = 2048
ROLLOUTS_PER_ENGINE = 256
# Task and verifier sandboxes are processes on the driver node. Twelve workers carried 592 concurrent
# baseline rollouts with idle CPU; the reservation leaves room for the policy actors on that node.
ROLLOUT_WORKERS = 12
CPUS_PER_ROLLOUT_WORKER = 8
VERIFIER_THREADS_PER_WORKER = 64


class Stage(StrEnum):
    BASELINE = "baseline"
    RL = "rl"


class BaselineEntrypoint(StrEnum):
    """``taskcompendium_generate`` evaluates without a policy update; ``taskcompendium`` is the
    fallback that trains one step after evaluating the base policy."""

    GENERATE = "taskcompendium_generate"
    TRAIN_ONE_STEP = "taskcompendium"


@dataclass(frozen=True)
class StagePlan:
    entrypoint: str
    num_nodes: int
    role_plan: SkyRLRolePlan
    max_steps: int


def disaggregated_role_plan(inference_nodes: int) -> SkyRLRolePlan:
    """Policy nodes with the reference colocated, plus ``inference_nodes`` nodes of TP2 engines."""
    return SkyRLRolePlan(
        colocate_all=False,
        colocate_policy_ref=True,
        policy_num_nodes=POLICY_NODES,
        policy_num_gpus_per_node=GPUS_PER_NODE,
        reference_num_nodes=POLICY_NODES,
        num_inference_engines=inference_nodes * ENGINES_PER_NODE,
        inference_engine_tensor_parallel_size=ENGINE_TENSOR_PARALLEL,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=1,
        inference_engine_expert_parallel_size=ENGINE_TENSOR_PARALLEL,
        train_batch_size=PROMPTS_PER_STEP,
        policy_mini_batch_size=PROMPTS_PER_STEP,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=SAMPLES_PER_PROMPT,
    )


def evaluation_role_plan(num_nodes: int) -> SkyRLRolePlan:
    """Engines on every GPU; the policy footprint is declared but an eval-only run never loads it."""
    return SkyRLRolePlan(
        colocate_all=True,
        colocate_policy_ref=True,
        policy_num_nodes=num_nodes,
        policy_num_gpus_per_node=GPUS_PER_NODE,
        reference_num_nodes=num_nodes,
        num_inference_engines=num_nodes * ENGINES_PER_NODE,
        inference_engine_tensor_parallel_size=ENGINE_TENSOR_PARALLEL,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=1,
        inference_engine_expert_parallel_size=ENGINE_TENSOR_PARALLEL,
        train_batch_size=PROMPTS_PER_STEP,
        policy_mini_batch_size=PROMPTS_PER_STEP,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=SAMPLES_PER_PROMPT,
    )


def stage_plan(
    stage: Stage,
    *,
    baseline_entrypoint: BaselineEntrypoint,
    baseline_nodes: int,
    inference_nodes: int,
    max_steps: int,
) -> StagePlan:
    if stage is Stage.RL:
        plan = disaggregated_role_plan(inference_nodes)
        return StagePlan("taskcompendium", POLICY_NODES + inference_nodes, plan, max_steps)
    if baseline_entrypoint is BaselineEntrypoint.GENERATE:
        return StagePlan(baseline_entrypoint.value, baseline_nodes, evaluation_role_plan(baseline_nodes), 1)
    return StagePlan(baseline_entrypoint.value, POLICY_NODES + 1, disaggregated_role_plan(1), 1)


ROLE_SETTINGS = frozenset(_role_plan_config_values(disaggregated_role_plan(RL_INFERENCE_NODES)))
LAUNCHER_SETTINGS = frozenset(
    {
        "entrypoint",
        "data",
        "trainer.strategy",
        "trainer.policy.megatron_config",
        "trainer.ref.megatron_config",
        "trainer.seed",
        "trainer.max_steps",
        "trainer.eval_interval",
        "trainer.eval_before_train",
        "trainer.ckpt_interval",
        "trainer.resume_mode",
        "trainer.callbacks",
        "generator.run_engines_locally",
        "generator.backend",
        "generator.chat_template",
    }
)


def apply_setting(config: dict, text: str) -> None:
    """Apply one ``dotted.key=value`` override to an existing recipe field."""
    key, separator, raw = text.partition("=")
    if not separator or not key:
        raise click.BadParameter(f"expected dotted.key=value, got {text!r}")
    owned = ROLE_SETTINGS | LAUNCHER_SETTINGS
    if any(key == name or key.startswith(f"{name}.") or name.startswith(f"{key}.") for name in owned):
        raise click.BadParameter(f"{key!r} is owned by the launcher or the role plan; use a typed option")
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        node = node.get(part)
        if not isinstance(node, dict):
            raise click.BadParameter(f"unknown setting {key!r}")
    if parts[-1] not in node:
        raise click.BadParameter(f"unknown setting {key!r}")
    node[parts[-1]] = yaml.safe_load(raw)


def recipe(
    stage: Stage,
    plan: StagePlan,
    *,
    chat_template: str,
    eval_samples: int,
    eval_interval: int,
    checkpoint_interval: int,
    export: bool,
    settings: tuple[str, ...],
) -> dict:
    """Render the MarinSkyRL source recipe for one stage."""
    training = stage is Stage.RL
    router_replay = not plan.role_plan.colocate_all
    megatron = {**POLICY_GEOMETRY, "moe_router_replay": router_replay}
    engine_kwargs = {
        "enable_auto_tool_choice": True,
        "tool_call_parser": "hermes",
        "disable_custom_all_reduce": True,
        "async_scheduling": False,
    }
    if router_replay:
        engine_kwargs["enable_return_routed_experts"] = True
    config: dict = {
        "entrypoint": plan.entrypoint,
        "context_budget": dict(CONTEXT_BUDGET),
        "trainer": {
            "strategy": "megatron",
            "attn_backend": "sdpa",
            "flash_attn": False,
            "gradient_checkpointing": True,
            "gradient_checkpointing_use_reentrant": True,
            "use_sample_packing": True,
            "epochs": 1000,
            "max_steps": plan.max_steps,
            "update_epochs_per_batch": 1,
            "micro_forward_batch_size_per_gpu": 1,
            "eval_batch_size": 64,
            # The baseline evaluates only before its single step; a positive interval loads the eval set.
            "eval_interval": eval_interval if training else BASELINE_EVAL_INTERVAL,
            "eval_before_train": True,
            "dump_eval_results": True,
            "resume_mode": "latest" if training else "none",
            "seed": SEED,
            "logger": "wandb",
            "project_name": WANDB_PROJECT,
            "tracker_commit_each_step": True,
            "log_level": "INFO",
            "algorithm": {
                "advantage_estimator": "rloo_n",
                "group_advantage_min_size": 4,
                "policy_loss_type": "behavior_clip",
                "loss_reduction": "sequence_mean",
                "eps_clip_low": 0.2,
                "eps_clip_high": 0.28,
                "use_kl_loss": False,
                "kl_loss_coef": 0.0,
                "use_kl_in_reward": False,
                "dynamic_sampling": {"type": None},
            },
            "policy": {
                "optimizer_config": {
                    "optimizer": "AdamW",
                    "lr": LEARNING_RATE,
                    "weight_decay": 0.0,
                    "adam_betas": [0.9, 0.999],
                    "max_grad_norm": 1.0,
                    "optimizer_kwargs": {"foreach": False},
                },
                "megatron_config": megatron,
            },
            "ref": {"megatron_config": dict(megatron)},
            "placement": {
                "policy_strict_spread_pg": True,
                "policy_per_gpu_bundles": True,
                "policy_force_cvd_mask": True,
            },
            "rollout_buffer": {
                # One step of staleness overlaps generation with the update when roles are disaggregated.
                "max_staleness_steps": 0 if plan.role_plan.colocate_all else 1,
                "batch_policy": "full_batch",
                "max_in_flight": 2 * PROMPTS_PER_STEP,
            },
        },
        "generator": {
            "backend": "vllm",
            "model_dtype": "bfloat16",
            "vllm_attention_backend": "FLASH_ATTN",
            "run_engines_locally": True,
            "weight_sync_backend": "nccl",
            "enforce_eager": False,
            "gpu_memory_utilization": 0.85 if training else 0.92,
            "max_num_seqs": ROLLOUTS_PER_ENGINE,
            "max_num_batched_tokens": 16384,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "inference_engine_mp_backend": False,
            "chat_template": {"source": "file", "name_or_path": chat_template},
            "engine_init_kwargs": engine_kwargs,
            "sampling_params": {**TRAIN_SAMPLING, "logprobs": 0},
            "eval_sampling_params": dict(EVAL_SAMPLING),
            "eval_n_samples_per_prompt": eval_samples,
            "inference_stats_interval": 1,
        },
        "environment": {"task_sessions": {"local": {}, "max_verifier_workers": VERIFIER_THREADS_PER_WORKER}},
        "trajectory_runner": {
            "max_concurrent_tasks": MAX_CONCURRENT_TASKS,
            "rollout_workers": {
                "num_workers": ROLLOUT_WORKERS,
                "cpus_per_worker": CPUS_PER_ROLLOUT_WORKER,
            },
        },
        "data": {"kind": "parquet", "shuffle": training, "train_data": [], "val_data": []},
        "extra_env": {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "NCCL_DEBUG": "WARN",
            "VLLM_MQ_MAX_CHUNK_BYTES_MB": "160",
            "VLLM_ROUTED_EXPERTS_SIDE_TIMEOUT_SECONDS": "1200",
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "900",
            "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "1800",
        },
    }
    _materialize_role_plan_config(config, plan.role_plan)
    for setting in settings:
        apply_setting(config, setting)
    trainer = config["trainer"]
    callbacks: list[dict] = [
        {
            "type": "evaluation",
            "eval_steps": trainer["eval_interval"],
            "eval_before_train": True,
            "eval_on_train_end": training,
        },
        {"type": "inference_stats", "log_every_steps": 1, "log_to_console": True, "log_to_tracker": True},
    ]
    if training:
        trainer["ckpt_interval"] = checkpoint_interval
        trainer["hf_save_interval"] = plan.max_steps
        callbacks.append({"type": "checkpoint", "save_steps": checkpoint_interval})
        if export:
            callbacks.append({"type": "hf_model_save", "save_steps": plan.max_steps})
    else:
        trainer["ckpt_interval"] = -1
    trainer["callbacks"] = callbacks
    return config


@cache
def adopted_directory(directory: str) -> ArtifactStep[Artifact]:
    """One handle per directory, so train and held-out files that share a directory share a step."""
    name = PurePosixPath(directory.removeprefix("s3://")).name
    return ArtifactStep.adopt(f"documents/{EXPERIMENT_NAME}/{name}", DATA_VERSION, directory, kind=Artifact)


def adopted_parquet(uri: str) -> ArtifactDataSource:
    """Adopt the directory holding ``uri`` and select the parquet inside it."""
    directory, _, filename = uri.rpartition("/")
    if not uri.startswith("s3://") or not filename.endswith(".parquet"):
        raise click.BadParameter(f"expected an s3:// .parquet file URI, got {uri!r}")
    return ArtifactDataSource(adopted_directory(directory), relative_path=filename)


def model_source() -> ArtifactHfModel:
    download = qwen3_30b_a3b_thinking_2507.build_config(
        StepContext.for_fingerprint(qwen3_30b_a3b_thinking_2507.runtime_args, qwen3_30b_a3b_thinking_2507.deps)
    )
    # A separate name keeps the provenance record out of the mirror directory the step resolves to.
    mirror = ArtifactStep.adopt(
        f"mirrors/{qwen3_30b_a3b_thinking_2507.name}", MODEL_MIRROR_VERSION, MODEL_SOURCE, kind=LevanterCheckpoint
    )
    return ArtifactHfModel(
        step=mirror,
        relative_path="",
        tokenizer_uri=download.hf_dataset_id,
        tokenizer_revision=download.revision,
    )


def build_run(
    stage: Stage,
    *,
    pool_uri: str = DEFAULT_POOL_URI,
    train_uri: str = DEFAULT_TRAIN_URI,
    heldout_uri: str = DEFAULT_HELDOUT_URI,
    baseline_entrypoint: BaselineEntrypoint = BaselineEntrypoint.GENERATE,
    baseline_nodes: int = BASELINE_NODES,
    inference_nodes: int = RL_INFERENCE_NODES,
    max_steps: int = RL_MAX_STEPS,
    chat_template: str = DEFAULT_CHAT_TEMPLATE,
    eval_samples: int | None = None,
    eval_interval: int = EVAL_INTERVAL,
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
    export: bool = False,
    job_timeout_seconds: int = 0,
    settings: tuple[str, ...] = (),
) -> ArtifactStep[SkyRLRun]:
    plan = stage_plan(
        stage,
        baseline_entrypoint=baseline_entrypoint,
        baseline_nodes=baseline_nodes,
        inference_nodes=inference_nodes,
        max_steps=max_steps,
    )
    if eval_samples is None:
        eval_samples = RL_EVAL_SAMPLES if stage is Stage.RL else BASELINE_EVAL_SAMPLES
    if stage is Stage.RL:
        train_data = (adopted_parquet(train_uri),)
        validation_data = (adopted_parquet(heldout_uri),)
    else:
        pool = adopted_parquet(pool_uri)
        train_data = () if baseline_entrypoint is BaselineEntrypoint.GENERATE else (pool,)
        validation_data = (pool,)
    config = recipe(
        stage,
        plan,
        chat_template=chat_template,
        eval_samples=eval_samples,
        eval_interval=eval_interval,
        checkpoint_interval=checkpoint_interval,
        export=export,
        settings=settings,
    )
    identity = fingerprint_hash(
        canonical_json(
            {
                "config": config,
                "train": [f"{source.step.adopt_source}/{source.relative_path}" for source in train_data],
                "validation": [f"{source.step.adopt_source}/{source.relative_path}" for source in validation_data],
            }
        )
    )
    label = PurePosixPath(validation_data[0].relative_path).stem
    name = user_owned_name(f"checkpoints/{EXPERIMENT_NAME}/{stage.value}-{label}-{identity}")
    return skyrl_step(
        SkyRLSpec(
            name=name,
            version=resolve_version(name, None),
            config_yaml=yaml.safe_dump(config, sort_keys=False),
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON),
            model=model_source(),
            train_data=train_data,
            validation_data=validation_data,
            topology=SkyRLTopology(
                num_nodes=plan.num_nodes,
                gpus_per_node=GPUS_PER_NODE,
                gpu_variant=GPU_VARIANT,
                role_plan=plan.role_plan,
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=SEED,
        ),
        IrisSkyRLExecution(
            cluster=CLUSTER,
            cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
            cpu=CPUS_PER_NODE,
            memory=MEMORY_PER_NODE,
            disk=DISK_PER_NODE,
            priority="interactive",
            max_retries=2,
            target_cluster=CLUSTER,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=72,
            wandb_entity=WANDB_ENTITY,
            job_timeout_seconds=job_timeout_seconds,
            container_profile=CONTAINER_PROFILE,
        ),
        export_hf=export,
    )


@click.command(help=__doc__)
@click.option("--stage", type=click.Choice([stage.value for stage in Stage]), required=True)
@click.option("--pool-uri", default=DEFAULT_POOL_URI, show_default=True, help="Baseline task_rollout_spec parquet.")
@click.option("--train-uri", default=DEFAULT_TRAIN_URI, show_default=True, help="RL training parquet.")
@click.option(
    "--heldout-uri", default=DEFAULT_HELDOUT_URI, show_default=True, help="RL held-out parquet, evaluated periodically."
)
@click.option(
    "--baseline-entrypoint",
    type=click.Choice([entrypoint.value for entrypoint in BaselineEntrypoint]),
    default=BaselineEntrypoint.GENERATE.value,
    show_default=True,
)
@click.option("--baseline-nodes", type=click.IntRange(min=1), default=BASELINE_NODES, show_default=True)
@click.option("--max-steps", type=click.IntRange(min=1), default=RL_MAX_STEPS, show_default=True)
@click.option("--chat-template", default=DEFAULT_CHAT_TEMPLATE, show_default=True)
@click.option("--inference-nodes", type=click.IntRange(min=1), default=RL_INFERENCE_NODES, show_default=True)
@click.option(
    "--eval-samples",
    type=click.IntRange(min=1),
    help=f"Samples per evaluated task [default: {RL_EVAL_SAMPLES} for rl, {BASELINE_EVAL_SAMPLES} for baseline]",
)
@click.option("--eval-interval", type=click.IntRange(min=1), default=EVAL_INTERVAL, show_default=True)
@click.option("--checkpoint-interval", type=click.IntRange(min=1), default=CHECKPOINT_INTERVAL, show_default=True)
@click.option("--export", is_flag=True, help="Export the final policy as a Hugging Face checkpoint.")
@click.option("--job-timeout-seconds", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--set", "settings", multiple=True, metavar="KEY=VALUE")
@rl_build_options
def main(
    stage: str,
    pool_uri: str,
    train_uri: str,
    heldout_uri: str,
    baseline_entrypoint: str,
    baseline_nodes: int,
    max_steps: int,
    chat_template: str,
    inference_nodes: int,
    eval_samples: int | None,
    eval_interval: int,
    checkpoint_interval: int,
    export: bool,
    job_timeout_seconds: int,
    settings: tuple[str, ...],
) -> ArtifactStep[SkyRLRun]:
    return build_run(
        Stage(stage),
        pool_uri=pool_uri,
        train_uri=train_uri,
        heldout_uri=heldout_uri,
        baseline_entrypoint=BaselineEntrypoint(baseline_entrypoint),
        baseline_nodes=baseline_nodes,
        max_steps=max_steps,
        chat_template=chat_template,
        inference_nodes=inference_nodes,
        eval_samples=eval_samples,
        eval_interval=eval_interval,
        checkpoint_interval=checkpoint_interval,
        export=export,
        job_timeout_seconds=job_timeout_seconds,
        settings=settings,
    )


if __name__ == "__main__":
    main()
