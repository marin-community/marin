# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch CatCountCanary synchronous and asynchronous runs through MarinSkyRL Megatron."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import click
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.fingerprint import fingerprint_hash
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

from experiments.models import qwen2_5_0_5b, qwen2_5_0_5b_instruct, qwen3_0_6b
from experiments.post_training.cat_count_canary.data import (
    DEFAULT_TRAIN_NS,
    ENV_CLASS,
    EXTRAPOLATION_NS,
    HELDOUT_NS,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
    cat_count_data_step,
)

EXPERIMENT_NAME = "cat-count-canary"
CLUSTER = "cw-rno2a"
GPU_VARIANT = "H100"
GPUS_PER_NODE = 2
# The 65-CPU request places the two tasks on separate 128-CPU hosts.
CPUS_PER_NODE = 65
TRAIN_BATCH_SIZE = 64
MICRO_TRAIN_BATCH_SIZE = 16
GROUP_SIZE = 8
SAMPLED_EVAL_SAMPLES = 8
SEED = 17
JOB_TIMEOUT_SECONDS = 7200


@dataclass(frozen=True)
class ModelChoice:
    step: ArtifactStep
    chat_template: str
    chat_template_kwargs: Mapping[str, object] | None = None


MODELS = MappingProxyType(
    {
        "qwen2.5-0.5b-instruct": ModelChoice(
            qwen2_5_0_5b_instruct,
            "qwen2_5_with_generation_tag_simplified",
        ),
        "qwen2.5-0.5b": ModelChoice(
            qwen2_5_0_5b,
            "qwen2_5_with_generation_tag_simplified",
        ),
        "qwen3-0.6b": ModelChoice(
            qwen3_0_6b,
            "qwen3_without_thinking",
            MappingProxyType({"enable_thinking": False}),
        ),
    }
)


@dataclass(frozen=True)
class Preset:
    async_steps: int
    sync_steps: int
    reward_rise: float | None


PRESETS = MappingProxyType(
    {
        "dry": Preset(1, 1, None),
        "calibrate": Preset(30, 30, None),
        "gate": Preset(50, 25, 0.2),
        "gate-filter": Preset(50, 25, 0.2),
        "on-policy": Preset(30, 30, None),
    }
)


def role_plan(
    *,
    batch_size: int = TRAIN_BATCH_SIZE,
    group_size: int = GROUP_SIZE,
    micro_train_batch_size: int = MICRO_TRAIN_BATCH_SIZE,
) -> SkyRLRolePlan:
    return SkyRLRolePlan(
        colocate_all=False,
        policy_num_nodes=1,
        policy_num_gpus_per_node=GPUS_PER_NODE,
        num_inference_engines=GPUS_PER_NODE,
        inference_engine_tensor_parallel_size=1,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=1,
        inference_engine_expert_parallel_size=1,
        train_batch_size=batch_size,
        policy_mini_batch_size=batch_size,
        micro_train_batch_size_per_gpu=micro_train_batch_size,
        n_samples_per_prompt=group_size,
    )


ROLE_SETTINGS = frozenset(_role_plan_config_values(role_plan()))
DERIVED_SETTINGS = frozenset(
    {
        "generator.max_input_length",
        "trainer.max_prompt_length",
        "generator.max_turns",
        "generator.sampling_params.max_generate_length",
        "generator.eval_sampling_params.max_generate_length",
        "generator.engine_init_kwargs.max_model_len",
        "generator.trajectory_reward_shaping.overlong.l_max",
        "generator.trajectory_reward_shaping.overlong.l_cache",
    }
)
PROTECTED_SETTINGS = (
    ROLE_SETTINGS
    | DERIVED_SETTINGS
    | frozenset(
        {
            "entrypoint",
            "trainer.strategy",
            "trainer.policy.model.path",
            "trainer.policy.megatron_config",
            "trainer.ref.megatron_config",
            "trainer.ref.model.path",
            "trainer.resume_mode",
            "trainer.seed",
            "trainer.max_ckpts_to_keep",
            "trainer.eval_before_train",
            "trainer.ckpt_interval",
            "trainer.callbacks",
            "generator.chat_template",
            "generator.chat_template_kwargs",
            "generator.run_engines_locally",
            "generator.enable_http_endpoint",
            "generator.backend",
            "generator.use_conversation_multi_turn",
            "generator.require_exact_chat_transport",
            "environment.env_class",
            "data.kind",
        }
    )
)


def apply_setting(config: dict, text: str) -> None:
    key, separator, raw = text.partition("=")
    if not separator or not key:
        raise click.BadParameter(f"expected dotted.key=value, got {text!r}")
    if any(
        key == protected or key.startswith(f"{protected}.") or protected.startswith(f"{key}.")
        for protected in PROTECTED_SETTINGS
    ) or key.startswith("data."):
        raise click.BadParameter(f"{key!r} is owned by the launcher; use a typed option")
    if not key.startswith(("trainer.", "generator.", "context_budget.")):
        raise click.BadParameter(f"{key!r} is outside the configurable recipe")
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        node = node.get(part)
        if not isinstance(node, dict):
            raise click.BadParameter(f"unknown setting {key!r}")
    if parts[-1] not in node:
        raise click.BadParameter(f"unknown setting {key!r}")
    value = yaml.safe_load(raw)
    if isinstance(node[parts[-1]], float) and isinstance(value, str):
        try:
            value = float(value)
        except ValueError as error:
            raise click.BadParameter(f"{key!r} requires a numeric value") from error
    node[parts[-1]] = value


def training_config(
    *,
    preset: str = "gate",
    lane: str = "async",
    model: str = "qwen2.5-0.5b-instruct",
    batch_size: int = TRAIN_BATCH_SIZE,
    group_size: int = GROUP_SIZE,
    micro_train_batch_size: int = MICRO_TRAIN_BATCH_SIZE,
    eval_reward_rise: float | None = None,
    train_ns: tuple[int, ...] = DEFAULT_TRAIN_NS,
    seed: int = SEED,
    settings: tuple[str, ...] = (),
) -> dict:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}")
    if lane not in ("async", "sync"):
        raise ValueError(f"unknown lane {lane!r}")
    if batch_size <= 0 or group_size <= 0 or micro_train_batch_size <= 0:
        raise ValueError("batch, group and micro-batch sizes must be positive")
    choice = MODELS[model]
    plan = role_plan(batch_size=batch_size, group_size=group_size, micro_train_batch_size=micro_train_batch_size)
    preset_config = PRESETS[preset]
    max_steps = preset_config.async_steps if lane == "async" else preset_config.sync_steps
    if eval_reward_rise is None:
        eval_reward_rise = preset_config.reward_rise
    geometry = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
    }
    config = {
        "entrypoint": "standard",
        "context_budget": {
            "request_window_tokens": 128,
            "max_new_tokens_per_turn": 64,
            "max_turns": 1,
        },
        "environment": {"env_class": ENV_CLASS},
        "trainer": {
            "strategy": "megatron",
            "flash_attn": False,
            "use_sample_packing": False,
            "gradient_checkpointing": False,
            "epochs": 1,
            "max_steps": max_steps,
            "update_epochs_per_batch": 1 if preset == "on-policy" else 2,
            "micro_forward_batch_size_per_gpu": micro_train_batch_size,
            "eval_batch_size": len(train_ns) + len(HELDOUT_NS) + len(EXTRAPOLATION_NS),
            "eval_interval": 1 if preset == "dry" else 5,
            "hf_save_interval": max_steps,
            "resume_mode": "latest",
            "max_ckpts_to_keep": 1,
            "seed": seed,
            "logger": "wandb",
            "project_name": "marin-cat-count-canary",
            "tracker_commit_each_step": True,
            "training_metrics": True,
            "policy_train_spans": True,
            "rollout_spans": True,
            "algorithm": {
                "advantage_estimator": "grpo",
                "policy_loss_type": "behavior_clip" if lane == "async" else "regular",
                "eps_clip_low": 0.2,
                "eps_clip_high": 0.2,
                "use_kl_loss": False,
                "use_kl_in_reward": False,
                "use_tis": lane == "sync",
                "tis_imp_ratio_cap": 2.0,
                "dynamic_sampling": {"type": "filter" if preset == "gate-filter" else None},
            },
            "policy": {
                "optimizer_config": {
                    "optimizer": "AdamW",
                    "lr": 2.0e-6,
                    "weight_decay": 0.01,
                    "max_grad_norm": 1.0,
                },
                "megatron_config": geometry,
            },
            "ref": {"megatron_config": geometry},
            "rollout_buffer": {
                "max_staleness_steps": 0 if lane == "sync" or preset == "on-policy" else 2,
                "batch_policy": "full_batch",
                "max_in_flight": batch_size,
                "object_store_root": None,
            },
        },
        "generator": {
            "backend": "vllm",
            "model_dtype": "bfloat16",
            "vllm_attention_backend": "FLASH_ATTN",
            "run_engines_locally": True,
            "enable_http_endpoint": False,
            "weight_sync_backend": "nccl",
            "use_conversation_multi_turn": False,
            "require_exact_chat_transport": False,
            "gpu_memory_utilization": 0.7,
            "enforce_eager": False,
            "chat_template": {"source": "name", "name_or_path": choice.chat_template},
            "sampling_params": {"temperature": 1.0, "top_p": 1.0, "logprobs": 0},
            "eval_sampling_params": {"temperature": 0.0},
            "eval_n_samples_per_prompt": 1,
            "inference_stats_interval": 1,
        },
        "data": {"kind": "parquet", "shuffle": False, "train_data": [], "val_data": []},
    }
    _materialize_role_plan_config(config, plan)
    if choice.chat_template_kwargs is not None:
        config["generator"]["chat_template_kwargs"] = dict(choice.chat_template_kwargs)
    for setting in settings:
        apply_setting(config, setting)
    trainer = config["trainer"]
    trainer["eval_before_train"] = trainer["eval_interval"] > 0
    trainer["ckpt_interval"] = max(1, trainer["eval_interval"])
    metric_groups = {}
    for profile in ("eval", "eval/sampled"):
        for split, counts in (("train", train_ns), ("heldout", HELDOUT_NS), ("extrapolation", EXTRAPOLATION_NS)):
            for metric in ("avg_score", "environment/exact"):
                source_metric = "environment/cat_count/exact" if metric == "environment/exact" else metric
                metric_groups[f"{profile}/{split}/{metric}"] = [
                    f"{profile}/cat_count_n{n}/{source_metric}" for n in counts
                ]
    if eval_reward_rise is not None and (
        not math.isfinite(eval_reward_rise) or eval_reward_rise <= 0 or trainer["eval_interval"] <= 0
    ):
        raise ValueError("evaluation reward rise requires a finite positive margin and periodic evaluation")
    trainer["callbacks"] = [
        {"type": "checkpoint", "save_steps": trainer["ckpt_interval"]},
        {
            "type": "evaluation",
            "eval_steps": trainer["eval_interval"],
            "eval_before_train": trainer["eval_before_train"],
            "additional_evaluations": {
                "sampled": {"sampling_params": {"temperature": 1.0}, "n_samples_per_prompt": SAMPLED_EVAL_SAMPLES}
            },
            "metric_groups": metric_groups,
            "stop_on_improvement": {"eval/train/avg_score": eval_reward_rise} if eval_reward_rise is not None else {},
        },
        {"type": "hf_model_save", "save_steps": trainer["hf_save_interval"]},
        {"type": "database_registration"},
        {
            "type": "inference_stats",
            "log_every_steps": config["generator"]["inference_stats_interval"],
            "log_to_console": True,
            "log_to_tracker": True,
        },
    ]
    if lane == "sync" and trainer["rollout_buffer"]["max_staleness_steps"] != 0:
        raise ValueError("the sync lane requires zero rollout staleness")
    return config


def build_run(
    *,
    preset: str = "gate",
    lane: str = "async",
    model: str = "qwen2.5-0.5b-instruct",
    batch_size: int = TRAIN_BATCH_SIZE,
    group_size: int = GROUP_SIZE,
    micro_train_batch_size: int = MICRO_TRAIN_BATCH_SIZE,
    eval_reward_rise: float | None = None,
    train_ns: tuple[int, ...] = DEFAULT_TRAIN_NS,
    seed: int = SEED,
    job_timeout_seconds: int = JOB_TIMEOUT_SECONDS,
    version: str | None = None,
    cluster: str = CLUSTER,
    settings: tuple[str, ...] = (),
) -> ArtifactStep[SkyRLRun]:
    if job_timeout_seconds <= 0:
        raise ValueError("job timeout must be positive")
    config = training_config(
        preset=preset,
        lane=lane,
        model=model,
        batch_size=batch_size,
        group_size=group_size,
        micro_train_batch_size=micro_train_batch_size,
        eval_reward_rise=eval_reward_rise,
        train_ns=train_ns,
        seed=seed,
        settings=settings,
    )
    max_steps = config["trainer"]["max_steps"]
    row_multiplier = 4 if preset == "gate-filter" else 1
    prefetch_rows = config["trainer"]["rollout_buffer"]["max_staleness_steps"] * batch_size
    train_rows = (batch_size * max_steps + prefetch_rows) * row_multiplier
    identity = f"{model}-{lane}-{preset}-{fingerprint_hash(yaml.safe_dump(config) + repr(train_ns))}"
    data_identity = fingerprint_hash(repr((train_ns, train_rows, seed)))
    data_name = f"documents/{EXPERIMENT_NAME}/{data_identity}"
    data = cat_count_data_step(
        data_name,
        version or resolve_version(data_name, None),
        train_ns=train_ns,
        train_rows=train_rows,
        seed=seed,
    )
    base_name = f"checkpoints/{EXPERIMENT_NAME}/{identity}"
    choice = MODELS[model]
    download = choice.step.build_config(StepContext.for_fingerprint(choice.step.runtime_args, choice.step.deps))
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name(base_name),
            version=version or resolve_version(base_name, None),
            config_yaml=yaml.safe_dump(config, sort_keys=False),
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON),
            model=ArtifactHfModel(
                step=choice.step,
                tokenizer_uri=download.hf_dataset_id,
                tokenizer_revision=download.revision,
            ),
            train_data=(ArtifactDataSource(data, relative_path=TRAIN_FILENAME),),
            validation_data=(ArtifactDataSource(data, relative_path=VALIDATION_FILENAME),),
            topology=SkyRLTopology(
                num_nodes=2,
                gpus_per_node=GPUS_PER_NODE,
                gpu_variant=GPU_VARIANT,
                role_plan=role_plan(
                    batch_size=batch_size, group_size=group_size, micro_train_batch_size=micro_train_batch_size
                ),
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=seed,
        ),
        IrisSkyRLExecution(
            cluster=cluster,
            cluster_config=f"lib/iris/config/{cluster}.yaml",
            cpu=CPUS_PER_NODE,
            memory="512GB",
            disk="1TB",
            priority="interactive",
            max_retries=1,
            target_cluster=cluster,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=24,
            wandb_entity=None,
            job_timeout_seconds=job_timeout_seconds,
        ),
        export_hf=True,
    )


@click.command(help=__doc__)
@click.option("--preset", type=click.Choice(tuple(PRESETS)), default="dry", show_default=True)
@click.option("--cluster", type=click.Choice(("cw-rno2a", "cw-us-east-02a")), default=CLUSTER, show_default=True)
@click.option("--lane", type=click.Choice(("sync", "async")), default="async", show_default=True)
@click.option("--model", type=click.Choice(tuple(MODELS)), default="qwen2.5-0.5b-instruct")
@click.option("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
@click.option("--group-size", type=int, default=GROUP_SIZE)
@click.option("--micro-train-batch-size", type=int, default=MICRO_TRAIN_BATCH_SIZE, show_default=True)
@click.option("--eval-reward-rise", type=float, help="Stop after this gain in greedy training evaluation reward.")
@click.option("--train-n", "train_ns", multiple=True, type=int)
@click.option("--seed", type=int, default=SEED)
@click.option("--job-timeout-seconds", type=int, default=JOB_TIMEOUT_SECONDS, show_default=True)
@click.option("--set", "settings", multiple=True, metavar="KEY=VALUE")
@rl_build_options
def main(
    preset: str,
    cluster: str,
    lane: str,
    model: str,
    batch_size: int,
    group_size: int,
    micro_train_batch_size: int,
    eval_reward_rise: float | None,
    train_ns: tuple[int, ...],
    seed: int,
    job_timeout_seconds: int,
    settings: tuple[str, ...],
) -> ArtifactStep[SkyRLRun]:
    return build_run(
        cluster=cluster,
        preset=preset,
        lane=lane,
        model=model,
        batch_size=batch_size,
        group_size=group_size,
        micro_train_batch_size=micro_train_batch_size,
        eval_reward_rise=eval_reward_rise,
        train_ns=train_ns or DEFAULT_TRAIN_NS,
        seed=seed,
        job_timeout_seconds=job_timeout_seconds,
        settings=settings,
    )


if __name__ == "__main__":
    main()
