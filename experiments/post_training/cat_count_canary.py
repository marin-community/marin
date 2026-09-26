# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch CatCountCanary standard and asynchronous runs through MarinSkyRL Megatron."""

from __future__ import annotations

from dataclasses import dataclass

import click
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import ArtifactStep
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
    _role_plan_config_values,
    skyrl_step,
)

from experiments.models import qwen2_5_0_5b, qwen2_5_0_5b_instruct, qwen3_0_6b
from experiments.post_training.cat_count_data import (
    DEFAULT_TRAIN_NS,
    HELDOUT_NS,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
    cat_count_data_step,
)

EXPERIMENT_NAME = "cat-count-canary"
CLUSTER = "cw-rno2a"
GPU_VARIANT = "H100"
GPUS_PER_NODE = 2
TRAIN_NS = DEFAULT_TRAIN_NS
TRAIN_BATCH_SIZE = 32
GROUP_SIZE = 8
SEED = 17


@dataclass(frozen=True)
class ModelChoice:
    step: ArtifactStep
    tokenizer_uri: str
    tokenizer_revision: str
    chat_template: str
    chat_template_kwargs: dict[str, object] | None = None


MODELS = {
    "qwen2.5-0.5b-instruct": ModelChoice(
        qwen2_5_0_5b_instruct,
        "Qwen/Qwen2.5-0.5B-Instruct",
        "7ae557604adf67be50417f59c2c2f167def9a775",
        "qwen2_5_with_generation_tag_simplified",
    ),
    "qwen2.5-0.5b": ModelChoice(
        qwen2_5_0_5b,
        "Qwen/Qwen2.5-0.5B",
        "060db6499f32faf8b98477b0a26969ef7d8b9987",
        "qwen2_5_with_generation_tag_simplified",
    ),
    "qwen3-0.6b": ModelChoice(
        qwen3_0_6b,
        "Qwen/Qwen3-0.6B",
        "c1899de",
        "qwen3_without_thinking",
        {"enable_thinking": False},
    ),
}

PRESET_STEPS = {"dry": 1, "calibrate": 30, "gate": 60, "gate-filter": 60, "on-policy": 30}
ROLE_SETTINGS = frozenset(
    _role_plan_config_values(
        SkyRLRolePlan(
            colocate_all=False,
            policy_num_nodes=1,
            policy_num_gpus_per_node=GPUS_PER_NODE,
            num_inference_engines=GPUS_PER_NODE,
            inference_engine_tensor_parallel_size=1,
            inference_engine_pipeline_parallel_size=1,
            inference_engine_data_parallel_size=1,
            inference_engine_expert_parallel_size=1,
            train_batch_size=TRAIN_BATCH_SIZE,
            policy_mini_batch_size=TRAIN_BATCH_SIZE,
            micro_train_batch_size_per_gpu=1,
            n_samples_per_prompt=GROUP_SIZE,
        )
    )
)
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
            "trainer.eval_before_train",
            "trainer.ckpt_interval",
            "generator.chat_template",
            "generator.chat_template_kwargs",
            "generator.run_engines_locally",
            "generator.backend",
            "generator.use_conversation_multi_turn",
            "environment.env_class",
            "data.kind",
        }
    )
)


def role_plan(*, batch_size: int = TRAIN_BATCH_SIZE, group_size: int = GROUP_SIZE) -> SkyRLRolePlan:
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
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=group_size,
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
    entrypoint: str = "fully_async",
    model: str = "qwen2.5-0.5b-instruct",
    batch_size: int = TRAIN_BATCH_SIZE,
    group_size: int = GROUP_SIZE,
    train_ns: tuple[int, ...] = TRAIN_NS,
    seed: int = SEED,
    settings: tuple[str, ...] = (),
) -> dict:
    if preset not in PRESET_STEPS:
        raise ValueError(f"unknown preset {preset!r}")
    if entrypoint not in ("fully_async", "standard"):
        raise ValueError(f"unknown entrypoint {entrypoint!r}")
    if batch_size <= 0 or group_size <= 0:
        raise ValueError("batch and group sizes must be positive")
    choice = MODELS[model]
    plan = role_plan(batch_size=batch_size, group_size=group_size)
    max_steps = PRESET_STEPS[preset]
    geometry = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
    }
    config = {
        "entrypoint": entrypoint,
        "context_budget": {
            "request_window_tokens": 128,
            "max_new_tokens_per_turn": 64,
            "max_turns": 1,
        },
        "environment": {"env_class": "cat_count"},
        "trainer": {
            "strategy": "megatron",
            "flash_attn": False,
            "use_sample_packing": False,
            "gradient_checkpointing": False,
            "epochs": 2,
            "max_steps": max_steps,
            "update_epochs_per_batch": 1 if preset == "on-policy" else 2,
            "train_batch_size": plan.train_batch_size,
            "policy_mini_batch_size": plan.policy_mini_batch_size,
            "micro_train_batch_size_per_gpu": plan.micro_train_batch_size_per_gpu,
            "micro_forward_batch_size_per_gpu": 1,
            "eval_batch_size": len(train_ns) + len(HELDOUT_NS),
            "eval_interval": 1 if preset == "dry" else 10,
            "hf_save_interval": -1,
            "resume_mode": "latest",
            "max_ckpts_to_keep": 1,
            "seed": seed,
            "logger": "wandb",
            "project_name": "marin-cat-count-canary",
            "tracker_commit_each_step": True,
            "algorithm": {
                "advantage_estimator": "grpo",
                "policy_loss_type": "regular",
                "eps_clip_low": 0.2,
                "eps_clip_high": 0.2,
                "use_kl_loss": False,
                "use_kl_in_reward": False,
                "use_tis": True,
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
            "placement": {
                "colocate_all": False,
                "colocate_policy_ref": True,
                "policy_num_nodes": 1,
                "policy_num_gpus_per_node": GPUS_PER_NODE,
                "ref_num_nodes": 1,
                "ref_num_gpus_per_node": GPUS_PER_NODE,
            },
            "fully_async": {
                "max_staleness_steps": 0 if preset == "on-policy" else 2,
                "num_parallel_generation_workers": batch_size,
                "max_buffered_groups": max(1, batch_size // 2),
            },
        },
        "generator": {
            "backend": "vllm",
            "model_dtype": "bfloat16",
            "vllm_attention_backend": "FLASH_ATTN",
            "run_engines_locally": True,
            "weight_sync_backend": "nccl",
            "async_engine": True,
            "batched": False,
            "use_conversation_multi_turn": True,
            "num_inference_engines": plan.num_inference_engines,
            "inference_engine_tensor_parallel_size": 1,
            "inference_engine_pipeline_parallel_size": 1,
            "inference_engine_data_parallel_size": 1,
            "inference_engine_expert_parallel_size": 1,
            "n_samples_per_prompt": group_size,
            "gpu_memory_utilization": 0.7,
            "enforce_eager": False,
            "chat_template": {"source": "name", "name_or_path": choice.chat_template},
            "sampling_params": {"temperature": 1.0, "top_p": 1.0, "logprobs": 0},
            "eval_sampling_params": {"temperature": 1.0 if preset == "dry" else 0.0},
            "eval_n_samples_per_prompt": group_size if preset == "dry" else 1,
        },
        "data": {"kind": "parquet", "shuffle": False, "train_data": [], "val_data": []},
    }
    if choice.chat_template_kwargs is not None:
        config["generator"]["chat_template_kwargs"] = choice.chat_template_kwargs
    for setting in settings:
        apply_setting(config, setting)
    trainer = config["trainer"]
    trainer["eval_before_train"] = trainer["eval_interval"] > 0
    trainer["ckpt_interval"] = max(1, trainer["eval_interval"])
    if entrypoint == "fully_async":
        workers = trainer["fully_async"]["num_parallel_generation_workers"]
        if workers < batch_size:
            raise click.BadParameter("generation workers cannot fill a training batch")
    return config


def build_run(
    *,
    preset: str = "gate",
    entrypoint: str = "fully_async",
    model: str = "qwen2.5-0.5b-instruct",
    batch_size: int = TRAIN_BATCH_SIZE,
    group_size: int = GROUP_SIZE,
    train_ns: tuple[int, ...] = TRAIN_NS,
    seed: int = SEED,
    version: str | None = None,
    settings: tuple[str, ...] = (),
) -> ArtifactStep[SkyRLRun]:
    config = training_config(
        preset=preset,
        entrypoint=entrypoint,
        model=model,
        batch_size=batch_size,
        group_size=group_size,
        train_ns=train_ns,
        seed=seed,
        settings=settings,
    )
    max_steps = config["trainer"]["max_steps"]
    row_multiplier = 4 if preset == "gate-filter" else 1
    prefetch_rows = 0
    if entrypoint == "fully_async":
        async_config = config["trainer"]["fully_async"]
        workers = async_config["num_parallel_generation_workers"]
        buffer = async_config["max_buffered_groups"] or workers
        in_flight_groups = workers + buffer
        prefetch_rows = ((in_flight_groups + batch_size - 1) // batch_size) * batch_size
    train_rows = (batch_size * max_steps + prefetch_rows) * row_multiplier
    identity = f"{model}-{entrypoint}-{preset}-{fingerprint_hash(yaml.safe_dump(config) + repr(train_ns))}"
    data_name = f"documents/{EXPERIMENT_NAME}/{identity}"
    data = cat_count_data_step(
        data_name,
        version or resolve_version(data_name, None),
        train_ns=train_ns,
        train_rows=train_rows,
        seed=seed,
    )
    base_name = f"checkpoints/{EXPERIMENT_NAME}/{identity}"
    choice = MODELS[model]
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name(base_name),
            version=version or resolve_version(base_name, None),
            config_yaml=yaml.safe_dump(config, sort_keys=False),
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON),
            model=ArtifactHfModel(
                step=choice.step,
                tokenizer_uri=choice.tokenizer_uri,
                tokenizer_revision=choice.tokenizer_revision,
            ),
            train_data=(ArtifactDataSource(data, relative_path=TRAIN_FILENAME),),
            validation_data=(ArtifactDataSource(data, relative_path=VALIDATION_FILENAME),),
            topology=SkyRLTopology(
                num_nodes=2,
                gpus_per_node=GPUS_PER_NODE,
                gpu_variant=GPU_VARIANT,
                role_plan=role_plan(batch_size=batch_size, group_size=group_size),
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=seed,
        ),
        IrisSkyRLExecution(
            cluster=CLUSTER,
            cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
            cpu=16,
            memory="512GB",
            disk="1TB",
            priority="interactive",
            max_retries=1,
            target_cluster=CLUSTER,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=24,
            wandb_entity=None,
        ),
        export_hf=True,
    )


@click.command(help=__doc__)
@click.option("--preset", type=click.Choice(tuple(PRESET_STEPS)), default="dry", show_default=True)
@click.option("--entrypoint", type=click.Choice(("fully_async", "standard")), default="fully_async")
@click.option("--model", type=click.Choice(tuple(MODELS)), default="qwen2.5-0.5b-instruct")
@click.option("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
@click.option("--group-size", type=int, default=GROUP_SIZE)
@click.option("--train-n", "train_ns", multiple=True, type=int)
@click.option("--seed", type=int, default=SEED)
@click.option("--set", "settings", multiple=True, metavar="KEY=VALUE")
@rl_build_options
def main(
    preset: str,
    entrypoint: str,
    model: str,
    batch_size: int,
    group_size: int,
    train_ns: tuple[int, ...],
    seed: int,
    settings: tuple[str, ...],
) -> ArtifactStep[SkyRLRun]:
    return build_run(
        preset=preset,
        entrypoint=entrypoint,
        model=model,
        batch_size=batch_size,
        group_size=group_size,
        train_ns=train_ns or TRAIN_NS,
        seed=seed,
        settings=settings,
    )


if __name__ == "__main__":
    main()
