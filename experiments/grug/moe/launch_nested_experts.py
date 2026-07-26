# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Four-arm nested expert-bank experiment on cw-us-east-08a GB200s.

The launcher keeps the current d1280 compute-optimal recipe fixed while varying
only total expert count and the fraction of batch rows restricted to an
extractable, interleaved E128 subset.

Required environment:

    NESTED_ARM    large | small | nested25 | nested50 | breakout25
    NESTED_PHASE  smoke | full | cooldown

Optional overrides:

    NESTED_STEPS        optimizer steps (phase defaults: 20 smoke, compute-derived
                        full, 50 cooldown)
    NESTED_BATCH        global batch size (default: 256)
    NESTED_NODES        four-GPU GB200 nodes (default: 16)
    NESTED_EXPERT_AXIS  expert-parallel axis size (default: 64)
    NESTED_CAPACITY_FACTOR  expert dispatch capacity factor (default: 1.0)
    NESTED_ATTENTION    attention backend (default: gpu_fa4_thd)
    NESTED_HIDDEN_DIM   model width (default: 1280)
    NESTED_SEQUENCE_LENGTH  sequence length (default: 8192)
    NESTED_RUN_SUFFIX   optional retry suffix for output and W&B IDs
    NESTED_MP           jmp policy (default: bf16 compute)
    NESTED_EVAL_EXPERTS  evaluate a fixed subset without restricting training
    NESTED_INIT_FROM    nested25 checkpoint root (required for breakout25)
"""

import dataclasses
import os
from enum import StrEnum
from typing import cast

from fray.cluster import ResourceConfig
from levanter.data.text.datasets import BlockShuffleConfig
from levanter.grug.attention import GrugAttentionImplementation
from levanter.tracker.wandb import WandbConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import experiment_main
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.data_configs import with_pack
from marin.training.training import LevanterCheckpoint

from experiments.datasets.paloma import paloma_datasets
from experiments.grug.moe.heuristic import MoeHeuristic, build_from_heuristic, compute_flops_per_token
from experiments.grug.moe.launch import (
    GrugMoeLaunchConfig,
    env_int,
    run_grug_moe_trial,
    slimpajama_6b_dataset,
)
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.llama import llama3_tokenizer

_BUDGET = 3.46e19
_TARGET_STEPS = 8192
_LARGE_EXPERTS = 256
_SMALL_EXPERTS = 128
_GPUS_PER_NODE = 4
_DEFAULT_NODES = 16
_DEFAULT_EXPERT_AXIS = 64
_DEFAULT_BATCH = 256
_DEFAULT_HIDDEN_DIM = 1280
_DEFAULT_SEQUENCE_LENGTH = 8192
_SMOKE_STEPS = 20
_COOLDOWN_STEPS = 50
_PROXY_WARMUP_STEPS = 5
_PROXY_EVAL_INTERVAL = 100
_COOLDOWN_EVAL_INTERVAL = 10
_OUTPUT_SUBDIR = "experiments/nested-moe"
_SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=256, perm_type="feistel")
_ATTENTION_IMPLEMENTATIONS = ("gpu_fa4_thd", "reference")
_DEFAULT_MP = "params=float32,compute=bfloat16,output=bfloat16"


class NestedArm(StrEnum):
    LARGE = "large"
    SMALL = "small"
    NESTED_25 = "nested25"
    NESTED_50 = "nested50"
    BREAKOUT_25 = "breakout25"

    @property
    def experiment_id(self) -> str:
        return {
            NestedArm.LARGE: "NEST-MOE-001",
            NestedArm.SMALL: "NEST-MOE-002",
            NestedArm.NESTED_25: "NEST-MOE-003",
            NestedArm.NESTED_50: "NEST-MOE-004",
            NestedArm.BREAKOUT_25: "NEST-MOE-005",
        }[self]


class _NestedPhase(StrEnum):
    SMOKE = "smoke"
    FULL = "full"
    COOLDOWN = "cooldown"


def _required_env(key: str) -> str:
    value = os.environ.get(key)
    if value is None:
        raise ValueError(f"{key} must be set")
    return value


def _attention_implementation() -> GrugAttentionImplementation:
    value = os.environ.get("NESTED_ATTENTION", "gpu_fa4_thd")
    if value not in _ATTENTION_IMPLEMENTATIONS:
        raise ValueError(f"NESTED_ATTENTION must be one of {_ATTENTION_IMPLEMENTATIONS}")
    return cast(GrugAttentionImplementation, value)


def _arm_model(
    base_model: GrugModelConfig,
    arm: NestedArm,
    attention_implementation: GrugAttentionImplementation,
) -> GrugModelConfig:
    common = dict(
        attention_implementation=attention_implementation,
        moe_implementation="ring",
        remat_mode="recompute_all",
    )
    if arm is NestedArm.LARGE:
        return dataclasses.replace(base_model, num_experts=_LARGE_EXPERTS, **common)
    if arm in (NestedArm.SMALL, NestedArm.BREAKOUT_25):
        return dataclasses.replace(base_model, num_experts=_SMALL_EXPERTS, **common)
    if arm is NestedArm.NESTED_25:
        return dataclasses.replace(
            base_model,
            num_experts=_LARGE_EXPERTS,
            nested_expert_count=_SMALL_EXPERTS,
            nested_batch_fraction=0.25,
            **common,
        )
    return dataclasses.replace(
        base_model,
        num_experts=_LARGE_EXPERTS,
        nested_expert_count=_SMALL_EXPERTS,
        nested_batch_fraction=0.5,
        **common,
    )


def build(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Build one preregistered nested-MoE arm."""
    arm = NestedArm(_required_env("NESTED_ARM"))
    phase = _NestedPhase(_required_env("NESTED_PHASE"))
    if (arm is NestedArm.BREAKOUT_25) != (phase is _NestedPhase.COOLDOWN):
        raise ValueError("breakout25 and cooldown must be selected together")

    attention_implementation = _attention_implementation()
    hidden_dim = env_int("NESTED_HIDDEN_DIM", _DEFAULT_HIDDEN_DIM)
    sequence_length = env_int("NESTED_SEQUENCE_LENGTH", _DEFAULT_SEQUENCE_LENGTH)
    capacity_factor = float(os.environ.get("NESTED_CAPACITY_FACTOR", "1.0"))
    heuristic = MoeHeuristic()
    base_model, _, _, _ = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=hidden_dim,
        heuristic=heuristic,
        target_steps=_TARGET_STEPS,
        seq_len=sequence_length,
    )
    model = _arm_model(
        dataclasses.replace(base_model, capacity_factor=capacity_factor),
        arm,
        attention_implementation,
    )
    eval_expert_count = os.environ.get("NESTED_EVAL_EXPERTS")
    if eval_expert_count is not None:
        if arm is not NestedArm.LARGE:
            raise ValueError("NESTED_EVAL_EXPERTS is only supported for the untreated large control")
        nested_expert_count = int(eval_expert_count)
        if nested_expert_count <= 0 or nested_expert_count >= model.num_experts:
            raise ValueError("NESTED_EVAL_EXPERTS must be positive and smaller than the full expert count")
        model = dataclasses.replace(
            model,
            nested_expert_count=nested_expert_count,
            nested_batch_fraction=0.0,
        )
    nodes = env_int("NESTED_NODES", _DEFAULT_NODES)
    expert_axis = env_int("NESTED_EXPERT_AXIS", _DEFAULT_EXPERT_AXIS)
    batch_size = env_int("NESTED_BATCH", _DEFAULT_BATCH)
    compute_optimal_tokens = _BUDGET / (3 * compute_flops_per_token(base_model))
    optimizer = heuristic.build_optimizer_config(
        batch_size,
        compute_optimal_tokens,
        hidden_dim,
        seq_len=sequence_length,
    )
    optimizer = dataclasses.replace(optimizer, warmup=_PROXY_WARMUP_STEPS)
    full_steps = max(1, round(compute_optimal_tokens / (batch_size * sequence_length)))
    if phase is _NestedPhase.SMOKE:
        default_steps = _SMOKE_STEPS
    elif phase is _NestedPhase.COOLDOWN:
        default_steps = _COOLDOWN_STEPS
    else:
        default_steps = full_steps
    steps = env_int("NESTED_STEPS", default_steps)

    total_devices = nodes * _GPUS_PER_NODE
    if total_devices % expert_axis != 0:
        raise ValueError(f"{total_devices=} must be divisible by {expert_axis=}")
    if model.num_experts % expert_axis != 0:
        raise ValueError(f"{model.num_experts=} must be divisible by {expert_axis=}")
    if batch_size % total_devices != 0:
        raise ValueError(f"{batch_size=} must be divisible by {total_devices=}")

    resources = ResourceConfig.with_gpu(
        "GB200",
        count=_GPUS_PER_NODE,
        cpu=64,
        ram="512g",
        disk="512g",
        replicas=nodes,
    )
    run_id = f"{arm.experiment_id.lower()}-{phase}-d{hidden_dim}-s{sequence_length}-e{model.num_experts}"
    run_suffix = os.environ.get("NESTED_RUN_SUFFIX")
    if run_suffix:
        run_id = f"{run_id}-{run_suffix}"
    step_name = f"{_OUTPUT_SUBDIR}/{run_id}"
    version = resolve_version(step_name, version)
    train_data = slimpajama_6b_dataset()
    validation = list(paloma_datasets(tokenizer=llama3_tokenizer).values())

    def build_config(ctx: StepContext) -> GrugMoeLaunchConfig:
        eval_batch_size = 64 if phase is _NestedPhase.SMOKE else 256
        nested_init_from = None
        nested_init_source_model = None
        if arm is NestedArm.BREAKOUT_25:
            nested_init_from = _required_env("NESTED_INIT_FROM")
            nested_init_source_model = _arm_model(
                dataclasses.replace(base_model, capacity_factor=capacity_factor),
                NestedArm.NESTED_25,
                attention_implementation,
            )
        data = mixture(ctx, {train_data: 1.0}, validation=validation, shuffle=_SHUFFLE)
        if model.attention_implementation == "gpu_fa4_thd":
            data = with_pack(data, 1)
        return GrugMoeLaunchConfig(
            model=model,
            data=data,
            output_path=ctx.output_path,
            run_id=run_id,
            resources=ctx.runtime_arg("train_resources"),
            steps=steps,
            batch_size=batch_size,
            seed=1 if phase is _NestedPhase.COOLDOWN else 0,
            mp=os.environ.get("NESTED_MP", _DEFAULT_MP),
            tracker=WandbConfig(
                project="marin_moe",
                tags=["moe", "nested-moe", arm.experiment_id, phase, "gb200"],
                group=f"NEST-MOE-20260726-{phase}",
                name=None,
                replicate_path=ctx.output_path,
            ),
            optimizer=optimizer,
            grug_trainer=GrugTrainerConfig(
                expert_axis_size=expert_axis,
                replica_axis_size=1,
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
            ),
            eval=GrugEvalConfig(
                eval_batch_size=eval_batch_size,
                steps_per_eval=(
                    steps
                    if phase is _NestedPhase.SMOKE
                    else _COOLDOWN_EVAL_INTERVAL if phase is _NestedPhase.COOLDOWN else _PROXY_EVAL_INTERVAL
                ),
                max_eval_batches=1,
                eval_current=True,
                eval_ema=False,
            ),
            processes_per_task=1,
            nested_init_from=nested_init_from,
            nested_init_source_model=nested_init_source_model,
        )

    return ArtifactStep(
        name=user_namespaced_name(step_name, version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_trial,
        build_config=build_config,
        deps=(train_data, *validation),
        runtime_args={"train_resources": resources},
    )


if __name__ == "__main__":
    experiment_main(build)()
