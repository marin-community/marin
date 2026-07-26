# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Four-arm nested expert-bank experiment on cw-us-east-08a GB200s.

The launcher keeps the current d1280 compute-optimal recipe fixed while varying
only total expert count and the fraction of batch rows restricted to an
extractable, interleaved E128 subset.

Required environment:

    NESTED_ARM    large | small | nested25 | nested50
    NESTED_PHASE  smoke | full

Optional overrides:

    NESTED_STEPS        optimizer steps (phase default: 20 or 5275)
    NESTED_BATCH        global batch size (default: 256)
    NESTED_NODES        four-GPU GB200 nodes (default: 16)
    NESTED_EXPERT_AXIS  expert-parallel axis size (default: 64)
"""

import dataclasses
import os
from enum import StrEnum

from fray.cluster import ResourceConfig
from levanter.data.text.datasets import BlockShuffleConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import experiment_main
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.datasets.paloma import paloma_datasets
from experiments.grug.moe.heuristic import build_from_heuristic
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
_HIDDEN_DIM = 1280
_TARGET_STEPS = 8192
_LARGE_EXPERTS = 256
_SMALL_EXPERTS = 128
_GPUS_PER_NODE = 4
_DEFAULT_NODES = 16
_DEFAULT_EXPERT_AXIS = 64
_DEFAULT_BATCH = 256
_SMOKE_STEPS = 20
_OUTPUT_SUBDIR = "experiments/nested-moe"
_SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=256, perm_type="feistel")


class NestedArm(StrEnum):
    LARGE = "large"
    SMALL = "small"
    NESTED_25 = "nested25"
    NESTED_50 = "nested50"

    @property
    def experiment_id(self) -> str:
        return {
            NestedArm.LARGE: "NEST-MOE-001",
            NestedArm.SMALL: "NEST-MOE-002",
            NestedArm.NESTED_25: "NEST-MOE-003",
            NestedArm.NESTED_50: "NEST-MOE-004",
        }[self]


def _required_env(key: str) -> str:
    value = os.environ.get(key)
    if value is None:
        raise ValueError(f"{key} must be set")
    return value


def _arm_model(base_model: GrugModelConfig, arm: NestedArm) -> GrugModelConfig:
    common = dict(
        attention_implementation="gpu_fa4_thd",
        moe_implementation="ring",
        remat_mode="recompute_all",
    )
    if arm is NestedArm.LARGE:
        return dataclasses.replace(base_model, num_experts=_LARGE_EXPERTS, **common)
    if arm is NestedArm.SMALL:
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
    phase = _required_env("NESTED_PHASE")
    if phase not in ("smoke", "full"):
        raise ValueError("NESTED_PHASE must be 'smoke' or 'full'")

    base_model, optimizer, _, full_steps = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        target_steps=_TARGET_STEPS,
    )
    model = _arm_model(base_model, arm)
    nodes = env_int("NESTED_NODES", _DEFAULT_NODES)
    expert_axis = env_int("NESTED_EXPERT_AXIS", _DEFAULT_EXPERT_AXIS)
    batch_size = env_int("NESTED_BATCH", _DEFAULT_BATCH)
    steps = env_int("NESTED_STEPS", _SMOKE_STEPS if phase == "smoke" else full_steps)

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
    run_id = f"{arm.experiment_id.lower()}-{phase}-d{_HIDDEN_DIM}-e{model.num_experts}"
    step_name = f"{_OUTPUT_SUBDIR}/{run_id}"
    version = resolve_version(step_name, version)
    train_data = slimpajama_6b_dataset()
    validation = list(paloma_datasets(tokenizer=llama3_tokenizer).values())

    def build_config(ctx: StepContext) -> GrugMoeLaunchConfig:
        eval_batch_size = 64 if phase == "smoke" else 256
        return GrugMoeLaunchConfig(
            model=model,
            data=mixture(ctx, {train_data: 1.0}, validation=validation, shuffle=_SHUFFLE),
            output_path=ctx.output_path,
            run_id=run_id,
            resources=ctx.runtime_arg("train_resources"),
            steps=steps,
            batch_size=batch_size,
            seed=0,
            mp="params=float32,compute=bfloat16,output=bfloat16",
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
                steps_per_eval=steps if phase == "smoke" else 1000,
                max_eval_batches=1,
                eval_current=True,
                eval_ema=False,
            ),
            processes_per_task=1,
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
