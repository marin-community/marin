# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Matched E256/fixed25 compute-optimal cell on the canonical datakit mixture.

Both arms use the same heuristic-derived d768/L8/E256 model, seed, 4.14e18
compute budget, data order, optimizer, mesh, and evaluation cadence.
The treatment restricts 25% of rows, alternating the fixed E128 and E16
subsets; the control always routes over E256.

The datakit source is explicitly pinned to its CoreWeave S3 cache so training
data, checkpoints, and W&B replicas remain local to the GB200 cluster.

Required environment:

    BURNIN_ARM  e256 | fixed25

Optional overrides:

    BURNIN_EXPERIMENT_ID
                         W&B group and run prefix (default: NEST-BURN-001)
    BURNIN_STEPS       training steps (default: heuristic-derived 16,840)
    BURNIN_DATA_STEPS  Datakit planning horizon (default: heuristic-derived 16,840)
    BURNIN_BATCH_SIZE  global sequence batch; an override requires BURNIN_OPTIMIZER_TOKENS
    BURNIN_OPTIMIZER_TOKENS
                         token horizon used to derive the optimizer hyperparameters
    BURNIN_NODES       four-GPU GB200 nodes (default: 8)
    BURNIN_TARGET_CLUSTER
                         Iris peer cluster for GB200 workers (default: cw-us-east-08a)
    BURNIN_REPLICA_AXIS_SIZE
                         replicated process groups (default: 1; use BURNIN_NODES for node-local FSDP)
    BURNIN_EXPERT_AXIS_SIZE
                         expert-parallel devices (default: 1)
    BURNIN_EVAL_INTERVAL
                         optimizer steps between evaluations (default: 1,000)
    BURNIN_RUN_SUFFIX  append a retry suffix to the run and artifact names
    BURNIN_MP          JMP policy (default: fp32 params, bf16 compute/output)
    BURNIN_ATTENTION   reference | cudnn | gpu_fa4_cute | gpu_fa4_thd
"""

import dataclasses
import os
from enum import StrEnum
from typing import cast

from fray.cluster import ResourceConfig
from levanter.data.text.datasets import ConcatDatasetComponent, DatasetComponent
from levanter.grug.attention import GrugAttentionImplementation
from levanter.tracker.wandb import WandbConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import experiment_main
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.data_configs import with_pack
from marin.training.training import LevanterCheckpoint

from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.moe.heuristic import MoeHeuristic, build_from_heuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig, env_int, run_grug_moe_trial
from experiments.grug.moe.launch_datakit_moe_mix import (
    _datakit_data_config,
    _val_component,
)
from experiments.grug.moe.launch_nested_experts import NestedArm, _arm_model
from experiments.grug.moe.model import NestedSubsetSchedule, RouterBalanceMode
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.marin_tokenizer import marin_tokenizer

_EXPERIMENT_ID = "NEST-BURN-001"
_DATKIT_STORE = "s3://marin-us-east-02a/marin/datakit/store_8ac06c74"
_OUTPUT_SUBDIR = "experiments/nested-moe-burnin"

_COMPUTE_BUDGET = 4.14e18
_HIDDEN_DIM = 768
_SEQUENCE_LENGTH = 8192
_HEURISTIC_TARGET_STEPS = 2**15
_CAPACITY_FACTOR = 1.25
_EXPERT_AXIS = 1
_GPUS_PER_NODE = 4
_DEFAULT_NODES = 8
_DEFAULT_TARGET_CLUSTER = "cw-us-east-08a"
_EVAL_INTERVAL = 1_000
_DEFAULT_MP = "params=float32,compute=bfloat16,output=bfloat16"
_ATTENTION_IMPLEMENTATIONS = ("reference", "cudnn", "gpu_fa4_cute", "gpu_fa4_thd")


class BurninArm(StrEnum):
    E256 = "e256"
    FIXED_25 = "fixed25"

    @property
    def nested_arm(self) -> NestedArm:
        if self is BurninArm.E256:
            return NestedArm.LARGE
        return NestedArm.FIXED_25


def _required_env(key: str) -> str:
    value = os.environ.get(key)
    if value is None:
        raise ValueError(f"{key} must be set")
    return value


def _validation_components(
    ctx: StepContext,
    validation: list[ArtifactStep],
) -> dict[str, DatasetComponent | ConcatDatasetComponent]:
    if ctx.is_fingerprint:
        return {item.name: _val_component(ctx.artifact_path(item)) for item in validation}
    return {item.name: ctx.resolved(item).as_component() for item in validation}


def build(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Build one matched burn-in arm."""
    experiment_id = os.environ.get("BURNIN_EXPERIMENT_ID", _EXPERIMENT_ID)
    arm = BurninArm(_required_env("BURNIN_ARM"))
    target_cluster = os.environ.get("BURNIN_TARGET_CLUSTER", _DEFAULT_TARGET_CLUSTER)
    if not target_cluster:
        raise ValueError("BURNIN_TARGET_CLUSTER must not be empty")
    nodes = env_int("BURNIN_NODES", _DEFAULT_NODES)
    if nodes <= 0:
        raise ValueError("BURNIN_NODES must be positive")
    replica_axis_size = env_int("BURNIN_REPLICA_AXIS_SIZE", 1)
    if replica_axis_size <= 0:
        raise ValueError("BURNIN_REPLICA_AXIS_SIZE must be positive")
    expert_axis_size = env_int("BURNIN_EXPERT_AXIS_SIZE", _EXPERT_AXIS)
    if expert_axis_size <= 0:
        raise ValueError("BURNIN_EXPERT_AXIS_SIZE must be positive")

    heuristic = MoeHeuristic()
    base_model, optimizer, heuristic_batch_size, heuristic_steps = build_from_heuristic(
        budget=_COMPUTE_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        heuristic=heuristic,
        target_steps=_HEURISTIC_TARGET_STEPS,
        seq_len=_SEQUENCE_LENGTH,
    )
    batch_size = env_int("BURNIN_BATCH_SIZE", heuristic_batch_size)
    if batch_size <= 0:
        raise ValueError("BURNIN_BATCH_SIZE must be positive")
    optimizer_tokens_value = os.environ.get("BURNIN_OPTIMIZER_TOKENS")
    if optimizer_tokens_value is None:
        if batch_size != heuristic_batch_size:
            raise ValueError("BURNIN_OPTIMIZER_TOKENS is required when overriding BURNIN_BATCH_SIZE")
    else:
        optimizer_tokens = int(optimizer_tokens_value)
        if optimizer_tokens <= 0:
            raise ValueError("BURNIN_OPTIMIZER_TOKENS must be positive")
        optimizer = heuristic.build_optimizer_config(
            batch_size,
            optimizer_tokens,
            _HIDDEN_DIM,
            seq_len=_SEQUENCE_LENGTH,
        )
    steps = env_int("BURNIN_STEPS", heuristic_steps)
    if steps <= 0:
        raise ValueError("BURNIN_STEPS must be positive")
    data_steps = env_int("BURNIN_DATA_STEPS", heuristic_steps)
    if data_steps < steps:
        raise ValueError("BURNIN_DATA_STEPS must be at least BURNIN_STEPS")
    eval_interval = env_int("BURNIN_EVAL_INTERVAL", _EVAL_INTERVAL)
    if eval_interval <= 0:
        raise ValueError("BURNIN_EVAL_INTERVAL must be positive")

    base_model = dataclasses.replace(
        base_model,
        capacity_factor=_CAPACITY_FACTOR,
        router_balance_mode=RouterBalanceMode.ELIGIBILITY_QB,
    )
    attention_implementation = os.environ.get("BURNIN_ATTENTION", "gpu_fa4_thd")
    if attention_implementation not in _ATTENTION_IMPLEMENTATIONS:
        raise ValueError(f"BURNIN_ATTENTION must be one of {_ATTENTION_IMPLEMENTATIONS}")
    model = _arm_model(
        base_model,
        arm.nested_arm,
        cast(GrugAttentionImplementation, attention_implementation),
    )
    if arm is BurninArm.FIXED_25:
        model = dataclasses.replace(model, nested_subset_schedule=NestedSubsetSchedule.PREFIX)

    total_devices = nodes * _GPUS_PER_NODE
    if base_model.num_experts % expert_axis_size != 0:
        raise ValueError(f"model experts {base_model.num_experts} must be divisible by expert axis {expert_axis_size}")
    if arm is BurninArm.FIXED_25 and expert_axis_size != 1:
        raise ValueError("contiguous prefix nesting requires BURNIN_EXPERT_AXIS_SIZE=1")

    fixed_mesh_axes = replica_axis_size * expert_axis_size
    if total_devices % fixed_mesh_axes != 0:
        raise ValueError(
            f"{total_devices=} must be divisible by replica axis {replica_axis_size} * expert axis {expert_axis_size}"
        )
    if batch_size % total_devices != 0:
        raise ValueError(f"batch size {batch_size} must be divisible by {total_devices=}")

    run_id = f"{experiment_id.lower()}-{arm.value}-d{_HIDDEN_DIM}-s{_SEQUENCE_LENGTH}-e256-c4p14e18"
    run_suffix = os.environ.get("BURNIN_RUN_SUFFIX")
    if run_suffix:
        run_id = f"{run_id}-{run_suffix}"
    step_name = f"{_OUTPUT_SUBDIR}/{run_id}"
    version = resolve_version(step_name, version)

    validation = [
        *paloma_datasets(tokenizer=marin_tokenizer).values(),
        *uncheatable_datasets(tokenizer=marin_tokenizer).values(),
    ]
    resources = ResourceConfig.with_gpu(
        "GB200",
        count=_GPUS_PER_NODE,
        cpu=64,
        ram="512g",
        disk="512g",
        target_cluster=target_cluster,
        replicas=nodes,
    )

    def build_config(ctx: StepContext) -> GrugMoeLaunchConfig:
        data = _datakit_data_config(
            total_steps=data_steps,
            batch_size=batch_size,
            max_seq_len=_SEQUENCE_LENGTH,
            enable_simulated_epoching=True,
            val_components=_validation_components(ctx, validation),
            store_prefix=_DATKIT_STORE,
        )
        data = with_pack(data, 1)
        if attention_implementation in ("reference", "cudnn"):
            # pack=1 contains at most one document per example. Removing the
            # redundant segment mask avoids materializing cross-document
            # metadata; padding remains excluded from the loss.
            data = dataclasses.replace(data, block_cross_document_attention=False)
        return GrugMoeLaunchConfig(
            model=model,
            data=data,
            output_path=ctx.output_path,
            run_id=run_id,
            resources=ctx.runtime_arg("train_resources"),
            steps=steps,
            batch_size=batch_size,
            seed=0,
            mp=os.environ.get("BURNIN_MP", _DEFAULT_MP),
            tracker=WandbConfig(
                project="marin",
                tags=[
                    "moe",
                    "nested-moe",
                    "burnin",
                    "datakit",
                    "gb200",
                    arm.value,
                    experiment_id,
                ],
                group=experiment_id,
                name=None,
                replicate_path=ctx.output_path,
            ),
            optimizer=optimizer,
            grug_trainer=GrugTrainerConfig(
                expert_axis_size=expert_axis_size,
                replica_axis_size=replica_axis_size,
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
            ),
            eval=GrugEvalConfig(
                eval_batch_size=256,
                steps_per_eval=eval_interval,
                max_eval_batches=4,
                eval_current=True,
                eval_ema=False,
                nested_eval_offset_count=1,
            ),
        )

    return ArtifactStep(
        name=user_namespaced_name(step_name, version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_trial,
        build_config=build_config,
        deps=tuple(validation),
        runtime_args={"train_resources": resources},
    )


if __name__ == "__main__":
    experiment_main(build)()
