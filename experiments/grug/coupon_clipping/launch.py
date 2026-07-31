# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Artifact builder for the 16-GB200 coupon-clipping experiments."""

import dataclasses
import os
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint, resolve_checkpointer_output_path

from experiments.grug.coupon_clipping.config import (
    NUM_EXPERTS,
    SELECTED_LEARNING_RATE,
    TRAIN_BATCH_SIZE,
    TRAIN_STEPS,
    CouponClippingArm,
    CouponClippingLearningRate,
    build_model_config,
    build_optimizer_config,
)
from experiments.grug.coupon_clipping.model import GrugModelConfig
from experiments.grug.coupon_clipping.train import GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.grug.depth_growth import DepthGrowthConfig
from experiments.grug.moe.launch_datakit_moe_mix import datakit_data_config

_TRAIN_RESOURCES = ResourceConfig.with_gpu("GB200", count=4, cpu=32, ram="256g", disk="256g", replicas=4)
_MIXED_PRECISION = "params=float32,compute=bfloat16,output=bfloat16"
_WANDB_PROJECT = "marin"
_WANDB_GROUP = "cc16-7836"
PILOT_STEPS = 128
EXPERT_AXIS_SIZE = 1

if NUM_EXPERTS % EXPERT_AXIS_SIZE != 0:
    raise AssertionError("the routed expert count must be divisible by the expert axis")
if TRAIN_BATCH_SIZE % 16 != 0:
    raise AssertionError("the global batch must be divisible by all 16 GB200 batch shards")


@dataclass(frozen=True)
class CouponClippingLaunchConfig:
    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    tracker: TrackerConfig
    steps: int
    optimizer_num_train_steps: int = TRAIN_STEPS
    initialize_from: str | None = None
    depth_growth: DepthGrowthConfig | None = None
    learning_rate: CouponClippingLearningRate = SELECTED_LEARNING_RATE
    watch_interval: int = 0


class CouponClippingRunKind(StrEnum):
    FULL = "full"
    PILOT = "pilot"


def run_coupon_clipping_trial(config: CouponClippingLaunchConfig) -> None:
    """Build the fixed Levanter runtime config and dispatch one experiment stage."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
    trainer = TrainerConfig(
        id=config.run_id,
        seed=0,
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=config.steps,
        profiler=ProfilerConfig(enabled=False),
        mp=jmp.get_policy(_MIXED_PRECISION),
        tracker=(
            dataclasses.replace(config.tracker, name=config.run_id)
            if isinstance(config.tracker, WandbConfig)
            else config.tracker
        ),
        watch=WatchConfig(
            watch_targets=["grads"],
            interval=config.watch_interval,
            include_per_parameter_norms=False,
            include_histograms=False,
            split_scan_layers=False,
        ),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=resolve_checkpointer_output_path(
            CheckpointerConfig(save_interval=timedelta(hours=4), keep=None),
            config.output_path,
        ),
        initialize_from=config.initialize_from,
    )
    grug_trainer = GrugTrainerConfig(
        trainer=trainer,
        expert_axis_size=EXPERT_AXIS_SIZE,
        replica_axis_size=1,
        z_loss_weight=1e-4,
        ema_beta=None,
        log_every=1,
    )
    run_grug(
        GrugRunConfig(
            model=config.model,
            data=config.data,
            resources=config.resources,
            optimizer=build_optimizer_config(config.learning_rate),
            optimizer_num_train_steps=config.optimizer_num_train_steps,
            depth_growth=config.depth_growth,
            trainer=grug_trainer,
            eval=None,
            processes_per_task=1,
        )
    )


def build_coupon_clipping_checkpoint(
    arm: CouponClippingArm,
    *,
    version: str | None = None,
    learning_rate: CouponClippingLearningRate = SELECTED_LEARNING_RATE,
) -> ArtifactStep[LevanterCheckpoint]:
    """Assemble one full coupon-clipping arm as a lazy checkpoint artifact."""
    return _build_coupon_clipping_checkpoint(
        arm,
        version=version,
        run_kind=CouponClippingRunKind.FULL,
        learning_rate=learning_rate,
    )


def build_coupon_clipping_pilot_checkpoint(
    arm: CouponClippingArm,
    *,
    version: str | None = None,
    learning_rate: CouponClippingLearningRate = SELECTED_LEARNING_RATE,
) -> ArtifactStep[LevanterCheckpoint]:
    """Assemble one 128-step coupon-clipping pilot artifact."""
    return _build_coupon_clipping_checkpoint(
        arm,
        version=version,
        run_kind=CouponClippingRunKind.PILOT,
        learning_rate=learning_rate,
    )


def _build_coupon_clipping_checkpoint(
    arm: CouponClippingArm,
    *,
    version: str | None,
    run_kind: CouponClippingRunKind,
    learning_rate: CouponClippingLearningRate,
) -> ArtifactStep[LevanterCheckpoint]:
    model = build_model_config(arm)
    if run_kind is CouponClippingRunKind.PILOT:
        steps = PILOT_STEPS
        run_id = f"{arm.value}-pilot128-{learning_rate.value}"
    else:
        steps = TRAIN_STEPS
        if learning_rate is not SELECTED_LEARNING_RATE:
            raise ValueError("full pyramid arms use the learning rate selected by the systems gate")
        run_id = arm.value
    step_name = f"grug/coupon-clipping/{run_id}"
    resolved_version = resolve_version(step_name, version)

    def build_config(ctx: StepContext) -> CouponClippingLaunchConfig:
        data = datakit_data_config(
            total_steps=TRAIN_STEPS,
            batch_size=TRAIN_BATCH_SIZE,
            max_seq_len=model.max_seq_len,
            enable_simulated_epoching=False,
            val_components={},
        )
        return CouponClippingLaunchConfig(
            model=model,
            data=data,
            output_path=ctx.output_path,
            run_id=run_id,
            resources=ctx.runtime_arg("train_resources"),
            tracker=WandbConfig(
                project=_WANDB_PROJECT,
                tags=["grug", "moe", "coupon-clipping", "gb200", arm.value, run_kind.value],
                group=_WANDB_GROUP,
                name=None,
                replicate_path=ctx.output_path,
            ),
            steps=steps,
            learning_rate=learning_rate,
            watch_interval=8 if run_kind is CouponClippingRunKind.PILOT else 0,
        )

    return ArtifactStep(
        name=user_namespaced_name(step_name, resolved_version),
        version=resolved_version,
        artifact_type=LevanterCheckpoint,
        run=run_coupon_clipping_trial,
        build_config=build_config,
        deps=(),
        runtime_args={"train_resources": _TRAIN_RESOURCES},
    )
