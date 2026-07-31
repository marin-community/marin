# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""L1-to-L48 artifact pipelines for the 16-GB200 coupon-clipping experiment."""

import dataclasses

from levanter.tracker.wandb import WandbConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint
from rigging.filesystem import prefix_join

from experiments.grug.coupon_clipping.config import (
    AVERAGE_SHARED_INTERMEDIATE_DIM,
    DEPTH_GROWTH_CONFIG,
    DEPTH_SOURCE_LAYERS,
    DEPTH_TRANSITION_STEP,
    TRAIN_BATCH_SIZE,
    TRAIN_STEPS,
    CouponClippingArm,
    build_model_config,
)
from experiments.grug.coupon_clipping.launch import (
    _TRAIN_RESOURCES,
    _WANDB_GROUP,
    _WANDB_PROJECT,
    CouponClippingLaunchConfig,
    CouponClippingRunKind,
    run_coupon_clipping_trial,
)
from experiments.grug.coupon_clipping.model import GrugModelConfig
from experiments.grug.depth_growth import DepthGrowthConfig
from experiments.grug.moe.launch_datakit_moe_mix import datakit_data_config

PILOT_SOURCE_STEPS = 32
PILOT_GROWN_STEPS = 16
L1_PILOT_STEPS = 128


def build_depth_source_model_config(source_layers: int = DEPTH_SOURCE_LAYERS) -> GrugModelConfig:
    """Build the uniform shallow source while retaining the production width and kernels."""
    return dataclasses.replace(
        build_model_config(CouponClippingArm.C0_P0),
        num_layers=source_layers,
        block_segment_lengths=(source_layers,),
        block_segment_shared_expert_intermediate_dims=(AVERAGE_SHARED_INTERMEDIATE_DIM,),
    )


def _build_source_checkpoint(
    *,
    run_id: str,
    steps: int,
    version: str | None,
    run_kind: CouponClippingRunKind,
) -> ArtifactStep[LevanterCheckpoint]:
    model = build_depth_source_model_config()
    step_name = f"grug/coupon-clipping/{run_id}"
    resolved_version = resolve_version(step_name, version)

    def build_config(ctx: StepContext) -> CouponClippingLaunchConfig:
        return CouponClippingLaunchConfig(
            model=model,
            data=datakit_data_config(
                total_steps=TRAIN_STEPS,
                batch_size=TRAIN_BATCH_SIZE,
                max_seq_len=model.max_seq_len,
                enable_simulated_epoching=False,
                val_components={},
            ),
            output_path=ctx.output_path,
            run_id=run_id,
            resources=ctx.runtime_arg("train_resources"),
            tracker=WandbConfig(
                project=_WANDB_PROJECT,
                tags=["grug", "moe", "coupon-clipping", "gb200", "depth-source", run_kind.value],
                group=_WANDB_GROUP,
                name=None,
                replicate_path=ctx.output_path,
            ),
            steps=steps,
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


def _build_growth_target_checkpoint(
    source: ArtifactStep[LevanterCheckpoint] | None,
    *,
    run_id: str,
    steps: int,
    growth: DepthGrowthConfig,
    version: str | None,
    run_kind: CouponClippingRunKind,
    source_checkpoint_root: str | None = None,
) -> ArtifactStep[LevanterCheckpoint]:
    if (source is None) == (source_checkpoint_root is None):
        raise ValueError("exactly one source artifact or checkpoint root is required")
    model = build_model_config(CouponClippingArm.C0_P0)
    step_name = f"grug/coupon-clipping/{run_id}"
    resolved_version = resolve_version(step_name, version)

    def build_config(ctx: StepContext) -> CouponClippingLaunchConfig:
        initialize_from = (
            prefix_join(ctx.artifact_path(source), "checkpoints") if source is not None else source_checkpoint_root
        )
        if initialize_from is None:
            raise AssertionError("source checkpoint root was validated before building the target")
        return CouponClippingLaunchConfig(
            model=model,
            data=datakit_data_config(
                total_steps=TRAIN_STEPS,
                batch_size=TRAIN_BATCH_SIZE,
                max_seq_len=model.max_seq_len,
                enable_simulated_epoching=False,
                val_components={},
            ),
            output_path=ctx.output_path,
            run_id=run_id,
            resources=ctx.runtime_arg("train_resources"),
            tracker=WandbConfig(
                project=_WANDB_PROJECT,
                tags=["grug", "moe", "coupon-clipping", "gb200", "depth-growth", run_kind.value],
                group=_WANDB_GROUP,
                name=None,
                replicate_path=ctx.output_path,
            ),
            steps=steps,
            initialize_from=initialize_from,
            depth_growth=growth,
            watch_interval=8 if run_kind is CouponClippingRunKind.PILOT else 0,
        )

    return ArtifactStep(
        name=user_namespaced_name(step_name, resolved_version),
        version=resolved_version,
        artifact_type=LevanterCheckpoint,
        run=run_coupon_clipping_trial,
        build_config=build_config,
        deps=(source,) if source is not None else (),
        runtime_args={"train_resources": _TRAIN_RESOURCES},
    )


def build_l1_pilot_checkpoint(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Run 128 L1 updates against the production optimizer and data horizons."""
    return _build_source_checkpoint(
        run_id="cc16-l1-pilot128",
        steps=L1_PILOT_STEPS,
        version=version,
        run_kind=CouponClippingRunKind.PILOT,
    )


def build_growth_pilot_checkpoint(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Run 32 L1 updates, grow to L48, and run 16 more updates."""
    source = _build_source_checkpoint(
        run_id="cc16-growth-pilot-source32",
        steps=PILOT_SOURCE_STEPS,
        version=version,
        run_kind=CouponClippingRunKind.PILOT,
    )
    growth = DepthGrowthConfig(
        source_layers=DEPTH_SOURCE_LAYERS,
        target_layers=build_model_config(CouponClippingArm.C0_P0).num_layers,
        expected_step=PILOT_SOURCE_STEPS,
        expected_data_offset=PILOT_SOURCE_STEPS * TRAIN_BATCH_SIZE,
    )
    return _build_growth_target_checkpoint(
        source,
        run_id="cc16-growth-pilot-l1-to-l48-16",
        steps=PILOT_SOURCE_STEPS + PILOT_GROWN_STEPS,
        growth=growth,
        version=version,
        run_kind=CouponClippingRunKind.PILOT,
    )


def build_growth_target_only_checkpoint(
    *,
    source_checkpoint_root: str,
    version: str | None = None,
) -> ArtifactStep[LevanterCheckpoint]:
    """Recover the pilot target from an already-complete L1 source checkpoint root."""
    growth = DepthGrowthConfig(
        source_layers=DEPTH_SOURCE_LAYERS,
        target_layers=build_model_config(CouponClippingArm.C0_P0).num_layers,
        expected_step=PILOT_SOURCE_STEPS,
        expected_data_offset=PILOT_SOURCE_STEPS * TRAIN_BATCH_SIZE,
    )
    return _build_growth_target_checkpoint(
        None,
        run_id="cc16-growth-pilot-l1-to-l48-16-recovery",
        steps=PILOT_SOURCE_STEPS + PILOT_GROWN_STEPS,
        growth=growth,
        version=version,
        run_kind=CouponClippingRunKind.PILOT,
        source_checkpoint_root=source_checkpoint_root,
    )


def build_d1_checkpoint(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Build the token-matched D1 pipeline with a 70% L1 transition."""
    source = _build_source_checkpoint(
        run_id=f"cc16-d1-l1-source-step{DEPTH_TRANSITION_STEP}",
        steps=DEPTH_TRANSITION_STEP,
        version=version,
        run_kind=CouponClippingRunKind.FULL,
    )
    return _build_growth_target_checkpoint(
        source,
        run_id="cc16-d1-l1-to-l48",
        steps=TRAIN_STEPS,
        growth=DEPTH_GROWTH_CONFIG,
        version=version,
        run_kind=CouponClippingRunKind.FULL,
    )
