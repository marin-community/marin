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
    AGGRESSIVE_DECAY_STEPS,
    AGGRESSIVE_GROWTH_CONFIG,
    AGGRESSIVE_TRANSITION_STEP,
    AVERAGE_SHARED_INTERMEDIATE_DIM,
    DECAY_STEPS,
    DEPTH_GROWTH_CONFIG,
    DEPTH_SOURCE_LAYERS,
    DEPTH_TRANSITION_STEP,
    EXTREME_DECAY_STEPS,
    EXTREME_GROWTH_CONFIG,
    EXTREME_TRANSITION_STEP,
    L4_DECAY_STEPS,
    L4_GROWTH_CONFIG,
    L4_TRANSITION_STEP,
    RANDOM_LAYER_DROPOUT_COUNT,
    RANDOM_LAYER_DROPOUT_GROWTH_CONFIG,
    SEGMENT_LENGTHS,
    TRAIN_BATCH_SIZE,
    TRAIN_STEPS,
    CouponClippingArm,
    build_growth_source_model_config,
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


def build_aggressive_source_model_config(
    growth: DepthGrowthConfig = AGGRESSIVE_GROWTH_CONFIG,
) -> GrugModelConfig:
    """Build the d1536/L1 source used by the >5x arm."""
    return build_growth_source_model_config(build_model_config(CouponClippingArm.C0_P0), growth)


def build_extreme_target_model_config() -> GrugModelConfig:
    """Build the WD2 target while preserving its expert widths and Q/KV head ratio."""
    return dataclasses.replace(
        build_model_config(CouponClippingArm.C0_P0),
        shared_expert_intermediate_dim=1536,
        num_kv_heads=8,
        block_segment_shared_expert_intermediate_dims=(1536,) * len(SEGMENT_LENGTHS),
    )


def build_extreme_source_model_config() -> GrugModelConfig:
    """Build the d768/L1 source with an exact factor-four path to the WD2 target."""
    target = build_extreme_target_model_config()
    return dataclasses.replace(
        target,
        hidden_dim=768,
        intermediate_dim=1536,
        shared_expert_intermediate_dim=1536,
        num_layers=1,
        num_heads=6,
        num_kv_heads=2,
        initializer_std=target.initializer_std * 2,
        block_segment_lengths=(1,),
        block_segment_shared_expert_intermediate_dims=(1536,),
    )


def build_l4_source_model_config(growth: DepthGrowthConfig = L4_GROWTH_CONFIG) -> GrugModelConfig:
    """Build the physical d1536/L4 efficiency control for the 80/20 arm."""
    return build_growth_source_model_config(build_model_config(CouponClippingArm.C0_P0), growth)


def build_random_layer_dropout_source_model_config(
    growth: DepthGrowthConfig = RANDOM_LAYER_DROPOUT_GROWTH_CONFIG,
) -> GrugModelConfig:
    """Build the d1536/L48 source that executes four uniformly sampled layers per update."""
    return build_growth_source_model_config(
        build_model_config(CouponClippingArm.C0_P0),
        growth,
    )


def _build_source_checkpoint(
    *,
    model: GrugModelConfig,
    run_id: str,
    steps: int,
    version: str | None,
    run_kind: CouponClippingRunKind,
    optimizer_decay_steps: int = DECAY_STEPS,
    random_layer_dropout_count: int | None = None,
) -> ArtifactStep[LevanterCheckpoint]:
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
            optimizer_decay_steps=optimizer_decay_steps,
            watch_interval=8 if run_kind is CouponClippingRunKind.PILOT else 0,
            random_layer_dropout_count=random_layer_dropout_count,
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
    model: GrugModelConfig,
    run_id: str,
    steps: int,
    growth: DepthGrowthConfig,
    version: str | None,
    run_kind: CouponClippingRunKind,
    source_checkpoint_root: str | None = None,
    optimizer_decay_steps: int = DECAY_STEPS,
) -> ArtifactStep[LevanterCheckpoint]:
    if (source is None) == (source_checkpoint_root is None):
        raise ValueError("exactly one source artifact or checkpoint root is required")
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
            optimizer_decay_steps=optimizer_decay_steps,
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
        model=build_depth_source_model_config(),
        run_id="cc16-l1-pilot128",
        steps=L1_PILOT_STEPS,
        version=version,
        run_kind=CouponClippingRunKind.PILOT,
    )


def build_growth_pilot_checkpoint(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Run 32 L1 updates, grow to L48, and run 16 more updates."""
    source = _build_source_checkpoint(
        model=build_depth_source_model_config(),
        run_id="cc16-growth-pilot-source32",
        steps=PILOT_SOURCE_STEPS,
        version=version,
        run_kind=CouponClippingRunKind.PILOT,
    )
    growth = DepthGrowthConfig(
        source_layers=DEPTH_SOURCE_LAYERS,
        target_layers=build_model_config(CouponClippingArm.C0_P0).num_layers,
        width_expansion_factor=1,
        new_layer_initialization=DEPTH_GROWTH_CONFIG.new_layer_initialization,
        expected_step=PILOT_SOURCE_STEPS,
        expected_data_offset=PILOT_SOURCE_STEPS * TRAIN_BATCH_SIZE,
    )
    return _build_growth_target_checkpoint(
        source,
        model=build_model_config(CouponClippingArm.C0_P0),
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
        width_expansion_factor=1,
        new_layer_initialization=DEPTH_GROWTH_CONFIG.new_layer_initialization,
        expected_step=PILOT_SOURCE_STEPS,
        expected_data_offset=PILOT_SOURCE_STEPS * TRAIN_BATCH_SIZE,
    )
    return _build_growth_target_checkpoint(
        None,
        model=build_model_config(CouponClippingArm.C0_P0),
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
        model=build_depth_source_model_config(),
        run_id=f"cc16-d1-l1-source-step{DEPTH_TRANSITION_STEP}",
        steps=DEPTH_TRANSITION_STEP,
        version=version,
        run_kind=CouponClippingRunKind.FULL,
    )
    return _build_growth_target_checkpoint(
        source,
        model=build_model_config(CouponClippingArm.C0_P0),
        run_id="cc16-d1-l1-to-l48",
        steps=TRAIN_STEPS,
        growth=DEPTH_GROWTH_CONFIG,
        version=version,
        run_kind=CouponClippingRunKind.FULL,
    )


def build_aggressive_source_pilot_checkpoint(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Measure the d1536/L1 source throughput for 128 updates."""
    return _build_source_checkpoint(
        model=build_aggressive_source_model_config(),
        run_id="ccx-wd1-d1536-l1-pilot128",
        steps=L1_PILOT_STEPS,
        version=version,
        run_kind=CouponClippingRunKind.PILOT,
    )


def build_extreme_source_pilot_checkpoint(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Measure the d768/L1 source with fixed expert count and wider selected experts."""
    return _build_source_checkpoint(
        model=build_extreme_source_model_config(),
        run_id="ccx-wd2-d768-l1-i1536-pilot128",
        steps=L1_PILOT_STEPS,
        version=version,
        run_kind=CouponClippingRunKind.PILOT,
    )


def build_l4_source_pilot_checkpoint(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Measure the physical d1536/L4 efficiency control for 128 updates."""
    return _build_source_checkpoint(
        model=build_l4_source_model_config(),
        run_id="ccx-l4-d1536-l4-pilot128",
        steps=L1_PILOT_STEPS,
        version=version,
        run_kind=CouponClippingRunKind.PILOT,
    )


def build_random_layer_dropout_source_pilot_checkpoint(
    *,
    version: str | None = None,
) -> ArtifactStep[LevanterCheckpoint]:
    """Measure d1536/L48 storage with four uniformly sampled active layers."""
    return _build_source_checkpoint(
        model=build_random_layer_dropout_source_model_config(),
        run_id="ccx-ld4-d1536-l48-sample4-pilot128",
        steps=L1_PILOT_STEPS,
        version=version,
        run_kind=CouponClippingRunKind.PILOT,
        random_layer_dropout_count=RANDOM_LAYER_DROPOUT_COUNT,
    )


def build_l4_growth_pilot_checkpoint(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Canary physical d1536/L4 growth to d3072/L48."""
    growth = dataclasses.replace(
        L4_GROWTH_CONFIG,
        expected_step=PILOT_SOURCE_STEPS,
        expected_data_offset=PILOT_SOURCE_STEPS * TRAIN_BATCH_SIZE,
    )
    source = _build_source_checkpoint(
        model=build_l4_source_model_config(growth),
        run_id="ccx-l4-growth-pilot-source32",
        steps=PILOT_SOURCE_STEPS,
        version=version,
        run_kind=CouponClippingRunKind.PILOT,
    )
    return _build_growth_target_checkpoint(
        source,
        model=build_model_config(CouponClippingArm.C0_P0),
        run_id="ccx-l4-growth-pilot-target16",
        steps=PILOT_SOURCE_STEPS + PILOT_GROWN_STEPS,
        growth=growth,
        version=version,
        run_kind=CouponClippingRunKind.PILOT,
    )


def build_random_layer_dropout_growth_pilot_checkpoint(
    *,
    version: str | None = None,
) -> ArtifactStep[LevanterCheckpoint]:
    """Canary sample-four d1536/L48 width growth to full d3072/L48."""
    growth = dataclasses.replace(
        RANDOM_LAYER_DROPOUT_GROWTH_CONFIG,
        expected_step=PILOT_SOURCE_STEPS,
        expected_data_offset=PILOT_SOURCE_STEPS * TRAIN_BATCH_SIZE,
    )
    source = _build_source_checkpoint(
        model=build_random_layer_dropout_source_model_config(growth),
        run_id="ccx-ld4-growth-pilot-source32",
        steps=PILOT_SOURCE_STEPS,
        version=version,
        run_kind=CouponClippingRunKind.PILOT,
        random_layer_dropout_count=RANDOM_LAYER_DROPOUT_COUNT,
    )
    return _build_growth_target_checkpoint(
        source,
        model=build_model_config(CouponClippingArm.C0_P0),
        run_id="ccx-ld4-growth-pilot-target16",
        steps=PILOT_SOURCE_STEPS + PILOT_GROWN_STEPS,
        growth=growth,
        version=version,
        run_kind=CouponClippingRunKind.PILOT,
    )


def build_aggressive_growth_pilot_checkpoint(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Canary width-and-depth growth after a 32-update d1536/L1 source."""
    growth = dataclasses.replace(
        AGGRESSIVE_GROWTH_CONFIG,
        expected_step=PILOT_SOURCE_STEPS,
        expected_data_offset=PILOT_SOURCE_STEPS * TRAIN_BATCH_SIZE,
    )
    source = _build_source_checkpoint(
        model=build_aggressive_source_model_config(growth),
        run_id="ccx-wd1-growth-pilot-source32",
        steps=PILOT_SOURCE_STEPS,
        version=version,
        run_kind=CouponClippingRunKind.PILOT,
    )
    return _build_growth_target_checkpoint(
        source,
        model=build_model_config(CouponClippingArm.C0_P0),
        run_id="ccx-wd1-growth-pilot-target16",
        steps=PILOT_SOURCE_STEPS + PILOT_GROWN_STEPS,
        growth=growth,
        version=version,
        run_kind=CouponClippingRunKind.PILOT,
    )


def build_aggressive_checkpoint(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Build the 95% d1536/L1 to 5% d3072/L48 aggressive growth arm."""
    source = _build_source_checkpoint(
        model=build_aggressive_source_model_config(),
        run_id=f"ccx-wd1-d1536-l1-source-step{AGGRESSIVE_TRANSITION_STEP}",
        steps=AGGRESSIVE_TRANSITION_STEP,
        version=version,
        run_kind=CouponClippingRunKind.FULL,
        optimizer_decay_steps=AGGRESSIVE_DECAY_STEPS,
    )
    return _build_growth_target_checkpoint(
        source,
        model=build_model_config(CouponClippingArm.C0_P0),
        run_id="ccx-wd1-d1536-l1-to-d3072-l48",
        steps=TRAIN_STEPS,
        growth=AGGRESSIVE_GROWTH_CONFIG,
        version=version,
        run_kind=CouponClippingRunKind.FULL,
        optimizer_decay_steps=AGGRESSIVE_DECAY_STEPS,
    )


def build_extreme_growth_pilot_checkpoint(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Canary exact factor-four width and depth growth after 32 WD2 source updates."""
    growth = dataclasses.replace(
        EXTREME_GROWTH_CONFIG,
        expected_step=PILOT_SOURCE_STEPS,
        expected_data_offset=PILOT_SOURCE_STEPS * TRAIN_BATCH_SIZE,
    )
    source = _build_source_checkpoint(
        model=build_extreme_source_model_config(),
        run_id="ccx-wd2-growth-pilot-source32",
        steps=PILOT_SOURCE_STEPS,
        version=version,
        run_kind=CouponClippingRunKind.PILOT,
    )
    return _build_growth_target_checkpoint(
        source,
        model=build_extreme_target_model_config(),
        run_id="ccx-wd2-growth-pilot-target16",
        steps=PILOT_SOURCE_STEPS + PILOT_GROWN_STEPS,
        growth=growth,
        version=version,
        run_kind=CouponClippingRunKind.PILOT,
    )


def build_extreme_checkpoint(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Build the 90% WD2 source to 10% wide/deep target arm."""
    source = _build_source_checkpoint(
        model=build_extreme_source_model_config(),
        run_id=f"ccx-wd2-d768-l1-i1536-source-step{EXTREME_TRANSITION_STEP}",
        steps=EXTREME_TRANSITION_STEP,
        version=version,
        run_kind=CouponClippingRunKind.FULL,
        optimizer_decay_steps=EXTREME_DECAY_STEPS,
    )
    return _build_growth_target_checkpoint(
        source,
        model=build_extreme_target_model_config(),
        run_id="ccx-wd2-d768-l1-to-d3072-l48-tail640",
        steps=TRAIN_STEPS,
        growth=EXTREME_GROWTH_CONFIG,
        version=version,
        run_kind=CouponClippingRunKind.FULL,
        optimizer_decay_steps=EXTREME_DECAY_STEPS,
    )


def build_l4_checkpoint(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Build the 80% physical d1536/L4 to 20% d3072/L48 efficiency control."""
    source = _build_source_checkpoint(
        model=build_l4_source_model_config(),
        run_id=f"ccx-l4-d1536-l4-source-step{L4_TRANSITION_STEP}",
        steps=L4_TRANSITION_STEP,
        version=version,
        run_kind=CouponClippingRunKind.FULL,
        optimizer_decay_steps=L4_DECAY_STEPS,
    )
    return _build_growth_target_checkpoint(
        source,
        model=build_model_config(CouponClippingArm.C0_P0),
        run_id="ccx-l4-d1536-l4-to-d3072-l48-tail1280",
        steps=TRAIN_STEPS,
        growth=L4_GROWTH_CONFIG,
        version=version,
        run_kind=CouponClippingRunKind.FULL,
        optimizer_decay_steps=L4_DECAY_STEPS,
    )


def build_random_layer_dropout_checkpoint(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Build the 80% sample-4 d1536/L48 to 20% full d3072/L48 arm."""
    source = _build_source_checkpoint(
        model=build_random_layer_dropout_source_model_config(),
        run_id=f"ccx-ld4-d1536-l48-sample4-source-step{L4_TRANSITION_STEP}",
        steps=L4_TRANSITION_STEP,
        version=version,
        run_kind=CouponClippingRunKind.FULL,
        optimizer_decay_steps=L4_DECAY_STEPS,
        random_layer_dropout_count=RANDOM_LAYER_DROPOUT_COUNT,
    )
    return _build_growth_target_checkpoint(
        source,
        model=build_model_config(CouponClippingArm.C0_P0),
        run_id="ccx-ld4-d1536-l48-sample4-to-d3072-l48-tail1280",
        steps=TRAIN_STEPS,
        growth=RANDOM_LAYER_DROPOUT_GROWTH_CONFIG,
        version=version,
        run_kind=CouponClippingRunKind.FULL,
        optimizer_decay_steps=L4_DECAY_STEPS,
    )
