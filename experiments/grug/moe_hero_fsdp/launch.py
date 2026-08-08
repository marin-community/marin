# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rack-scaled GB200 launcher for the FSDP MoE hero configuration."""

import dataclasses
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, get_args

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig
from levanter.callbacks.progress_watchdog import ProgressWatchdogConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import BlockShuffleConfig
from levanter.recovery.types import AblationSpec
from levanter.tracker.telemetry import TelemetryConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.data import mixture, tokenized
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.tokenize import TokenizedCache
from rigging.filesystem import prefix_join

from experiments.grug.moe_hero_fsdp.heuristic import build_hero_configs
from experiments.grug.moe_hero_fsdp.model import GrugModelConfig, RematMode, SmallParamSharding
from experiments.grug.moe_hero_fsdp.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe_hero_fsdp.train import (
    GrugAblationSweepConfig,
    GrugRunConfig,
    GrugRunMode,
    GrugTrainerConfig,
    run_grug,
    run_grug_ablation_sweep,
)
from experiments.llama import llama3_tokenizer

DEFAULT_HERO_STEPS = 25
DEFAULT_WANDB_PROJECT = "marin_moe"
HERO_FSDP_BATCH_SIZE = 1024
HERO_NODES_PER_RACK = 16
HERO_GPUS_PER_TASK = 4
HERO_PROCESSES_PER_TASK = HERO_GPUS_PER_TASK
HERO_MIXED_PRECISION = "params=float32,compute=bfloat16,output=bfloat16"
HERO_CHECKPOINT_INTERVAL = timedelta(minutes=30)
# This must exceed XLA's 10-minute collective timeout.
HERO_TRAIN_STEP_TIMEOUT = timedelta(minutes=15)
# Evaluation, checkpointing, and other hooks use this process-wide deadline.
HERO_PROCESS_STALL_TIMEOUT = timedelta(hours=1)
HERO_STALL_DIAGNOSTIC_TIMEOUT = timedelta(seconds=20)
# Grad/param norm reductions run outside the scanned step, cost a visible slice of a short run's
# wall clock, and can require a 117 GiB temporary buffer on this model. The hero default leaves
# them off; --watch-interval re-enables them every N steps for a run that wants the diagnostic.
HERO_WATCH_INTERVAL = 0
# One process writes the XPlane capture. Every rank of a 16-node gang would upload a
# multi-hundred-MB session into the same directory for the same timeline.
HERO_PROFILE_PROCESS_INDEX = 0
HERO_PROFILE_NUM_STEPS = 3

_SLIMPAJAMA_TOKENIZE_RESOURCES = ResourceConfig(ram="64g", disk="64g")
_SLIMPAJAMA_SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=256, perm_type="feistel")


def _slimpajama_6b_dataset() -> ArtifactStep[TokenizedCache]:
    return tokenized(
        "slimpajama-6b",
        source="DKYoon/SlimPajama-6B",
        tokenizer=llama3_tokenizer,
        resources=_SLIMPAJAMA_TOKENIZE_RESOURCES,
        version="2026.06.28",
    )


class HeroThroughputResult(Artifact):
    """Metrics and resumable checkpoints from the rack-scale throughput hero run."""


def _hero_run_config(
    *,
    ctx: StepContext,
    run_id: str,
    batch_size: int,
    num_steps: int,
    model: GrugModelConfig,
    optimizer: GrugMoeMuonHConfig,
    grug_trainer: GrugTrainerConfig,
    wandb_project: str,
    slim: ArtifactStep[TokenizedCache],
    run_mode: GrugRunMode,
    watch_interval: int = HERO_WATCH_INTERVAL,
    profile_start_step: int | None = None,
    profile_num_steps: int = HERO_PROFILE_NUM_STEPS,
) -> GrugRunConfig:
    """Assemble one hero trainer config; ``run_id`` names both the trainer and its W&B run.

    ``profile_start_step`` captures an XPlane trace over ``profile_num_steps`` steps from a single
    process and uploads it to the temp-bucket XProf store; leave it ``None`` to disable profiling.
    ``watch_interval`` re-enables the grad/param norm dumps every N steps and defaults to off.
    """
    trainer = TrainerConfig(
        id=run_id,
        seed=0,
        train_batch_size=batch_size,
        num_train_steps=num_steps,
        profiler=ProfilerConfig(
            enabled=profile_start_step is not None,
            start_step=profile_start_step or 0,
            num_steps=profile_num_steps,
            process_index=HERO_PROFILE_PROCESS_INDEX,
            profile_options=ProfileOptionsConfig(host_tracer_level=1, python_tracer_level=0, enable_hlo_proto=True),
        ),
        mp=jmp.get_policy(HERO_MIXED_PRECISION),
        tracker=(
            WandbConfig(
                entity="marin-community",
                project=wandb_project,
                tags=["grug", "moe", "hero", "fsdp", "gb200"],
                group="moe-hero-fsdp",
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            TelemetryConfig(),
        ),
        # Watch statistics can require a 117 GiB temporary buffer on this model, so leave them off
        # unless the caller opts in with a positive interval.
        watch=WatchConfig(interval=watch_interval) if watch_interval > 0 else WatchConfig(watch_targets=[]),
        progress_watchdog=ProgressWatchdogConfig(
            step_timeout=HERO_TRAIN_STEP_TIMEOUT,
            process_timeout=HERO_PROCESS_STALL_TIMEOUT,
            diagnostic_timeout=HERO_STALL_DIAGNOSTIC_TIMEOUT,
        ),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=prefix_join(ctx.output_path, "checkpoints"),
            temporary_base_path=None,
            save_interval=HERO_CHECKPOINT_INTERVAL,
            keep=None,
            append_run_id_to_base_path=False,
            delete_old_temp_checkpoints=True,
            keep_last_temporary_checkpoints=1,
        ),
    )
    return GrugRunConfig(
        model=model,
        data=mixture(ctx, {slim: 1.0}, shuffle=_SLIMPAJAMA_SHUFFLE),
        resources=ctx.runtime_arg("train_resources"),
        optimizer=optimizer,
        trainer=dataclasses.replace(grug_trainer, trainer=trainer),
        eval=None,
        processes_per_task=HERO_PROCESSES_PER_TASK,
        run_mode=run_mode,
    )


def _hero_resources(dp_racks: int) -> ResourceConfig:
    return ResourceConfig.with_gpu(
        "GB200",
        count=HERO_GPUS_PER_TASK,
        cpu=120,
        ram="850g",
        disk="1t",
        replicas=HERO_NODES_PER_RACK * dp_racks,
    )


def _validate_hero_args(run_id: str, dp_racks: int, num_steps: int) -> None:
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if dp_racks <= 0:
        raise ValueError(f"dp_racks must be positive, got {dp_racks}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")


@dataclass(frozen=True)
class _HeroRunParts:
    batch_size: int
    model: GrugModelConfig
    optimizer: GrugMoeMuonHConfig
    wandb_project: str
    grug_trainer: GrugTrainerConfig
    train_resources: ResourceConfig
    name: str
    version: str
    slim: ArtifactStep[TokenizedCache]


@dataclasses.dataclass(frozen=True)
class HeroOverrides:
    """The hero performance knobs a sweep varies; ``None`` keeps the hero value."""

    expert_chunks: int | None = None
    small_param_sharding: SmallParamSharding | None = None
    remat_mode: RematMode | None = None
    ce_b_block_size: int | None = None


def apply_hero_overrides(model: GrugModelConfig, overrides: HeroOverrides) -> GrugModelConfig:
    """Return ``model`` with every set knob in ``overrides`` applied."""
    fields: dict[str, Any] = {
        name: value
        for name, value in dataclasses.asdict(overrides).items()
        if value is not None and name != "ce_b_block_size"
    }
    if overrides.ce_b_block_size is not None:
        fields["ce_block_sizes"] = dataclasses.replace(model.ce_block_sizes, b_block_size=overrides.ce_b_block_size)
    return dataclasses.replace(model, **fields) if fields else model


def _hero_run_parts(
    *,
    run_id: str,
    dp_racks: int,
    num_steps: int,
    save_checkpoints: bool,
    version: str | None,
    batch_size: int | None = None,
    overrides: HeroOverrides = HeroOverrides(),
) -> _HeroRunParts:
    _validate_hero_args(run_id, dp_racks, num_steps)
    batch_size = batch_size if batch_size is not None else dp_racks * HERO_FSDP_BATCH_SIZE
    model, optimizer = build_hero_configs(num_train_steps=num_steps, batch_size=batch_size)
    model = apply_hero_overrides(model, overrides)
    name = f"grug/{run_id}"
    return _HeroRunParts(
        batch_size=batch_size,
        model=model,
        optimizer=optimizer,
        wandb_project=os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT,
        grug_trainer=GrugTrainerConfig(
            data_seed=None,
            log_every=1,
            ema_beta=None,
            z_loss_weight=1e-4,
            offload_opt_state=True,
            save_checkpoints=save_checkpoints,
            expert_axis_size=1,
            replica_axis_size=dp_racks,
            sharding_dump_path=None,
        ),
        train_resources=_hero_resources(dp_racks),
        name=name,
        version=resolve_version(name, version),
        slim=_slimpajama_6b_dataset(),
    )


def build_hero_run(
    *,
    run_id: str,
    dp_racks: int,
    num_steps: int,
    save_checkpoints: bool = True,
    run_mode: GrugRunMode = GrugRunMode.DEFAULT,
    watch_interval: int = HERO_WATCH_INTERVAL,
    profile_start_step: int | None = None,
    profile_num_steps: int = HERO_PROFILE_NUM_STEPS,
    overrides: HeroOverrides = HeroOverrides(),
    batch_size: int | None = None,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """Build the rack-local FSDP hero throughput run.

    A short throughput gate sets ``save_checkpoints=False``. The final forced checkpoint writes the
    parameters and the offloaded optimizer state, about 2.7 TiB at the hero shape, which a run that
    only reports MFU does not need.

    ``profile_start_step`` captures an XPlane trace over ``profile_num_steps`` steps from process 0
    and uploads it to the temp-bucket XProf store. Start it well after PGLE's own profiling recompile
    so the capture covers the steady-state program.
    """
    if watch_interval < 0:
        raise ValueError(f"watch_interval must be non-negative, got {watch_interval}")
    if profile_start_step is not None and not 0 < profile_start_step < num_steps:
        raise ValueError(f"profile_start_step must fall inside (0, {num_steps}), got {profile_start_step}")

    parts = _hero_run_parts(
        run_id=run_id,
        dp_racks=dp_racks,
        num_steps=num_steps,
        save_checkpoints=save_checkpoints,
        version=version,
        batch_size=batch_size,
        overrides=overrides,
    )
    batch_size = parts.batch_size
    model, optimizer = parts.model, parts.optimizer
    wandb_project = parts.wandb_project
    grug_trainer = parts.grug_trainer
    train_resources = parts.train_resources
    name, version, slim = parts.name, parts.version, parts.slim

    def build_config(ctx: StepContext) -> GrugRunConfig:
        return _hero_run_config(
            ctx=ctx,
            run_id=run_id,
            batch_size=batch_size,
            num_steps=num_steps,
            model=model,
            optimizer=optimizer,
            grug_trainer=grug_trainer,
            wandb_project=wandb_project,
            slim=slim,
            run_mode=run_mode,
            watch_interval=watch_interval,
            profile_start_step=profile_start_step,
            profile_num_steps=profile_num_steps,
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=HeroThroughputResult,
        run=run_grug,
        build_config=build_config,
        deps=(slim,),
        runtime_args={"train_resources": train_resources},
    )


@dataclass(frozen=True)
class HeroSweepArm:
    """One sweep arm: an ``AblationSpec`` for the process env, plus the model-side deltas.

    ``AblationSpec`` carries only process-start environment, which covers NCCL, XLA_FLAGS, and the
    allocator fraction. ``overrides`` and ``batch_size`` reach the knobs that live in the model
    config instead, so one sweep can vary both.
    """

    spec: AblationSpec
    overrides: HeroOverrides = HeroOverrides()
    batch_size: int | None = None


def build_hero_sweep_run(
    *,
    run_id: str,
    dp_racks: int,
    steps_per_arm: int,
    arms: Sequence[HeroSweepArm],
    priority: int,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """Build one allocation that runs ``arms`` back to back.

    Each arm gets a fresh trainer subprocess (so process-start env vars take effect) and its own
    W&B run named ``<run_id>-<arm>``. One arm's fault does not end the sweep. A sweep is a
    diagnostic, so it never checkpoints.

    ``priority`` is the Iris band for the training gang. It rides as a runtime arg, so rescheduling
    the same arms at a different band reuses the cached result rather than rebuilding.
    """
    if not arms:
        raise ValueError("a sweep needs at least one arm")
    parts = _hero_run_parts(
        run_id=run_id, dp_racks=dp_racks, num_steps=steps_per_arm, save_checkpoints=False, version=version
    )
    wandb_project = parts.wandb_project
    grug_trainer = parts.grug_trainer
    train_resources = parts.train_resources
    name, version, slim = parts.name, parts.version, parts.slim
    # The optimizer's LR schedule is derived from the batch, so an arm that changes the batch needs
    # its own pair rather than the allocation's.
    configs = []
    for arm in arms:
        batch_size = arm.batch_size if arm.batch_size is not None else parts.batch_size
        steps = arm.spec.num_steps or steps_per_arm
        model, optimizer = build_hero_configs(num_train_steps=steps, batch_size=batch_size)
        configs.append((arm, batch_size, steps, apply_hero_overrides(model, arm.overrides), optimizer))

    def build_config(ctx: StepContext) -> GrugAblationSweepConfig:
        runs = tuple(
            _hero_run_config(
                ctx=ctx,
                run_id=f"{run_id}-{arm.spec.name}",
                batch_size=batch_size,
                num_steps=steps,
                model=model,
                optimizer=optimizer,
                grug_trainer=grug_trainer,
                wandb_project=wandb_project,
                slim=slim,
                run_mode=GrugRunMode.SUPERVISED,
            )
            for arm, batch_size, steps, model, optimizer in configs
        )
        return GrugAblationSweepConfig(
            run_id=run_id,
            arms=tuple(arm.spec for arm in arms),
            runs=runs,
            resources=ctx.runtime_arg("train_resources"),
            processes_per_task=HERO_PROCESSES_PER_TASK,
            priority=ctx.runtime_arg("priority"),
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=HeroThroughputResult,
        run=run_grug_ablation_sweep,
        build_config=build_config,
        deps=(slim,),
        runtime_args={"train_resources": train_resources, "priority": priority},
    )


@click.command()
@click.option("--run-id", required=True, help="Run identifier for artifact and W&B names.")
@click.option("--dp-racks", type=click.IntRange(min=1), required=True, help="Data-parallel NVL72 rack count.")
@click.option(
    "--num-steps",
    type=click.IntRange(min=1),
    default=DEFAULT_HERO_STEPS,
    show_default=True,
    help="Number of training steps.",
)
@click.option(
    "--save-checkpoints/--no-save-checkpoints",
    default=True,
    show_default=True,
    help="Write resumable checkpoints. Use --no-save-checkpoints for a metrics-only diagnostic.",
)
@click.option(
    "--mode",
    type=click.Choice([mode.value for mode in GrugRunMode]),
    default=GrugRunMode.DEFAULT.value,
    show_default=True,
)
@click.option(
    "--watch-interval",
    type=click.IntRange(min=0),
    default=HERO_WATCH_INTERVAL,
    show_default=True,
    help="Steps between grad/param norm dumps. 0 disables them.",
)
@click.option(
    "--profile-start-step",
    type=click.IntRange(min=1),
    default=None,
    help="First step of the XPlane capture window. Unset disables profiling.",
)
@click.option(
    "--profile-num-steps",
    type=click.IntRange(min=1),
    default=HERO_PROFILE_NUM_STEPS,
    show_default=True,
    help="Steps to capture once --profile-start-step is reached.",
)
@click.option(
    "--expert-chunks",
    type=click.IntRange(min=1),
    default=None,
    help="Expert-weight gather chunks for the sonic_cute MoE. Unset keeps the hero value.",
)
@click.option(
    "--small-param-sharding",
    type=click.Choice(get_args(SmallParamSharding)),
    default=None,
    help="Shard or replicate the router, attention gate, and GatedNorm factors.",
)
@click.option(
    "--remat-mode",
    type=click.Choice(get_args(RematMode)),
    default=None,
    help="Rematerialization policy for the transformer block.",
)
@click.option(
    "--ce-b-block-size",
    type=click.IntRange(min=1),
    default=None,
    help="Token-axis tile for the fused cross-entropy. Unset keeps the hero value.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=None,
    help="Global batch in sequences. Unset derives it as dp_racks x 1024. Must divide the device count.",
)
@build_options
def main(
    run_id: str,
    dp_racks: int,
    num_steps: int,
    save_checkpoints: bool,
    mode: str,
    watch_interval: int,
    profile_start_step: int | None,
    profile_num_steps: int,
    expert_chunks: int | None,
    small_param_sharding: SmallParamSharding | None,
    remat_mode: RematMode | None,
    ce_b_block_size: int | None,
    batch_size: int | None,
) -> ArtifactStep[HeroThroughputResult]:
    return build_hero_run(
        run_id=run_id,
        dp_racks=dp_racks,
        num_steps=num_steps,
        save_checkpoints=save_checkpoints,
        run_mode=GrugRunMode(mode),
        watch_interval=watch_interval,
        profile_start_step=profile_start_step,
        profile_num_steps=profile_num_steps,
        overrides=HeroOverrides(
            expert_chunks=expert_chunks,
            small_param_sharding=small_param_sharding,
            remat_mode=remat_mode,
            ce_b_block_size=ce_b_block_size,
        ),
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()
