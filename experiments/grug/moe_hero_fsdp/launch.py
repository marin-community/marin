# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rack-scaled GB200 launcher for the FSDP MoE hero configuration."""

import dataclasses
import os
from collections.abc import Callable
from datetime import timedelta

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.progress_watchdog import ProgressWatchdogConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import BlockShuffleConfig
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
from experiments.grug.moe_hero_fsdp.train import (
    GrugAblationSweepConfig,
    GrugRunConfig,
    GrugTrainerConfig,
    run_grug,
    run_grug_ablation_sweep,
    run_grug_failsafe_control,
    run_grug_stock_control,
    run_grug_supervised,
)
from experiments.grug.recovery.ablation_catalog import environment_ablations, selected_ablations
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
    model,
    optimizer,
    grug_trainer: GrugTrainerConfig,
    wandb_project: str,
    slim: ArtifactStep[TokenizedCache],
) -> GrugRunConfig:
    """Assemble one hero trainer config; ``run_id`` names both the trainer and its W&B run."""
    trainer = TrainerConfig(
        id=run_id,
        seed=0,
        train_batch_size=batch_size,
        num_train_steps=num_steps,
        profiler=ProfilerConfig(enabled=False, start_step=8, num_steps=0),
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
        # Watch stats are off. The `compute_watch` variant of the train step needs a 117 GiB
        # buffer that the cuda_async pool cannot always satisfy once fragmented, so at 2 racks
        # the run dies on a watch step instead of on whatever it was there to measure.
        watch=WatchConfig(watch_targets=[]),
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


def _build_hero_run(
    *,
    run_id: str,
    dp_racks: int,
    num_steps: int,
    save_checkpoints: bool,
    run: Callable[[GrugRunConfig], None],
    version: str | None,
) -> ArtifactStep[HeroThroughputResult]:
    """Build the rack-local FSDP hero throughput run.

    A short throughput gate sets ``save_checkpoints=False``. The final forced checkpoint writes the
    parameters and the offloaded optimizer state, about 2.7 TiB at the hero shape, which a run that
    only reports MFU does not need.
    """
    _validate_hero_args(run_id, dp_racks, num_steps)

    batch_size = dp_racks * HERO_FSDP_BATCH_SIZE
    model, optimizer = build_hero_configs(num_train_steps=num_steps, batch_size=batch_size)
    wandb_project = os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT
    grug_trainer = GrugTrainerConfig(
        data_seed=None,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        offload_opt_state=True,
        save_checkpoints=save_checkpoints,
        expert_axis_size=1,
        replica_axis_size=dp_racks,
        sharding_dump_path=None,
    )
    train_resources = _hero_resources(dp_racks)
    name = f"grug/{run_id}"
    version = resolve_version(name, version)
    slim = _slimpajama_6b_dataset()

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
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=HeroThroughputResult,
        run=run,
        build_config=build_config,
        deps=(slim,),
        runtime_args={"train_resources": train_resources},
    )


def build_hero_run(
    *, run_id: str, dp_racks: int, num_steps: int, save_checkpoints: bool = True, version: str | None = None
) -> ArtifactStep[HeroThroughputResult]:
    """Build the ordinary rack-local FSDP hero run."""
    return _build_hero_run(
        run_id=run_id,
        dp_racks=dp_racks,
        num_steps=num_steps,
        save_checkpoints=save_checkpoints,
        run=run_grug,
        version=version,
    )


def build_supervised_hero_run(
    *, run_id: str, dp_racks: int, num_steps: int, save_checkpoints: bool = True, version: str | None = None
) -> ArtifactStep[HeroThroughputResult]:
    """Build the rack-local FSDP hero run with one crash supervisor per task."""
    return _build_hero_run(
        run_id=run_id,
        dp_racks=dp_racks,
        num_steps=num_steps,
        save_checkpoints=save_checkpoints,
        run=run_grug_supervised,
        version=version,
    )


def build_stock_control_hero_run(
    *, run_id: str, dp_racks: int, num_steps: int, save_checkpoints: bool = True, version: str | None = None
) -> ArtifactStep[HeroThroughputResult]:
    """Build the hero run with no recovery instrumentation, for measuring what it costs the hazard."""
    return _build_hero_run(
        run_id=run_id,
        dp_racks=dp_racks,
        num_steps=num_steps,
        save_checkpoints=save_checkpoints,
        run=run_grug_stock_control,
        version=version,
    )


def build_ablation_sweep_hero_run(
    *,
    run_id: str,
    dp_racks: int,
    steps_per_arm: int,
    ablation_names: tuple[str, ...],
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """Build one allocation that runs the named environment arms back to back.

    Each arm gets a fresh trainer subprocess (so process-start env vars take effect) and its
    own W&B run named ``<run_id>-<arm>``. A sweep is a diagnostic, so it never checkpoints.
    """
    _validate_hero_args(run_id, dp_racks, steps_per_arm)
    arms = tuple(selected_ablations(environment_ablations(num_steps=steps_per_arm), ablation_names))

    batch_size = dp_racks * HERO_FSDP_BATCH_SIZE
    model, optimizer = build_hero_configs(num_train_steps=steps_per_arm, batch_size=batch_size)
    wandb_project = os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT
    grug_trainer = GrugTrainerConfig(
        data_seed=None,
        log_every=1,
        ema_beta=None,
        z_loss_weight=1e-4,
        offload_opt_state=True,
        save_checkpoints=False,
        expert_axis_size=1,
        replica_axis_size=dp_racks,
        sharding_dump_path=None,
    )
    train_resources = _hero_resources(dp_racks)
    name = f"grug/{run_id}"
    version = resolve_version(name, version)
    slim = _slimpajama_6b_dataset()

    def build_config(ctx: StepContext) -> GrugAblationSweepConfig:
        runs = tuple(
            _hero_run_config(
                ctx=ctx,
                run_id=f"{run_id}-{arm.name}",
                batch_size=batch_size,
                num_steps=arm.num_steps or steps_per_arm,
                model=model,
                optimizer=optimizer,
                grug_trainer=grug_trainer,
                wandb_project=wandb_project,
                slim=slim,
            )
            for arm in arms
        )
        return GrugAblationSweepConfig(
            run_id=run_id,
            arms=arms,
            runs=runs,
            resources=ctx.runtime_arg("train_resources"),
            processes_per_task=HERO_PROCESSES_PER_TASK,
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=HeroThroughputResult,
        run=run_grug_ablation_sweep,
        build_config=build_config,
        deps=(slim,),
        runtime_args={"train_resources": train_resources},
    )


def build_failsafe_control_hero_run(
    *, run_id: str, dp_racks: int, num_steps: int, save_checkpoints: bool = True, version: str | None = None
) -> ArtifactStep[HeroThroughputResult]:
    """Build the hero run with XLA failsafes but no recovery supervisor parent."""
    return _build_hero_run(
        run_id=run_id,
        dp_racks=dp_racks,
        num_steps=num_steps,
        save_checkpoints=save_checkpoints,
        run=run_grug_failsafe_control,
        version=version,
    )


def hero_launch_command(
    build_run: Callable[..., ArtifactStep[HeroThroughputResult]],
) -> click.Command:
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
    @build_options
    def command(
        run_id: str, dp_racks: int, num_steps: int, save_checkpoints: bool
    ) -> ArtifactStep[HeroThroughputResult]:
        return build_run(
            run_id=run_id,
            dp_racks=dp_racks,
            num_steps=num_steps,
            save_checkpoints=save_checkpoints,
        )

    return command


main = hero_launch_command(build_hero_run)


if __name__ == "__main__":
    main()
