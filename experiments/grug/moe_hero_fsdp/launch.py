# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rack-scaled GB200 launcher for the FSDP MoE hero configuration."""

import dataclasses
import os
from dataclasses import dataclass
from datetime import timedelta

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig
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

from experiments.datasets.paloma import paloma_datasets
from experiments.grug.moe_hero_fsdp.heuristic import build_hero_configs
from experiments.grug.moe_hero_fsdp.model import GrugModelConfig
from experiments.grug.moe_hero_fsdp.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe_hero_fsdp.train import (
    GrugAblationSweepConfig,
    GrugEvalConfig,
    GrugRunConfig,
    GrugRunMode,
    GrugTrainerConfig,
    run_grug,
    run_grug_ablation_sweep,
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


# Paloma only, matching the EP launcher. `paloma_dataset` and `uncheatable_dataset` both hardcode a
# `-llama3` cache suffix regardless of the tokenizer argument, so the two suites collide on one cache
# name; paloma alone keeps the eval sets consistent with the EP arm they are compared against.
def _validation_datasets() -> list[ArtifactStep[TokenizedCache]]:
    return list(paloma_datasets(tokenizer=llama3_tokenizer).values())


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
    schedule_steps: int | None = None,
    validation: list[ArtifactStep[TokenizedCache]] | None = None,
    eval_every: int = 0,
    watch_interval: int = HERO_WATCH_INTERVAL,
    profile_start_step: int | None = None,
    profile_num_steps: int = HERO_PROFILE_NUM_STEPS,
) -> GrugRunConfig:
    """Assemble one hero trainer config; ``run_id`` names both the trainer and its W&B run.

    ``profile_start_step`` captures an XPlane trace over ``profile_num_steps`` steps from a single
    process and uploads it to the temp-bucket XProf store; leave it ``None`` to disable profiling.
    ``watch_interval`` re-enables the grad/param norm dumps every N steps and defaults to off.
    ``schedule_steps`` sizes the learning-rate schedule; the run still stops after ``num_steps``.
    """
    if schedule_steps is not None and schedule_steps < num_steps:
        raise ValueError(f"schedule_steps={schedule_steps} must be at least num_steps={num_steps}")
    trainer = TrainerConfig(
        id=run_id,
        seed=0,
        train_batch_size=batch_size,
        # Warmup and decay are fractions of this field, so it carries the whole schedule length.
        # `stop_after_steps` below is what actually bounds the run.
        num_train_steps=schedule_steps if schedule_steps is not None else num_steps,
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
        data=mixture(ctx, {slim: 1.0}, validation=validation or [], shuffle=_SLIMPAJAMA_SHUFFLE),
        resources=ctx.runtime_arg("train_resources"),
        optimizer=optimizer,
        trainer=dataclasses.replace(grug_trainer, trainer=trainer),
        # Off by default so a throughput run stays a throughput run. Turn it on to make a run
        # scoreable: comparing configs needs held-out loss, not train loss.
        eval=(GrugEvalConfig(steps_per_eval=eval_every, eval_ema=False, compute_bpb=True) if eval_every > 0 else None),
        stop_after_steps=num_steps,
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


def _hero_run_parts(
    *,
    run_id: str,
    dp_racks: int,
    num_steps: int,
    save_checkpoints: bool,
    version: str | None,
    schedule_steps: int | None = None,
) -> _HeroRunParts:
    _validate_hero_args(run_id, dp_racks, num_steps)
    if schedule_steps is not None and schedule_steps <= 0:
        raise ValueError(f"schedule_steps must be positive, got {schedule_steps}")
    batch_size = dp_racks * HERO_FSDP_BATCH_SIZE
    # The optimizer heuristic scales learning rate, adam_lr, and epsilon from the token budget
    # implied by `num_train_steps * batch * seq`, so `schedule_steps` sets the budget the schedule is
    # built for while `num_steps` bounds the run. That trains the head of a long run's schedule at
    # the rate the long run would use. An EP/FSDP pair must pass the same value or the two arms train
    # at different rates. Default keeps them equal, the previous behavior.
    model, optimizer = build_hero_configs(
        num_train_steps=schedule_steps if schedule_steps is not None else num_steps,
        batch_size=batch_size,
    )
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
    schedule_steps: int | None = None,
    eval_every: int = 0,
    save_checkpoints: bool = True,
    run_mode: GrugRunMode = GrugRunMode.DEFAULT,
    watch_interval: int = HERO_WATCH_INTERVAL,
    profile_start_step: int | None = None,
    profile_num_steps: int = HERO_PROFILE_NUM_STEPS,
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
        schedule_steps=schedule_steps,
        save_checkpoints=save_checkpoints,
        version=version,
    )
    batch_size = parts.batch_size
    model, optimizer = parts.model, parts.optimizer
    wandb_project = parts.wandb_project
    grug_trainer = parts.grug_trainer
    train_resources = parts.train_resources
    name, version, slim = parts.name, parts.version, parts.slim
    validation = _validation_datasets() if eval_every > 0 else []

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
            schedule_steps=schedule_steps,
            validation=validation,
            eval_every=eval_every,
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
        deps=(slim, *validation),
        runtime_args={"train_resources": train_resources},
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
    arms = tuple(selected_ablations(environment_ablations(num_steps=steps_per_arm), ablation_names))
    parts = _hero_run_parts(
        run_id=run_id, dp_racks=dp_racks, num_steps=steps_per_arm, save_checkpoints=False, version=version
    )
    batch_size = parts.batch_size
    model, optimizer = parts.model, parts.optimizer
    wandb_project = parts.wandb_project
    grug_trainer = parts.grug_trainer
    train_resources = parts.train_resources
    name, version, slim = parts.name, parts.version, parts.slim

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
                run_mode=GrugRunMode.SUPERVISED,
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
    "--eval-every",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Run the paloma suite every N steps. 0 disables evaluation.",
)
@click.option(
    "--schedule-steps",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "Build the learning-rate schedule for this many steps instead of --num-steps. The optimizer "
        "heuristic scales its rates from the implied token budget, so this trains the head of a long "
        "run's schedule. Defaults to --num-steps."
    ),
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
@build_options
def main(
    run_id: str,
    dp_racks: int,
    num_steps: int,
    schedule_steps: int | None,
    eval_every: int,
    save_checkpoints: bool,
    mode: str,
    watch_interval: int,
    profile_start_step: int | None,
    profile_num_steps: int,
) -> ArtifactStep[HeroThroughputResult]:
    return build_hero_run(
        run_id=run_id,
        dp_racks=dp_racks,
        num_steps=num_steps,
        schedule_steps=schedule_steps,
        eval_every=eval_every,
        save_checkpoints=save_checkpoints,
        run_mode=GrugRunMode(mode),
        watch_interval=watch_interval,
        profile_start_step=profile_start_step,
        profile_num_steps=profile_num_steps,
    )


if __name__ == "__main__":
    main()
