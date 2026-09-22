# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-rack hero benchmarks restored from a completed production checkpoint."""

import dataclasses
import json
import signal
from collections.abc import Callable

import click
import jax
from iris.runtime.jax_init import initialize_jax
from levanter.checkpoint import latest_checkpoint_path
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.utils.jax_utils import multihost_broadcast_sync
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe_hero_ep.hero_recipe import HeroThroughputResult
from experiments.grug.moe_hero_ep.heuristic import build_hero_configs
from experiments.grug.moe_hero_ep.launch_diagnostics import build_diagnostic_run
from experiments.grug.moe_hero_ep.train import (
    GrugRunConfig,
    TrainingDataMode,
    WatchMode,
    _apply_hero_ep_runtime_defaults,
    _run_grug_local,
)

PRODUCTION_BATCH_SIZE = 11264
PRODUCTION_SCHEDULE_STEPS = 390251
TRAINING_TIMEOUT = 55 * 60


def resolve_checkpoint(checkpoint: tuple[str, ...]) -> tuple[str, int]:
    resolved = latest_checkpoint_path(*checkpoint)
    with (StoragePath(resolved) / "metadata.json").open("r") as source:
        metadata = json.load(source)
    return resolved, int(metadata["step"])


def _run_restored_local(config: GrugRunConfig) -> None:
    # Resolve only after the gang is allocated: hourly checkpoint rotation can outlive a queue wait.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(TRAINING_TIMEOUT)
    initialize_jax()
    trainer = config.trainer.trainer
    checkpoint = trainer.load_checkpoint_path
    assert isinstance(checkpoint, list) and checkpoint
    resolved, checkpoint_step = multihost_broadcast_sync(
        resolve_checkpoint(tuple(checkpoint)) if jax.process_index() == 0 else ("", 0)
    )
    assert config.stop_after_steps is not None
    stop_step = checkpoint_step + config.stop_after_steps
    if stop_step > trainer.num_train_steps:
        raise ValueError(f"Benchmark stop step {stop_step} exceeds the production schedule")
    trainer = dataclasses.replace(
        trainer,
        load_checkpoint_path=resolved,
        profiler=dataclasses.replace(trainer.profiler, start_step=checkpoint_step + trainer.profiler.start_step),
    )
    config = dataclasses.replace(
        config, trainer=dataclasses.replace(config.trainer, trainer=trainer), stop_after_steps=stop_step
    )
    if jax.process_index() == 0:
        click.echo(
            json.dumps({"benchmark_checkpoint": resolved, "checkpoint_step": checkpoint_step, "stop_step": stop_step})
        )
    try:
        _run_grug_local(config)
    finally:
        signal.alarm(0)


def run_restored_benchmark(
    config: GrugRunConfig, *, local_entrypoint: Callable[[GrugRunConfig], None] = _run_restored_local
) -> None:
    trainer = config.trainer.trainer
    assert trainer.id is not None
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=trainer.watch.is_enabled and config.trainer.watch_mode == WatchMode.INLINE,
        processes_per_task=config.processes_per_task,
        moe_implementation=config.model.moe_implementation,
        remat_mode=config.model.remat_mode,
    )
    dispatch_grug_training_run(
        run_id=trainer.id,
        config=config,
        local_entrypoint=local_entrypoint,
        resources=config.resources,
        processes_per_task=config.processes_per_task,
        max_retries_failure=0,
    )


def build_restored_benchmark(
    *,
    run_id: str,
    checkpoint: tuple[str, ...],
    num_steps: int,
    profile_steps: int,
    warmup_steps: int,
    training_data_mode: TrainingDataMode,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    if not checkpoint:
        raise ValueError("At least one checkpoint root is required")
    if warmup_steps + profile_steps >= num_steps:
        raise ValueError("num_steps must leave an unprofiled step after warmup and capture")
    step = build_diagnostic_run(
        run_id=run_id,
        dp_racks=1,
        num_steps=num_steps,
        schedule_steps=PRODUCTION_SCHEDULE_STEPS,
        profile_start_step=warmup_steps,
        profile_steps=profile_steps,
        training_data_mode=training_data_mode,
        save_checkpoints=False,
        version=version,
    )
    # Keep the production LR/epsilon schedule when reducing only the data-parallel rack count.
    _, optimizer = build_hero_configs(
        num_train_steps=PRODUCTION_SCHEDULE_STEPS,
        batch_size=PRODUCTION_BATCH_SIZE,
    )
    optimizer = dataclasses.replace(optimizer, gate_router_weight_decay=0.02)

    def build_config(ctx: StepContext) -> GrugRunConfig:
        config = step.build_config(ctx)
        trainer = dataclasses.replace(
            config.trainer.trainer,
            load_checkpoint_path=list(checkpoint),
            load_checkpoint=True,
            tracker=JsonLoggerConfig(),
        )
        return dataclasses.replace(
            config,
            optimizer=optimizer,
            trainer=dataclasses.replace(config.trainer, trainer=trainer),
            max_retries_failure=0,
        )

    return dataclasses.replace(step, build_config=build_config, run=run_restored_benchmark)


@click.command()
@click.option("--run-id", required=True)
@click.option(
    "--checkpoint",
    multiple=True,
    required=True,
    help="Exact checkpoint or root; rank 0 resolves latest after GPU allocation.",
)
@click.option("--num-steps", type=click.IntRange(min=1), default=30, show_default=True)
@click.option("--warmup-steps", type=click.IntRange(min=1), default=5, show_default=True)
@click.option("--profile-steps", type=click.IntRange(min=0), default=2, show_default=True)
@click.option("--training-data", type=click.Choice([mode.value for mode in TrainingDataMode]), default="mixture")
@build_options
def main(
    run_id: str,
    checkpoint: tuple[str, ...],
    num_steps: int,
    warmup_steps: int,
    profile_steps: int,
    training_data: str,
) -> ArtifactStep[HeroThroughputResult]:
    return build_restored_benchmark(
        run_id=run_id,
        checkpoint=checkpoint,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        profile_steps=profile_steps,
        training_data_mode=TrainingDataMode(training_data),
    )


if __name__ == "__main__":
    main()
