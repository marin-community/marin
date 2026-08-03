# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure GB200 checkpoint staging and object-store commit time.

Unlike the throughput hero run, this benchmark uses a smaller 52.85B-total,
1.71B-active MoE and checkpoints at deterministic steps. It retains rack-local
FSDP and optimizer offload while disabling W&B, profiling, and Python allocation
tracing, then writes the disposable checkpoints to a one-day temporary bucket.
"""

import dataclasses
from datetime import timedelta

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointDebugConfig, CheckpointerConfig
from levanter.tracker.telemetry import TelemetryConfig
from levanter.trainer import TrainerConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from rigging.filesystem import marin_temp_bucket, prefix_join

from experiments.grug.moe_hero_fsdp.heuristic import build_checkpoint_benchmark_configs
from experiments.grug.moe_hero_fsdp.launch import (
    _SLIMPAJAMA_SHUFFLE,
    HERO_FSDP_BATCH_SIZE,
    HERO_MIXED_PRECISION,
    HERO_NODES_PER_RACK,
    HERO_PROCESSES_PER_TASK,
    HERO_TRAINING_STALL_TIMEOUT,
    _slimpajama_6b_dataset,
)
from experiments.grug.moe_hero_fsdp.train import GrugRunConfig, hero_grug_trainer_config, run_grug

DEFAULT_BENCHMARK_STEPS = 12
DEFAULT_CHECKPOINT_EVERY_STEPS = 8
CHECKPOINT_OUTPUT_TTL_DAYS = 1
CHECKPOINT_DEBUG_INTERVAL = 5.0
CHECKPOINT_STACK_DUMP_AFTER = timedelta(minutes=5).total_seconds()


class CheckpointBenchmarkResult(Artifact):
    """Timing logs and checkpoints from the GB200 checkpoint benchmark."""


def build_checkpoint_benchmark_run(
    *,
    run_id: str,
    dp_racks: int,
    num_steps: int,
    checkpoint_every_steps: int,
    version: str | None = None,
) -> ArtifactStep[CheckpointBenchmarkResult]:
    """Build a deterministic checkpoint benchmark on one or more GB200 racks."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if dp_racks <= 0:
        raise ValueError(f"dp_racks must be positive, got {dp_racks}")
    if num_steps <= 1:
        raise ValueError(f"num_steps must be greater than one, got {num_steps}")
    if checkpoint_every_steps <= 0 or checkpoint_every_steps >= num_steps:
        raise ValueError(
            "checkpoint_every_steps must be positive and less than num_steps, "
            f"got checkpoint_every_steps={checkpoint_every_steps}, num_steps={num_steps}"
        )

    batch_size = dp_racks * HERO_FSDP_BATCH_SIZE
    model, optimizer = build_checkpoint_benchmark_configs(num_train_steps=num_steps, batch_size=batch_size)
    grug_trainer = hero_grug_trainer_config(replica_axis_size=dp_racks)
    train_resources = ResourceConfig.with_gpu(
        "GB200",
        count=4,
        cpu=120,
        ram="850g",
        disk="1t",
        replicas=HERO_NODES_PER_RACK * dp_racks,
    )
    name = f"grug/checkpoint-benchmark/{run_id}"
    version = resolve_version(name, version)
    step_name = user_namespaced_name(name, version)
    slim = _slimpajama_6b_dataset()
    output_path = marin_temp_bucket(
        ttl_days=CHECKPOINT_OUTPUT_TTL_DAYS,
        prefix=prefix_join(step_name, version),
    )

    def build_config(ctx: StepContext) -> GrugRunConfig:
        trainer = TrainerConfig(
            id=run_id,
            seed=0,
            train_batch_size=batch_size,
            num_train_steps=num_steps,
            profiler=ProfilerConfig(enabled=False, start_step=8, num_steps=0),
            mp=jmp.get_policy(HERO_MIXED_PRECISION),
            tracker=TelemetryConfig(training_stall_timeout=HERO_TRAINING_STALL_TIMEOUT),
            watch=WatchConfig(interval=20),
            use_explicit_mesh_axes=True,
            require_accelerator=True,
            allow_nondivisible_batch_size=False,
            checkpointer=CheckpointerConfig(
                base_path=prefix_join(ctx.output_path, "checkpoints"),
                temporary_base_path=None,
                save_interval=None,
                keep=[{"every": checkpoint_every_steps}],
                append_run_id_to_base_path=False,
                delete_old_temp_checkpoints=False,
                keep_last_temporary_checkpoints=0,
                debug=CheckpointDebugConfig(
                    enabled=True,
                    log_interval=CHECKPOINT_DEBUG_INTERVAL,
                    dump_stacks_after=CHECKPOINT_STACK_DUMP_AFTER,
                    tracemalloc_frames=None,
                    top_allocations=0,
                    force_gc_before_serialize=False,
                ),
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

    return ArtifactStep(
        name=step_name,
        version=version,
        artifact_type=CheckpointBenchmarkResult,
        run=run_grug,
        build_config=build_config,
        deps=(slim,),
        runtime_args={"train_resources": train_resources},
        override_path=output_path,
    )


@click.command()
@click.option("--run-id", required=True, help="Run identifier for artifact and telemetry names.")
@click.option(
    "--dp-racks",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Data-parallel NVL72 rack count.",
)
@click.option(
    "--num-steps",
    type=click.IntRange(min=2),
    default=DEFAULT_BENCHMARK_STEPS,
    show_default=True,
    help="Number of training steps.",
)
@click.option(
    "--checkpoint-every-steps",
    type=click.IntRange(min=1),
    default=DEFAULT_CHECKPOINT_EVERY_STEPS,
    show_default=True,
    help="Deterministic checkpoint interval for the benchmark.",
)
@build_options
def main(
    run_id: str,
    dp_racks: int,
    num_steps: int,
    checkpoint_every_steps: int,
) -> ArtifactStep[CheckpointBenchmarkResult]:
    return build_checkpoint_benchmark_run(
        run_id=run_id,
        dp_racks=dp_racks,
        num_steps=num_steps,
        checkpoint_every_steps=checkpoint_every_steps,
    )


if __name__ == "__main__":
    main()
