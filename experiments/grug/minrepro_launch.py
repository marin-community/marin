# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin launcher for the #7344 minimal repro.

WHY THIS EXISTS: submitting `minimal_wedge_repro.py` with a bare `iris job run` gives a
CPU-ONLY jaxlib ("a CUDA-enabled jaxlib is not installed. Falling back to cpu"), because the
GPU extra is selected by `extras_for_resources(resources)` inside `dispatch_grug_training_run`.
Going through the same dispatch path as moe_hero_fsdp means the repro runs on the SAME image,
deps and launcher as the wedging job, so any behavioural difference is attributable to the
program rather than to the environment.

jax.distributed is brought up by the Iris runtime here exactly as it is for the hero, so the
repro is invoked with --distributed off.
"""

import dataclasses
from dataclasses import dataclass, field

import click
from fray.cluster import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name

from experiments.grug.dispatch import dispatch_grug_training_run

HERO_NODES_PER_RACK = 16  # launch.py: HERO_NODES_PER_RACK


@dataclass(frozen=True)
class MinReproConfig:
    run_id: str
    resources: ResourceConfig
    argv: tuple[str, ...] = field(default_factory=tuple)


class MinReproResult(Artifact):
    """Marker artifact for the minimal-repro run."""


def _run_local(config: MinReproConfig) -> None:
    # The Iris runtime does NOT bring up jax.distributed for us; the hero gets it via
    # levanter's trainer.initialize(). Without this every task comes up standalone and the
    # repro sees device_count()=4 (its own node) instead of 128, i.e. no cross-node
    # collectives at all -- which would be a null repro that still "ran".
    from iris.runtime.jax_init import initialize_jax

    initialize_jax()

    from experiments.grug.minimal_wedge_repro import main

    main(list(config.argv))


def run_minrepro(config: MinReproConfig) -> None:
    dispatch_grug_training_run(
        run_id=config.run_id,
        config=config,
        local_entrypoint=_run_local,
        resources=config.resources,
        max_retries_failure=0,
        max_task_failures=0,
        processes_per_task=1,
    )


def build_minrepro_run(*, run_id: str, dp_racks: int, num_steps: int, version: str | None = None) -> ArtifactStep:
    resources = ResourceConfig.with_gpu(
        "GB200", count=4, cpu=120, ram="850g", disk="1t", replicas=HERO_NODES_PER_RACK * dp_racks
    )
    argv = ("--dp-racks", str(dp_racks), "--num-steps", str(num_steps), "--distributed", "off")
    name = f"grug/{run_id}"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> MinReproConfig:
        return MinReproConfig(run_id=run_id, resources=ctx.runtime_arg("train_resources"), argv=argv)

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=MinReproResult,
        run=run_minrepro,
        build_config=build_config,
        deps=(),
        runtime_args={"train_resources": resources},
    )


@click.command()
@click.option("--run-id", required=True)
@click.option("--dp-racks", type=click.IntRange(min=1), required=True)
@click.option("--num-steps", type=int, default=20000, show_default=True)
@build_options
def main(run_id: str, dp_racks: int, num_steps: int) -> ArtifactStep:
    return build_minrepro_run(run_id=run_id, dp_racks=dp_racks, num_steps=num_steps)


if __name__ == "__main__":
    main()
