# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the rack-scaled FSDP MoE hero under the GPU hang supervisor."""

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options

from experiments.grug.moe_hero_fsdp.launch import DEFAULT_HERO_STEPS, HeroThroughputResult, build_supervised_hero_run


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
def main(run_id: str, dp_racks: int, num_steps: int, save_checkpoints: bool) -> ArtifactStep[HeroThroughputResult]:
    return build_supervised_hero_run(
        run_id=run_id,
        dp_racks=dp_racks,
        num_steps=num_steps,
        save_checkpoints=save_checkpoints,
    )


if __name__ == "__main__":
    main()
