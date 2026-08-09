# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep process-environment arms against the 300B FSDP hero on one allocation."""

import click
from iris.rpc.proto_display import PRIORITY_BAND_NAMES, priority_band_value
from marin.experiment.cli import build_options

from experiments.grug.moe_hero_fsdp.launch import HeroSweepArm, build_hero_sweep_run
from experiments.grug.recovery.ablation_catalog import (
    BASELINE_ABLATION_NAME,
    environment_ablation_names,
    environment_ablations,
    selected_ablations,
)


@click.command()
@click.option("--run-id", required=True, help="Sweep identifier; each arm runs as <run-id>-<arm>.")
@click.option("--dp-racks", type=click.IntRange(min=1), required=True, help="Data-parallel NVL72 rack count.")
@click.option(
    "--steps-per-arm",
    type=click.IntRange(min=1),
    default=250,
    show_default=True,
    help="Steps each arm runs before the sweep advances.",
)
@click.option(
    "--ablation",
    "ablation_names",
    type=click.Choice(environment_ablation_names()),
    multiple=True,
    default=(BASELINE_ABLATION_NAME,),
    show_default=True,
    help="Environment arm to run; repeat the option to sweep several arms on one allocation.",
)
@click.option(
    "--priority",
    type=click.Choice(PRIORITY_BAND_NAMES),
    default="interactive",
    show_default=True,
    help="Iris band for the training gang. 'production' is admin-only and never preempted.",
)
@build_options
def main(run_id: str, dp_racks: int, steps_per_arm: int, ablation_names: tuple[str, ...], priority: str):
    specs = selected_ablations(environment_ablations(num_steps=steps_per_arm), ablation_names)
    return build_hero_sweep_run(
        run_id=run_id,
        dp_racks=dp_racks,
        steps_per_arm=steps_per_arm,
        arms=[HeroSweepArm(spec=spec, run_id=f"{run_id}-{spec.name}") for spec in specs],
        priority=priority_band_value(priority),
    )


if __name__ == "__main__":
    main()
