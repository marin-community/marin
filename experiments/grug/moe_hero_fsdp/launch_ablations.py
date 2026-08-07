# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep environment arms against the 300B FSDP hero on one allocation.

Every arm runs on the same nodes under one ``GPUHangSupervisor`` per task, in a fresh
trainer subprocess so process-start variables take effect. A wedged arm is ended by the
XLA execution deadman and the sweep advances to the next arm, so the whole matrix runs
without a scheduler round trip or an operator between arms.
"""

import click
from marin.experiment.cli import build_options

from experiments.grug.moe_hero_fsdp.launch import build_ablation_sweep_hero_run
from experiments.grug.recovery.ablation_catalog import environment_ablation_names


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
    default=("baseline",),
    show_default=True,
    help="Environment arm to run; repeat the option to sweep several arms on one allocation.",
)
@build_options
def main(run_id: str, dp_racks: int, steps_per_arm: int, ablation_names: tuple[str, ...]):
    return build_ablation_sweep_hero_run(
        run_id=run_id,
        dp_racks=dp_racks,
        steps_per_arm=steps_per_arm,
        ablation_names=ablation_names,
    )


if __name__ == "__main__":
    main()
