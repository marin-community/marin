# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one CODA RMS-GatedNorm rack arm at the FSDP64 hero shape.

Submit the baseline and treatment as separate top-level Marin coordinators. Sibling arms under one
root share a JAX coordinator registry namespace and can join the wrong 64-rank world. Each invocation
contains one supervised sweep arm, writes no checkpoints, and records an explicitly named W&B run.
Score steady-state metrics over steps 5 through the end. Confirm the arm supervisor and W&B rows;
the sweep wrapper can exit zero after a contained arm failure.

Usage:
    uv run python -m experiments.grug.moe_hero_fsdp.coda_rms_epilogue_rack_ab \
        --run-id coda-rms-rack-baseline --arm xla --version dev --max-concurrent 1

    uv run python -m experiments.grug.moe_hero_fsdp.coda_rms_epilogue_rack_ab \
        --run-id coda-rms-rack-treatment --arm quack-coda --version dev --max-concurrent 1

Add ``--run`` only after reviewing each printed plan.
"""

import click
from iris.rpc.proto_display import PRIORITY_BAND_NAMES, priority_band_value
from levanter.recovery.types import AblationSpec
from marin.experiment.cli import build_options

from experiments.grug.moe_hero_fsdp.launch import HeroOverrides, HeroSweepArm, build_hero_sweep_run
from experiments.grug.moe_hero_fsdp.model import RmsGatedNormImplementation

DEFAULT_STEPS = 25
DP_RACKS = 1


def _rack_arm(*, run_id: str, name: str, implementation: RmsGatedNormImplementation, num_steps: int) -> HeroSweepArm:
    return HeroSweepArm(
        spec=AblationSpec(
            name=name,
            env={},
            num_steps=num_steps,
            notes=f"FSDP64 d6144/L48/B1024/S4096/E128/top4; RMS-GatedNorm={implementation}",
        ),
        run_id=f"{run_id}-{name}",
        overrides=HeroOverrides(rms_gated_norm_implementation=implementation),
    )


@click.command()
@click.option("--run-id", required=True, help="Arm identifier; the W&B run appends -xla or -quack-coda.")
@click.option(
    "--num-steps",
    type=click.IntRange(min=20),
    default=DEFAULT_STEPS,
    show_default=True,
    help="Training steps per rack arm.",
)
@click.option(
    "--priority",
    type=click.Choice(PRIORITY_BAND_NAMES),
    default="production",
    show_default=True,
    help="Iris band for each rack-sized training gang.",
)
@click.option(
    "--arm",
    type=click.Choice(["xla", "quack-coda"]),
    required=True,
    help="Select the single rack arm for this top-level coordinator.",
)
@build_options
def main(run_id: str, num_steps: int, priority: str, arm: str):
    """Build one rack step for an isolated top-level coordinator."""
    band = priority_band_value(priority)
    implementation: RmsGatedNormImplementation = "xla" if arm == "xla" else "quack_coda"
    return [
        build_hero_sweep_run(
            run_id=f"{run_id}-{arm}",
            dp_racks=DP_RACKS,
            steps_per_arm=num_steps,
            arms=[_rack_arm(run_id=run_id, name=arm, implementation=implementation, num_steps=num_steps)],
            priority=band,
        )
    ]


if __name__ == "__main__":
    main()
