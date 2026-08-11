# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the CODA RMS-GatedNorm epilogue A/B back to back on one GB200 node.

Both arms use the d6144/L48/B64/S4096 hero shape with eight experts and sonic_cute. The baseline
uses the XLA RMS-GatedNorm boundary; the treatment uses the QuACK CODA kernel. The shared sweep
allocation runs each arm in a fresh subprocess and never writes checkpoints.

Compare the median W&B ``throughput/duration`` over steps 5 through the end; the earlier steps
include compilation, PGLE recompilation, and data-loader warmup.

Usage:
    uv run python -m experiments.grug.moe_hero_fsdp.coda_rms_epilogue_ab \
        --run-id coda-rms-epilogue --version dev

Add ``--run`` only after reviewing the printed one-node plan.
"""

import click
from iris.rpc.proto_display import PRIORITY_BAND_NAMES, priority_band_value
from levanter.recovery.types import AblationSpec
from marin.experiment.cli import build_options

from experiments.grug.moe_hero_fsdp.launch import HeroOverrides, HeroSweepArm, build_hero_sweep_run
from experiments.grug.moe_hero_fsdp.model import RmsGatedNormImplementation

DEFAULT_STEPS = 25
NUM_NODES = 1
NUM_EXPERTS = 8
BATCH_SIZE = 64


def _arm(*, run_id: str, name: str, implementation: RmsGatedNormImplementation, num_steps: int) -> HeroSweepArm:
    return HeroSweepArm(
        spec=AblationSpec(
            name=name,
            env={},
            num_steps=num_steps,
            notes=f"d6144/L48/B64/S4096/E8 sonic_cute; RMS-GatedNorm={implementation}",
        ),
        run_id=f"{run_id}-{name}",
        overrides=HeroOverrides(
            num_experts=NUM_EXPERTS,
            rms_gated_norm_implementation=implementation,
        ),
        batch_size=BATCH_SIZE,
    )


@click.command()
@click.option("--run-id", required=True, help="Sweep identifier; arms use <run-id>-xla and <run-id>-quack-coda.")
@click.option(
    "--num-steps",
    type=click.IntRange(min=20, max=25),
    default=DEFAULT_STEPS,
    show_default=True,
    help="Training steps per arm.",
)
@click.option(
    "--priority",
    type=click.Choice(PRIORITY_BAND_NAMES),
    default="interactive",
    show_default=True,
    help="Iris band for the one-node training allocation.",
)
@build_options
def main(run_id: str, num_steps: int, priority: str):
    arms = [
        _arm(run_id=run_id, name="xla", implementation="xla", num_steps=num_steps),
        _arm(run_id=run_id, name="quack-coda", implementation="quack_coda", num_steps=num_steps),
    ]
    return build_hero_sweep_run(
        run_id=run_id,
        dp_racks=1,
        steps_per_arm=num_steps,
        arms=arms,
        priority=priority_band_value(priority),
        num_nodes=NUM_NODES,
    )


if __name__ == "__main__":
    main()
