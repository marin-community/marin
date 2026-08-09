# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""d1024 EP shape probe: narrower experts, higher top-k, matched compute.

A one-off EP64 shape at d1024 that trades expert width for routed top-k. Against the default EP rung
(192 experts, top-4, expert width = hidden = 1,024, latent = hidden/2 = 512, capacity 1.33) it:

- narrows the routed expert width to 768,
- routes top-6 of 192 instead of top-4,
- raises the fixed all-to-all capacity factor to 1.45.

Routed active parameters scale as ``top_k * expert_width`` (the latent width is unchanged), so this is
``(6 * 768) / (4 * 1024) = 1.125`` -- 12.5% more active parameters, ignoring the latent projections. To
hold total training compute equal to the default d1024 rung, the step count drops by the same factor:
``round(26,016 / 1.125) = 23,125`` (about 11% fewer). Everything else matches the default EP: LatentMoE
with the RMSNorm gain, histogram QB estimator, no gate/up reinit, window 2048, one GB200 rack.
"""

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options

from experiments.grug.moe_hero_ep.ladder_launch import LADDER_BATCH_PER_RACK, LADDER_SEQ_LEN, LADDER_WATCH_INTERVAL
from experiments.grug.moe_hero_ep.launch import HeroThroughputResult
from experiments.grug.moe_hero_ep.small_scale_abl_launch import SMALL_SHAPES, build_small_run

SIZE = "d1024"
NUM_EXPERTS = 192
NUM_EXPERTS_PER_TOKEN = 6
EXPERT_INTERMEDIATE_DIM = 768
CAPACITY_FACTOR = 1.45
# Default d1024 rung is 26,016 steps; +12.5% active params -> 26,016 / 1.125 to hold compute equal.
NUM_TRAIN_STEPS = 23125


def build_wide_top6_run(*, run_id: str, version: str | None = None) -> ArtifactStep[HeroThroughputResult]:
    """The d1024 narrow-expert / top-6 EP run at matched compute."""
    hidden = SMALL_SHAPES[SIZE].hidden_dim
    return build_small_run(
        run_id=run_id,
        size=SIZE,
        target="gb200-rack",
        flavor="ep",
        seq_len=LADDER_SEQ_LEN,
        tokens_per_step=LADDER_BATCH_PER_RACK * LADDER_SEQ_LEN,
        capacity_factor=CAPACITY_FACTOR,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=NUM_EXPERTS_PER_TOKEN,
        intermediate_dim=EXPERT_INTERMEDIATE_DIM,
        latent_dim=hidden // 2,
        latent_reinit_gate_up=False,
        qb_use_histogram=True,
        num_train_steps_override=NUM_TRAIN_STEPS,
        watch_interval=LADDER_WATCH_INTERVAL,
        dp_racks=1,
        steps_per_eval=3000,
        version=version,
    )


@click.command()
@click.option("--run-id", required=True, help="Run identifier for artifact and W&B names.")
@build_options
def main(run_id: str) -> ArtifactStep[HeroThroughputResult]:
    return build_wide_top6_run(run_id=run_id)


if __name__ == "__main__":
    main()
