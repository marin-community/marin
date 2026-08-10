# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dynamic-latent experiment (issue #8105): does a per-token latent subspace beat the fixed one?

The d768 EP default (192 experts, top-4, LatentMoE latent=384, histogram QB, no gate/up reinit,
capacity 1.33) run two ways on one H100 EP64 fleet (8 nodes x 8 GPUs = 64):

- ``--no-dynamic-latent`` (baseline): the standard fixed hidden->latent down-projection + RMSNorm.
- ``--dynamic-latent`` (treatment): a full hidden->hidden projection whose two halves are mixed
  per token by ``s = sigmoid(gate . x)`` (gate zero-init, Adam), then RMSNorm -- each token composes
  its own latent subspace instead of sharing one.

Runs on ``cw-us-east-02a`` (H100). The ``h100-8node`` target sets the H100 differences from B200:
reference attention (``gpu_fa4_cute`` is Blackwell-only) and no QuACK SM100 ``use_syrk`` for MuonH.
"""

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options

from experiments.grug.moe_hero_ep.ladder_launch import (
    LADDER_BATCH_PER_RACK,
    LADDER_SEQ_LEN,
    LADDER_TOKENS_PER_ACTIVE_PARAM,
    LADDER_WATCH_INTERVAL,
)
from experiments.grug.moe_hero_ep.launch import HeroThroughputResult
from experiments.grug.moe_hero_ep.small_scale_abl_launch import SMALL_SHAPES, build_small_run

SIZE = "d768"
TARGET = "h100-8node"  # H100 EP64: 8 nodes x 8 GPUs, reference attention, no syrk
NUM_EXPERTS = 192
NUM_EXPERTS_PER_TOKEN = 4
CAPACITY_FACTOR = 1.33


def build_dynamic_latent_run(
    *, run_id: str, dynamic_latent: bool, version: str | None = None
) -> ArtifactStep[HeroThroughputResult]:
    """The d768 EP default on one H100 rack, with or without the dynamic latent."""
    hidden = SMALL_SHAPES[SIZE].hidden_dim
    return build_small_run(
        run_id=run_id,
        size=SIZE,
        target=TARGET,
        flavor="ep",
        seq_len=LADDER_SEQ_LEN,
        tokens_per_step=LADDER_BATCH_PER_RACK * LADDER_SEQ_LEN,
        capacity_factor=CAPACITY_FACTOR,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=NUM_EXPERTS_PER_TOKEN,
        intermediate_dim=hidden,  # routed expert width = hidden (LatentMoE default)
        latent_dim=hidden // 2,
        latent_reinit_gate_up=False,
        qb_use_histogram=True,
        dynamic_latent=dynamic_latent,
        tokens_per_active_param=LADDER_TOKENS_PER_ACTIVE_PARAM,
        watch_interval=LADDER_WATCH_INTERVAL,
        dp_racks=1,
        steps_per_eval=1000,
        version=version,
    )


@click.command()
@click.option("--run-id", required=True, help="Run identifier for artifact and W&B names.")
@click.option(
    "--dynamic-latent/--no-dynamic-latent",
    default=False,
    show_default=True,
    help="Per-token subspace-mixing latent (treatment) vs the fixed down-projection (baseline).",
)
@build_options
def main(run_id: str, dynamic_latent: bool) -> ArtifactStep[HeroThroughputResult]:
    return build_dynamic_latent_run(run_id=run_id, dynamic_latent=dynamic_latent)


if __name__ == "__main__":
    main()
