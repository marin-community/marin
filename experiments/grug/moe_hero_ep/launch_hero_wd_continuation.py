# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Continue the EP hero from a pinned permanent checkpoint with decoupled weight decay on the
``attn_gate`` and ``router`` weights, under its own W&B id.

Forks ``hero-12d8b6f0-dee637`` at ``step-54000`` and continues to the hero's end (num_train_steps
unchanged), decaying the two tensors by ``0.05 * (1 - step/num_train_steps)``. The decay reads the
Adam step count, so on the resume it evaluates at the restored global step. See ``build_ladder_run``
and ``GrugMoeMuonHConfig.gate_router_weight_decay``.

    uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources \\
        --target-cluster cw-us-east-02a --priority production \\
        -- python -m experiments.grug.moe_hero_ep.launch_hero_wd_continuation --version <v> --run
"""

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options

from experiments.grug.moe_hero_ep.hero_recipe import HeroThroughputResult
from experiments.grug.moe_hero_ep.launch_scaling_ladder import build_ladder_run

RUN_ID = "hero-wd-gate-router-0p05-decay"
GATE_ROUTER_WEIGHT_DECAY = 0.05
# The hero permanent checkpoint to fork from (permanent checkpoints are kept every 6000 steps).
SOURCE_CHECKPOINT_PATH = "s3://marin-us-east-02a/marin/grug/hero-12d8b6f0-dee637/2026.08.19.2/checkpoints/step-54000"


def build_hero_wd_continuation(*, version: str | None = None) -> ArtifactStep[HeroThroughputResult]:
    """The d6144 hero forked at ``SOURCE_CHECKPOINT_PATH`` with gate/router weight decay, under its own id."""
    return build_ladder_run(
        run_id=RUN_ID,
        size="d6144",
        gate_router_weight_decay=GATE_ROUTER_WEIGHT_DECAY,
        source_checkpoint_path=SOURCE_CHECKPOINT_PATH,
        version=version,
    )


@click.command()
@build_options
def main() -> ArtifactStep[HeroThroughputResult]:
    return build_hero_wd_continuation()


if __name__ == "__main__":
    main()
