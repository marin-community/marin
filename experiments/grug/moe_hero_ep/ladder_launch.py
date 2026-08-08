# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue #8062 small-scale ladder: d768 / d1024 / d1536 / d2048 across EP64 and FSDP.

The ladder produces the loss-vs-scale signal that decides between the EP and FSDP hero. Every rung
holds EP=64, batch 1024/rack, and seq_len 4096 fixed, and trains to 750 tokens per active parameter
so the small runs mirror the hero's token/parameter ratio and drop dynamics. Three flavors run at
each width:

- ``ep64``: 192 routed experts with LatentMoE (latent = hidden/2, expert width = hidden), capacity
  factor 1.33 -- the EP hero downsized.
- ``fsdp-chunk1``: 128 experts, no latent, the dropless ``scatter`` local path.
- ``fsdp-chunk4``: 128 experts, no latent, four ``sonic_cute`` chunks -- the FSDP hero's
  minor-dropping reference.

The three narrowest rungs run on one GB200 rack; d2048 spans four racks (4x batch) to hold its wider
step. Parameter and gradient norms log every 10 steps, and each rung evals paloma/uncheatable and
writes a permanent checkpoint every 1,000 steps (3,000 for d1024 and up) so a downstream job can
reload each checkpoint and re-eval under a different MoE path.

``main`` returns one artifact per selected ``(size, flavor)`` pair, so a single invocation submits a
whole row, a whole column, or the full 12-run grid.
"""

import dataclasses

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options

from experiments.grug.moe_hero_ep.launch import HeroThroughputResult
from experiments.grug.moe_hero_ep.small_scale_abl_launch import SMALL_SHAPES, build_small_run

# Issue #8062 holds these fixed across the whole ladder so only width and depth move.
LADDER_SIZES = ("d768", "d1024", "d1536", "d2048")
LADDER_SEQ_LEN = 4096
LADDER_BATCH_PER_RACK = 1024
LADDER_TOKENS_PER_ACTIVE_PARAM = 750
LADDER_WATCH_INTERVAL = 10
# The widest rung needs four racks to hold its 4x batch; the rest fit on one.
LADDER_RACKS: dict[str, int] = {"d768": 1, "d1024": 1, "d1536": 1, "d2048": 4}


def _steps_per_eval(size: str) -> int:
    """Eval and permanent-checkpoint cadence: dense at the cheap rung, sparser as steps get long."""
    return 1000 if size == "d768" else 3000


@dataclasses.dataclass(frozen=True)
class LadderFlavor:
    """One column of the ladder: how the MoE shards and the model shape that goes with it.

    ``build_flavor`` selects the sharding path in ``small_scale_abl_launch.FLAVORS``. ``use_latent``
    switches the EP LatentMoE shape (expert width = hidden, latent = hidden/2) on; the FSDP columns
    leave it off and take the standard hidden/2 expert width. ``capacity_factor`` only bites the EP
    ``fixed_all_to_all`` path -- the FSDP paths drop by chunk count, not capacity.
    """

    build_flavor: str
    num_experts: int
    use_latent: bool
    capacity_factor: float


LADDER_FLAVORS: dict[str, LadderFlavor] = {
    "ep64": LadderFlavor("ep", num_experts=192, use_latent=True, capacity_factor=1.33),
    # Same EP shape, but standard MoE (no latent): the routed experts read full hidden width, so their
    # intermediate halves to hidden/2 (the `_small_model` default) to hold the routed compute constant.
    "ep64-nolatent": LadderFlavor("ep", num_experts=192, use_latent=False, capacity_factor=1.33),
    "fsdp-chunk1": LadderFlavor("fsdp-nodrop", num_experts=128, use_latent=False, capacity_factor=1.0),
    "fsdp-chunk4": LadderFlavor("fsdp-chunk4", num_experts=128, use_latent=False, capacity_factor=1.0),
}


def build_ladder_run(
    *,
    run_id: str,
    size: str,
    flavor: str,
    latent_reinit_gate_up: bool = True,
    qb_use_histogram: bool = False,
    capacity_factor: float | None = None,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """One ladder rung: shape ``size`` under column ``flavor`` at the issue #8062 fixed settings.

    ``latent_reinit_gate_up`` only bites the EP (LatentMoE) column; set False to ablate the
    sqrt(hidden/latent) reinit of the routed gate/up projections. ``qb_use_histogram`` swaps the QB
    router's top-k threshold estimate for the 1k-bin live-range histogram quantile. ``capacity_factor``
    overrides the flavor's default (EP only -- the FSDP paths drop by chunk count, not capacity).
    """
    if size not in LADDER_RACKS:
        raise ValueError(f"size must be one of {sorted(LADDER_RACKS)}, got {size!r}")
    if flavor not in LADDER_FLAVORS:
        raise ValueError(f"flavor must be one of {sorted(LADDER_FLAVORS)}, got {flavor!r}")
    shape = SMALL_SHAPES[size]
    column = LADDER_FLAVORS[flavor]
    dp_racks = LADDER_RACKS[size]
    return build_small_run(
        run_id=run_id,
        size=size,
        target="gb200-rack",
        flavor=column.build_flavor,
        seq_len=LADDER_SEQ_LEN,
        tokens_per_step=LADDER_BATCH_PER_RACK * dp_racks * LADDER_SEQ_LEN,
        capacity_factor=column.capacity_factor if capacity_factor is None else capacity_factor,
        num_experts=column.num_experts,
        num_experts_per_token=4,
        intermediate_dim=shape.hidden_dim if column.use_latent else None,
        latent_dim=shape.hidden_dim // 2 if column.use_latent else None,
        latent_reinit_gate_up=latent_reinit_gate_up,
        qb_use_histogram=qb_use_histogram,
        tokens_per_active_param=LADDER_TOKENS_PER_ACTIVE_PARAM,
        watch_interval=LADDER_WATCH_INTERVAL,
        dp_racks=dp_racks,
        steps_per_eval=_steps_per_eval(size),
        version=version,
    )


@click.command()
@click.option("--run-prefix", default="mhep-ladder", show_default=True, help="Prefix for run and W&B names.")
@click.option(
    "--flavor",
    type=click.Choice([*sorted(LADDER_FLAVORS), "all"]),
    default="all",
    show_default=True,
    help="Ladder column to submit, or all three.",
)
@click.option(
    "--size",
    type=click.Choice([*LADDER_SIZES, "all"]),
    default="all",
    show_default=True,
    help="Ladder rung to submit, or all four.",
)
@click.option(
    "--latent-reinit/--no-latent-reinit",
    default=True,
    show_default=True,
    help="EP column only: rescale the routed gate/up init by sqrt(hidden/latent). Off ablates it.",
)
@click.option(
    "--qb-histogram/--no-qb-histogram",
    default=False,
    show_default=True,
    help="QB router threshold via the 1k-bin live-range histogram quantile instead of per-device top-k.",
)
@click.option(
    "--capacity-factor",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help="Override the EP flavor's fixed all-to-all capacity factor (default keeps the flavor's value).",
)
@build_options
def main(
    run_prefix: str, flavor: str, size: str, latent_reinit: bool, qb_histogram: bool, capacity_factor: float | None
) -> list[ArtifactStep[HeroThroughputResult]]:
    flavors = list(LADDER_FLAVORS) if flavor == "all" else [flavor]
    sizes = list(LADDER_SIZES) if size == "all" else [size]
    return [
        build_ladder_run(
            run_id=f"{run_prefix}-{f}-{s}",
            size=s,
            flavor=f,
            latent_reinit_gate_up=latent_reinit,
            qb_use_histogram=qb_histogram,
            capacity_factor=capacity_factor,
        )
        for f in flavors
        for s in sizes
    ]


if __name__ == "__main__":
    main()
