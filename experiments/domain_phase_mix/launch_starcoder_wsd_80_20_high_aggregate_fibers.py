# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample six high-aggregate fibers on the StarCoder WSD80 response surface."""

from __future__ import annotations

import argparse
import logging
import os

from marin.execution.lazy import lower, run

from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base
from experiments.domain_phase_mix.launch_starcoder_wsd_80_20_fixed_aggregate_fiber import (
    fixed_aggregate_coordinates as aggregate_030_coordinates,
)
from experiments.domain_phase_mix.launch_starcoder_wsd_80_20_global_optimum_fiber import (
    fixed_aggregate_coordinates as aggregate_018_coordinates,
)
from experiments.domain_phase_mix.launch_starcoder_wsd_80_20_repeat_panel import SCHEDULES
from experiments.domain_phase_mix.starcoder_wsd_80_20_refinement_coordinates import (
    DRIFT_ANCHOR_COORDINATE,
    REFINEMENT_COORDINATES,
)

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_wsd80_highagg_fibers6x31_20260728"
WANDB_EXPERIMENT_TAG = "starcoder_wsd80_highagg_fibers6x31"
PANEL_TAG = "highagg_fibers6x31"
NOMINAL_PHASE_0_FRACTION = 0.80
NOMINAL_PHASE_1_FRACTION = 0.20
AGGREGATE_STARCODER_WEIGHTS = (0.35, 0.40, 0.50, 0.60, 0.70, 0.80)
NUM_POINTS_PER_FIBER = 31
FIRST_SELECTION_RANK = 260
EXPECTED_REUSED_COORDINATES = (
    (0.40, 0.40),
    (0.50, 0.50),
    (0.60, 0.60),
    (0.70, 0.70),
    (0.80, 0.80),
    (1.00, 0.00),
)
EXPECTED_NUM_COORDINATES = len(AGGREGATE_STARCODER_WEIGHTS) * NUM_POINTS_PER_FIBER
EXPECTED_NUM_RUNS = EXPECTED_NUM_COORDINATES - len(EXPECTED_REUSED_COORDINATES)
DEFAULT_MAX_CONCURRENT = EXPECTED_NUM_RUNS
COORDINATE_TOLERANCE = 1e-12


def _coordinates_match(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return max(abs(left[0] - right[0]), abs(left[1] - right[1])) <= COORDINATE_TOLERANCE


def _phase_0_weight(aggregate: float, phase_1_weight: float) -> float:
    return (aggregate - NOMINAL_PHASE_1_FRACTION * phase_1_weight) / NOMINAL_PHASE_0_FRACTION


def fixed_aggregate_coordinates(aggregate: float) -> tuple[tuple[float, float], ...]:
    """Return 31 points spanning one slope-minus-four aggregate fiber."""
    phase_1_weights = [index / (NUM_POINTS_PER_FIBER - 1) for index in range(NUM_POINTS_PER_FIBER)]
    if not any(abs(weight - aggregate) <= COORDINATE_TOLERANCE for weight in phase_1_weights):
        nearest_index = min(
            range(len(phase_1_weights)),
            key=lambda index: abs(phase_1_weights[index] - aggregate),
        )
        phase_1_weights[nearest_index] = aggregate
    return tuple(
        (_phase_0_weight(aggregate, phase_1_weight), phase_1_weight) for phase_1_weight in sorted(phase_1_weights)
    )


def all_panel_coordinates() -> tuple[tuple[float, float, float, int], ...]:
    """Return aggregate, phase weights, and within-fiber index for the panel."""
    return tuple(
        (aggregate, phase_0, phase_1, fiber_index)
        for aggregate in AGGREGATE_STARCODER_WEIGHTS
        for fiber_index, (phase_0, phase_1) in enumerate(fixed_aggregate_coordinates(aggregate))
    )


def _prior_coordinates() -> tuple[tuple[float, float], ...]:
    return (
        *base.SURFACE_COORDINATES,
        *REFINEMENT_COORDINATES,
        DRIFT_ANCHOR_COORDINATE,
        *aggregate_030_coordinates(),
        *aggregate_018_coordinates(),
        *((phase_0, phase_1) for _, phase_0, phase_1 in SCHEDULES),
    )


def build_run_specs() -> tuple[base.SurfaceRunSpec, ...]:
    """Return the new coordinates after strict geometry and overlap checks."""
    coordinates = all_panel_coordinates()
    if len(coordinates) != EXPECTED_NUM_COORDINATES:
        raise ValueError(f"Expected {EXPECTED_NUM_COORDINATES} coordinates, got {len(coordinates)}")
    rounded_coordinates = {(round(phase_0, 12), round(phase_1, 12)) for _, phase_0, phase_1, _ in coordinates}
    if len(rounded_coordinates) != len(coordinates):
        raise ValueError("High-aggregate fiber coordinates must be unique")

    for aggregate, phase_0, phase_1, _ in coordinates:
        if not 0.0 <= phase_0 <= 1.0 or not 0.0 <= phase_1 <= 1.0:
            raise ValueError(f"Coordinate outside simplex interval: {(aggregate, phase_0, phase_1)}")
        recovered_aggregate = NOMINAL_PHASE_0_FRACTION * phase_0 + NOMINAL_PHASE_1_FRACTION * phase_1
        if abs(recovered_aggregate - aggregate) > COORDINATE_TOLERANCE:
            raise ValueError(
                "Coordinate does not preserve its nominal 80/20 aggregate: "
                f"{(aggregate, phase_0, phase_1, recovered_aggregate)}"
            )

    for aggregate in AGGREGATE_STARCODER_WEIGHTS:
        fiber = fixed_aggregate_coordinates(aggregate)
        tied_matches = [coordinate for coordinate in fiber if _coordinates_match(coordinate, (aggregate, aggregate))]
        if len(tied_matches) != 1:
            raise ValueError(f"Aggregate {aggregate:.2f} must contain exactly one tied coordinate")
        if not _coordinates_match(fiber[0], (_phase_0_weight(aggregate, 0.0), 0.0)):
            raise ValueError(f"Aggregate {aggregate:.2f} must include its phase-1 zero endpoint")
        if not _coordinates_match(fiber[-1], (_phase_0_weight(aggregate, 1.0), 1.0)):
            raise ValueError(f"Aggregate {aggregate:.2f} must include its phase-1 one endpoint")

    prior_coordinates = _prior_coordinates()
    overlaps = tuple(
        (phase_0, phase_1)
        for _, phase_0, phase_1, _ in coordinates
        if any(_coordinates_match((phase_0, phase_1), prior) for prior in prior_coordinates)
    )
    if len(overlaps) != len(EXPECTED_REUSED_COORDINATES) or any(
        not any(_coordinates_match(overlap, expected) for expected in EXPECTED_REUSED_COORDINATES)
        for overlap in overlaps
    ):
        raise ValueError(f"Unexpected overlap with prior StarCoder panels: {overlaps}")

    new_coordinates = tuple(
        coordinate
        for coordinate in coordinates
        if not any(_coordinates_match((coordinate[1], coordinate[2]), prior) for prior in prior_coordinates)
    )
    specs = tuple(
        base.SurfaceRunSpec(
            rank=FIRST_SELECTION_RANK + index,
            phase_0_starcoder=phase_0,
            phase_1_starcoder=phase_1,
            run_name_override=(
                f"highagg_a{base._weight_slug(aggregate)}_i{fiber_index:02d}"
                f"_p0_{base._weight_slug(phase_0)}_p1_{base._weight_slug(phase_1)}"
            ),
        )
        for index, (aggregate, phase_0, phase_1, fiber_index) in enumerate(new_coordinates)
    )
    if len(specs) != EXPECTED_NUM_RUNS:
        raise ValueError(f"Expected {EXPECTED_NUM_RUNS} new runs, got {len(specs)}")
    if len({spec.run_name for spec in specs}) != len(specs):
        raise ValueError("High-aggregate fiber run names must be unique")
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--marin-prefix", default=base.DEFAULT_MARIN_PREFIX)
    parser.add_argument("--tpu-type", default=base.DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=base.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=base.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--data-seed", type=int, default=base.DEFAULT_DATA_SEED)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if os.getenv("CI") is not None:
        logger.info("Skipping StarCoder WSD80 high-aggregate fibers in CI")
        return
    if args.marin_prefix != base.DEFAULT_MARIN_PREFIX:
        raise ValueError(f"This historical StarCoder experiment is central1-local: got {args.marin_prefix!r}")
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError(
            "StarCoder child TPU placement must remain central1-local: "
            f"got region={args.tpu_region!r}, zone={args.tpu_zone!r}"
        )
    if args.data_seed != base.DEFAULT_DATA_SEED:
        raise ValueError(f"The reference-seed fibers are pinned to data seed {base.DEFAULT_DATA_SEED}")

    specs = build_run_specs()
    if not 1 <= args.max_concurrent <= len(specs):
        raise ValueError(f"max_concurrent must be in [1, {len(specs)}], got {args.max_concurrent}")

    schedule = base._schedule_summary()
    realized_phase_0_fraction = schedule["boundary_step"] / schedule["total_steps"]
    realized_aggregates = tuple(
        (
            aggregate,
            realized_phase_0_fraction * phase_0 + (1.0 - realized_phase_0_fraction) * phase_1,
        )
        for aggregate, phase_0, phase_1, _ in all_panel_coordinates()
    )
    realized_ranges = {
        aggregate: (
            min(value for nominal, value in realized_aggregates if nominal == aggregate),
            max(value for nominal, value in realized_aggregates if nominal == aggregate),
        )
        for aggregate in AGGREGATE_STARCODER_WEIGHTS
    }
    logger.info(
        "Prepared %d new reference-seed runs across %d fixed-aggregate fibers "
        "(%d coordinates, %d reused controls): aggregates=%s nominal_slope=%.1f "
        "realized_phase_0_fraction=%.6f realized_aggregate_ranges=%s",
        len(specs),
        len(AGGREGATE_STARCODER_WEIGHTS),
        EXPECTED_NUM_COORDINATES,
        len(EXPECTED_REUSED_COORDINATES),
        AGGREGATE_STARCODER_WEIGHTS,
        -NOMINAL_PHASE_0_FRACTION / NOMINAL_PHASE_1_FRACTION,
        realized_phase_0_fraction,
        realized_ranges,
    )
    os.environ["MARIN_PREFIX"] = args.marin_prefix
    steps = base.build_training_steps(
        name_prefix=args.name_prefix,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        data_seed=args.data_seed,
        run_specs=specs,
        wandb_experiment_tag=WANDB_EXPERIMENT_TAG,
        panel_tag=PANEL_TAG,
    )
    if args.dry_run:
        for step in steps:
            lower(step)
        logger.info("Dry-run lowering passed for all %d training handles", len(steps))
        return
    run(*steps, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
