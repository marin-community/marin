# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample the fixed-aggregate fiber through the best tied StarCoder WSD80 policy."""

from __future__ import annotations

import argparse
import logging
import os

from marin.execution.lazy import lower, run

from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base
from experiments.domain_phase_mix.launch_starcoder_wsd_80_20_repeat_panel import (
    NEW_DATA_SEEDS,
    SCHEDULES,
)
from experiments.domain_phase_mix.starcoder_wsd_80_20_refinement_coordinates import (
    DRIFT_ANCHOR_COORDINATE,
    REFINEMENT_COORDINATES,
)

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_wsd80_fixedagg03_fiber31_repeat5x4_20260727"
WANDB_EXPERIMENT_TAG = "starcoder_wsd80_fixedagg03_fiber31_repeat5x4"
PANEL_TAG = "fixedagg03_fiber31_repeat5x4"
NOMINAL_PHASE_0_FRACTION = 0.80
NOMINAL_PHASE_1_FRACTION = 0.20
TIED_STARCODER_WEIGHT = 0.30
NUM_FIBER_POINTS = 31
TIED_FIBER_INDEX = 9
REPEATED_FIBER_INDICES = (3, 6, TIED_FIBER_INDEX, 12, 15)
REUSED_REFERENCE_INDEX = 15
FIRST_SELECTION_RANK = 109
NUM_REFERENCE_RUNS = NUM_FIBER_POINTS - 1
NUM_REPEAT_RUNS = len(REPEATED_FIBER_INDICES) * len(NEW_DATA_SEEDS)
NUM_RUNS = NUM_REFERENCE_RUNS + NUM_REPEAT_RUNS
DEFAULT_MAX_CONCURRENT = NUM_RUNS
COORDINATE_TOLERANCE = 1e-12


def realized_phase_fractions() -> tuple[float, float]:
    """Return the token-weighted phase fractions after step alignment."""
    schedule = base._schedule_summary()
    phase_0 = schedule["boundary_step"] / schedule["total_steps"]
    return phase_0, 1.0 - phase_0


def fixed_aggregate_coordinates() -> tuple[tuple[float, float], ...]:
    """Return a uniform phase-1 grid on the nominal 80/20 aggregate fiber."""
    coordinates = []
    for index in range(NUM_FIBER_POINTS):
        phase_1_starcoder = index / (NUM_FIBER_POINTS - 1)
        phase_0_starcoder = (
            TIED_STARCODER_WEIGHT - NOMINAL_PHASE_1_FRACTION * phase_1_starcoder
        ) / NOMINAL_PHASE_0_FRACTION
        coordinates.append((phase_0_starcoder, phase_1_starcoder))
    return tuple(coordinates)


def _coordinates_match(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return max(abs(left[0] - right[0]), abs(left[1] - right[1])) <= COORDINATE_TOLERANCE


def build_run_specs() -> tuple[base.SurfaceRunSpec, ...]:
    """Return the fixed fiber panel after geometric and overlap checks."""
    coordinates = fixed_aggregate_coordinates()
    if len(coordinates) != NUM_FIBER_POINTS:
        raise ValueError(f"Expected {NUM_FIBER_POINTS} coordinates, got {len(coordinates)}")
    rounded_coordinates = {(round(p0, 12), round(p1, 12)) for p0, p1 in coordinates}
    if len(rounded_coordinates) != len(coordinates):
        raise ValueError("Fiber coordinates must be unique")

    for p0, p1 in coordinates:
        if not 0.0 <= p0 <= 1.0 or not 0.0 <= p1 <= 1.0:
            raise ValueError(f"Coordinate outside simplex interval: {(p0, p1)}")
        aggregate = NOMINAL_PHASE_0_FRACTION * p0 + NOMINAL_PHASE_1_FRACTION * p1
        if abs(aggregate - TIED_STARCODER_WEIGHT) > COORDINATE_TOLERANCE:
            raise ValueError(f"Coordinate does not preserve the nominal 80/20 aggregate: {(p0, p1, aggregate)}")

    previous_coordinates = (
        *base.SURFACE_COORDINATES,
        *REFINEMENT_COORDINATES,
        DRIFT_ANCHOR_COORDINATE,
        *((p0, p1) for _, p0, p1 in SCHEDULES),
    )
    overlaps = [
        coordinate
        for coordinate in coordinates
        if any(_coordinates_match(coordinate, previous) for previous in previous_coordinates)
    ]
    expected_overlaps = (coordinates[TIED_FIBER_INDEX], coordinates[REUSED_REFERENCE_INDEX])
    if len(overlaps) != len(expected_overlaps) or any(
        not any(_coordinates_match(overlap, expected) for overlap in overlaps) for expected in expected_overlaps
    ):
        raise ValueError(f"Unexpected overlap with prior panels: {overlaps}")
    tied_coordinate = coordinates[TIED_FIBER_INDEX]
    if not _coordinates_match(tied_coordinate, (TIED_STARCODER_WEIGHT, TIED_STARCODER_WEIGHT)):
        raise ValueError(f"Fiber must pass through the tied anchor: {tied_coordinate}")
    for early_index, late_index in ((3, 15), (6, 12)):
        midpoint = tuple(
            (left + right) / 2 for left, right in zip(coordinates[early_index], coordinates[late_index], strict=True)
        )
        if not _coordinates_match(midpoint, tied_coordinate):
            raise ValueError(f"Repeat coordinates must be antithetic around the tied anchor: {midpoint}")

    fiber_specs = tuple(
        base.SurfaceRunSpec(
            rank=FIRST_SELECTION_RANK + index,
            phase_0_starcoder=p0,
            phase_1_starcoder=p1,
            run_name_override=(f"fiber31_i{index:02d}_p0_{base._weight_slug(p0)}_p1_{base._weight_slug(p1)}"),
        )
        for index, (p0, p1) in enumerate(coordinates)
        if index != REUSED_REFERENCE_INDEX
    )
    repeat_specs = tuple(
        base.SurfaceRunSpec(
            rank=FIRST_SELECTION_RANK + NUM_FIBER_POINTS + repeat_index,
            phase_0_starcoder=coordinates[fiber_index][0],
            phase_1_starcoder=coordinates[fiber_index][1],
            run_name_override=f"fiberrep_i{fiber_index:02d}_seed{seed}",
            data_seed_override=seed,
        )
        for repeat_index, (seed, fiber_index) in enumerate(
            (seed, fiber_index) for seed in NEW_DATA_SEEDS for fiber_index in REPEATED_FIBER_INDICES
        )
    )
    specs = (*fiber_specs, *repeat_specs)
    if len(specs) != NUM_RUNS:
        raise ValueError(f"Expected {NUM_RUNS} runs, got {len(specs)}")
    if len({spec.run_name for spec in specs}) != len(specs):
        raise ValueError("Fiber run names must be unique")
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
        logger.info("Skipping StarCoder WSD80 fixed-aggregate fiber in CI")
        return
    if args.marin_prefix != base.DEFAULT_MARIN_PREFIX:
        raise ValueError(f"This historical StarCoder experiment is central1-local: got {args.marin_prefix!r}")
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError(
            "StarCoder child TPU placement must remain central1-local: "
            f"got region={args.tpu_region!r}, zone={args.tpu_zone!r}"
        )
    if args.data_seed != base.DEFAULT_DATA_SEED:
        raise ValueError(f"The reference-seed fiber is pinned to data seed {base.DEFAULT_DATA_SEED}")

    specs = build_run_specs()
    if not 1 <= args.max_concurrent <= len(specs):
        raise ValueError(f"max_concurrent must be in [1, {len(specs)}], got {args.max_concurrent}")

    phase_0_fraction, phase_1_fraction = realized_phase_fractions()
    coordinates = fixed_aggregate_coordinates()
    realized_aggregates = [phase_0_fraction * phase_0 + phase_1_fraction * phase_1 for phase_0, phase_1 in coordinates]
    logger.info(
        "Prepared %d fixed-aggregate runs (%d reference-seed requests, 1 reused reference coordinate, "
        "%d matched repeats): "
        "nominal_aggregate=%.6f nominal_slope=%.6f realized_phase_fractions=(%.6f, %.6f) "
        "realized_aggregate_range=(%.6f, %.6f) new_coordinates=%d drift_controls=1",
        len(specs),
        NUM_REFERENCE_RUNS,
        NUM_REPEAT_RUNS,
        TIED_STARCODER_WEIGHT,
        -NOMINAL_PHASE_0_FRACTION / NOMINAL_PHASE_1_FRACTION,
        phase_0_fraction,
        phase_1_fraction,
        min(realized_aggregates),
        max(realized_aggregates),
        NUM_FIBER_POINTS - 2,
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
