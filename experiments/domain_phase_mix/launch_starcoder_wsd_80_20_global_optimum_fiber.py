# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample the fixed-aggregate fiber through the best observed StarCoder WSD80 policy."""

from __future__ import annotations

import argparse
import logging
import os

from marin.execution.lazy import lower, run

from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base
from experiments.domain_phase_mix.launch_starcoder_wsd_80_20_fixed_aggregate_fiber import (
    fixed_aggregate_coordinates as prior_fiber_coordinates,
)
from experiments.domain_phase_mix.launch_starcoder_wsd_80_20_repeat_panel import (
    NEW_DATA_SEEDS,
    SCHEDULES,
)
from experiments.domain_phase_mix.starcoder_wsd_80_20_refinement_coordinates import (
    DRIFT_ANCHOR_COORDINATE,
    REFINEMENT_COORDINATES,
)

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_wsd80_fixedagg018_optfiber32_repeat6x4_20260728"
WANDB_EXPERIMENT_TAG = "starcoder_wsd80_fixedagg018_optfiber32_repeat6x4"
PANEL_TAG = "fixedagg018_optfiber32_repeat6x4"
NOMINAL_PHASE_0_FRACTION = 0.80
NOMINAL_PHASE_1_FRACTION = 0.20
AGGREGATE_STARCODER_WEIGHT = 0.18
PHASE_1_STEP = 0.03
PHASE_1_GRID_STEPS = 30
OBSERVED_OPTIMUM = (0.10, 0.50)
TIED_COORDINATE = (AGGREGATE_STARCODER_WEIGHT, AGGREGATE_STARCODER_WEIGHT)
REPEATED_PHASE_1_WEIGHTS = (0.09, 0.15, 0.18, 0.21, 0.27, OBSERVED_OPTIMUM[1])
REUSED_REFERENCE_COORDINATES = (OBSERVED_OPTIMUM, (0.0, 0.90))
FIRST_SELECTION_RANK = 160
NUM_FIBER_COORDINATES = PHASE_1_GRID_STEPS + 2
NUM_REFERENCE_RUNS = NUM_FIBER_COORDINATES - len(REUSED_REFERENCE_COORDINATES)
NUM_REPEAT_RUNS = len(REPEATED_PHASE_1_WEIGHTS) * len(NEW_DATA_SEEDS)
NUM_RUNS = NUM_REFERENCE_RUNS + NUM_REPEAT_RUNS
DEFAULT_MAX_CONCURRENT = NUM_RUNS
COORDINATE_TOLERANCE = 1e-12


def realized_phase_fractions() -> tuple[float, float]:
    """Return the token-weighted phase fractions after step alignment."""
    schedule = base._schedule_summary()
    phase_0 = schedule["boundary_step"] / schedule["total_steps"]
    return phase_0, 1.0 - phase_0


def _phase_0_weight(phase_1_weight: float) -> float:
    return (AGGREGATE_STARCODER_WEIGHT - NOMINAL_PHASE_1_FRACTION * phase_1_weight) / NOMINAL_PHASE_0_FRACTION


def _coordinates_match(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return max(abs(left[0] - right[0]), abs(left[1] - right[1])) <= COORDINATE_TOLERANCE


def fixed_aggregate_coordinates() -> tuple[tuple[float, float], ...]:
    """Return the aggregate-0.18 fiber with exact tied and observed-optimum points."""
    coordinates = [
        (_phase_0_weight(index * PHASE_1_STEP), index * PHASE_1_STEP) for index in range(PHASE_1_GRID_STEPS + 1)
    ]
    if not any(_coordinates_match(coordinate, OBSERVED_OPTIMUM) for coordinate in coordinates):
        coordinates.append(OBSERVED_OPTIMUM)
    return tuple(sorted(coordinates, key=lambda coordinate: coordinate[1]))


def _coordinate_index(coordinates: tuple[tuple[float, float], ...], target: tuple[float, float]) -> int:
    matches = [index for index, coordinate in enumerate(coordinates) if _coordinates_match(coordinate, target)]
    if len(matches) != 1:
        raise ValueError(f"Expected one match for {target}, found {matches}")
    return matches[0]


def build_run_specs() -> tuple[base.SurfaceRunSpec, ...]:
    """Return the global-optimum fiber panel after geometric and overlap checks."""
    coordinates = fixed_aggregate_coordinates()
    if len(coordinates) != NUM_FIBER_COORDINATES:
        raise ValueError(f"Expected {NUM_FIBER_COORDINATES} coordinates, got {len(coordinates)}")
    rounded_coordinates = {(round(p0, 12), round(p1, 12)) for p0, p1 in coordinates}
    if len(rounded_coordinates) != len(coordinates):
        raise ValueError("Fiber coordinates must be unique")

    for p0, p1 in coordinates:
        if not 0.0 <= p0 <= 1.0 or not 0.0 <= p1 <= 1.0:
            raise ValueError(f"Coordinate outside simplex interval: {(p0, p1)}")
        aggregate = NOMINAL_PHASE_0_FRACTION * p0 + NOMINAL_PHASE_1_FRACTION * p1
        if abs(aggregate - AGGREGATE_STARCODER_WEIGHT) > COORDINATE_TOLERANCE:
            raise ValueError(f"Coordinate does not preserve the nominal 80/20 aggregate: {(p0, p1, aggregate)}")

    previous_coordinates = (
        *base.SURFACE_COORDINATES,
        *REFINEMENT_COORDINATES,
        DRIFT_ANCHOR_COORDINATE,
        *prior_fiber_coordinates(),
        *((p0, p1) for _, p0, p1 in SCHEDULES),
    )
    overlaps = [
        coordinate
        for coordinate in coordinates
        if any(_coordinates_match(coordinate, previous) for previous in previous_coordinates)
    ]
    if len(overlaps) != len(REUSED_REFERENCE_COORDINATES) or any(
        not any(_coordinates_match(overlap, expected) for overlap in overlaps)
        for expected in REUSED_REFERENCE_COORDINATES
    ):
        raise ValueError(f"Unexpected overlap with prior panels: {overlaps}")

    tied_index = _coordinate_index(coordinates, TIED_COORDINATE)
    observed_optimum_index = _coordinate_index(coordinates, OBSERVED_OPTIMUM)
    repeated_indices = tuple(
        _coordinate_index(coordinates, (_phase_0_weight(phase_1), phase_1)) for phase_1 in REPEATED_PHASE_1_WEIGHTS
    )
    for early_phase_1, late_phase_1 in ((0.09, 0.27), (0.15, 0.21)):
        early = coordinates[_coordinate_index(coordinates, (_phase_0_weight(early_phase_1), early_phase_1))]
        late = coordinates[_coordinate_index(coordinates, (_phase_0_weight(late_phase_1), late_phase_1))]
        midpoint = tuple((left + right) / 2 for left, right in zip(early, late, strict=True))
        if not _coordinates_match(midpoint, TIED_COORDINATE):
            raise ValueError(f"Repeat coordinates must be antithetic around the tied anchor: {midpoint}")
    if tied_index not in repeated_indices or observed_optimum_index not in repeated_indices:
        raise ValueError("The tied control and observed optimum must both receive matched repeats")

    fiber_specs = tuple(
        base.SurfaceRunSpec(
            rank=FIRST_SELECTION_RANK + index,
            phase_0_starcoder=p0,
            phase_1_starcoder=p1,
            run_name_override=(f"optfiber_i{index:02d}_p0_{base._weight_slug(p0)}_p1_{base._weight_slug(p1)}"),
        )
        for index, (p0, p1) in enumerate(coordinates)
        if not any(_coordinates_match((p0, p1), reused) for reused in REUSED_REFERENCE_COORDINATES)
    )
    repeat_specs = tuple(
        base.SurfaceRunSpec(
            rank=FIRST_SELECTION_RANK + NUM_FIBER_COORDINATES + repeat_index,
            phase_0_starcoder=coordinates[fiber_index][0],
            phase_1_starcoder=coordinates[fiber_index][1],
            run_name_override=f"optfiberrep_i{fiber_index:02d}_seed{seed}",
            data_seed_override=seed,
        )
        for repeat_index, (seed, fiber_index) in enumerate(
            (seed, fiber_index) for seed in NEW_DATA_SEEDS for fiber_index in repeated_indices
        )
    )
    specs = (*fiber_specs, *repeat_specs)
    if len(fiber_specs) != NUM_REFERENCE_RUNS:
        raise ValueError(f"Expected {NUM_REFERENCE_RUNS} reference runs, got {len(fiber_specs)}")
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
        logger.info("Skipping StarCoder WSD80 global-optimum fiber in CI")
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
        "Prepared %d fixed-aggregate runs (%d reference-seed requests, %d reused reference coordinates, "
        "%d matched repeats): nominal_aggregate=%.6f nominal_slope=%.6f "
        "realized_phase_fractions=(%.6f, %.6f) realized_aggregate_range=(%.6f, %.6f) "
        "fiber_coordinates=%d tied=%s observed_optimum=%s",
        len(specs),
        NUM_REFERENCE_RUNS,
        len(REUSED_REFERENCE_COORDINATES),
        NUM_REPEAT_RUNS,
        AGGREGATE_STARCODER_WEIGHT,
        -NOMINAL_PHASE_0_FRACTION / NOMINAL_PHASE_1_FRACTION,
        phase_0_fraction,
        phase_1_fraction,
        min(realized_aggregates),
        max(realized_aggregates),
        len(coordinates),
        TIED_COORDINATE,
        OBSERVED_OPTIMUM,
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
