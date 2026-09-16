# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Refine the StarCoder 80/20 WSD surface around its observed low-BPB valley."""

from __future__ import annotations

import argparse
import logging
import os

from marin.execution.lazy import lower, run

from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base
from experiments.domain_phase_mix.starcoder_wsd_80_20_refinement_coordinates import (
    DRIFT_ANCHOR_COORDINATE,
    REFINEMENT_COORDINATES,
)

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_wsd80_20_refinement44_20260714"
WANDB_EXPERIMENT_TAG = "starcoder_wsd80_20_refine44"
PANEL_TAG = "refinement44"
DEFAULT_MAX_CONCURRENT = 44


def build_run_specs() -> tuple[base.SurfaceRunSpec, ...]:
    """Return the immutable refinement manifest after checking it is new."""
    if len(REFINEMENT_COORDINATES) != 43:
        raise ValueError(f"Expected 43 refinement coordinates, got {len(REFINEMENT_COORDINATES)}")
    if len(set(REFINEMENT_COORDINATES)) != len(REFINEMENT_COORDINATES):
        raise ValueError("Refinement coordinates must be unique")
    overlap = set(REFINEMENT_COORDINATES).intersection(base.SURFACE_COORDINATES)
    if overlap:
        raise ValueError(f"Refinement duplicates completed surface points: {sorted(overlap)}")
    for p0, p1 in REFINEMENT_COORDINATES:
        if not 0.0 <= p0 < p1 <= 1.0:
            raise ValueError(f"Refinement must stay on the p1 > p0 half-plane: {(p0, p1)}")
    refinement_specs = tuple(
        base.SurfaceRunSpec(rank=65 + index, phase_0_starcoder=p0, phase_1_starcoder=p1)
        for index, (p0, p1) in enumerate(REFINEMENT_COORDINATES)
    )
    if DRIFT_ANCHOR_COORDINATE not in base.SURFACE_COORDINATES:
        raise ValueError("The drift anchor must duplicate a completed surface coordinate")
    anchor = base.SurfaceRunSpec(
        rank=65 + len(REFINEMENT_COORDINATES),
        phase_0_starcoder=DRIFT_ANCHOR_COORDINATE[0],
        phase_1_starcoder=DRIFT_ANCHOR_COORDINATE[1],
    )
    return (*refinement_specs, anchor)


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
    args = parse_args()
    if os.getenv("CI") is not None:
        logger.info("Skipping StarCoder WSD refinement in CI")
        return
    if args.marin_prefix != base.DEFAULT_MARIN_PREFIX:
        raise ValueError(f"This historical StarCoder experiment is central1-local: got {args.marin_prefix!r}")
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError(
            "StarCoder child TPU placement must remain central1-local: "
            f"got region={args.tpu_region!r}, zone={args.tpu_zone!r}"
        )
    specs = build_run_specs()
    if not 1 <= args.max_concurrent <= len(specs):
        raise ValueError(f"max_concurrent must be in [1, {len(specs)}], got {args.max_concurrent}")

    schedule = base._schedule_summary()
    logger.info(
        "Prepared %d matched-seed WSD refinement runs: total_steps=%d boundary_step=%d shared_data_seed=%d",
        len(specs),
        schedule["total_steps"],
        schedule["boundary_step"],
        args.data_seed,
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
