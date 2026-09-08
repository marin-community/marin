# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Launch paired repeats for three candidate schedules on the StarCoder WSD surface."""

from __future__ import annotations

import argparse
import logging
import os

from marin.execution.lazy import lower, run

from experiments.domain_phase_mix.launch_starcoder_wsd_80_20_surface import (
    DEFAULT_MARIN_PREFIX,
    DEFAULT_TPU_REGION,
    DEFAULT_TPU_TYPE,
    DEFAULT_TPU_ZONE,
    SurfaceRunSpec,
    build_training_steps,
)

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_wsd80_20_repeat3x4_20260711"
WANDB_EXPERIMENT_TAG = "starcoder_wsd80_repeat3x4"
PANEL_TAG = "repeat3x4"
DEFAULT_MAX_CONCURRENT = 12
REFERENCE_DATA_SEED = 20_260_711
NEW_DATA_SEEDS = (20_260_712, 20_260_713, 20_260_714, 20_260_715)
SCHEDULES = (
    ("global", 0.1452468603730965, 0.517364768878253),
    ("boundary", 0.0, 0.6),
    ("constant", 0.3, 0.3),
)


def build_repeat_run_specs() -> tuple[SurfaceRunSpec, ...]:
    """Return the immutable 12-run paired repeat panel."""
    specs = tuple(
        SurfaceRunSpec(
            rank=index,
            phase_0_starcoder=phase_0,
            phase_1_starcoder=phase_1,
            run_name_override=f"wsd80_repeat_{schedule}_seed{seed}",
            data_seed_override=seed,
        )
        for index, (seed, (schedule, phase_0, phase_1)) in enumerate(
            ((seed, schedule) for seed in NEW_DATA_SEEDS for schedule in SCHEDULES),
            start=1,
        )
    )
    if len(specs) != 12:
        raise ValueError(f"Expected exactly 12 repeat runs, got {len(specs)}")
    if len({spec.run_name for spec in specs}) != len(specs):
        raise ValueError("Repeat run names must be unique")
    if {spec.data_seed_override for spec in specs} != set(NEW_DATA_SEEDS):
        raise ValueError("Every configured repeat seed must be represented")
    for seed in NEW_DATA_SEEDS:
        if sum(spec.data_seed_override == seed for spec in specs) != len(SCHEDULES):
            raise ValueError(f"Seed {seed} must have one run for every schedule")
    return specs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--marin-prefix", default=DEFAULT_MARIN_PREFIX)
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if os.getenv("CI") is not None:
        logger.info("Skipping StarCoder WSD repeat panel in CI")
        return
    if args.marin_prefix != DEFAULT_MARIN_PREFIX:
        raise ValueError(f"This historical StarCoder experiment is central1-local: got {args.marin_prefix!r}")
    if args.tpu_region != DEFAULT_TPU_REGION or args.tpu_zone != DEFAULT_TPU_ZONE:
        raise ValueError(
            "StarCoder child TPU placement must remain central1-local: "
            f"got region={args.tpu_region!r}, zone={args.tpu_zone!r}"
        )

    run_specs = build_repeat_run_specs()
    if args.max_concurrent < 1 or args.max_concurrent > len(run_specs):
        raise ValueError(f"max_concurrent must be in [1, {len(run_specs)}], got {args.max_concurrent}")

    logger.info(
        "Prepared %d paired repeats across %d schedules and %d new seeds; "
        "existing seed %d remains the first matched observation",
        len(run_specs),
        len(SCHEDULES),
        len(NEW_DATA_SEEDS),
        REFERENCE_DATA_SEED,
    )
    os.environ["MARIN_PREFIX"] = args.marin_prefix
    steps = build_training_steps(
        name_prefix=args.name_prefix,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        data_seed=REFERENCE_DATA_SEED,
        run_specs=run_specs,
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
