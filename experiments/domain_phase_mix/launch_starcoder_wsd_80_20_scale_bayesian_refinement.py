# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the frozen first-stage Bayesian refinement of WSD80 optima."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from marin.execution.lazy import ArtifactStep, lower, run
from marin.training.training import LevanterCheckpoint

from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_fixed_model_token_scaling as scaling
from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_wsd80_scale_bo_stage1_20260731"
WANDB_EXPERIMENT_TAG = "starcoder_wsd80_scale_bo_stage1"
PANEL_TAG = "scale_bo_stage1_20260731"
DESIGN_PATH = Path(__file__).with_name("starcoder_wsd80_scale_bayesian_refinement_design_20260731.json")
EXPECTED_RUN_COUNT = 52
DEFAULT_MAX_CONCURRENT = 32


@dataclass(frozen=True)
class RefinementRun:
    """One frozen acquisition or incumbent-repeat checkpoint."""

    token_budget: int
    phase_0_starcoder: float
    phase_1_starcoder: float
    run_name: str
    trainer_data_seed: int
    simulated_epoch_subset_seed: int

    def surface_spec(self, rank: int) -> base.SurfaceRunSpec:
        data_seed_override = self.trainer_data_seed if self.trainer_data_seed != scaling.REFERENCE_SEED else None
        subset_seed_override = (
            self.simulated_epoch_subset_seed if self.simulated_epoch_subset_seed != scaling.REFERENCE_SEED else None
        )
        return base.SurfaceRunSpec(
            rank=rank,
            phase_0_starcoder=self.phase_0_starcoder,
            phase_1_starcoder=self.phase_1_starcoder,
            run_name_override=self.run_name,
            data_seed_override=data_seed_override,
            simulated_epoch_subset_seed_override=subset_seed_override,
        )


def load_design() -> tuple[RefinementRun, ...]:
    """Load and validate the immutable source-controlled design."""
    payload = json.loads(DESIGN_PATH.read_text())
    if payload.get("design_version") != "2026-07-31":
        raise ValueError(f"Unexpected Bayesian-refinement design version in {DESIGN_PATH}")
    if payload.get("objective_metric") != scaling.OBJECTIVE_METRIC:
        raise ValueError("Frozen design targets an unexpected objective")
    raw_rows = payload.get("runs")
    if not isinstance(raw_rows, list) or len(raw_rows) != EXPECTED_RUN_COUNT:
        raise ValueError(f"Expected {EXPECTED_RUN_COUNT} frozen runs")
    runs = tuple(
        RefinementRun(
            token_budget=int(row["token_budget_requested"]),
            phase_0_starcoder=float(row["phase_0_starcoder"]),
            phase_1_starcoder=float(row["phase_1_starcoder"]),
            run_name=str(row["run_name"]),
            trainer_data_seed=int(row["trainer_data_seed"]),
            simulated_epoch_subset_seed=int(row["simulated_epoch_subset_seed"]),
        )
        for row in raw_rows
    )
    if len({item.run_name for item in runs}) != len(runs):
        raise ValueError("Frozen Bayesian-refinement run names are not unique")
    if {item.token_budget for item in runs} != set(scaling.TOKEN_BUDGETS):
        raise ValueError("Frozen design does not cover every token-budget rung")
    for item in runs:
        if not 0.0 <= item.phase_0_starcoder <= 1.0 or not 0.0 <= item.phase_1_starcoder <= 1.0:
            raise ValueError(f"Invalid frozen coordinate: {item}")
    return runs


def build_training_steps(
    *,
    name_prefix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
) -> tuple[ArtifactStep[LevanterCheckpoint], ...]:
    """Build training handles grouped by token budget."""
    requested_runs = load_design()
    steps: list[ArtifactStep[LevanterCheckpoint]] = []
    rank = 1
    for token_budget in scaling.TOKEN_BUDGETS:
        budget_runs = [item for item in requested_runs if item.token_budget == token_budget]
        specs = tuple(item.surface_spec(rank + index) for index, item in enumerate(budget_runs))
        rank += len(specs)
        steps.extend(
            base.build_training_steps(
                name_prefix=name_prefix,
                tpu_type=tpu_type,
                tpu_region=tpu_region,
                tpu_zone=tpu_zone,
                data_seed=scaling.REFERENCE_SEED,
                run_specs=specs,
                wandb_experiment_tag=WANDB_EXPERIMENT_TAG,
                panel_tag=f"{PANEL_TAG}_{token_budget // 1_000_000_000}b",
                experiment_budget=token_budget,
                target_budget=base.TARGET_BUDGET,
            )
        )
    if len(steps) != len(requested_runs):
        raise ValueError(f"Expected {len(requested_runs)} training handles, got {len(steps)}")
    return tuple(steps)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--marin-prefix", default=base.DEFAULT_MARIN_PREFIX)
    parser.add_argument("--tpu-type", default=base.DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=base.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=base.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if os.getenv("CI") is not None:
        logger.info("Skipping StarCoder WSD80 Bayesian refinement in CI")
        return
    if args.marin_prefix != base.DEFAULT_MARIN_PREFIX:
        raise ValueError(f"This historical StarCoder experiment is central1-local: got {args.marin_prefix!r}")
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError(
            "StarCoder child TPU placement must remain central1-local: "
            f"got region={args.tpu_region!r}, zone={args.tpu_zone!r}"
        )

    requested_runs = load_design()
    if not 1 <= args.max_concurrent <= len(requested_runs):
        raise ValueError(f"max_concurrent must be in [1, {len(requested_runs)}], got {args.max_concurrent}")
    logger.info(
        "Prepared %d first-stage Bayesian-refinement runs: %s",
        len(requested_runs),
        {
            token_budget: sum(item.token_budget == token_budget for item in requested_runs)
            for token_budget in scaling.TOKEN_BUDGETS
        },
    )
    os.environ["MARIN_PREFIX"] = args.marin_prefix
    steps = build_training_steps(
        name_prefix=args.name_prefix,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )
    if args.dry_run:
        for step in steps:
            lower(step)
        logger.info("Dry-run lowering passed for all %d training handles", len(steps))
        return
    run(*steps, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
