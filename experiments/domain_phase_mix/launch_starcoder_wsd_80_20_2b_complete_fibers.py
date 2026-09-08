# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete both measured 2B WSD80 fibers on a 0.05 contrast grid."""

from __future__ import annotations

import argparse
import csv
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

NAME = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_wsd80_2b_complete_fibers_20260731"
WANDB_EXPERIMENT_TAG = "starcoder_wsd80_2b_complete_fibers"
PANEL_TAG = "2b_complete_fibers_20260731"
TOKEN_BUDGET = 2_000_000_000
ANCHOR_AGGREGATES = (0.35, 0.40)
EXISTING_SIGNED_CONTRASTS = frozenset({-0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25})
SIGNED_CONTRASTS_BY_ANCHOR = {
    0.35: tuple(index / 20 for index in range(-8, 17)),
    0.40: tuple(index / 20 for index in range(-10, 16)),
}
EXPECTED_RUN_COUNT = 29
DEFAULT_MAX_CONCURRENT = EXPECTED_RUN_COUNT
COORDINATE_TOLERANCE = 1e-12
MANIFEST_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "starcoder_wsd80_2b_complete_fibers_20260731"
)


def _phase_fractions() -> tuple[float, float]:
    schedule = base._schedule_summary(TOKEN_BUDGET)
    phase_0 = float(schedule["realized_phase_0_fraction"])
    return phase_0, 1.0 - phase_0


def _coordinate(aggregate: float, signed_contrast: float) -> tuple[float, float]:
    phase_0_fraction, phase_1_fraction = _phase_fractions()
    return (
        aggregate - phase_1_fraction * signed_contrast,
        aggregate + phase_0_fraction * signed_contrast,
    )


def _weight_slug(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


def _signed_slug(value: float) -> str:
    sign = "p" if value >= 0 else "m"
    return f"{sign}{_weight_slug(abs(value))}"


@dataclass(frozen=True)
class FiberRun:
    """One new reference-seed point on a measured 2B aggregate fiber."""

    anchor_index: int
    aggregate: float
    signed_contrast: float

    @property
    def coordinate(self) -> tuple[float, float]:
        return _coordinate(self.aggregate, self.signed_contrast)

    @property
    def run_name(self) -> str:
        return (
            f"d2b_cf{self.anchor_index:02d}_a{_weight_slug(self.aggregate)}"
            f"_d{_signed_slug(self.signed_contrast)}_ref_s{scaling.REFERENCE_SEED}"
        )

    def surface_spec(self, rank: int) -> base.SurfaceRunSpec:
        phase_0_starcoder, phase_1_starcoder = self.coordinate
        return base.SurfaceRunSpec(
            rank=rank,
            phase_0_starcoder=phase_0_starcoder,
            phase_1_starcoder=phase_1_starcoder,
            run_name_override=self.run_name,
        )


def build_runs() -> tuple[FiberRun, ...]:
    """Return every not-yet-measured point on the two complete contrast grids."""
    runs = tuple(
        FiberRun(anchor_index, aggregate, signed_contrast)
        for anchor_index, aggregate in enumerate(ANCHOR_AGGREGATES, start=1)
        for signed_contrast in SIGNED_CONTRASTS_BY_ANCHOR[aggregate]
        if signed_contrast not in EXISTING_SIGNED_CONTRASTS
    )
    if len(runs) != EXPECTED_RUN_COUNT:
        raise ValueError(f"Expected {EXPECTED_RUN_COUNT} new fiber runs, got {len(runs)}")
    if len({item.run_name for item in runs}) != len(runs):
        raise ValueError("Complete-fiber run names are not unique")
    phase_0_fraction, phase_1_fraction = _phase_fractions()
    for item in runs:
        p0, p1 = item.coordinate
        if not 0.0 <= p0 <= 1.0 or not 0.0 <= p1 <= 1.0:
            raise ValueError(f"Infeasible complete-fiber point: {item}")
        aggregate = phase_0_fraction * p0 + phase_1_fraction * p1
        if abs(aggregate - item.aggregate) > COORDINATE_TOLERANCE:
            raise ValueError(f"Complete-fiber point changes aggregate: {item}")
        if abs((p1 - p0) - item.signed_contrast) > COORDINATE_TOLERANCE:
            raise ValueError(f"Complete-fiber contrast is inconsistent: {item}")
    return runs


def manifest() -> dict[str, object]:
    """Return the frozen geometry and requested runs."""
    phase_0_fraction, phase_1_fraction = _phase_fractions()
    rows = []
    for item in build_runs():
        p0, p1 = item.coordinate
        rows.append(
            {
                "run_name": item.run_name,
                "token_budget_requested": TOKEN_BUDGET,
                "anchor_index": item.anchor_index,
                "anchor_aggregate_starcoder": item.aggregate,
                "phase_0_fraction_realized": phase_0_fraction,
                "phase_1_fraction_realized": phase_1_fraction,
                "phase_0_starcoder": p0,
                "phase_1_starcoder": p1,
                "signed_contrast_phase1_minus_phase0": item.signed_contrast,
                "trainer_data_seed": scaling.REFERENCE_SEED,
                "simulated_epoch_subset_seed": scaling.REFERENCE_SEED,
            }
        )
    return {
        "experiment": "StarCoder WSD80 complete 2B tied-optimum fibers",
        "design_version": "2026-07-31",
        "objective_metric": scaling.OBJECTIVE_METRIC,
        "design": {
            "new_runs": len(rows),
            "anchors": list(ANCHOR_AGGREGATES),
            "contrast_step": 0.05,
            "existing_contrasts_reused": sorted(EXISTING_SIGNED_CONTRASTS),
            "requested_contrasts_by_anchor": {
                str(anchor): list(SIGNED_CONTRASTS_BY_ANCHOR[anchor]) for anchor in ANCHOR_AGGREGATES
            },
            "aggregate_matching": "p0=a-beta1*d and p1=a+beta0*d using the realized 2B phase fractions",
        },
        "interpretation_boundary": (
            "The grids approach both feasible endpoints without adding near-duplicate exact-boundary points. "
            "They complete the two prespecified 2B fibers but do not search other aggregates."
        ),
        "runs": rows,
    }


def write_manifest() -> tuple[Path, Path]:
    """Persist JSON and flat CSV copies of the frozen design."""
    payload = manifest()
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    json_path = MANIFEST_DIR / "design_manifest.json"
    csv_path = MANIFEST_DIR / "run_manifest.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    rows = payload["runs"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest contains no runs")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def build_training_steps(
    *, name_prefix: str, tpu_type: str, tpu_region: str, tpu_zone: str
) -> tuple[ArtifactStep[LevanterCheckpoint], ...]:
    """Build all new reference-seed training handles."""
    specs = tuple(item.surface_spec(index) for index, item in enumerate(build_runs(), start=1))
    steps = base.build_training_steps(
        name_prefix=name_prefix,
        tpu_type=tpu_type,
        tpu_region=tpu_region,
        tpu_zone=tpu_zone,
        data_seed=scaling.REFERENCE_SEED,
        run_specs=specs,
        wandb_experiment_tag=WANDB_EXPERIMENT_TAG,
        panel_tag=PANEL_TAG,
        experiment_budget=TOKEN_BUDGET,
        target_budget=base.TARGET_BUDGET,
    )
    if len(steps) != EXPECTED_RUN_COUNT:
        raise ValueError(f"Expected {EXPECTED_RUN_COUNT} training handles, got {len(steps)}")
    return tuple(steps)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--marin-prefix", default=base.DEFAULT_MARIN_PREFIX)
    parser.add_argument("--tpu-type", default=base.DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=base.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=base.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if os.getenv("CI") is not None:
        logger.info("Skipping StarCoder WSD80 complete 2B fibers in CI")
        return
    if args.marin_prefix != base.DEFAULT_MARIN_PREFIX:
        raise ValueError(f"This historical StarCoder experiment is central1-local: got {args.marin_prefix!r}")
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError(
            "StarCoder child TPU placement must remain central1-local: "
            f"got region={args.tpu_region!r}, zone={args.tpu_zone!r}"
        )
    requested_runs = build_runs()
    if not 1 <= args.max_concurrent <= len(requested_runs):
        raise ValueError(f"max_concurrent must be in [1, {len(requested_runs)}], got {args.max_concurrent}")
    if args.write_manifest:
        json_path, csv_path = write_manifest()
        logger.info("Wrote complete-fiber manifests to %s and %s", json_path, csv_path)
    logger.info(
        "Prepared %d complete 2B fiber runs: %s",
        len(requested_runs),
        {aggregate: sum(item.aggregate == aggregate for item in requested_runs) for aggregate in ANCHOR_AGGREGATES},
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
