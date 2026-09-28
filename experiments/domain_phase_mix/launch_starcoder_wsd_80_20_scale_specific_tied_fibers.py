# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Measure antithetic phase fibers through the tied optimum at each WSD80 token rung."""

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

NAME = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_wsd80_scale_tied_fibers_1b8b_20260731"
WANDB_EXPERIMENT_TAG = "starcoder_wsd80_scale_tied_fibers"
PANEL_TAG = "scale_tied_fibers_1b8b"
REFERENCE_SEED = scaling.REFERENCE_SEED
JOINT_RANDOMNESS_SEEDS = scaling.JOINT_RANDOMNESS_SEEDS
COMMON_SIGNED_CONTRASTS = (-0.25, -0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20, 0.25)
REPEATED_SIGNED_CONTRASTS = (-0.20, 0.0, 0.20)
DEFAULT_MAX_CONCURRENT = 48
EXPECTED_NUM_RUNS = 132
COORDINATE_TOLERANCE = 1e-12
MANIFEST_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "starcoder_wsd80_scale_specific_tied_fibers_20260731"
)


@dataclass(frozen=True)
class FiberAnchor:
    """One tied policy whose aggregate-matched phase fiber will be measured."""

    index: int
    token_budget: int
    aggregate: float
    role: str
    tied_control_wandb_id: str

    @property
    def budget_slug(self) -> str:
        return f"{self.token_budget // 1_000_000_000}b"

    @property
    def anchor_slug(self) -> str:
        return f"a{base._weight_slug(self.aggregate)}"

    @property
    def phase_fractions(self) -> tuple[float, float]:
        schedule = base._schedule_summary(self.token_budget)
        phase_0 = float(schedule["realized_phase_0_fraction"])
        return phase_0, 1.0 - phase_0

    def coordinate(self, signed_contrast: float) -> tuple[float, float]:
        """Return p0,p1 with p1-p0=contrast and exact realized aggregate."""
        phase_0_fraction, phase_1_fraction = self.phase_fractions
        phase_0_starcoder = self.aggregate - phase_1_fraction * signed_contrast
        phase_1_starcoder = self.aggregate + phase_0_fraction * signed_contrast
        return phase_0_starcoder, phase_1_starcoder


ANCHORS: tuple[FiberAnchor, ...] = (
    FiberAnchor(
        index=1,
        token_budget=1_000_000_000,
        aggregate=0.30,
        role="measured_grid_minimum",
        tied_control_wandb_id="surface64_r34_p0_0p3000_p1_0p3000",
    ),
    FiberAnchor(
        index=2,
        token_budget=2_000_000_000,
        aggregate=0.35,
        role="measured_grid_minimum",
        tied_control_wandb_id="d2b_c06_ref_s20260711",
    ),
    FiberAnchor(
        index=3,
        token_budget=2_000_000_000,
        aggregate=0.40,
        role="broad_basin_sensitivity",
        tied_control_wandb_id="d2b_tieddiag_p0p4000_ref_s20260711",
    ),
    FiberAnchor(
        index=4,
        token_budget=4_000_000_000,
        aggregate=0.55,
        role="measured_grid_minimum",
        tied_control_wandb_id="d4b_tieddiag_p0p5500_ref_s20260711",
    ),
    FiberAnchor(
        index=5,
        token_budget=8_000_000_000,
        aggregate=0.80,
        role="measured_grid_minimum",
        tied_control_wandb_id="d8b_tieddiag_p0p8000_ref_s20260711",
    ),
    FiberAnchor(
        index=6,
        token_budget=8_000_000_000,
        aggregate=0.75,
        role="broad_basin_sensitivity",
        tied_control_wandb_id="d8b_tieddiag_p0p7500_ref_s20260711",
    ),
)


def _signed_slug(value: float) -> str:
    sign = "p" if value >= 0 else "m"
    return f"{sign}{base._weight_slug(abs(value))}"


@dataclass(frozen=True)
class FiberRun:
    """One new checkpoint at a frozen anchor, contrast, and joint seed."""

    anchor: FiberAnchor
    signed_contrast: float
    replicate_kind: str
    seed: int

    @property
    def run_name(self) -> str:
        kind_slug = "ref" if self.replicate_kind == "reference" else "joint"
        return (
            f"d{self.anchor.budget_slug}_f{self.anchor.index:02d}_{self.anchor.anchor_slug}"
            f"_d{_signed_slug(self.signed_contrast)}_{kind_slug}_s{self.seed}"
        )

    def surface_spec(self, rank: int) -> base.SurfaceRunSpec:
        phase_0_starcoder, phase_1_starcoder = self.anchor.coordinate(self.signed_contrast)
        seed_override = self.seed if self.seed != REFERENCE_SEED else None
        return base.SurfaceRunSpec(
            rank=rank,
            phase_0_starcoder=phase_0_starcoder,
            phase_1_starcoder=phase_1_starcoder,
            run_name_override=self.run_name,
            data_seed_override=seed_override,
            simulated_epoch_subset_seed_override=seed_override,
        )


def build_fiber_runs() -> tuple[FiberRun, ...]:
    """Return the immutable 132-checkpoint scale-specific fiber design."""
    runs: list[FiberRun] = []
    for anchor in ANCHORS:
        runs.extend(
            FiberRun(
                anchor=anchor,
                signed_contrast=signed_contrast,
                replicate_kind="reference",
                seed=REFERENCE_SEED,
            )
            for signed_contrast in COMMON_SIGNED_CONTRASTS
        )
        runs.extend(
            FiberRun(
                anchor=anchor,
                signed_contrast=signed_contrast,
                replicate_kind="joint_randomness",
                seed=seed,
            )
            for signed_contrast in REPEATED_SIGNED_CONTRASTS
            for seed in JOINT_RANDOMNESS_SEEDS
        )
    _validate_runs(runs)
    return tuple(runs)


def _validate_runs(runs: list[FiberRun]) -> None:
    if len(ANCHORS) != 6:
        raise ValueError(f"Expected six fiber anchors, got {len(ANCHORS)}")
    if len(runs) != EXPECTED_NUM_RUNS:
        raise ValueError(f"Expected {EXPECTED_NUM_RUNS} requested runs, got {len(runs)}")
    if len({item.run_name for item in runs}) != len(runs):
        raise ValueError("Fiber run names must be unique")

    expected_runs_per_anchor = len(COMMON_SIGNED_CONTRASTS) + (
        len(REPEATED_SIGNED_CONTRASTS) * len(JOINT_RANDOMNESS_SEEDS)
    )
    for anchor in ANCHORS:
        anchor_runs = [item for item in runs if item.anchor == anchor]
        if len(anchor_runs) != expected_runs_per_anchor:
            raise ValueError(f"Anchor {anchor.index} has {len(anchor_runs)} runs")

        reference_contrasts = {item.signed_contrast for item in anchor_runs if item.replicate_kind == "reference"}
        if reference_contrasts != set(COMMON_SIGNED_CONTRASTS):
            raise ValueError(f"Anchor {anchor.index} has an incomplete reference contrast grid")
        for signed_contrast in REPEATED_SIGNED_CONTRASTS:
            observed_seeds = {
                item.seed
                for item in anchor_runs
                if item.replicate_kind == "joint_randomness" and item.signed_contrast == signed_contrast
            }
            if observed_seeds != set(JOINT_RANDOMNESS_SEEDS):
                raise ValueError(f"Anchor {anchor.index}, contrast {signed_contrast} has wrong repeat seeds")

        phase_0_fraction, phase_1_fraction = anchor.phase_fractions
        for signed_contrast in (*COMMON_SIGNED_CONTRASTS, *REPEATED_SIGNED_CONTRASTS):
            phase_0_starcoder, phase_1_starcoder = anchor.coordinate(signed_contrast)
            if not 0.0 <= phase_0_starcoder <= 1.0 or not 0.0 <= phase_1_starcoder <= 1.0:
                raise ValueError(
                    f"Anchor {anchor.index}, contrast {signed_contrast} is infeasible: "
                    f"{(phase_0_starcoder, phase_1_starcoder)}"
                )
            aggregate = phase_0_fraction * phase_0_starcoder + phase_1_fraction * phase_1_starcoder
            if abs(aggregate - anchor.aggregate) > COORDINATE_TOLERANCE:
                raise ValueError(f"Anchor {anchor.index}, contrast {signed_contrast} changes aggregate")
            if abs((phase_1_starcoder - phase_0_starcoder) - signed_contrast) > COORDINATE_TOLERANCE:
                raise ValueError(f"Anchor {anchor.index}, contrast {signed_contrast} is misparameterized")

        for magnitude in (0.05, 0.10, 0.15, 0.20, 0.25):
            negative = anchor.coordinate(-magnitude)
            positive = anchor.coordinate(magnitude)
            midpoint = tuple((left + right) / 2 for left, right in zip(negative, positive, strict=True))
            if max(abs(value - anchor.aggregate) for value in midpoint) > COORDINATE_TOLERANCE:
                raise ValueError(f"Anchor {anchor.index}, magnitude {magnitude} is not antithetic")


def _manifest_row(item: FiberRun) -> dict[str, int | float | str]:
    schedule = base._schedule_summary(item.anchor.token_budget)
    materialized_tokens = int(schedule["materialized_tokens"])
    phase_0_fraction, phase_1_fraction = item.anchor.phase_fractions
    phase_0_starcoder, phase_1_starcoder = item.anchor.coordinate(item.signed_contrast)
    aggregate = phase_0_fraction * phase_0_starcoder + phase_1_fraction * phase_1_starcoder
    return {
        "run_name": item.run_name,
        "anchor_index": item.anchor.index,
        "anchor_role": item.anchor.role,
        "token_budget_requested": item.anchor.token_budget,
        "materialized_tokens": materialized_tokens,
        "total_steps": int(schedule["total_steps"]),
        "boundary_step": int(schedule["boundary_step"]),
        "phase_0_fraction_realized": phase_0_fraction,
        "phase_1_fraction_realized": phase_1_fraction,
        "anchor_aggregate_starcoder": item.anchor.aggregate,
        "aggregate_starcoder_realized": aggregate,
        "phase_0_starcoder": phase_0_starcoder,
        "phase_1_starcoder": phase_1_starcoder,
        "signed_contrast_phase1_minus_phase0": item.signed_contrast,
        "replicate_kind": item.replicate_kind,
        "trainer_data_seed": item.seed,
        "simulated_epoch_subset_seed": item.seed,
        "tied_control_wandb_id": item.anchor.tied_control_wandb_id,
        "total_parameter_tpp": materialized_tokens / scaling.TOTAL_TRAINABLE_PARAMETERS,
        "non_embedding_parameter_tpp": materialized_tokens / scaling.NON_EMBEDDING_PARAMETERS,
        "estimated_training_flops": 6 * scaling.TOTAL_TRAINABLE_PARAMETERS * materialized_tokens,
    }


def manifest() -> dict[str, object]:
    """Return the frozen design and preregistered fiber estimands."""
    rows = [_manifest_row(item) for item in build_fiber_runs()]
    return {
        "experiment": "StarCoder WSD80 scale-specific tied-optimum phase fibers",
        "design_version": "2026-07-31",
        "objective_metric": scaling.OBJECTIVE_METRIC,
        "model": {
            "architecture": "Llama, 10 layers, d_model=768, d_ff=1536, 8 Q/KV heads",
            "total_trainable_parameters": scaling.TOTAL_TRAINABLE_PARAMETERS,
            "non_embedding_parameters": scaling.NON_EMBEDDING_PARAMETERS,
        },
        "invariants": {
            "target_budget": base.TARGET_BUDGET,
            "phase_boundary_nominal": base.PHASE_BOUNDARY,
            "batch_size": base.BATCH_SIZE,
            "sequence_length": base.SEQ_LEN,
            "mixture_block_size": base.MIXTURE_BLOCK_SIZE,
            "warmup_fraction": base.WARMUP_FRACTION,
            "optimizer": "MuonH, peak Muon LR 0.02, Adam LR 0.008",
            "lr_schedule": "1% warmup, stable through phase 0, cosine decay over phase 1",
            "region": base.DEFAULT_TPU_REGION,
            "zone": base.DEFAULT_TPU_ZONE,
            "tpu_type": base.DEFAULT_TPU_TYPE,
        },
        "design": {
            "new_runs": len(rows),
            "anchors": [
                {
                    "index": anchor.index,
                    "token_budget": anchor.token_budget,
                    "aggregate": anchor.aggregate,
                    "role": anchor.role,
                    "phase_fractions_realized": list(anchor.phase_fractions),
                    "tied_control_wandb_id": anchor.tied_control_wandb_id,
                }
                for anchor in ANCHORS
            ],
            "contrast_definition": "phase_1_starcoder - phase_0_starcoder",
            "reference_signed_contrasts": list(COMMON_SIGNED_CONTRASTS),
            "matched_repeat_signed_contrasts": list(REPEATED_SIGNED_CONTRASTS),
            "matched_repeat_seeds": [REFERENCE_SEED, *JOINT_RANDOMNESS_SEEDS],
            "joint_randomness_repeat_semantics": "trainer seed, data seed, and simulated-support seed change together",
            "aggregate_matching": "p0=a-beta1*d and p1=a+beta0*d using each rung's realized phase fractions",
            "anchor_selection": (
                "Primary anchors are the measured regular-grid tied minima from the completed diagonal. "
                "The 2B a=.40 and 8B a=.75 anchors are frozen sensitivity checks for broad tied basins."
            ),
        },
        "primary_estimands": [
            "At each anchor and |d|: odd ordering effect o(d)=[L(+d)-L(-d)]/2.",
            "At each anchor and |d|: even asymmetry cost c(d)=[L(+d)+L(-d)]/2-L(0).",
            "At each anchor and |d|: better-ordering delta c(d)-|o(d)| relative to tied.",
            "At |d|=.20: five-seed paired odd, even, and better-ordering effects with uncertainty.",
            "Across measured-grid anchors: change in fiber gain and best contrast with log materialized tokens.",
        ],
        "interpretation_boundary": (
            "This panel tests local fixed-aggregate phase order around data-selected tied anchors. "
            "It does not resolve the global two-phase optimum, and the best sampled fiber point is "
            "selection-biased unless confirmed separately."
        ),
        "runs": rows,
    }


def write_manifest(output_dir: Path = MANIFEST_DIR) -> tuple[Path, Path]:
    """Persist the frozen JSON and flat CSV manifests."""
    payload = manifest()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "design_manifest.json"
    csv_path = output_dir / "run_manifest.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    rows = payload["runs"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest contains no run rows")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def check_manifest(output_dir: Path = MANIFEST_DIR) -> None:
    """Verify that frozen files exactly represent the current launcher design."""
    expected = manifest()
    json_path = output_dir / "design_manifest.json"
    csv_path = output_dir / "run_manifest.csv"
    if json.loads(json_path.read_text()) != expected:
        raise ValueError(f"Frozen design manifest is stale: {json_path}")
    expected_runs = expected["runs"]
    if not isinstance(expected_runs, list):
        raise ValueError("Expected run manifest rows to be a list")
    with csv_path.open(newline="") as handle:
        observed_run_names = [row["run_name"] for row in csv.DictReader(handle)]
    expected_run_names = [str(row["run_name"]) for row in expected_runs]
    if observed_run_names != expected_run_names:
        raise ValueError(f"Frozen CSV manifest is stale: {csv_path}")


def build_training_steps(
    *,
    name_prefix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    include_run_names: frozenset[str] = frozenset(),
    wandb_run_id_suffix: str = "",
) -> tuple[ArtifactStep[LevanterCheckpoint], ...]:
    """Build the requested fibers, grouped by explicit token budget."""
    requested_runs = build_fiber_runs()
    if include_run_names:
        known_run_names = {item.run_name for item in requested_runs}
        missing_run_names = sorted(include_run_names - known_run_names)
        if missing_run_names:
            raise ValueError(f"Unknown requested run names: {missing_run_names}")
        requested_runs = tuple(item for item in requested_runs if item.run_name in include_run_names)

    steps: list[ArtifactStep[LevanterCheckpoint]] = []
    rank = 1
    for token_budget in scaling.TOKEN_BUDGETS:
        budget_runs = [item for item in requested_runs if item.anchor.token_budget == token_budget]
        if not budget_runs:
            continue
        specs = tuple(item.surface_spec(rank + index) for index, item in enumerate(budget_runs))
        rank += len(specs)
        steps.extend(
            base.build_training_steps(
                name_prefix=name_prefix,
                tpu_type=tpu_type,
                tpu_region=tpu_region,
                tpu_zone=tpu_zone,
                data_seed=REFERENCE_SEED,
                run_specs=specs,
                wandb_experiment_tag=WANDB_EXPERIMENT_TAG,
                panel_tag=f"{PANEL_TAG}_{token_budget // 1_000_000_000}b",
                experiment_budget=token_budget,
                target_budget=base.TARGET_BUDGET,
                wandb_run_id_suffix=wandb_run_id_suffix,
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
    parser.add_argument("--include-run-name", action="append", default=[])
    parser.add_argument("--wandb-run-id-suffix", default="")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--check-manifest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if os.getenv("CI") is not None:
        logger.info("Skipping StarCoder WSD80 scale-specific tied fibers in CI")
        return
    if args.marin_prefix != base.DEFAULT_MARIN_PREFIX:
        raise ValueError(f"This historical StarCoder experiment is central1-local: got {args.marin_prefix!r}")
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError(
            "StarCoder child TPU placement must remain central1-local: "
            f"got region={args.tpu_region!r}, zone={args.tpu_zone!r}"
        )

    all_requested_runs = build_fiber_runs()
    include_run_names = frozenset(args.include_run_name)
    if args.wandb_run_id_suffix and not include_run_names:
        raise ValueError("--wandb-run-id-suffix requires at least one --include-run-name")
    if include_run_names and not args.wandb_run_id_suffix:
        raise ValueError("Targeted recovery requires --wandb-run-id-suffix to avoid reusing a failed W&B run ID")
    requested_runs = tuple(
        item for item in all_requested_runs if not include_run_names or item.run_name in include_run_names
    )
    if not 1 <= args.max_concurrent <= len(requested_runs):
        raise ValueError(f"max_concurrent must be in [1, {len(requested_runs)}], got {args.max_concurrent}")
    if args.write_manifest:
        json_path, csv_path = write_manifest()
        logger.info("Wrote frozen manifests to %s and %s", json_path, csv_path)
    if args.check_manifest:
        check_manifest()
        logger.info("Frozen manifest matches the current launcher design")

    logger.info(
        "Prepared %d scale-specific fiber runs: anchors=%d, reference_contrasts=%d, "
        "new_joint_repeats_per_anchor=%d, rung_counts=%s",
        len(requested_runs),
        len(ANCHORS),
        len(COMMON_SIGNED_CONTRASTS),
        len(REPEATED_SIGNED_CONTRASTS) * len(JOINT_RANDOMNESS_SEEDS),
        {
            token_budget: sum(item.anchor.token_budget == token_budget for item in requested_runs)
            for token_budget in scaling.TOKEN_BUDGETS
        },
    )
    os.environ["MARIN_PREFIX"] = args.marin_prefix
    steps = build_training_steps(
        name_prefix=args.name_prefix,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        include_run_names=include_run_names,
        wandb_run_id_suffix=args.wandb_run_id_suffix,
    )
    if args.dry_run:
        for step in steps:
            lower(step)
        logger.info("Dry-run lowering passed for all %d training handles", len(steps))
        return
    run(*steps, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
