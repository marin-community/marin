# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Launch the fixed-model StarCoder WSD80 token-scaling trend panel."""

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

from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_wsd80_fixedn_tokenscale_1b8b_20260728"
WANDB_EXPERIMENT_TAG = "starcoder_wsd80_fixedn_tokenscale"
PANEL_TAG = "fixedn_tokenscale_1b8b"
OBJECTIVE_METRIC = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
REFERENCE_SEED = 20_260_711
JOINT_RANDOMNESS_SEEDS = (20_260_712, 20_260_713, 20_260_714, 20_260_715)
SUPPORT_PROBE_SEED = 20_260_729
TOKEN_BUDGETS = (1_000_000_000, 2_000_000_000, 4_000_000_000, 8_000_000_000)
NEW_TOKEN_BUDGETS = TOKEN_BUDGETS[1:]
DEFAULT_MAX_CONCURRENT = 48
TOTAL_TRAINABLE_PARAMETERS = 157_499_136
NON_EMBEDDING_PARAMETERS = 58_998_528
STARCODER_CORPUS_TOKENS = 216_567_300_822
STARCODER_CORPUS_TOKEN_SOURCE = (
    "experiments/domain_phase_mix/domains.py:DOLMA_TOKENS; central1 tokenized cache queried 2025-01-28"
)
MANIFEST_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "starcoder_wsd80_fixed_model_token_scaling_20260728"
)


@dataclass(frozen=True)
class ScalingCoordinate:
    """One aggregate-and-contrast point in the sparse scaling design."""

    index: int
    aggregate: float
    contrast: float
    role: str

    @property
    def phase_0_starcoder(self) -> float:
        value = self.aggregate - (1.0 - base.PHASE_BOUNDARY) * self.contrast
        return 0.0 if abs(value) < 1e-12 else round(value, 12)

    @property
    def phase_1_starcoder(self) -> float:
        value = self.aggregate + base.PHASE_BOUNDARY * self.contrast
        return 1.0 if abs(value - 1.0) < 1e-12 else round(value, 12)


@dataclass(frozen=True)
class ScalingRun:
    """One requested checkpoint, including its two independent seed controls."""

    token_budget: int
    coordinate: ScalingCoordinate
    replicate_kind: str
    trainer_data_seed: int
    simulated_epoch_subset_seed: int

    @property
    def budget_slug(self) -> str:
        return f"{self.token_budget // 1_000_000_000}b"

    @property
    def run_name(self) -> str:
        kind_slug = {
            "reference": "ref",
            "backfill": "ref",
            "joint_randomness": "joint",
            "support_seed": "subset",
        }[self.replicate_kind]
        seed = self.simulated_epoch_subset_seed if self.replicate_kind == "support_seed" else self.trainer_data_seed
        return f"d{self.budget_slug}_c{self.coordinate.index:02d}_{kind_slug}_s{seed}"

    def surface_spec(self, rank: int) -> base.SurfaceRunSpec:
        subset_override = self.simulated_epoch_subset_seed if self.replicate_kind == "support_seed" else None
        data_override = self.trainer_data_seed if self.trainer_data_seed != REFERENCE_SEED else None
        return base.SurfaceRunSpec(
            rank=rank,
            phase_0_starcoder=self.coordinate.phase_0_starcoder,
            phase_1_starcoder=self.coordinate.phase_1_starcoder,
            run_name_override=self.run_name,
            data_seed_override=data_override,
            simulated_epoch_subset_seed_override=subset_override,
        )


COORDINATES: tuple[ScalingCoordinate, ...] = (
    ScalingCoordinate(1, 0.10, 0.00, "tied_low"),
    ScalingCoordinate(2, 0.18, 0.00, "matched_tied_control"),
    ScalingCoordinate(3, 0.22, 0.00, "tied_spine"),
    ScalingCoordinate(4, 0.26, 0.00, "tied_spine"),
    ScalingCoordinate(5, 0.30, 0.00, "best_sampled_tied"),
    ScalingCoordinate(6, 0.35, 0.00, "tied_high"),
    ScalingCoordinate(7, 0.10, 0.50, "phase_0_boundary"),
    ScalingCoordinate(8, 0.18, 0.20, "optimum_fiber"),
    ScalingCoordinate(9, 0.18, 0.40, "matched_two_phase_candidate"),
    ScalingCoordinate(10, 0.18, 0.60, "optimum_fiber"),
    ScalingCoordinate(11, 0.18, 0.80, "outward_drift_probe"),
    ScalingCoordinate(12, 0.22, 0.20, "aggregate_contrast_interaction"),
    ScalingCoordinate(13, 0.22, 0.40, "aggregate_contrast_interaction"),
    ScalingCoordinate(14, 0.22, 0.60, "aggregate_contrast_interaction"),
    ScalingCoordinate(15, 0.26, 0.20, "aggregate_contrast_interaction"),
    ScalingCoordinate(16, 0.26, 0.40, "aggregate_contrast_interaction"),
    ScalingCoordinate(17, 0.30, 0.20, "frontier_fiber"),
    ScalingCoordinate(18, 0.30, 0.40, "frontier_fiber"),
)

# These seven coordinates already have reference-seed 1B observations in the
# completed surface and fiber panels. The other eleven are the only 1B backfills.
REUSED_ONE_BILLION_COORDINATE_INDICES = frozenset({1, 2, 5, 6, 7, 9, 10})
FIVE_SEED_COORDINATE_INDICES = frozenset({2, 5, 9})
SUPPORT_PROBE_COORDINATE_INDICES = frozenset({2, 5, 9})


def build_scaling_runs() -> tuple[ScalingRun, ...]:
    """Return and validate the immutable 104-run request manifest."""
    runs: list[ScalingRun] = []
    for coordinate in COORDINATES:
        if coordinate.index in REUSED_ONE_BILLION_COORDINATE_INDICES:
            continue
        runs.append(
            ScalingRun(
                token_budget=TOKEN_BUDGETS[0],
                coordinate=coordinate,
                replicate_kind="backfill",
                trainer_data_seed=REFERENCE_SEED,
                simulated_epoch_subset_seed=REFERENCE_SEED,
            )
        )

    for token_budget in NEW_TOKEN_BUDGETS:
        for coordinate in COORDINATES:
            runs.append(
                ScalingRun(
                    token_budget=token_budget,
                    coordinate=coordinate,
                    replicate_kind="reference",
                    trainer_data_seed=REFERENCE_SEED,
                    simulated_epoch_subset_seed=REFERENCE_SEED,
                )
            )
        for coordinate in COORDINATES:
            if coordinate.index in FIVE_SEED_COORDINATE_INDICES:
                repeat_seeds = JOINT_RANDOMNESS_SEEDS
            else:
                continue
            for repeat_seed in repeat_seeds:
                runs.append(
                    ScalingRun(
                        token_budget=token_budget,
                        coordinate=coordinate,
                        replicate_kind="joint_randomness",
                        trainer_data_seed=repeat_seed,
                        simulated_epoch_subset_seed=repeat_seed,
                    )
                )

    coordinate_by_index = {coordinate.index: coordinate for coordinate in COORDINATES}
    for coordinate_index in sorted(SUPPORT_PROBE_COORDINATE_INDICES):
        runs.append(
            ScalingRun(
                token_budget=4_000_000_000,
                coordinate=coordinate_by_index[coordinate_index],
                replicate_kind="support_seed",
                trainer_data_seed=REFERENCE_SEED,
                simulated_epoch_subset_seed=SUPPORT_PROBE_SEED,
            )
        )

    _validate_runs(runs)
    return tuple(runs)


def _validate_runs(runs: list[ScalingRun]) -> None:
    if len(COORDINATES) != 18:
        raise ValueError(f"Expected 18 scaling coordinates, got {len(COORDINATES)}")
    if {coordinate.index for coordinate in COORDINATES} != set(range(1, 19)):
        raise ValueError("Coordinate indices must be exactly 1 through 18")
    for coordinate in COORDINATES:
        p0 = coordinate.phase_0_starcoder
        p1 = coordinate.phase_1_starcoder
        if not 0.0 <= p0 <= 1.0 or not 0.0 <= p1 <= 1.0:
            raise ValueError(f"Coordinate {coordinate.index} is infeasible: {(p0, p1)}")
        aggregate = base.PHASE_BOUNDARY * p0 + (1.0 - base.PHASE_BOUNDARY) * p1
        if abs(aggregate - coordinate.aggregate) > 1e-12:
            raise ValueError(f"Coordinate {coordinate.index} does not preserve its aggregate")
        if abs((p1 - p0) - coordinate.contrast) > 1e-12:
            raise ValueError(f"Coordinate {coordinate.index} does not preserve its contrast")

    if len(runs) != 104:
        raise ValueError(f"Expected 104 requested runs, got {len(runs)}")
    if len({item.run_name for item in runs}) != len(runs):
        raise ValueError("Scaling run names must be unique")
    expected_counts = {
        1_000_000_000: 11,
        2_000_000_000: 30,
        4_000_000_000: 33,
        8_000_000_000: 30,
    }
    observed_counts = {
        token_budget: sum(item.token_budget == token_budget for item in runs) for token_budget in TOKEN_BUDGETS
    }
    if observed_counts != expected_counts:
        raise ValueError(f"Unexpected rung counts: {observed_counts}")

    for token_budget in NEW_TOKEN_BUDGETS:
        reference_indices = {
            item.coordinate.index
            for item in runs
            if item.token_budget == token_budget and item.replicate_kind == "reference"
        }
        if reference_indices != set(range(1, 19)):
            raise ValueError(f"Rung {token_budget} does not contain all 18 reference coordinates")


def _manifest_row(item: ScalingRun) -> dict[str, int | float | str]:
    schedule = base._schedule_summary(item.token_budget)
    materialized_tokens = int(schedule["materialized_tokens"])
    phase_0_fraction = float(schedule["realized_phase_0_fraction"])
    phase_1_fraction = 1.0 - phase_0_fraction
    p0 = item.coordinate.phase_0_starcoder
    p1 = item.coordinate.phase_1_starcoder
    realized_aggregate = phase_0_fraction * p0 + phase_1_fraction * p1
    starcoder_epoch_scale = base.TARGET_BUDGET / STARCODER_CORPUS_TOKENS
    return {
        "run_name": item.run_name,
        "token_budget_requested": item.token_budget,
        "materialized_tokens": materialized_tokens,
        "total_steps": int(schedule["total_steps"]),
        "boundary_step": int(schedule["boundary_step"]),
        "phase_0_fraction_realized": phase_0_fraction,
        "phase_1_fraction_realized": phase_1_fraction,
        "coordinate_index": item.coordinate.index,
        "coordinate_role": item.coordinate.role,
        "phase_0_starcoder": p0,
        "phase_1_starcoder": p1,
        "aggregate_starcoder_nominal": item.coordinate.aggregate,
        "aggregate_starcoder_realized": realized_aggregate,
        "phase_contrast": item.coordinate.contrast,
        "replicate_kind": item.replicate_kind,
        "trainer_data_seed": item.trainer_data_seed,
        "simulated_epoch_subset_seed": item.simulated_epoch_subset_seed,
        "support_fraction": materialized_tokens / base.TARGET_BUDGET,
        "total_parameter_tpp": materialized_tokens / TOTAL_TRAINABLE_PARAMETERS,
        "non_embedding_parameter_tpp": materialized_tokens / NON_EMBEDDING_PARAMETERS,
        "estimated_training_flops": 6 * TOTAL_TRAINABLE_PARAMETERS * materialized_tokens,
        "phase_0_starcoder_simulated_epochs": phase_0_fraction * p0 * starcoder_epoch_scale,
        "phase_1_starcoder_simulated_epochs": phase_1_fraction * p1 * starcoder_epoch_scale,
        "aggregate_starcoder_simulated_epochs": realized_aggregate * starcoder_epoch_scale,
        "phase_0_nemotron_simulated_epochs": phase_0_fraction * (1.0 - p0),
        "phase_1_nemotron_simulated_epochs": phase_1_fraction * (1.0 - p1),
        "aggregate_nemotron_simulated_epochs": 1.0 - realized_aggregate,
    }


def manifest() -> dict[str, object]:
    """Return the frozen design and preregistered estimands."""
    rows = [_manifest_row(item) for item in build_scaling_runs()]
    return {
        "experiment": "Fixed-model StarCoder WSD80 token-scaling trend",
        "design_version": "2026-07-28",
        "objective_metric": OBJECTIVE_METRIC,
        "model": {
            "architecture": "Llama, 10 layers, d_model=768, d_ff=1536, 8 Q/KV heads",
            "total_trainable_parameters": TOTAL_TRAINABLE_PARAMETERS,
            "non_embedding_parameters": NON_EMBEDDING_PARAMETERS,
        },
        "invariants": {
            "target_budget": base.TARGET_BUDGET,
            "starcoder_corpus_tokens": STARCODER_CORPUS_TOKENS,
            "starcoder_corpus_token_source": STARCODER_CORPUS_TOKEN_SOURCE,
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
            "token_budgets": list(TOKEN_BUDGETS),
            "coordinates_per_new_rung": len(COORDINATES),
            "new_runs": len(rows),
            "reused_1b_reference_coordinates": len(REUSED_ONE_BILLION_COORDINATE_INDICES),
            "reused_1b_reference_coordinate_indices": sorted(REUSED_ONE_BILLION_COORDINATE_INDICES),
            "new_1b_backfills": len(COORDINATES) - len(REUSED_ONE_BILLION_COORDINATE_INDICES),
            "matched_repeat_coordinate_indices": sorted(FIVE_SEED_COORDINATE_INDICES),
            "matched_repeat_seeds": [REFERENCE_SEED, *JOINT_RANDOMNESS_SEEDS],
            "joint_randomness_repeat_semantics": (
                "trainer seed, data seed, and simulated-support seed change together, matching the historical repeats"
            ),
            "support_probe_semantics": (
                "trainer and data seed stay at 20260711; only simulated-support seed changes to 20260729"
            ),
            "support_nesting": (
                "reference runs share subset seed 20260711, so increasing support fractions are nested per component"
            ),
            "simulated_epoch_alignment": (
                "exact across rungs on the tied spine; block alignment changes the most asymmetric coordinate "
                "by at most 1.1%"
            ),
            "lr_policy": (
                "fixed historical optimizer and fractional WSD schedule; no LR-retuning arm without a frozen rule"
            ),
        },
        "coordinates": [
            {
                "index": coordinate.index,
                "aggregate_starcoder_nominal": coordinate.aggregate,
                "phase_contrast": coordinate.contrast,
                "phase_0_starcoder": coordinate.phase_0_starcoder,
                "phase_1_starcoder": coordinate.phase_1_starcoder,
                "role": coordinate.role,
                "reused_at_1b": coordinate.index in REUSED_ONE_BILLION_COORDINATE_INDICES,
            }
            for coordinate in COORDINATES
        ],
        "primary_estimands": [
            "matched phase gain: BPB(c09: a=.18, delta=.40) - BPB(c02: a=.18, delta=0)",
            "aggregate penalty: BPB(c02: a=.18, delta=0) - BPB(c05: a=.30, delta=0)",
            "net two-phase advantage: BPB(c09) - BPB(c05)",
            "coordinate-wise BPB change with log materialized tokens",
            "best tied and best two-phase location by rung, reported as selection-biased secondary summaries",
        ],
        "conditional_extension": (
            "Add a 16B rung only if the 8B coordinate spread remains at least three times the matched-repeat noise SD."
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
    """Verify that the frozen files exactly represent the current design code."""
    expected = manifest()
    json_path = output_dir / "design_manifest.json"
    csv_path = output_dir / "run_manifest.csv"
    observed = json.loads(json_path.read_text())
    if observed != expected:
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
    """Build all requested rungs with one explicit budget per graph group."""
    requested_runs = build_scaling_runs()
    if include_run_names:
        known_run_names = {item.run_name for item in requested_runs}
        missing_run_names = sorted(include_run_names - known_run_names)
        if missing_run_names:
            raise ValueError(f"Unknown requested run names: {missing_run_names}")
        requested_runs = tuple(item for item in requested_runs if item.run_name in include_run_names)
    steps: list[ArtifactStep[LevanterCheckpoint]] = []
    rank = 1
    for token_budget in TOKEN_BUDGETS:
        budget_runs = [item for item in requested_runs if item.token_budget == token_budget]
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
        logger.info("Skipping StarCoder WSD80 token scaling in CI")
        return
    if args.marin_prefix != base.DEFAULT_MARIN_PREFIX:
        raise ValueError(f"This historical StarCoder experiment is central1-local: got {args.marin_prefix!r}")
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError(
            "StarCoder child TPU placement must remain central1-local: "
            f"got region={args.tpu_region!r}, zone={args.tpu_zone!r}"
        )

    all_requested_runs = build_scaling_runs()
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
        "Prepared %d fixed-model token-scaling runs: rung_counts=%s, target_budget=%d, "
        "joint_randomness_seeds=%s, support_probe_seed=%d",
        len(requested_runs),
        {
            token_budget: sum(item.token_budget == token_budget for item in requested_runs)
            for token_budget in TOKEN_BUDGETS
        },
        base.TARGET_BUDGET,
        JOINT_RANDOMNESS_SEEDS,
        SUPPORT_PROBE_SEED,
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
