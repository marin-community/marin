# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Complete the tied StarCoder WSD80 diagonal at every 1B--8B token rung."""

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

NAME = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_wsd80_fixedn_tieddiag_1b8b_20260730"
WANDB_EXPERIMENT_TAG = "starcoder_wsd80_fixedn_tieddiag"
PANEL_TAG = "fixedn_tieddiag_1b8b"
REFERENCE_SEED = scaling.REFERENCE_SEED
TOKEN_BUDGETS = scaling.TOKEN_BUDGETS
REGULAR_TIED_WEIGHTS = tuple(index / 20 for index in range(21))
DEFAULT_MAX_CONCURRENT = 48
EXPECTED_RUN_COUNTS = {
    1_000_000_000: 6,
    2_000_000_000: 18,
    4_000_000_000: 18,
    8_000_000_000: 18,
}
EXPECTED_NUM_RUNS = sum(EXPECTED_RUN_COUNTS.values())
MANIFEST_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "starcoder_wsd80_fixed_model_tied_diagonal_20260730"
)

# The 1B surface already contains these regular-grid tied coordinates. The
# completed scaling panel contains the three listed coordinates at every larger
# rung. Irregular tied probes at 0.18, 0.22, and 0.26 remain supplemental.
EXISTING_REGULAR_TIED_WEIGHTS = {
    1_000_000_000: frozenset(
        {
            0.00,
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            1.00,
        }
    ),
    2_000_000_000: frozenset({0.10, 0.30, 0.35}),
    4_000_000_000: frozenset({0.10, 0.30, 0.35}),
    8_000_000_000: frozenset({0.10, 0.30, 0.35}),
}


@dataclass(frozen=True)
class TiedDiagonalRun:
    """One missing tied coordinate at one token-budget rung."""

    token_budget: int
    weight: float

    @property
    def budget_slug(self) -> str:
        return f"{self.token_budget // 1_000_000_000}b"

    @property
    def run_name(self) -> str:
        return f"d{self.budget_slug}_tieddiag_p{base._weight_slug(self.weight)}_ref_s{REFERENCE_SEED}"

    def surface_spec(self, rank: int) -> base.SurfaceRunSpec:
        return base.SurfaceRunSpec(
            rank=rank,
            phase_0_starcoder=self.weight,
            phase_1_starcoder=self.weight,
            run_name_override=self.run_name,
        )


def build_diagonal_runs() -> tuple[TiedDiagonalRun, ...]:
    """Return exactly the missing regular-grid tied coordinates."""
    runs = tuple(
        TiedDiagonalRun(token_budget=token_budget, weight=weight)
        for token_budget in TOKEN_BUDGETS
        for weight in REGULAR_TIED_WEIGHTS
        if weight not in EXISTING_REGULAR_TIED_WEIGHTS[token_budget]
    )
    _validate_runs(runs)
    return runs


def _validate_runs(runs: tuple[TiedDiagonalRun, ...]) -> None:
    if len(runs) != EXPECTED_NUM_RUNS:
        raise ValueError(f"Expected {EXPECTED_NUM_RUNS} missing tied runs, got {len(runs)}")
    if len({item.run_name for item in runs}) != len(runs):
        raise ValueError("Tied-diagonal run names must be unique")

    observed_counts = {
        token_budget: sum(item.token_budget == token_budget for item in runs) for token_budget in TOKEN_BUDGETS
    }
    if observed_counts != EXPECTED_RUN_COUNTS:
        raise ValueError(f"Unexpected tied-diagonal rung counts: {observed_counts}")

    for token_budget in TOKEN_BUDGETS:
        requested = {item.weight for item in runs if item.token_budget == token_budget}
        existing = EXISTING_REGULAR_TIED_WEIGHTS[token_budget]
        if requested & existing:
            raise ValueError(f"Rung {token_budget} requests existing tied coordinates: {sorted(requested & existing)}")
        if requested | existing != set(REGULAR_TIED_WEIGHTS):
            raise ValueError(f"Rung {token_budget} does not complete the regular tied diagonal")

    for item in runs:
        if not 0.0 <= item.weight <= 1.0:
            raise ValueError(f"Infeasible tied weight: {item.weight}")


def _manifest_row(item: TiedDiagonalRun) -> dict[str, int | float | str]:
    schedule = base._schedule_summary(item.token_budget)
    materialized_tokens = int(schedule["materialized_tokens"])
    return {
        "run_name": item.run_name,
        "token_budget_requested": item.token_budget,
        "materialized_tokens": materialized_tokens,
        "total_steps": int(schedule["total_steps"]),
        "boundary_step": int(schedule["boundary_step"]),
        "phase_0_fraction_realized": float(schedule["realized_phase_0_fraction"]),
        "phase_1_fraction_realized": 1.0 - float(schedule["realized_phase_0_fraction"]),
        "phase_0_starcoder": item.weight,
        "phase_1_starcoder": item.weight,
        "aggregate_starcoder_nominal": item.weight,
        "phase_contrast": 0.0,
        "trainer_data_seed": REFERENCE_SEED,
        "simulated_epoch_subset_seed": REFERENCE_SEED,
        "total_parameter_tpp": materialized_tokens / scaling.TOTAL_TRAINABLE_PARAMETERS,
        "non_embedding_parameter_tpp": materialized_tokens / scaling.NON_EMBEDDING_PARAMETERS,
        "estimated_training_flops": 6 * scaling.TOTAL_TRAINABLE_PARAMETERS * materialized_tokens,
    }


def manifest() -> dict[str, object]:
    """Return the frozen tied-diagonal completion design."""
    rows = [_manifest_row(item) for item in build_diagonal_runs()]
    return {
        "experiment": "StarCoder WSD80 fixed-model tied-diagonal completion",
        "design_version": "2026-07-30",
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
            "token_budgets": list(TOKEN_BUDGETS),
            "regular_tied_weights": list(REGULAR_TIED_WEIGHTS),
            "new_runs": len(rows),
            "new_runs_by_rung": {str(key): value for key, value in EXPECTED_RUN_COUNTS.items()},
            "existing_regular_tied_weights": {
                str(key): sorted(value) for key, value in EXISTING_REGULAR_TIED_WEIGHTS.items()
            },
            "purpose": (
                "Resolve an interior tied optimum at every token rung before testing whether antithetic "
                "phase contrasts can improve that optimum on its fixed-aggregate fiber."
            ),
            "followup_boundary": (
                "Do not choose or launch the phase-fiber anchors until this diagonal is complete. "
                "Fiber outcomes are a separate phase-order test."
            ),
        },
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
    """Verify that frozen files exactly represent the current design."""
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
    """Build the missing tied runs, grouped by explicit token budget."""
    requested_runs = build_diagonal_runs()
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
        logger.info("Skipping StarCoder WSD80 tied-diagonal completion in CI")
        return
    if args.marin_prefix != base.DEFAULT_MARIN_PREFIX:
        raise ValueError(f"This historical StarCoder experiment is central1-local: got {args.marin_prefix!r}")
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError(
            "StarCoder child TPU placement must remain central1-local: "
            f"got region={args.tpu_region!r}, zone={args.tpu_zone!r}"
        )

    all_requested_runs = build_diagonal_runs()
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
        "Prepared %d tied-diagonal completion runs: rung_counts=%s, target_budget=%d",
        len(requested_runs),
        {
            token_budget: sum(item.token_budget == token_budget for item in requested_runs)
            for token_budget in TOKEN_BUDGETS
        },
        base.TARGET_BUDGET,
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
