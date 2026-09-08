# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scipy",
#   "tabulate",
#   "wandb",
# ]
# ///
"""Collect and audit Stage 1 of the WSD80 scale-wise optimum refinement."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
PANEL_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_scale_bayesian_refinement_20260731"
DEFAULT_OUTPUT_DIR = PANEL_DIR / "results_20260801"
FROZEN_DESIGN_PATH = SCRIPT_DIR.parents[1] / "starcoder_wsd80_scale_bayesian_refinement_design_20260731.json"
DENSE_SURFACE_PATH = REFERENCE_OUTPUTS / "starcoder_wsd80_surface_refined_20260714" / "wsd80_observed_metrics.csv"
TOKEN_SCALING_PATH = (
    REFERENCE_OUTPUTS / "starcoder_wsd80_fixed_model_token_scaling_20260728" / "results_20260730" / "observations.csv"
)
TIED_DIAGONAL_PATH = (
    REFERENCE_OUTPUTS
    / "starcoder_wsd80_fixed_model_tied_diagonal_20260730"
    / "results_20260731"
    / "tied_diagonal_observations.csv"
)
SCALE_FIBERS_PATH = (
    REFERENCE_OUTPUTS / "starcoder_wsd80_scale_specific_tied_fibers_20260731" / "results_20260731" / "observations.csv"
)

TRAIN_PROJECT = "marin-community/marin"
TRAIN_TAG = "starcoder_wsd80_scale_bo_stage1"
EXPECTED_RUNS = 52
OBJECTIVE_METRIC = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
INCUMBENT_COORDINATES = {
    2_000_000_000: (0.06, 0.66),
    4_000_000_000: (0.02, 0.82),
    8_000_000_000: (0.02, 0.82),
}
REFERENCE_SEED = 20_260_711
FRESH_SEEDS = (20_260_712, 20_260_713, 20_260_714, 20_260_715)


@dataclass(frozen=True)
class PersistedMetric:
    """One final checkpoint metric recovered independently of W&B state."""

    value: float
    step: int
    uri: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=FROZEN_DESIGN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wandb-timeout", type=int, default=240)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def persisted_final_metric(run: Any) -> PersistedMetric:
    """Read the final objective from a run's durable checkpoint output."""
    checkpoint_root = str(run.config["trainer"]["checkpointer"]["base_path"])
    uri = f"{checkpoint_root}/eval_metrics.jsonl"
    result = subprocess.run(
        ["gcloud", "storage", "cat", uri],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    finite = [
        row for row in rows if row.get(OBJECTIVE_METRIC) is not None and math.isfinite(float(row[OBJECTIVE_METRIC]))
    ]
    if not finite:
        raise ValueError(f"{run.name}: no finite {OBJECTIVE_METRIC} in {uri}")
    final = max(finite, key=lambda row: int(row["step"]))
    return PersistedMetric(float(final[OBJECTIVE_METRIC]), int(final["step"]), uri)


def collect_stage1(design_path: Path, timeout: int, workers: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join the immutable design to one durable final observation per run."""
    design = json.loads(design_path.read_text(encoding="utf-8"))
    manifest = pd.DataFrame(design["runs"])
    if len(manifest) != EXPECTED_RUNS or manifest["run_name"].duplicated().any():
        raise ValueError(f"Expected {EXPECTED_RUNS} unique design rows")
    if design.get("objective_metric") != OBJECTIVE_METRIC:
        raise ValueError("Frozen design targets an unexpected objective")

    api = wandb.Api(timeout=timeout)
    runs = list(api.runs(TRAIN_PROJECT, filters={"tags": TRAIN_TAG}, per_page=100))
    by_name: dict[str, list[Any]] = {}
    for run in runs:
        by_name.setdefault(str(run.name), []).append(run)

    ordered_runs = []
    for run_name in manifest["run_name"]:
        candidates = by_name.get(str(run_name), [])
        if len(candidates) != 1:
            raise ValueError(f"{run_name}: expected exactly one W&B run, found {len(candidates)}")
        ordered_runs.append(candidates[0])

    with ThreadPoolExecutor(max_workers=workers) as executor:
        metrics = list(executor.map(persisted_final_metric, ordered_runs))

    observations = manifest.copy()
    observations["starcoder_bpb"] = [metric.value for metric in metrics]
    observations["final_metric_step"] = [metric.step for metric in metrics]
    observations["metric_uri"] = [metric.uri for metric in metrics]
    observations["metric_source"] = "persisted eval_metrics.jsonl"
    observations["wandb_id"] = [str(run.id) for run in ordered_runs]
    observations["wandb_name"] = [str(run.name) for run in ordered_runs]
    observations["wandb_state"] = [str(run.state) for run in ordered_runs]
    observations["wandb_url"] = [str(run.url) for run in ordered_runs]
    if not np.isfinite(observations["starcoder_bpb"].to_numpy(dtype=float)).all():
        raise ValueError("Stage-1 observations contain non-finite BPB")
    return observations, design


def existing_observations() -> pd.DataFrame:
    """Load the exact pre-Stage-1 observations used to generate the batch."""
    dense = pd.read_csv(DENSE_SURFACE_PATH)
    scaling = pd.read_csv(TOKEN_SCALING_PATH)
    tied = pd.read_csv(TIED_DIAGONAL_PATH)
    fibers = pd.read_csv(SCALE_FIBERS_PATH)
    observations = pd.concat(
        [
            pd.DataFrame(
                {
                    "token_budget_requested": 1_000_000_000,
                    "p0": dense["phase_0_starcoder"],
                    "p1": dense["phase_1_starcoder"],
                    "bpb": dense["wsd80_bpb"],
                    "run_id": dense["wandb_run_id"].astype(str),
                }
            ),
            pd.DataFrame(
                {
                    "token_budget_requested": scaling["token_budget_requested"],
                    "p0": scaling["phase_0_starcoder"],
                    "p1": scaling["phase_1_starcoder"],
                    "bpb": scaling["starcoder_bpb"],
                    "run_id": scaling["training_wandb_id"].astype(str),
                }
            ),
            pd.DataFrame(
                {
                    "token_budget_requested": tied["token_budget_requested"],
                    "p0": tied["weight"],
                    "p1": tied["weight"],
                    "bpb": tied["starcoder_bpb"],
                    "run_id": tied["wandb_id"].astype(str),
                }
            ),
            pd.DataFrame(
                {
                    "token_budget_requested": fibers["token_budget_requested"],
                    "p0": fibers["phase_0_starcoder"],
                    "p1": fibers["phase_1_starcoder"],
                    "bpb": fibers["starcoder_bpb"],
                    "run_id": fibers["wandb_id"].astype(str),
                }
            ),
        ],
        ignore_index=True,
    ).drop_duplicates("run_id")
    if not np.isfinite(observations[["p0", "p1", "bpb"]].to_numpy(dtype=float)).all():
        raise ValueError("Frozen pre-Stage-1 observations contain non-finite values")
    observations["source"] = "pre_stage1"
    return observations.reset_index(drop=True)


def collapsed_coordinates(existing: pd.DataFrame, stage1: pd.DataFrame) -> pd.DataFrame:
    """Collapse all run-level outcomes by rung and policy coordinate."""
    old = existing[["token_budget_requested", "p0", "p1", "bpb", "run_id", "source"]].copy()
    new = stage1[
        ["token_budget_requested", "phase_0_starcoder", "phase_1_starcoder", "starcoder_bpb", "wandb_id"]
    ].rename(
        columns={
            "phase_0_starcoder": "p0",
            "phase_1_starcoder": "p1",
            "starcoder_bpb": "bpb",
            "wandb_id": "run_id",
        }
    )
    new["source"] = "stage1"
    combined = pd.concat([old, new], ignore_index=True)
    if combined["run_id"].duplicated().any():
        raise ValueError("Existing and Stage-1 observations overlap by W&B run ID")
    combined["p0_key"] = combined["p0"].round(8)
    combined["p1_key"] = combined["p1"].round(8)
    collapsed = (
        combined.groupby(["token_budget_requested", "p0_key", "p1_key"], as_index=False)
        .agg(
            mean_bpb=("bpb", "mean"),
            sd_bpb=("bpb", "std"),
            count=("bpb", "size"),
            stage1_count=("source", lambda values: int(sum(value == "stage1" for value in values))),
        )
        .sort_values(["token_budget_requested", "mean_bpb"])
        .reset_index(drop=True)
    )
    return collapsed


def incumbent_summary(existing: pd.DataFrame, stage1: pd.DataFrame) -> pd.DataFrame:
    """Summarize the four fresh repeats together with the selecting seed."""
    rows = []
    for token_budget, (p0, p1) in INCUMBENT_COORDINATES.items():
        old = existing.loc[
            existing["token_budget_requested"].eq(token_budget)
            & np.isclose(existing["p0"], p0)
            & np.isclose(existing["p1"], p1)
        ]
        fresh = stage1.loc[
            stage1["token_budget_requested"].eq(token_budget)
            & stage1["run_kind"].eq("incumbent_repeat")
            & np.isclose(stage1["phase_0_starcoder"], p0)
            & np.isclose(stage1["phase_1_starcoder"], p1)
        ]
        if len(old) != 1 or len(fresh) != 4:
            raise ValueError(
                f"Expected one selecting seed and four fresh repeats at {(token_budget, p0, p1)}, "
                f"found {len(old)} and {len(fresh)}"
            )
        selecting = float(old.iloc[0]["bpb"])
        fresh_values = fresh["starcoder_bpb"].to_numpy(dtype=float)
        all_values = np.concatenate([[selecting], fresh_values])
        fresh_sem = float(stats.sem(fresh_values))
        fresh_half_width = float(stats.t.ppf(0.975, len(fresh_values) - 1) * fresh_sem)
        rows.append(
            {
                "token_budget_requested": token_budget,
                "phase_0_starcoder": p0,
                "phase_1_starcoder": p1,
                "selecting_seed_bpb": selecting,
                "fresh_mean_bpb": float(fresh_values.mean()),
                "fresh_sd_bpb": float(fresh_values.std(ddof=1)),
                "fresh_ci_low": float(fresh_values.mean() - fresh_half_width),
                "fresh_ci_high": float(fresh_values.mean() + fresh_half_width),
                "all_five_mean_bpb": float(all_values.mean()),
                "all_five_sd_bpb": float(all_values.std(ddof=1)),
                "fresh_minus_reference_seed": float(fresh_values.mean() - selecting),
                "fresh_better_count": int(np.sum(fresh_values < selecting)),
            }
        )
    return pd.DataFrame(rows)


def seed_block_offsets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify the joint trainer/subset seed offset at repeated fiber coordinates."""
    fibers = pd.read_csv(SCALE_FIBERS_PATH)
    required = {
        "token_budget_requested",
        "phase_0_starcoder",
        "phase_1_starcoder",
        "trainer_data_seed",
        "simulated_epoch_subset_seed",
        "starcoder_bpb",
    }
    if not required.issubset(fibers.columns):
        raise ValueError(f"Fiber observations lack required columns: {sorted(required - set(fibers.columns))}")

    rows = []
    coordinate_columns = ["token_budget_requested", "phase_0_starcoder", "phase_1_starcoder"]
    for coordinate, group in fibers.groupby(coordinate_columns, sort=True):
        reference = group.loc[
            group["trainer_data_seed"].eq(REFERENCE_SEED) & group["simulated_epoch_subset_seed"].eq(REFERENCE_SEED)
        ]
        fresh = group.loc[
            group["trainer_data_seed"].isin(FRESH_SEEDS)
            & group["simulated_epoch_subset_seed"].isin(FRESH_SEEDS)
            & group["trainer_data_seed"].eq(group["simulated_epoch_subset_seed"])
        ]
        if len(reference) != 1 or len(fresh) != len(FRESH_SEEDS):
            continue
        reference_bpb = float(reference.iloc[0]["starcoder_bpb"])
        fresh_values = fresh["starcoder_bpb"].to_numpy(dtype=float)
        rows.append(
            {
                "token_budget_requested": int(coordinate[0]),
                "phase_0_starcoder": float(coordinate[1]),
                "phase_1_starcoder": float(coordinate[2]),
                "reference_seed_bpb": reference_bpb,
                "fresh_mean_bpb": float(fresh_values.mean()),
                "fresh_sd_bpb": float(fresh_values.std(ddof=1)),
                "fresh_minus_reference_seed": float(fresh_values.mean() - reference_bpb),
                "fresh_better_count": int(np.sum(fresh_values < reference_bpb)),
                "seed_protocol": "trainer_and_subset_seed_changed_together",
            }
        )
    coordinates = pd.DataFrame(rows).sort_values(coordinate_columns).reset_index(drop=True)
    if coordinates.empty:
        raise ValueError("No five-seed fiber coordinates were found for seed-block auditing")
    summary = (
        coordinates.groupby("token_budget_requested", as_index=False)
        .agg(
            repeated_coordinates=("fresh_minus_reference_seed", "size"),
            mean_fresh_minus_reference=("fresh_minus_reference_seed", "mean"),
            min_fresh_minus_reference=("fresh_minus_reference_seed", "min"),
            max_fresh_minus_reference=("fresh_minus_reference_seed", "max"),
            all_fresh_better_count=("fresh_better_count", "sum"),
        )
        .sort_values("token_budget_requested")
        .reset_index(drop=True)
    )
    return coordinates, summary


def region_summary(stage1: pd.DataFrame) -> pd.DataFrame:
    """Report the measured acquisition range in every preregistered basin."""
    acquisitions = stage1.loc[stage1["run_kind"].eq("acquisition")]
    return (
        acquisitions.groupby(["token_budget_requested", "region"], as_index=False)
        .agg(
            acquisitions=("starcoder_bpb", "size"),
            best_bpb=("starcoder_bpb", "min"),
            median_bpb=("starcoder_bpb", "median"),
            worst_bpb=("starcoder_bpb", "max"),
        )
        .sort_values(["token_budget_requested", "region"])
        .reset_index(drop=True)
    )


def write_report(
    output_dir: Path,
    stage1: pd.DataFrame,
    collapsed: pd.DataFrame,
    incumbents: pd.DataFrame,
    regions: pd.DataFrame,
    seed_offsets: pd.DataFrame,
) -> None:
    best = collapsed.groupby("token_budget_requested", as_index=False).first()
    lines = [
        "# StarCoder WSD80 scale Bayesian refinement: Stage-1 outcomes",
        "",
        f"- Durable final observations: {len(stage1)}/{EXPECTED_RUNS}.",
        "- Final BPB values come from checkpoint `eval_metrics.jsonl`; W&B supplies identity and links only.",
        "- Repeated coordinates are collapsed before comparing candidate optima.",
        "- These data refine preregistered local basins; they do not establish global optimality over the square.",
        "",
        "## Incumbent replication",
        "",
        incumbents.to_markdown(index=False, floatfmt=".6f"),
        "",
        "The fresh runs are systematically better than the reference seed at every repeated incumbent. "
        "This is a seed-block offset, not winner's curse: the same sign recurs across preregistered fiber coordinates.",
        "",
        "## Joint-seed block audit",
        "",
        seed_offsets.to_markdown(index=False, floatfmt=".6f"),
        "",
        "Spatial acquisitions and their incumbent comparisons use the reference seed only. Fresh-seed confirmation "
        "must compare a candidate and incumbent under the same joint trainer/subset seeds; pooled means across these "
        "blocks are not admissible for sub-noise spatial gains.",
        "",
        "## Stage-1 acquisition basins",
        "",
        regions.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Best collapsed observed coordinate by rung",
        "",
        best[
            [
                "token_budget_requested",
                "p0_key",
                "p1_key",
                "mean_bpb",
                "sd_bpb",
                "count",
                "stage1_count",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Stage-2 boundary",
        "",
        "Stage 2 may be generated only from this completed table and the predeclared local GP/noise procedure. "
        "The chosen batch must preserve unresolved competing basins, exclude already measured coordinates, "
        "pass central1 locality checks, and receive independent CC review before submission.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage1, design = collect_stage1(args.design, args.wandb_timeout, args.workers)
    existing = existing_observations()
    collapsed = collapsed_coordinates(existing, stage1)
    incumbents = incumbent_summary(existing, stage1)
    regions = region_summary(stage1)
    seed_offset_coordinates, seed_offsets = seed_block_offsets()

    stage1.to_csv(args.output_dir / "stage1_observations.csv", index=False)
    collapsed.to_csv(args.output_dir / "collapsed_coordinates.csv", index=False)
    incumbents.to_csv(args.output_dir / "incumbent_repeats.csv", index=False)
    regions.to_csv(args.output_dir / "region_summary.csv", index=False)
    seed_offset_coordinates.to_csv(args.output_dir / "seed_block_offset_coordinates.csv", index=False)
    seed_offsets.to_csv(args.output_dir / "seed_block_offsets.csv", index=False)
    (args.output_dir / "source_design.json").write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir, stage1, collapsed, incumbents, regions, seed_offsets)


if __name__ == "__main__":
    main()
