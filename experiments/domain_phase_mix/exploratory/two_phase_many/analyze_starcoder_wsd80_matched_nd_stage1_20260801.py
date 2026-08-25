# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "tabulate",
#   "wandb",
# ]
# ///

"""Collect and audit Stage 1 of the matched-compute StarCoder WSD80 N-D panel."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_starcoder_wsd80_scale_bo_stage1_20260801 as scale_analysis,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    starcoder_wsd80_training_identity as stream_identity,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
PANEL_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_matched_nd_stage1_20260731"
DEFAULT_OUTPUT_DIR = PANEL_DIR / "results_20260801"
FROZEN_DESIGN_PATH = SCRIPT_DIR.parents[1] / "starcoder_wsd80_matched_nd_stage1_design_20260731.json"

TRAIN_PROJECT = "marin-community/marin"
TRAIN_TAG = "starcoder_wsd80_matched_nd_stage1"
EXPECTED_RUNS = 180
EXPECTED_CELLS = 10
EXPECTED_COORDINATES = 18
PHASE_0_FRACTION = 0.8
TIED_CONTROL_BY_OFF_DIAGONAL = {
    "off_low_d040": "diag_agg018",
    "off_low_d080": "diag_agg018",
    "off_mid_plus": "diag_agg035",
    "off_mid_minus": "diag_agg035",
    "off_high_plus": "diag_agg075",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=FROZEN_DESIGN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wandb-timeout", type=int, default=240)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def collect_stage1(design_path: Path, timeout: int, workers: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join the immutable manifest to one durable final observation per run."""
    design = json.loads(design_path.read_text(encoding="utf-8"))
    manifest = pd.DataFrame(design["runs"])
    if len(manifest) != EXPECTED_RUNS or manifest["run_name"].duplicated().any():
        raise ValueError(f"Expected {EXPECTED_RUNS} unique design rows")
    if design.get("cell_count") != EXPECTED_CELLS or design.get("coordinate_count_per_cell") != EXPECTED_COORDINATES:
        raise ValueError("Frozen design has unexpected cell or coordinate counts")

    api = wandb.Api(timeout=timeout)
    runs = list(api.runs(TRAIN_PROJECT, filters={"tags": TRAIN_TAG}, per_page=250))
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
        metrics = list(executor.map(scale_analysis.persisted_final_metric, ordered_runs))

    observations = manifest.copy()
    observations["starcoder_bpb"] = [metric.value for metric in metrics]
    observations["final_metric_step"] = [metric.step for metric in metrics]
    observations["expected_final_metric_step"] = observations["total_steps"].astype(int) - 1
    observations["metric_uri"] = [metric.uri for metric in metrics]
    observations["metric_source"] = "persisted eval_metrics.jsonl"
    observations["wandb_id"] = [str(run.id) for run in ordered_runs]
    observations["wandb_state"] = [str(run.state) for run in ordered_runs]
    observations["wandb_url"] = [str(run.url) for run in ordered_runs]
    observations["boundary_step"] = (observations["total_steps"].astype(int) * PHASE_0_FRACTION).astype(int)

    stream_digests = []
    for (_, row), run in zip(observations.iterrows(), ordered_runs, strict=True):
        expected_policy = [
            {"boundary_step": 0, "starcoder_weight": float(row["phase_0_starcoder"])},
            {"boundary_step": int(row["boundary_step"]), "starcoder_weight": float(row["phase_1_starcoder"])},
        ]
        observed_policy = stream_identity.policy_coordinates(run.config)
        differences = stream_identity.identity_differences(observed_policy, expected_policy)
        if differences:
            raise ValueError(f"{row['run_name']}: persisted training policy disagrees with the manifest: {differences}")
        stream_digests.append(stream_identity.canonical_sha256(stream_identity.wandb_stream_identity(run.config)))
    observations["stream_identity_sha256"] = stream_digests
    coordinate_metadata = pd.DataFrame(design["coordinates"])
    cell_metadata = pd.DataFrame(design["cells"])
    observations = observations.merge(
        coordinate_metadata,
        on="coordinate_id",
        validate="many_to_one",
        suffixes=("", "_coordinate"),
    )
    for column in ("phase_0_starcoder", "phase_1_starcoder"):
        coordinate_column = f"{column}_coordinate"
        if not np.allclose(observations[column], observations[coordinate_column], rtol=0.0, atol=1e-12):
            raise ValueError(f"Run and coordinate metadata disagree for {column}")
        observations = observations.drop(columns=coordinate_column)
    observations = observations.merge(cell_metadata, on="cell_id", validate="many_to_one", suffixes=("", "_cell"))
    for column in ("hidden_size", "total_steps", "materialized_tokens"):
        cell_column = f"{column}_cell"
        if not observations[column].eq(observations[cell_column]).all():
            raise ValueError(f"Run and cell metadata disagree for {column}")
        observations = observations.drop(columns=cell_column)
    if not np.isfinite(observations["starcoder_bpb"].to_numpy(dtype=float)).all():
        raise ValueError("Stage-1 observations contain non-finite BPB")
    incomplete = observations.loc[observations["final_metric_step"].ne(observations["expected_final_metric_step"])]
    if not incomplete.empty:
        details = incomplete[["run_name", "final_metric_step", "expected_final_metric_step"]].to_dict("records")
        raise ValueError(f"Stage-1 contains partial checkpoints: {details}")
    stream_counts = observations.groupby("cell_id")["stream_identity_sha256"].nunique()
    if not stream_counts.eq(1).all():
        raise ValueError(f"Stage-1 policies within a cell do not share one policy-free training stream: {stream_counts}")
    return observations, design


def cell_summary(observations: pd.DataFrame) -> pd.DataFrame:
    """Summarize the best measured tied and unrestricted policies in each N-D cell."""
    rows = []
    for cell_id, group in observations.groupby("cell_id", sort=True):
        tied = group.loc[np.isclose(group["phase_0_starcoder"], group["phase_1_starcoder"])]
        if len(tied) != 13 or len(group) != EXPECTED_COORDINATES:
            raise ValueError(f"Cell {cell_id} has unexpected tied/full counts: {len(tied)}/{len(group)}")
        best_tied = tied.loc[tied["starcoder_bpb"].idxmin()]
        best_all = group.loc[group["starcoder_bpb"].idxmin()]
        metadata = group.iloc[0]
        rows.append(
            {
                "cell_id": cell_id,
                "rung": int(metadata["rung"]),
                "track_memberships": ",".join(metadata["track_memberships"]),
                "hidden_size": int(metadata["hidden_size"]),
                "materialized_tokens": int(metadata["materialized_tokens"]),
                "total_parameters": int(metadata["total_parameters"]),
                "non_embedding_parameters": int(metadata["non_embedding_parameters"]),
                "total_parameter_tpp": float(metadata["materialized_tokens"] / metadata["total_parameters"]),
                "best_tied_coordinate": str(best_tied["coordinate_id"]),
                "best_tied_weight": float(best_tied["phase_0_starcoder"]),
                "best_tied_bpb": float(best_tied["starcoder_bpb"]),
                "best_observed_coordinate": str(best_all["coordinate_id"]),
                "best_observed_p0": float(best_all["phase_0_starcoder"]),
                "best_observed_p1": float(best_all["phase_1_starcoder"]),
                "best_observed_bpb": float(best_all["starcoder_bpb"]),
                "observed_policy_class_gap_bpb": float(best_all["starcoder_bpb"] - best_tied["starcoder_bpb"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["rung", "cell_id"]).reset_index(drop=True)


def aggregate_matched_effects(observations: pd.DataFrame) -> pd.DataFrame:
    """Compute every preregistered off-diagonal effect against its exact tied aggregate."""
    rows = []
    for cell_id, group in observations.groupby("cell_id", sort=True):
        by_coordinate = group.set_index("coordinate_id")
        for off_diagonal, tied_control in TIED_CONTROL_BY_OFF_DIAGONAL.items():
            candidate = by_coordinate.loc[off_diagonal]
            tied = by_coordinate.loc[tied_control]
            rows.append(
                {
                    "cell_id": cell_id,
                    "rung": int(candidate["rung"]),
                    "track_memberships": ",".join(candidate["track_memberships"]),
                    "off_diagonal_coordinate": off_diagonal,
                    "tied_control_coordinate": tied_control,
                    "aggregate_starcoder": float(candidate["aggregate_starcoder"]),
                    "phase_contrast": float(candidate["phase_contrast"]),
                    "off_diagonal_bpb": float(candidate["starcoder_bpb"]),
                    "tied_control_bpb": float(tied["starcoder_bpb"]),
                    "off_diagonal_minus_tied_bpb": float(candidate["starcoder_bpb"] - tied["starcoder_bpb"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["rung", "cell_id", "off_diagonal_coordinate"]).reset_index(drop=True)


def write_report(output_dir: Path, observations: pd.DataFrame, cells: pd.DataFrame, effects: pd.DataFrame) -> None:
    improving = effects.loc[effects["off_diagonal_minus_tied_bpb"].lt(0)]
    lines = [
        "# StarCoder WSD80 matched-N,D Stage-1 outcomes",
        "",
        f"- Durable final observations: {len(observations)}/{EXPECTED_RUNS} across {len(cells)} cells.",
        "- Final BPB values come from checkpoint `eval_metrics.jsonl`; W&B supplies identity and links only.",
        "- All comparisons below use the shared reference trainer/subset seed, so cross-policy differences are "
        "not confounded by the seed block found in the scale-refinement panel.",
        f"- Aggregate-matched off-diagonal improvements: {len(improving)}/{len(effects)} measured contrasts.",
        "",
        "## Measured optima by cell",
        "",
        cells.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Aggregate-matched phase effects",
        "",
        effects.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Stage-2 boundary",
        "",
        "Stage 2 is one frozen 50-run batch: two tied-diagonal and three unrestricted acquisitions per cell. "
        "The across-cell model may borrow strength in `(aggregate, contrast, log N, log D)`, but every proposal "
        "must be audited against an independent per-cell fit, remain feasible, and satisfy the frozen minimum-"
        "separation rule before submission.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations, design = collect_stage1(args.design, args.wandb_timeout, args.workers)
    cells = cell_summary(observations)
    effects = aggregate_matched_effects(observations)
    observations.to_csv(args.output_dir / "stage1_observations.csv", index=False)
    cells.to_csv(args.output_dir / "cell_summary.csv", index=False)
    effects.to_csv(args.output_dir / "aggregate_matched_effects.csv", index=False)
    (args.output_dir / "source_design.json").write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir, observations, cells, effects)


if __name__ == "__main__":
    main()
