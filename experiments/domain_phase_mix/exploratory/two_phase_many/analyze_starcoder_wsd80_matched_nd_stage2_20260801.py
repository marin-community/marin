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

"""Collect matched-N,D Stage 2 and apply its frozen discovery gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_starcoder_wsd80_scale_bo_stage2_20260801 as scale_analysis,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_starcoder_wsd80_matched_nd_stage2_20260801 as frozen_designer,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    starcoder_wsd80_training_identity as stream_identity,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
PANEL_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_matched_nd_stage1_20260731"
DEFAULT_OUTPUT_DIR = PANEL_DIR / "stage2_results_20260801"
FROZEN_DESIGN_PATH = SCRIPT_DIR.parents[1] / "starcoder_wsd80_matched_nd_stage2_design_20260801.json"
STAGE1_OBSERVATIONS_PATH = PANEL_DIR / "results_20260801" / "stage1_observations.csv"

TRAIN_PROJECT = "marin-community/marin"
TRAIN_TAG = "starcoder_wsd80_matched_nd_stage2"
EXPECTED_STAGE1_RUNS = 180
EXPECTED_STAGE2_RUNS = 50
EXPECTED_CELLS = 10
PROMOTION_GAIN_THRESHOLD = 0.005


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=FROZEN_DESIGN_PATH)
    parser.add_argument("--stage1", type=Path, default=STAGE1_OBSERVATIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wandb-timeout", type=int, default=240)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def _verify_frozen_design(design: dict[str, Any], stage1_path: Path) -> pd.DataFrame:
    if design.get("design_version") != "2026-08-01":
        raise ValueError("Unexpected Stage-2 design version")
    if design.get("objective_metric") != scale_analysis.OBJECTIVE_METRIC:
        raise ValueError("Frozen Stage-2 objective does not match the durable-metric reader")
    if design.get("expected_run_count") != EXPECTED_STAGE2_RUNS or design.get("cell_count") != EXPECTED_CELLS:
        raise ValueError("Frozen Stage-2 design has unexpected run or cell counts")
    promotion_rule = str(design["confirmation_boundary"]["promotion_rule"])
    if f"at least {PROMOTION_GAIN_THRESHOLD:.3f} BPB" not in promotion_rule:
        raise ValueError("Frozen promotion threshold no longer matches the analyzer")
    stage1_key = str(STAGE1_OBSERVATIONS_PATH.relative_to(REPO_ROOT))
    expected_stage1_hash = design.get("data_use", {}).get("source_sha256", {}).get(stage1_key)
    if expected_stage1_hash is None:
        raise ValueError(f"Frozen design does not pin {stage1_key}")
    actual_stage1_hash = _sha256(stage1_path)
    if actual_stage1_hash != expected_stage1_hash:
        raise ValueError(
            f"Stage-1 observations differ from the frozen acquisition input: "
            f"{actual_stage1_hash} != {expected_stage1_hash}"
        )
    rows = design.get("runs")
    if not isinstance(rows, list) or len(rows) != EXPECTED_STAGE2_RUNS:
        raise ValueError("Frozen Stage-2 design does not contain 50 runs")
    manifest = pd.DataFrame(rows)
    if manifest["run_name"].duplicated().any() or manifest["cell_id"].nunique() != EXPECTED_CELLS:
        raise ValueError("Frozen Stage-2 run names or cell coverage are invalid")
    expected_hash = design.get("design", {}).get("launch_manifest_sha256")
    actual_hash = stream_identity.canonical_sha256(frozen_designer.launch_manifest(rows))
    if actual_hash != expected_hash:
        raise ValueError(f"Frozen Stage-2 launch manifest hash is invalid: {actual_hash} != {expected_hash}")
    return manifest


def _ordered_wandb_runs(manifest: pd.DataFrame, timeout: int) -> list[Any]:
    api = wandb.Api(timeout=timeout)
    runs = list(api.runs(TRAIN_PROJECT, filters={"tags": TRAIN_TAG}, per_page=100))
    by_name: dict[str, list[Any]] = {}
    for run in runs:
        by_name.setdefault(str(run.name), []).append(run)
    ordered = []
    for run_name in manifest["run_name"]:
        candidates = by_name.get(str(run_name), [])
        if len(candidates) != 1:
            raise ValueError(f"{run_name}: expected exactly one W&B run, found {len(candidates)}")
        ordered.append(candidates[0])
    return ordered


def _verify_training_streams(manifest: pd.DataFrame, runs: list[Any], stage1: pd.DataFrame) -> list[str]:
    digests_by_cell: dict[str, set[str]] = {}
    digests = []
    for row, run in zip(manifest.to_dict("records"), runs, strict=True):
        expected_policy = [
            {"boundary_step": 0, "starcoder_weight": float(row["phase_0_starcoder"])},
            {"boundary_step": int(row["boundary_step"]), "starcoder_weight": float(row["phase_1_starcoder"])},
        ]
        differences = stream_identity.identity_differences(
            stream_identity.policy_coordinates(run.config), expected_policy
        )
        if differences:
            raise ValueError(f"{row['run_name']}: persisted policy disagrees with the frozen manifest: {differences}")
        digest = stream_identity.canonical_sha256(stream_identity.wandb_stream_identity(run.config))
        digests.append(digest)
        digests_by_cell.setdefault(str(row["cell_id"]), set()).add(digest)

    inconsistent = {cell_id: values for cell_id, values in digests_by_cell.items() if len(values) != 1}
    if inconsistent:
        raise ValueError(f"Stage-2 rows within a cell do not share one policy-free stream: {inconsistent}")
    stage1_digests = stage1.groupby("cell_id")["stream_identity_sha256"].agg(lambda values: set(values))
    for cell_id, stage2_digests in digests_by_cell.items():
        if stage1_digests.loc[cell_id] != stage2_digests:
            raise ValueError(f"Stage-1 and Stage-2 stream identities differ for {cell_id}")
    return digests


def collect_stage2(
    design_path: Path,
    stage1_path: Path,
    timeout: int,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Join the frozen manifest to one durable final metric per Stage-2 run."""
    design = json.loads(design_path.read_text(encoding="utf-8"))
    manifest = _verify_frozen_design(design, stage1_path)
    stage1 = pd.read_csv(stage1_path)
    if len(stage1) != EXPECTED_STAGE1_RUNS or stage1["cell_id"].nunique() != EXPECTED_CELLS:
        raise ValueError("Stage-1 source does not contain 180 rows over ten cells")
    stage1_counts = stage1.groupby("cell_id").size()
    stage2_counts = manifest.groupby("cell_id").size()
    if not stage1_counts.eq(18).all() or not stage2_counts.eq(5).all():
        raise ValueError(
            f"Unexpected per-cell Stage-1/Stage-2 counts: {stage1_counts.to_dict()}, {stage2_counts.to_dict()}"
        )
    if set(stage1_counts.index) != set(stage2_counts.index):
        raise ValueError("Stage-1 and Stage-2 do not cover the same N,D cells")
    runs = _ordered_wandb_runs(manifest, timeout)
    stream_digests = _verify_training_streams(manifest, runs, stage1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        metrics = list(executor.map(scale_analysis.persisted_final_metric, runs))

    observations = manifest.copy()
    observations["starcoder_bpb"] = [metric.value for metric in metrics]
    observations["final_metric_step"] = [metric.step for metric in metrics]
    observations["expected_final_metric_step"] = observations["total_steps"].astype(int) - 1
    observations["metric_uri"] = [metric.uri for metric in metrics]
    observations["metric_source"] = "persisted eval_metrics.jsonl"
    observations["wandb_id"] = [str(run.id) for run in runs]
    observations["wandb_state"] = [str(run.state) for run in runs]
    observations["wandb_url"] = [str(run.url) for run in runs]
    observations["stream_identity_sha256"] = stream_digests
    if observations["metric_uri"].nunique() != EXPECTED_STAGE2_RUNS:
        raise ValueError("Stage-2 rows do not resolve to distinct durable metric files")
    misplaced_metrics = observations.loc[
        ~observations.apply(lambda row: str(row["run_name"]) in str(row["metric_uri"]), axis=1)
    ]
    if not misplaced_metrics.empty:
        raise ValueError(
            "A durable metric path is not anchored under its frozen run name: "
            f"{misplaced_metrics[['run_name', 'metric_uri']].to_dict('records')}"
        )
    if not np.isfinite(observations["starcoder_bpb"].to_numpy(dtype=float)).all():
        raise ValueError("Stage-2 observations contain non-finite BPB")
    incomplete = observations.loc[observations["final_metric_step"].ne(observations["expected_final_metric_step"])]
    if not incomplete.empty:
        details = incomplete[["run_name", "final_metric_step", "expected_final_metric_step"]].to_dict("records")
        raise ValueError(f"Stage-2 contains partial checkpoints: {details}")
    return observations, stage1, design


def discovery_summary(stage1: pd.DataFrame, stage2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare the best tied and untied policies under the frozen per-cell rule."""
    first = stage1.copy()
    first["source_stage"] = "stage1"
    first["selection_label"] = first["coordinate_id"].astype(str)
    second = stage2.copy()
    second["source_stage"] = "stage2"
    second["selection_label"] = second["acquisition_kind"].astype(str)
    shared = sorted(set(first.columns) & set(second.columns))
    combined = pd.concat([first[shared], second[shared]], ignore_index=True)
    tied_mask = np.abs(combined["phase_0_starcoder"] - combined["phase_1_starcoder"]) <= 1e-12
    combined["policy_class"] = np.where(tied_mask, "tied", "untied")

    rows = []
    for cell_id, group in combined.groupby("cell_id", sort=True):
        tied = group.loc[group["policy_class"].eq("tied")]
        untied = group.loc[group["policy_class"].eq("untied")]
        if tied.empty or untied.empty:
            raise ValueError(f"{cell_id}: missing tied or untied discovery candidates")
        best_tied = tied.loc[tied["starcoder_bpb"].idxmin()]
        best_untied = untied.loc[untied["starcoder_bpb"].idxmin()]
        gain = float(best_tied["starcoder_bpb"] - best_untied["starcoder_bpb"])
        rows.append(
            {
                "cell_id": cell_id,
                "rung": int(best_tied["rung"]),
                "hidden_size": int(best_tied["hidden_size"]),
                "materialized_tokens": int(best_tied["materialized_tokens"]),
                "total_parameters": int(best_tied["total_parameters"]),
                "total_parameter_tpp": float(best_tied["materialized_tokens"] / best_tied["total_parameters"]),
                "best_tied_source_stage": str(best_tied["source_stage"]),
                "best_tied_label": str(best_tied["selection_label"]),
                "best_tied_p0": float(best_tied["phase_0_starcoder"]),
                "best_tied_p1": float(best_tied["phase_1_starcoder"]),
                "best_tied_bpb": float(best_tied["starcoder_bpb"]),
                "best_untied_source_stage": str(best_untied["source_stage"]),
                "best_untied_label": str(best_untied["selection_label"]),
                "best_untied_p0": float(best_untied["phase_0_starcoder"]),
                "best_untied_p1": float(best_untied["phase_1_starcoder"]),
                "best_untied_bpb": float(best_untied["starcoder_bpb"]),
                "discovery_gain_tied_minus_untied_bpb": gain,
                "promoted": bool(gain >= PROMOTION_GAIN_THRESHOLD),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["rung", "cell_id"]).reset_index(drop=True)
    promotions = summary.loc[summary["promoted"]].copy()
    return combined, summary, promotions


def write_report(
    output_dir: Path,
    combined: pd.DataFrame,
    summary: pd.DataFrame,
    promotions: pd.DataFrame,
    design: dict[str, Any],
) -> None:
    lines = [
        "# StarCoder WSD80 matched-N,D Stage-2 outcomes",
        "",
        (
            f"- Complete discovery panel: {len(combined)} observations over "
            f"{combined['cell_id'].nunique()} fixed N,D cells."
        ),
        "- Final BPB values come from durable checkpoint `eval_metrics.jsonl`; W&B supplies identity, links, and "
        "the checkpoint root, which is cross-checked against the frozen run name.",
        f"- Frozen promotion threshold: tied minus untied discovery BPB >= {PROMOTION_GAIN_THRESHOLD:.3f}.",
        f"- Cells promoted to fresh-seed confirmation: {len(promotions)}/{combined['cell_id'].nunique()}.",
        "",
        "## Discovery result",
        "",
        summary.to_markdown(index=False, floatfmt=".7f"),
        "",
        "## Frozen interpretation boundary",
        "",
        str(design["interpretation_boundary"]),
        "",
        "## Adaptive-design provenance",
        "",
        str(design["design_provenance"]),
        "",
        "The promotion statistic compares two selected single-seed minima. The acquisition design assumed a "
        f"{design['acquisition']['noise_sd_bpb']:.3f}-BPB run-noise SD; promotion authorizes fresh-seed testing but "
        "is not itself evidence of a phase effect.",
        "",
    ]
    if promotions.empty:
        lines.extend(
            [
                "No cell clears the frozen discovery threshold, so this panel authorizes no positive confirmation "
                "launch.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Promoted cells must use the eight fresh seeds and paired success rule frozen in the source design. "
                "Passing confirms only the selected discrete untied policy against its selected tied comparator.",
                "",
            ]
        )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    stage2, stage1, design = collect_stage2(args.design, args.stage1, args.wandb_timeout, args.workers)
    combined, summary, promotions = discovery_summary(stage1, stage2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage2.to_csv(args.output_dir / "stage2_observations.csv", index=False)
    combined.to_csv(args.output_dir / "combined_discovery_observations.csv", index=False)
    summary.to_csv(args.output_dir / "cell_discovery_summary.csv", index=False)
    promotions.to_csv(args.output_dir / "promotion_candidates.csv", index=False)
    (args.output_dir / "source_design.json").write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir, combined, summary, promotions, design)


if __name__ == "__main__":
    main()
