# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas",
#   "tabulate",
# ]
# ///

"""Materialize the frozen fresh-seed confirmation for promoted matched-N,D cells."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    starcoder_wsd80_training_identity as stream_identity,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
PANEL_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_matched_nd_stage1_20260731"
STAGE2_RESULTS_DIR = PANEL_DIR / "stage2_results_20260801"
CELL_SUMMARY_PATH = STAGE2_RESULTS_DIR / "cell_discovery_summary.csv"
STAGE1_OBSERVATIONS_PATH = PANEL_DIR / "results_20260801" / "stage1_observations.csv"
STAGE2_OBSERVATIONS_PATH = STAGE2_RESULTS_DIR / "stage2_observations.csv"
SOURCE_DESIGN_PATH = SCRIPT_DIR.parents[1] / "starcoder_wsd80_matched_nd_stage2_design_20260801.json"
OUTPUT_DIR = PANEL_DIR / "confirmation_design_20260801"
FROZEN_DESIGN_PATH = SCRIPT_DIR.parents[1] / "starcoder_wsd80_matched_nd_confirmation_design_20260801.json"
LAUNCHER_PATH = SCRIPT_DIR.parents[1] / "launch_starcoder_wsd_80_20_matched_nd_confirmation.py"
ANALYZER_PATH = SCRIPT_DIR / "analyze_starcoder_wsd80_matched_nd_confirmation_20260801.py"
STAGE2_ANALYZER_PATH = SCRIPT_DIR / "analyze_starcoder_wsd80_matched_nd_stage2_20260801.py"
STAGE2_DESIGNER_PATH = SCRIPT_DIR / "design_starcoder_wsd80_matched_nd_stage2_20260801.py"
BASE_LAUNCHER_PATH = SCRIPT_DIR.parents[1] / "launch_starcoder_wsd_80_20_surface.py"
STAGE1_LAUNCHER_PATH = SCRIPT_DIR.parents[1] / "launch_starcoder_wsd_80_20_matched_nd_stage1.py"
STREAM_IDENTITY_PATH = SCRIPT_DIR / "starcoder_wsd80_training_identity.py"
COMPLETED_ADAMH_PATH = REPO_ROOT / "experiments/scaling_law_sweeps/completed_adamh.py"
TRAIN_LM_PATH = REPO_ROOT / "lib/marin/src/marin/experiment/train.py"
DATASET_PATHS = (
    REPO_ROOT / "experiments/datasets/dolma.py",
    REPO_ROOT / "experiments/datasets/nemotron.py",
    REPO_ROOT / "experiments/datasets/paloma.py",
    REPO_ROOT / "experiments/datasets/uncheatable.py",
    REPO_ROOT / "experiments/llama.py",
)

DESIGN_VERSION = "2026-08-01"
PHASE_0_FRACTION = 0.8
PROMOTION_GAIN_THRESHOLD = 0.005
ROLES = ("untied_candidate", "tied_comparator")
LAUNCH_FIELDS = (
    "run_name",
    "cell_id",
    "role",
    "pair_seed",
    "phase_0_starcoder",
    "phase_1_starcoder",
    "total_steps",
    "boundary_step",
    "data_seed",
    "simulated_epoch_subset_seed",
    "pair_stream_identity_sha256",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weight_slug(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


def launch_manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the fields that fully determine emitted confirmation training runs."""
    return [{field: row[field] for field in LAUNCH_FIELDS} for row in rows]


def pair_stream_identity(cell: Mapping[str, Any], pair_seed: int) -> str:
    """Hash policy-free stream inputs shared by both arms of one pair."""
    payload = {
        "cell_id": str(cell["cell_id"]),
        "hidden_size": int(cell["hidden_size"]),
        "num_layers": int(cell["num_layers"]),
        "num_heads": int(cell["num_heads"]),
        "total_steps": int(cell["total_steps"]),
        "materialized_tokens": int(cell["materialized_tokens"]),
        "total_parameters": int(cell["total_parameters"]),
        "non_embedding_parameters": int(cell["non_embedding_parameters"]),
        "flops_per_token": float(cell["flops_per_token"]),
        "phase_boundary": PHASE_0_FRACTION,
        "data_seed": int(pair_seed),
        "simulated_epoch_subset_seed": int(pair_seed),
    }
    return stream_identity.canonical_sha256(payload)


def _promoted_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin(("true", "false"))
    if invalid.any():
        raise ValueError(f"Invalid promoted values: {sorted(series.loc[invalid].astype(str).unique())}")
    return normalized.eq("true")


def build_rows(
    summary: pd.DataFrame,
    source_design: dict[str, Any],
    discovery_seeds: set[int],
) -> list[dict[str, Any]]:
    """Apply the frozen promotion rule and pair each selected policy on eight fresh seeds."""
    expected_columns = {
        "cell_id",
        "rung",
        "promoted",
        "discovery_gain_tied_minus_untied_bpb",
        "best_tied_p0",
        "best_tied_p1",
        "best_tied_bpb",
        "best_untied_p0",
        "best_untied_p1",
        "best_untied_bpb",
    }
    missing = expected_columns - set(summary)
    if missing:
        raise ValueError(f"Discovery summary is missing columns: {sorted(missing)}")
    promoted_mask = _promoted_mask(summary["promoted"])
    promoted = summary.loc[promoted_mask].copy()
    if promoted.empty:
        raise ValueError("No cell clears the frozen discovery threshold; confirmation is not authorized")
    if not promoted["discovery_gain_tied_minus_untied_bpb"].ge(PROMOTION_GAIN_THRESHOLD).all():
        raise ValueError("A promoted cell does not clear the frozen discovery threshold")
    if summary.loc[~promoted_mask, "discovery_gain_tied_minus_untied_bpb"].ge(PROMOTION_GAIN_THRESHOLD).any():
        raise ValueError("The discovery summary omits an eligible promoted cell")

    boundary = source_design["confirmation_boundary"]
    promotion_rule = str(boundary["promotion_rule"])
    if f"at least {PROMOTION_GAIN_THRESHOLD:.3f} BPB" not in promotion_rule:
        raise ValueError("Frozen promotion prose does not match the numeric threshold")
    seeds = tuple(int(seed) for seed in boundary["fresh_seeds"])
    if len(seeds) != 8 or len(set(seeds)) != 8:
        raise ValueError("Frozen confirmation must contain eight unique fresh seeds")
    if set(seeds) & discovery_seeds:
        raise ValueError("Confirmation seeds overlap the adaptive discovery panel")
    source_cells = {str(row["cell_id"]): row for row in source_design["source_cells"]}
    rows: list[dict[str, Any]] = []
    for selection in promoted.sort_values(["rung", "cell_id"]).to_dict("records"):
        cell_id = str(selection["cell_id"])
        cell = source_cells[cell_id]
        policies = {
            "untied_candidate": (float(selection["best_untied_p0"]), float(selection["best_untied_p1"])),
            "tied_comparator": (float(selection["best_tied_p0"]), float(selection["best_tied_p1"])),
        }
        if abs(policies["untied_candidate"][0] - policies["untied_candidate"][1]) <= 1e-12:
            raise ValueError(f"{cell_id}: selected candidate is phase tied")
        if abs(policies["tied_comparator"][0] - policies["tied_comparator"][1]) > 1e-12:
            raise ValueError(f"{cell_id}: selected comparator is not phase tied")
        for seed in seeds:
            for role in ROLES:
                p0, p1 = policies[role]
                rows.append(
                    {
                        "run_name": f"confirm_{cell_id}_{role}_p0{_weight_slug(p0)}_p1{_weight_slug(p1)}_s{seed}",
                        "cell_id": cell_id,
                        "role": role,
                        "pair_seed": seed,
                        "phase_0_starcoder": p0,
                        "phase_1_starcoder": p1,
                        "total_steps": int(cell["total_steps"]),
                        "boundary_step": int(int(cell["total_steps"]) * PHASE_0_FRACTION),
                        "materialized_tokens": int(cell["materialized_tokens"]),
                        "hidden_size": int(cell["hidden_size"]),
                        "total_parameters": int(cell["total_parameters"]),
                        "non_embedding_parameters": int(cell["non_embedding_parameters"]),
                        "data_seed": seed,
                        "simulated_epoch_subset_seed": seed,
                        "pair_stream_identity_sha256": pair_stream_identity(cell, seed),
                        "discovery_gain_tied_minus_untied_bpb": float(selection["discovery_gain_tied_minus_untied_bpb"]),
                        "selected_candidate_bpb": float(selection["best_untied_bpb"]),
                        "selected_comparator_bpb": float(selection["best_tied_bpb"]),
                    }
                )
    expected_count = len(promoted) * len(seeds) * len(ROLES)
    if len(rows) != expected_count or len({row["run_name"] for row in rows}) != expected_count:
        raise ValueError("Confirmation materialization did not produce unique paired runs")
    return rows


def discovery_seeds() -> set[int]:
    """Return every data-order and simulated-subset seed used in discovery."""
    seeds = pd.concat(
        [
            pd.read_csv(
                STAGE1_OBSERVATIONS_PATH,
                usecols=["data_seed", "simulated_epoch_subset_seed"],
            ).stack(),
            pd.read_csv(
                STAGE2_OBSERVATIONS_PATH,
                usecols=["data_seed", "simulated_epoch_subset_seed"],
            ).stack(),
        ],
        ignore_index=True,
    )
    if seeds.isna().any():
        raise ValueError("Discovery observations contain missing data seeds")
    return set(seeds.astype("int64").tolist())


def regenerate_rows() -> list[dict[str, Any]]:
    """Rebuild the exact confirmation rows from hash-pinned discovery inputs."""
    summary = pd.read_csv(CELL_SUMMARY_PATH)
    source_design = json.loads(SOURCE_DESIGN_PATH.read_text(encoding="utf-8"))
    return build_rows(summary, source_design, discovery_seeds())


def write_outputs() -> None:
    """Persist the exact paired confirmation authorized by the completed discovery panel."""
    source_design = json.loads(SOURCE_DESIGN_PATH.read_text(encoding="utf-8"))
    rows = regenerate_rows()
    promoted_cells = sorted({str(row["cell_id"]) for row in rows})
    source_cells = [row for row in source_design["source_cells"] if str(row["cell_id"]) in promoted_cells]
    discovery_rows = len(pd.read_csv(STAGE1_OBSERVATIONS_PATH)) + len(pd.read_csv(STAGE2_OBSERVATIONS_PATH))
    source_paths = (
        Path(__file__).resolve(),
        LAUNCHER_PATH,
        ANALYZER_PATH,
        STAGE2_ANALYZER_PATH,
        STAGE2_DESIGNER_PATH,
        BASE_LAUNCHER_PATH,
        STAGE1_LAUNCHER_PATH,
        STREAM_IDENTITY_PATH,
        COMPLETED_ADAMH_PATH,
        TRAIN_LM_PATH,
        *DATASET_PATHS,
        CELL_SUMMARY_PATH,
        STAGE1_OBSERVATIONS_PATH,
        STAGE2_OBSERVATIONS_PATH,
        SOURCE_DESIGN_PATH,
    )
    payload = {
        "design_version": DESIGN_VERSION,
        "description": "Fresh-seed paired confirmation for matched-N,D cells clearing the frozen discovery gate.",
        "objective_metric": source_design["objective_metric"],
        "phase_0_fraction": PHASE_0_FRACTION,
        "expected_run_count": len(rows),
        "cell_count": len(promoted_cells),
        "pair_count_per_cell": len(source_design["confirmation_boundary"]["fresh_seeds"]),
        "training_environment": source_design["training_environment"],
        "selection": {
            "promotion_threshold_bpb": PROMOTION_GAIN_THRESHOLD,
            "promotion_rule": source_design["confirmation_boundary"]["promotion_rule"],
            "promoted_cells": promoted_cells,
            "discovery_rows": discovery_rows,
        },
        "analysis_plan": {
            "success_rule": source_design["confirmation_boundary"]["success_rule"],
            "claim_limit": source_design["confirmation_boundary"]["claim_limit"],
            "paired_seeds": source_design["confirmation_boundary"]["fresh_seeds"],
            "multiple_testing": "Holm-adjust one-sided paired-t p-values across all promoted cells.",
            "non_pass_interpretation": (
                "A non-pass means the selected contrast was not confirmed at the frozen eight-pair power; it is not "
                "evidence of zero effect. Report the realized paired-difference SD regardless of the decision."
            ),
            "provenance_disclosure": (
                "The 0.005-BPB promotion threshold was chosen after Stage-1 outcomes were available. Fresh-seed "
                "confirmation preserves test validity for this selected discrete comparison, but the discovery rule "
                "was not fully preregistered."
            ),
            "estimand": (
                "The candidate and comparator differ in aggregate StarCoder exposure as well as phase order. A pass "
                "establishes only that the selected two-phase policy beats the selected tied policy."
            ),
        },
        "data_use": {
            "source_sha256": {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in source_paths},
        },
        "design": {"launch_manifest_sha256": stream_identity.canonical_sha256(launch_manifest(rows))},
        "source_cells": source_cells,
        "runs": rows,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    FROZEN_DESIGN_PATH.write_text(serialized, encoding="utf-8")
    (OUTPUT_DIR / "design_manifest.json").write_text(serialized, encoding="utf-8")
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "run_manifest.csv", index=False)
    selected = frame[
        [
            "cell_id",
            "role",
            "phase_0_starcoder",
            "phase_1_starcoder",
            "discovery_gain_tied_minus_untied_bpb",
        ]
    ].drop_duplicates()
    report = [
        "# StarCoder WSD80 matched-N,D fresh-seed confirmation design",
        "",
        f"- Promoted cells: {len(promoted_cells)}.",
        f"- Frozen runs: {len(rows)} ({len(rows) // 2} same-seed candidate/comparator pairs).",
        f"- Discovery gate: tied minus untied BPB >= {PROMOTION_GAIN_THRESHOLD:.3f}.",
        "- Each promoted cell uses the eight fresh seeds frozen before Stage-2 outcomes were observed.",
        "- Inference uses the frozen per-cell paired rule and Holm adjustment across promoted cells.",
        "- A non-pass is reported as not confirmed at the frozen power, not as evidence of zero effect.",
        "- The selected policies differ in aggregate exposure as well as phase order; this is not a pure ordering test.",
        "",
        "## Selected policies",
        "",
        selected.to_markdown(index=False, floatfmt=".7f"),
        "",
        "## Claim boundary",
        "",
        str(payload["analysis_plan"]["claim_limit"]),
        "",
        "## Discovery provenance",
        "",
        str(payload["analysis_plan"]["provenance_disclosure"]),
        "",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    write_outputs()
