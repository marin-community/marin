# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas",
#   "tabulate",
# ]
# ///

"""Inventory deduplicated one-dimensional StarCoder single-phase curves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from starcoder_wsd80_atomic_metrics import ATOMIC_METRICS

SCRIPT_DIR = Path(__file__).resolve().parent
EXPLORATORY_DIR = SCRIPT_DIR.parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "starcoder_single_phase_curve_inventory_20260902"

PRIMARY_TARGET = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
UNCHEATABLE_TARGET = "eval/uncheatable_eval/bpb"
MIN_UNIQUE_WEIGHTS = 13
MIN_WEIGHT_SPAN = 0.8
TIED_TOLERANCE = 1e-9
VALUE_TOLERANCE = 1e-9

FIXED_DIAGONAL_PATH = (
    REFERENCE_OUTPUTS
    / "starcoder_wsd80_fixed_model_tied_diagonal_20260730/results_20260731/tied_diagonal_observations.csv"
)
TOKEN_SURFACE_PATH = REFERENCE_OUTPUTS / "starcoder_wsd80_token_budget_surfaces_20260731/surface_coordinates.csv"
TOKEN_PANEL_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_fixed_model_token_scaling_20260728"
TOKEN_OBSERVATIONS_PATH = TOKEN_PANEL_DIR / "results_20260730/observations.csv"
TOKEN_DESIGN_PATH = TOKEN_PANEL_DIR / "design_manifest.json"
MATCHED_ND_PATH = (
    REFERENCE_OUTPUTS
    / "starcoder_wsd80_matched_nd_stage1_20260731"
    / "stage3_dense_surface_results_20260802"
    / "combined_discovery_observations.csv"
)
DENSE_REPLAY_PATH = (
    REFERENCE_OUTPUTS
    / "starcoder_wsd80_dense_support_calibration_results_20260813/coverage_with_calibration_weights.csv"
)
DENSE_ATOMIC_ENDPOINT_PATH = (
    REFERENCE_OUTPUTS
    / "starcoder_wsd80_full_pool_atomic_surface_explorer_20260811/full_pool_atomic_metric_observations.csv"
)
DENSE_ATOMIC_CHECKPOINT_PATH = (
    REFERENCE_OUTPUTS
    / "starcoder_wsd80_full_pool_atomic_surface_explorer_20260811/phase0_atomic_metric_observations.csv"
)
DENSE_AGGREGATE_ENDPOINT_PATH = (
    REFERENCE_OUTPUTS / "starcoder_wsd80_full_pool_joint_objective_20260811/full_pool_metric_observations.csv"
)
COUPLED_ONSET_PATH = REFERENCE_OUTPUTS / "starcoder_wsd80_coupled_onset_dense_surface_results_20260901/observations.csv"
HISTORICAL_TRAJECTORY_PATH = (
    REFERENCE_OUTPUTS / "intermediate_target_trajectory_audit_20260731/wsd80_target_histories.csv"
)

ATOMIC_LABELS = dict(ATOMIC_METRICS)
AGGREGATE_COLUMNS = {
    "code": PRIMARY_TARGET,
    "c4_100": "eval/paloma/c4_100_domains-llama3/bpb",
    "c4_en": "eval/paloma/c4_en-llama3/bpb",
    "refinedweb": "eval/paloma/falcon-refinedweb-llama3/bpb",
    "dolma15": "eval/paloma/dolma-v1_5-llama3/bpb",
    "paloma_macro": "eval/paloma/macro_bpb",
    "uncheatable": UNCHEATABLE_TARGET,
}
COUPLED_ONSET_COLUMNS = {
    "programming_languages_bpb": PRIMARY_TARGET,
    "c4_bpb": "eval/paloma/c4_en-llama3/bpb",
    "uncheatable_bpb": UNCHEATABLE_TARGET,
    "github_cpp_bpb": "eval/uncheatable_eval/github_cpp-llama3/bpb",
    "github_python_bpb": "eval/uncheatable_eval/github_python-llama3/bpb",
}

CANONICAL_SOURCE_ROLES = {
    FIXED_DIAGONAL_PATH: (
        "canonical_curve_geometry",
        "Regular 21-point endpoint grid for the fixed-model token ladder.",
    ),
    TOKEN_SURFACE_PATH: (
        "canonical_curve_geometry_supplement",
        "Adds measured irregular tied coordinates to the regular fixed-model grids.",
    ),
    MATCHED_ND_PATH: (
        "canonical_curve_geometry",
        "Latest union of all measured matched-N,D discovery stages.",
    ),
    DENSE_REPLAY_PATH: (
        "canonical_curve_geometry",
        "Audited dense horizon-by-support observations; calibration weights are ignored.",
    ),
    DENSE_ATOMIC_ENDPOINT_PATH: (
        "canonical_joined_target_payload",
        "Atomic endpoint BPBs joined to dense curves by canonical physical run name.",
    ),
    DENSE_ATOMIC_CHECKPOINT_PATH: (
        "canonical_joined_target_payload",
        "Atomic pre-boundary checkpoint BPBs joined to dense curves by physical run name.",
    ),
    DENSE_AGGREGATE_ENDPOINT_PATH: (
        "canonical_joined_target_payload",
        "Exact aggregate endpoint BPBs for the four full-support dense curves.",
    ),
    COUPLED_ONSET_PATH: (
        "canonical_curve_geometry_and_targets",
        "Measured endpoint surfaces for three coupled phase/LR-decay onsets.",
    ),
    HISTORICAL_TRAJECTORY_PATH: (
        "canonical_checkpoint_targets",
        "Three pre-endpoint tied curves and endpoint target augmentation for the 1B panel.",
    ),
}

SOURCE_PAIR_COLUMNS = (
    ("phase_0_starcoder", "phase_1_starcoder"),
    ("p0", "p1"),
    ("phase0_rare", "phase1_rare"),
    ("phase0_rare_weight", "phase1_rare_weight"),
    ("phase_0_rare", "phase_1_rare"),
    ("rare_weight_phase_0", "rare_weight_phase_1"),
    ("first_phase_starcoder", "second_phase_starcoder"),
    ("weight", "weight"),
)
SOURCE_TARGET_COLUMNS = {
    "programming_languages_bpb",
    "starcoder_bpb",
    "wsd80_bpb",
    "actual_bpb",
    "bpb",
}
SOURCE_ID_COLUMNS = (
    "wandb_id",
    "wandb_run_id",
    "training_wandb_id",
    "run_id",
    "run_name",
    "coordinate_id",
    "row_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_id(path: Path) -> str:
    return str(path.resolve().relative_to(SCRIPT_DIR.parents[3]))


def target_label(target_id: str) -> str:
    labels = {
        PRIMARY_TARGET: "Programming Languages BPB",
        UNCHEATABLE_TARGET: "Uncheatable macro BPB",
        "eval/paloma/macro_bpb": "Paloma macro BPB",
    }
    return labels.get(target_id, ATOMIC_LABELS.get(target_id, target_id))


def target_family(target_id: str) -> str:
    if target_id == PRIMARY_TARGET:
        return "programming_languages"
    if target_id == UNCHEATABLE_TARGET:
        return "uncheatable_macro"
    if target_id.startswith("eval/uncheatable_eval/"):
        return "uncheatable_component"
    if target_id == "eval/paloma/macro_bpb":
        return "paloma_macro"
    if target_id.startswith("eval/paloma/"):
        return "paloma_component"
    return "other"


def classify_source(path: Path, *, unique_p: int, has_identity: bool) -> tuple[str, str]:
    resolved = path.resolve()
    if resolved in CANONICAL_SOURCE_ROLES:
        return CANONICAL_SOURCE_ROLES[resolved]

    text = str(path)
    name = path.name
    if "three_phase_starcoder" in text:
        return "out_of_scope", "Three-phase policy; equality of only the first two phase weights is not single-phase."
    if "surrogate_search" in text or "predictions" in name:
        return "derived_prediction", "Predicted values are not measured training observations."
    if name in {"selected_optima.csv", "selected_policies.csv"} or "summary" in name:
        return "derived_summary", "Selection or summary table, not a complete observation curve."
    if "dense_support_empirical_optimum_confirmation_design_20260811/coverage_observations.csv" in text:
        return (
            "duplicate_snapshot",
            "Superseded by the calibrated dense-support table with the same 3,500 geometry rows.",
        )
    if "starcoder_wsd80_matched_nd_stage1_20260731" in text:
        return "nested_subset", "Earlier matched-N,D stage or stage-local view contained in the latest combined table."
    if "wsd80_surface_refined_20260714/wsd80_observed_metrics.csv" in text:
        return "nested_subset", "Its tied endpoint rows are included in the fixed-model 1B union curve."
    if (
        "two_phase_surrogate_collaborator_packet_20260721/data/raw/starcoder/wsd_80_20/wsd80_observed_metrics.csv"
        in text
    ):
        return "nested_subset", "All 18 tied W&B runs are included in the fixed-model token-surface export."
    if "starcoder_wsd80_surface_analysis_20260711" in text or "surrogate_packet" in text:
        return "duplicate_snapshot", "Earlier or copied snapshot of the fixed-model 1B surface."
    if "intermediate_target_trajectory_audit" in text:
        return "canonical_checkpoint_targets", "Checkpoint target history used by the normalized registry."
    if "measured_fiber_observations" in name or "repeat_observations" in name:
        return "replicate_bank", "Useful for noise estimation but not a complete distinct tied curve."
    if not has_identity:
        return "derived_summary", "No durable run identity is available for run-level deduplication."
    if unique_p < MIN_UNIQUE_WEIGHTS:
        return (
            "sparse_control",
            f"Only {unique_p} unique tied weights; below the {MIN_UNIQUE_WEIGHTS}-weight protocol floor.",
        )
    return "not_selected", "Measured table is superseded or requires a noncanonical join; retained for audit."


def discover_candidate_sources() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(EXPLORATORY_DIR.rglob("*.csv")):
        lowered = str(path).lower()
        if "starcoder" not in lowered and "wsd80" not in lowered:
            continue
        try:
            with path.open(newline="", errors="replace") as handle:
                columns = next(csv.reader(handle), [])
        except OSError:
            continue
        pair = next(((left, right) for left, right in SOURCE_PAIR_COLUMNS if left in columns and right in columns), None)
        if pair is None:
            continue
        targets = [
            column
            for column in columns
            if column in SOURCE_TARGET_COLUMNS
            or ("programing_languages" in column.lower() and column.lower().endswith("/bpb"))
        ]
        if not targets:
            continue
        identities = [column for column in SOURCE_ID_COLUMNS if column in columns]
        phase_0_column, phase_1_column = pair
        use_columns = list(dict.fromkeys([phase_0_column, phase_1_column, *targets, *identities]))
        try:
            frame = pd.read_csv(path, usecols=use_columns)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        phase_0 = pd.to_numeric(frame[phase_0_column], errors="coerce")
        phase_1 = pd.to_numeric(frame[phase_1_column], errors="coerce")
        tied = frame.loc[(phase_0 - phase_1).abs() <= TIED_TOLERANCE].copy()
        weights = phase_0.loc[tied.index]
        finite_target_rows = sum(int(pd.to_numeric(tied[column], errors="coerce").notna().sum()) for column in targets)
        if finite_target_rows == 0:
            continue
        unique_p = int(weights.nunique())
        role, reason = classify_source(path, unique_p=unique_p, has_identity=bool(identities))
        rows.append(
            {
                "source_id": source_id(path),
                "source_role": role,
                "reason": reason,
                "row_count": len(frame),
                "tied_row_count": len(tied),
                "unique_tied_weights_tablewide": unique_p,
                "tied_weight_min": float(weights.min()) if len(weights) else None,
                "tied_weight_max": float(weights.max()) if len(weights) else None,
                "target_columns": ";".join(targets),
                "identity_columns": ";".join(identities),
                "sha256": file_sha256(path),
            }
        )

    discovered = {row["source_id"] for row in rows}
    for path, (role, reason) in CANONICAL_SOURCE_ROLES.items():
        identifier = source_id(path)
        if identifier in discovered:
            continue
        frame = pd.read_csv(path)
        rows.append(
            {
                "source_id": identifier,
                "source_role": role,
                "reason": reason,
                "row_count": len(frame),
                "tied_row_count": None,
                "unique_tied_weights_tablewide": None,
                "tied_weight_min": None,
                "tied_weight_max": None,
                "target_columns": ";".join(column for column in frame.columns if column.endswith("/bpb")),
                "identity_columns": ";".join(column for column in SOURCE_ID_COLUMNS if column in frame.columns),
                "sha256": file_sha256(path),
            }
        )
    return pd.DataFrame(rows).sort_values(["source_role", "source_id"]).reset_index(drop=True)


def collapse_sources(values: Iterable[str]) -> str:
    return ";".join(sorted({value for value in values if value}))


def register_curve(curves: dict[str, dict[str, Any]], curve_id: str, **metadata: Any) -> None:
    row = {"curve_id": curve_id, **metadata}
    previous = curves.setdefault(curve_id, row)
    if previous != row:
        raise ValueError(f"Conflicting metadata for {curve_id}")


def add_membership(
    memberships: list[dict[str, Any]],
    *,
    curve_id: str,
    weight: float,
    training_run_id: str,
    observation_id: str,
    source_path: Path,
    source_row_id: str,
) -> None:
    memberships.append(
        {
            "curve_id": curve_id,
            "starcoder_weight": float(weight),
            "training_run_id": training_run_id,
            "observation_id": observation_id,
            "source_ids": source_id(source_path),
            "source_row_ids": source_row_id,
        }
    )


def add_target(
    observations: list[dict[str, Any]],
    *,
    observation_id: str,
    training_run_id: str,
    observation_step: str,
    target_id: str,
    value: float,
    source_path: Path,
) -> None:
    observations.append(
        {
            "observation_id": observation_id,
            "training_run_id": training_run_id,
            "observation_step": observation_step,
            "target_id": target_id,
            "target_label": target_label(target_id),
            "target_family": target_family(target_id),
            "bpb": float(value),
            "source_ids": source_id(source_path),
        }
    )


def fixed_model_metadata() -> tuple[dict[str, Any], dict[int, dict[str, int]]]:
    design = json.loads(TOKEN_DESIGN_PATH.read_text(encoding="utf-8"))
    observations = pd.read_csv(TOKEN_OBSERVATIONS_PATH)
    by_budget: dict[int, dict[str, int]] = {}
    for budget, block in observations.groupby("token_budget_requested"):
        by_budget[int(budget)] = {
            "materialized_tokens": round(float(block["materialized_tokens"].median())),
            "total_steps": round(float(block["total_steps"].median())),
            "boundary_step": round(float(block["boundary_step"].median())),
        }
    return design, by_budget


def load_fixed_model_curves(
    curves: dict[str, dict[str, Any]],
    memberships: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> None:
    design, by_budget = fixed_model_metadata()
    model = design["model"]
    invariants = design["invariants"]
    clean = pd.read_csv(FIXED_DIAGONAL_PATH)
    surface = pd.read_csv(TOKEN_SURFACE_PATH)
    surface = surface.loc[(surface["p0"] - surface["p1"]).abs() <= TIED_TOLERANCE]

    for budget in sorted(clean["token_budget_requested"].unique()):
        budget = int(budget)
        curve_id = f"fixed_model_wsd80_{budget // 1_000_000_000}b__endpoint"
        metadata = by_budget[budget]
        register_curve(
            curves,
            curve_id,
            family="fixed_model_token_ladder",
            state_role="endpoint",
            protocol_group="core_endpoint",
            model_label=str(model["architecture"]),
            hidden_size=768,
            total_parameters=int(model["total_trainable_parameters"]),
            non_embedding_parameters=int(model["non_embedding_parameters"]),
            planned_materialized_tokens=metadata["materialized_tokens"],
            total_steps=metadata["total_steps"],
            observation_step="endpoint",
            training_progress=1.0,
            phase_boundary_fraction=0.8,
            lr_decay_onset_fraction=0.8,
            support_id="historical_simulated_support",
            support_description="Epoch-aligned simulated StarCoder support",
            region=str(invariants["region"]),
            zone=str(invariants["zone"]),
            accelerator=str(invariants["tpu_type"]),
            seed_semantics="Reference trainer/data/support seed 20260711",
            notes="Union of the regular tied-diagonal export and irregular measured token-surface coordinates.",
        )

    for path, frame, weight_column, value_column, run_id_column in (
        (FIXED_DIAGONAL_PATH, clean, "weight", "starcoder_bpb", "wandb_id"),
        (TOKEN_SURFACE_PATH, surface, "p0", "bpb", "wandb_id"),
    ):
        for row in frame.itertuples(index=False):
            budget = int(row.token_budget_requested)
            curve_id = f"fixed_model_wsd80_{budget // 1_000_000_000}b__endpoint"
            wandb_id = str(getattr(row, run_id_column))
            training_run_id = f"wandb:{wandb_id}"
            observation_id = f"{training_run_id}@endpoint"
            add_membership(
                memberships,
                curve_id=curve_id,
                weight=float(getattr(row, weight_column)),
                training_run_id=training_run_id,
                observation_id=observation_id,
                source_path=path,
                source_row_id=wandb_id,
            )
            add_target(
                observations,
                observation_id=observation_id,
                training_run_id=training_run_id,
                observation_step="endpoint",
                target_id=PRIMARY_TARGET,
                value=float(getattr(row, value_column)),
                source_path=path,
            )


def load_historical_checkpoint_curves(
    curves: dict[str, dict[str, Any]],
    memberships: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> None:
    history = pd.read_csv(HISTORICAL_TRAJECTORY_PATH)
    history = history.loc[(history["phase_0_starcoder"] - history["phase_1_starcoder"]).abs() <= TIED_TOLERANCE]
    endpoint_memberships = {
        row["observation_id"] for row in memberships if row["curve_id"] == "fixed_model_wsd80_1b__endpoint"
    }
    total_steps = int(history["global_step"].max())
    for step, block in history.groupby("global_step", sort=True):
        step = int(step)
        is_endpoint = step == total_steps
        if not is_endpoint:
            curve_id = f"fixed_model_wsd80_1b__checkpoint_{step:04d}"
            register_curve(
                curves,
                curve_id,
                family="fixed_model_token_ladder",
                state_role="checkpoint",
                protocol_group="trajectory_supplement",
                model_label="Llama, 10 layers, d_model=768, d_ff=1536, 8 Q/KV heads",
                hidden_size=768,
                total_parameters=157_499_136,
                non_embedding_parameters=58_998_528,
                planned_materialized_tokens=round(1_000_000_000 * float(block["run_progress"].median())),
                total_steps=total_steps,
                observation_step=str(step),
                training_progress=float(block["run_progress"].median()),
                phase_boundary_fraction=0.8,
                lr_decay_onset_fraction=0.8,
                support_id="historical_simulated_support",
                support_description="Epoch-aligned simulated StarCoder support",
                region="us-central1",
                zone="us-central1-a",
                accelerator="v5p-8",
                seed_semantics="Same physical training runs as the fixed-model 1B endpoint curve",
                notes="Intermediate readout; group by training_run_id to prevent checkpoint leakage.",
            )
        for _, row in block.iterrows():
            training_run_id = f"wandb:{row['wandb_run_id']}"
            observation_id = f"{training_run_id}@{'endpoint' if is_endpoint else f'step:{step}'}"
            if is_endpoint:
                if observation_id not in endpoint_memberships:
                    raise ValueError(f"Historical endpoint {observation_id} is absent from the fixed-model 1B union")
            else:
                add_membership(
                    memberships,
                    curve_id=curve_id,
                    weight=float(row["phase_0_starcoder"]),
                    training_run_id=training_run_id,
                    observation_id=observation_id,
                    source_path=HISTORICAL_TRAJECTORY_PATH,
                    source_row_id=f"{row['wandb_run_id']}:{step}",
                )
            add_target(
                observations,
                observation_id=observation_id,
                training_run_id=training_run_id,
                observation_step="endpoint" if is_endpoint else str(step),
                target_id=PRIMARY_TARGET,
                value=float(row[PRIMARY_TARGET]),
                source_path=HISTORICAL_TRAJECTORY_PATH,
            )
            add_target(
                observations,
                observation_id=observation_id,
                training_run_id=training_run_id,
                observation_step="endpoint" if is_endpoint else str(step),
                target_id=UNCHEATABLE_TARGET,
                value=float(row[UNCHEATABLE_TARGET]),
                source_path=HISTORICAL_TRAJECTORY_PATH,
            )


def load_matched_nd_curves(
    curves: dict[str, dict[str, Any]],
    memberships: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> None:
    frame = pd.read_csv(MATCHED_ND_PATH)
    frame = frame.loc[(frame["phase_0_starcoder"] - frame["phase_1_starcoder"]).abs() <= TIED_TOLERANCE]
    for cell_id, block in frame.groupby("cell_id", sort=True):
        first = block.iloc[0]
        curve_id = f"matched_nd__{cell_id}__endpoint"
        register_curve(
            curves,
            curve_id,
            family="matched_nd",
            state_role="endpoint",
            protocol_group="core_endpoint",
            model_label=f"Llama h{int(first.hidden_size)} matched-N,D cell",
            hidden_size=int(first.hidden_size),
            total_parameters=int(first.total_parameters),
            non_embedding_parameters=int(first.non_embedding_parameters),
            planned_materialized_tokens=int(first.materialized_tokens),
            total_steps=int(first.total_steps),
            observation_step="endpoint",
            training_progress=1.0,
            phase_boundary_fraction=0.8,
            lr_decay_onset_fraction=0.8,
            support_id="matched_nd_reference_support",
            support_description="Cell-local simulated support under the matched-N,D design",
            region="us-central1",
            zone="us-central1-a",
            accelerator="v5p-8",
            seed_semantics="Shared reference trainer/data/support seed 20260711 within each cell",
            notes="Latest combined Stage 1-3 discovery union; earlier stage exports are nested subsets.",
        )
        for row in block.itertuples(index=False):
            training_run_id = f"wandb:{row.wandb_id}"
            observation_id = f"{training_run_id}@endpoint"
            add_membership(
                memberships,
                curve_id=curve_id,
                weight=float(row.phase_0_starcoder),
                training_run_id=training_run_id,
                observation_id=observation_id,
                source_path=MATCHED_ND_PATH,
                source_row_id=str(row.run_name),
            )
            add_target(
                observations,
                observation_id=observation_id,
                training_run_id=training_run_id,
                observation_step="endpoint",
                target_id=PRIMARY_TARGET,
                value=float(row.starcoder_bpb),
                source_path=MATCHED_ND_PATH,
            )


def canonical_dense_run(row: Any) -> str:
    if bool(row.is_alias):
        return str(row.alias_of_run_name)
    return str(row.run_name)


def load_dense_replay_curves(
    curves: dict[str, dict[str, Any]],
    memberships: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> None:
    frame = pd.read_csv(DENSE_REPLAY_PATH)
    frame = frame.loc[(frame["phase_0_starcoder"] - frame["phase_1_starcoder"]).abs() <= TIED_TOLERANCE]
    checkpoint_metrics = pd.read_csv(DENSE_ATOMIC_CHECKPOINT_PATH).set_index("run_name")

    for (cell_id, support_id), block in frame.groupby(["cell_id", "support_id"], sort=True):
        first = block.iloc[0]
        physical_runs = [canonical_dense_run(row) for row in block.itertuples(index=False)]
        checkpoint_steps = checkpoint_metrics.loc[physical_runs, "phase0_step"].astype(int).unique()
        if len(checkpoint_steps) != 1:
            raise ValueError(f"{cell_id}/{support_id} has inconsistent checkpoint steps: {checkpoint_steps}")
        checkpoint_step = int(checkpoint_steps[0])
        common = {
            "family": "dense_horizon_replay",
            "model_label": f"Llama h{int(first.hidden_size)} fixed-N horizon cell",
            "hidden_size": int(first.hidden_size),
            "total_parameters": int(first.total_parameters),
            "non_embedding_parameters": int(first.non_embedding_parameters),
            "total_steps": int(first.total_steps),
            "phase_boundary_fraction": float(first.boundary_step / first.total_steps),
            "lr_decay_onset_fraction": float(first.boundary_step / first.total_steps),
            "support_id": str(support_id),
            "support_description": str(first.support_role),
            "region": "us-central1",
            "zone": "us-central1-a",
            "accelerator": "v5p-8",
            "seed_semantics": "Reference trainer/data seed 20260711; support changes define separate curves",
        }
        endpoint_curve_id = f"dense_replay__{cell_id}__{support_id}__endpoint"
        register_curve(
            curves,
            endpoint_curve_id,
            **common,
            state_role="endpoint",
            protocol_group="core_endpoint",
            planned_materialized_tokens=int(first.materialized_tokens),
            observation_step="endpoint",
            training_progress=1.0,
            notes="Measured endpoint curve; inverse-variance calibration weights are not part of this registry.",
        )
        checkpoint_curve_id = f"dense_replay__{cell_id}__{support_id}__checkpoint_{checkpoint_step}"
        register_curve(
            curves,
            checkpoint_curve_id,
            **common,
            state_role="checkpoint",
            protocol_group="trajectory_supplement",
            planned_materialized_tokens=round(first.materialized_tokens * checkpoint_step / first.total_steps),
            observation_step=str(checkpoint_step),
            training_progress=float(checkpoint_step / first.total_steps),
            notes="Closest scheduled atomic evaluation before the 0.80T boundary; group by training_run_id.",
        )

        for row in block.itertuples(index=False):
            run_name = canonical_dense_run(row)
            training_run_id = f"run:{run_name}"
            endpoint_observation_id = f"{training_run_id}@endpoint"
            checkpoint_observation_id = f"{training_run_id}@step:{checkpoint_step}"
            source_row = str(row.run_name)
            add_membership(
                memberships,
                curve_id=endpoint_curve_id,
                weight=float(row.phase_0_starcoder),
                training_run_id=training_run_id,
                observation_id=endpoint_observation_id,
                source_path=DENSE_REPLAY_PATH,
                source_row_id=source_row,
            )
            add_membership(
                memberships,
                curve_id=checkpoint_curve_id,
                weight=float(row.phase_0_starcoder),
                training_run_id=training_run_id,
                observation_id=checkpoint_observation_id,
                source_path=DENSE_REPLAY_PATH,
                source_row_id=source_row,
            )
            add_target(
                observations,
                observation_id=endpoint_observation_id,
                training_run_id=training_run_id,
                observation_step="endpoint",
                target_id=PRIMARY_TARGET,
                value=float(row.bpb),
                source_path=DENSE_REPLAY_PATH,
            )

    referenced_runs = {
        membership["training_run_id"].removeprefix("run:")
        for membership in memberships
        if membership["curve_id"].startswith("dense_replay__")
    }
    endpoint_metrics = pd.read_csv(DENSE_ATOMIC_ENDPOINT_PATH)
    endpoint_metrics = endpoint_metrics.loc[endpoint_metrics["run_name"].isin(referenced_runs)]
    checkpoint_frame = pd.read_csv(DENSE_ATOMIC_CHECKPOINT_PATH)
    checkpoint_frame = checkpoint_frame.loc[checkpoint_frame["run_name"].isin(referenced_runs)]
    if set(endpoint_metrics["run_name"]) != referenced_runs or set(checkpoint_frame["run_name"]) != referenced_runs:
        raise ValueError("Dense tied memberships do not have complete endpoint and checkpoint atomic payloads")

    for _, row in endpoint_metrics.iterrows():
        training_run_id = f"run:{row['run_name']}"
        observation_id = f"{training_run_id}@endpoint"
        for target_id, _label in ATOMIC_METRICS:
            add_target(
                observations,
                observation_id=observation_id,
                training_run_id=training_run_id,
                observation_step="endpoint",
                target_id=target_id,
                value=float(row[target_id]),
                source_path=DENSE_ATOMIC_ENDPOINT_PATH,
            )

    for _, row in checkpoint_frame.iterrows():
        step = int(row["phase0_step"])
        training_run_id = f"run:{row['run_name']}"
        observation_id = f"{training_run_id}@step:{step}"
        for target_id, _label in ATOMIC_METRICS:
            add_target(
                observations,
                observation_id=observation_id,
                training_run_id=training_run_id,
                observation_step=str(step),
                target_id=target_id,
                value=float(row[target_id]),
                source_path=DENSE_ATOMIC_CHECKPOINT_PATH,
            )

    aggregate = pd.read_csv(DENSE_AGGREGATE_ENDPOINT_PATH)
    aggregate = aggregate.loc[aggregate["run_name"].isin(referenced_runs)]
    for row in aggregate.itertuples(index=False):
        training_run_id = f"run:{row.run_name}"
        observation_id = f"{training_run_id}@endpoint"
        for column, target_id in AGGREGATE_COLUMNS.items():
            add_target(
                observations,
                observation_id=observation_id,
                training_run_id=training_run_id,
                observation_step="endpoint",
                target_id=target_id,
                value=float(getattr(row, column)),
                source_path=DENSE_AGGREGATE_ENDPOINT_PATH,
            )


def load_coupled_onset_curves(
    curves: dict[str, dict[str, Any]],
    memberships: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> None:
    frame = pd.read_csv(COUPLED_ONSET_PATH)
    frame = frame.loc[(frame["phase_0_starcoder"] - frame["phase_1_starcoder"]).abs() <= TIED_TOLERANCE]
    for onset, block in frame.groupby("requested_onset_fraction", sort=True):
        first = block.iloc[0]
        slug = f"{float(onset):.2f}".replace(".", "p")
        curve_id = f"coupled_onset__{slug}__endpoint"
        register_curve(
            curves,
            curve_id,
            family="coupled_lr_onset",
            state_role="endpoint",
            protocol_group="core_endpoint",
            model_label=f"Llama h{int(first.hidden_size)} fixed-N 8B cell",
            hidden_size=int(first.hidden_size),
            total_parameters=int(first.total_parameters),
            non_embedding_parameters=int(first.non_embedding_parameters),
            planned_materialized_tokens=int(first.materialized_tokens),
            total_steps=int(first.total_steps),
            observation_step="endpoint",
            training_progress=1.0,
            phase_boundary_fraction=float(first.realized_onset_fraction),
            lr_decay_onset_fraction=float(first.decay_onset_fraction),
            support_id=str(first.support_id),
            support_description=str(first.support_role),
            region="us-central2",
            zone="us-central2-b",
            accelerator="v4-8",
            seed_semantics="Trainer/data seed 20260711; newly trained independently for each onset arm",
            notes="Phase boundary and cosine-decay onset are coupled; tied data make the phase boundary inert.",
        )
        for row in block.itertuples(index=False):
            training_run_id = f"run:{row.run_name}"
            observation_id = f"{training_run_id}@endpoint"
            add_membership(
                memberships,
                curve_id=curve_id,
                weight=float(row.phase_0_starcoder),
                training_run_id=training_run_id,
                observation_id=observation_id,
                source_path=COUPLED_ONSET_PATH,
                source_row_id=str(row.row_id),
            )
            for column, target_id in COUPLED_ONSET_COLUMNS.items():
                add_target(
                    observations,
                    observation_id=observation_id,
                    training_run_id=training_run_id,
                    observation_step="endpoint",
                    target_id=target_id,
                    value=float(getattr(row, column)),
                    source_path=COUPLED_ONSET_PATH,
                )


def collapse_memberships(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    keys = ["curve_id", "starcoder_weight", "training_run_id", "observation_id"]
    collapsed = (
        frame.groupby(keys, as_index=False, sort=True)
        .agg(source_ids=("source_ids", collapse_sources), source_row_ids=("source_row_ids", collapse_sources))
        .sort_values(["curve_id", "starcoder_weight", "observation_id"])
        .reset_index(drop=True)
    )
    observation_curve_counts = collapsed.groupby("observation_id")["curve_id"].nunique()
    collapsed["shared_across_curves"] = collapsed["observation_id"].map(observation_curve_counts).gt(1)
    return collapsed


def collapse_target_observations(rows: list[dict[str, Any]], membership_ids: set[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame = frame.loc[frame["observation_id"].isin(membership_ids)].copy()
    keys = [
        "observation_id",
        "training_run_id",
        "observation_step",
        "target_id",
        "target_label",
        "target_family",
    ]
    consistency = frame.groupby(keys)["bpb"].agg(["min", "max"])
    conflicts = consistency.loc[(consistency["max"] - consistency["min"]).abs() > VALUE_TOLERANCE]
    if not conflicts.empty:
        raise ValueError(f"Conflicting duplicate target observations:\n{conflicts.head(20)}")
    return (
        frame.groupby(keys, as_index=False, sort=True)
        .agg(bpb=("bpb", "first"), source_ids=("source_ids", collapse_sources))
        .sort_values(["observation_id", "target_id"])
        .reset_index(drop=True)
    )


def build_target_coverage(memberships: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    joined = memberships.merge(observations, on=["observation_id", "training_run_id"], validate="many_to_many")
    rows: list[dict[str, Any]] = []
    for (curve_id, target_id), block in joined.groupby(["curve_id", "target_id"], sort=True):
        weight_means = block.groupby("starcoder_weight", as_index=False)["bpb"].mean()
        best = weight_means.loc[weight_means["bpb"].idxmin()]
        unique_weights = int(block["starcoder_weight"].nunique())
        weight_min = float(block["starcoder_weight"].min())
        weight_max = float(block["starcoder_weight"].max())
        weight_span = weight_max - weight_min
        protocol_ready = unique_weights >= MIN_UNIQUE_WEIGHTS and weight_span >= MIN_WEIGHT_SPAN
        rows.append(
            {
                "curve_id": curve_id,
                "target_id": target_id,
                "target_label": str(block["target_label"].iloc[0]),
                "target_family": str(block["target_family"].iloc[0]),
                "observation_count": len(block),
                "unique_weights": unique_weights,
                "weight_min": weight_min,
                "weight_max": weight_max,
                "weight_span": weight_span,
                "observed_argmin_weight": float(best["starcoder_weight"]),
                "observed_min_bpb": float(best["bpb"]),
                "protocol_ready": protocol_ready,
                "readiness_reason": (
                    "passes density and span floors"
                    if protocol_ready
                    else f"needs >= {MIN_UNIQUE_WEIGHTS} weights and span >= {MIN_WEIGHT_SPAN:.1f}"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_curve_inventory(
    curves: dict[str, dict[str, Any]],
    memberships: pd.DataFrame,
    target_coverage: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for curve_id, metadata in sorted(curves.items()):
        block = memberships.loc[memberships["curve_id"].eq(curve_id)]
        targets = target_coverage.loc[target_coverage["curve_id"].eq(curve_id)]
        primary = targets.loc[targets["target_id"].eq(PRIMARY_TARGET)]
        if len(primary) != 1:
            raise ValueError(f"{curve_id} has {len(primary)} primary target rows")
        rows.append(
            {
                **metadata,
                "curve_point_memberships": len(block),
                "unique_state_observations": int(block["observation_id"].nunique()),
                "unique_training_runs": int(block["training_run_id"].nunique()),
                "unique_weights": int(block["starcoder_weight"].nunique()),
                "weight_min": float(block["starcoder_weight"].min()),
                "weight_max": float(block["starcoder_weight"].max()),
                "weight_span": float(block["starcoder_weight"].max() - block["starcoder_weight"].min()),
                "shared_observation_memberships": int(block["shared_across_curves"].sum()),
                "target_count": len(targets),
                "protocol_ready_target_count": int(targets["protocol_ready"].sum()),
                "primary_target_points": int(primary["unique_weights"].iloc[0]),
                "primary_observed_argmin_weight": float(primary["observed_argmin_weight"].iloc[0]),
                "primary_target_ready": bool(primary["protocol_ready"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["protocol_group", "family", "curve_id"]).reset_index(drop=True)


def validate_registry(
    curves: pd.DataFrame,
    memberships: pd.DataFrame,
    observations: pd.DataFrame,
    target_coverage: pd.DataFrame,
    sources: pd.DataFrame,
) -> dict[str, Any]:
    if len(curves) != 76:
        raise ValueError(f"Expected 76 deduplicated physical curves, found {len(curves)}")
    core = curves.loc[curves["protocol_group"].eq("core_endpoint")]
    trajectory = curves.loc[curves["protocol_group"].eq("trajectory_supplement")]
    if len(core) != 45 or len(trajectory) != 31:
        raise ValueError(f"Expected 45 core and 31 trajectory curves, found {len(core)} and {len(trajectory)}")
    if not core["primary_target_ready"].all() or not trajectory["primary_target_ready"].all():
        raise ValueError("Every retained curve must have a protocol-ready Programming Languages target")
    if not memberships.groupby("curve_id")["starcoder_weight"].nunique().ge(MIN_UNIQUE_WEIGHTS).all():
        raise ValueError("A retained curve falls below the unique-weight floor")
    spans = memberships.groupby("curve_id")["starcoder_weight"].agg(lambda values: values.max() - values.min())
    if not spans.ge(MIN_WEIGHT_SPAN - TIED_TOLERANCE).all():
        raise ValueError("A retained curve falls below the weight-span floor")
    if observations.duplicated(["observation_id", "target_id"]).any():
        raise ValueError("Target observations were not deduplicated")
    if memberships.duplicated(["curve_id", "starcoder_weight", "observation_id"]).any():
        raise ValueError("Curve memberships were not deduplicated")
    family_counts = core.groupby("family")["curve_id"].nunique().to_dict()
    expected_family_counts = {
        "coupled_lr_onset": 3,
        "dense_horizon_replay": 28,
        "fixed_model_token_ladder": 4,
        "matched_nd": 10,
    }
    if family_counts != expected_family_counts:
        raise ValueError(f"Unexpected core family counts: {family_counts}")
    protocol_ready_curve_targets = int(target_coverage["protocol_ready"].sum())
    if protocol_ready_curve_targets != 1332:
        raise ValueError(f"Expected 1,332 protocol-ready curve-target pairs, found {protocol_ready_curve_targets}")
    unresolved_sources = sources.loc[sources["source_role"].eq("not_selected"), "source_id"]
    if not unresolved_sources.empty:
        unresolved = unresolved_sources.to_string(index=False)
        raise ValueError(f"Dense measured candidate sources remain unresolved:\n{unresolved}")
    return {
        "physical_curve_count": len(curves),
        "core_endpoint_curve_count": len(core),
        "trajectory_supplement_curve_count": len(trajectory),
        "unique_training_run_count": int(memberships["training_run_id"].nunique()),
        "unique_state_observation_count": int(memberships["observation_id"].nunique()),
        "curve_point_membership_count": len(memberships),
        "target_observation_count": len(observations),
        "curve_target_count": len(target_coverage),
        "protocol_ready_curve_target_count": protocol_ready_curve_targets,
        "core_family_counts": family_counts,
    }


def write_report(
    output_dir: Path,
    curves: pd.DataFrame,
    sources: pd.DataFrame,
    target_coverage: pd.DataFrame,
    summary: dict[str, Any],
) -> Path:
    core = curves.loc[curves["protocol_group"].eq("core_endpoint")]
    family = (
        core.groupby("family", as_index=False)
        .agg(
            curves=("curve_id", "nunique"),
            min_weights=("unique_weights", "min"),
            max_weights=("unique_weights", "max"),
            target_curves=("protocol_ready_target_count", "sum"),
            regions=("region", collapse_sources),
            accelerators=("accelerator", collapse_sources),
        )
        .sort_values("family")
    )
    source_roles = sources.groupby("source_role").size().rename("tables").reset_index()
    core_target_families = (
        target_coverage.merge(core[["curve_id"]], on="curve_id", validate="many_to_one")
        .loc[lambda frame: frame["protocol_ready"]]
        .groupby("target_family")
        .size()
        .rename("curve_targets")
        .reset_index()
        .sort_values("target_family")
    )
    lines = [
        "# StarCoder one-dimensional single-phase curve inventory",
        "",
        "A physical curve holds model, data source, support, trainer, hardware, learning-rate schedule, and "
        "training state fixed while setting `phase_0_starcoder = phase_1_starcoder = p`. Only the StarCoder "
        "mixture weight `p` varies, so the data policy has one degree of freedom. Targets are catalogued "
        "separately because one physical curve can have several measured BPB readouts.",
        "",
        "## Result",
        "",
        f"- **{summary['core_endpoint_curve_count']} dense endpoint curves** are ready for the core shape gate. "
        "Every one has at least 13 distinct weights, spans at least 0.8 of the simplex edge, and has measured "
        "Programming Languages BPB.",
        f"- **{summary['trajectory_supplement_curve_count']} checkpoint curves** are also usable, but reuse "
        "the endpoint training runs. They belong in a trajectory supplement with splits grouped by physical "
        "training run, not as independent endpoint evidence.",
        f"- The normalized registry has **{summary['physical_curve_count']} physical curves**, "
        f"**{summary['unique_training_run_count']} unique training runs**, "
        f"**{summary['curve_point_membership_count']} curve-point memberships**, and "
        f"**{summary['protocol_ready_curve_target_count']} usable curve-target pairs** after run-level "
        "deduplication.",
        "- None of these StarCoder runs has the paper's 51-component Table 9 payload. They are sanity and "
        "mechanism panels, not substitutes for the mandatory Delphi Table 9 benchmark.",
        "",
        "## Core endpoint families",
        "",
        family.to_markdown(index=False),
        "",
        "The 28 replay curves are a crossed panel: four fixed model-size/training-horizon cells by seven "
        "StarCoder support sizes. The three coupled-onset curves use new Central2 v4-8 trainings and remain "
        "separate from the Central1 v5p-8 replay curve even at the nominal 0.80T onset.",
        "",
        "## Deduplication",
        "",
        "- The regular 4x21 tied-diagonal export and the fixed-model token-surface export share 78 W&B runs. "
        "They are unioned into four curves with 26, 24, 24, and 24 distinct measured weights.",
        "- Matched-N,D Stage 1 and Stage 2 tables are nested in the latest Stage 3 combined export; only the "
        "latest union contributes memberships.",
        "- The dense-support design snapshot and calibrated observation table carry the same geometry; only "
        "the calibrated table is used, and its fitting weights are ignored.",
        "- Dense-support alias rows resolve to the physical `alias_of_run_name`. A physical state may therefore "
        "belong to several support curves at a coordinate where support cannot change the sampled data, but "
        "it appears only once in the target-observation registry.",
        "- Copied collaborator packets, old surface snapshots, selection summaries, predicted surfaces, and "
        "off-diagonal fibers do not create additional single-phase curves.",
        "",
        "## Target coverage",
        "",
        core_target_families.to_markdown(index=False),
        "",
        "All 45 core curves support Programming Languages BPB. The dense replay panel additionally has 23 "
        "atomic BPBs; its four full-support endpoint curves also have exact Paloma and Uncheatable macro BPBs. "
        "The coupled-onset curves have Programming Languages, C4, Uncheatable macro, GitHub C++, and GitHub "
        "Python BPB. The historical 1B endpoint has Uncheatable macro BPB at 20 of its 26 mixture weights.",
        "",
        "## Protocol recommendation",
        "",
        "1. Add the **45 endpoint Programming Languages curves** as the mandatory StarCoder shape gate. Report "
        "metrics per curve, then macro-average within each of the four families and macro-average the family "
        "scores so the 28 replay curves do not dominate by count.",
        "2. Keep the **31 checkpoint curves** as a trajectory supplement. Any split spanning checkpoints must "
        "group on `training_run_id`; otherwise the same trained model leaks between train and test.",
        "3. Keep the full target expansion outside the five-minute Certify tier. Use it for diagnostic ablations "
        "after a model passes the 45-curve primary gate.",
        "4. Treat a curve as usable only when it has at least 13 unique tied weights and weight span at least "
        "0.8. Historical cosine/WSD50 schedules, batch/repetition controls, and confirmations remain in the "
        "source ledger but fail this density rule.",
        "",
        "## Source audit",
        "",
        source_roles.to_markdown(index=False),
        "",
        "The machine-readable files distinguish physical runs, state observations, curve memberships, target "
        "coverage, and rejected source tables. This prevents copied reports or shared replay aliases from "
        "silently increasing benchmark weight.",
        "",
    ]
    path = output_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    curves: dict[str, dict[str, Any]] = {}
    membership_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    load_fixed_model_curves(curves, membership_rows, observation_rows)
    load_historical_checkpoint_curves(curves, membership_rows, observation_rows)
    load_matched_nd_curves(curves, membership_rows, observation_rows)
    load_dense_replay_curves(curves, membership_rows, observation_rows)
    load_coupled_onset_curves(curves, membership_rows, observation_rows)

    memberships = collapse_memberships(membership_rows)
    observations = collapse_target_observations(observation_rows, set(memberships["observation_id"]))
    target_coverage = build_target_coverage(memberships, observations)
    curve_inventory = build_curve_inventory(curves, memberships, target_coverage)
    candidate_sources = discover_candidate_sources()
    summary = validate_registry(curve_inventory, memberships, observations, target_coverage, candidate_sources)

    output_paths = {
        "curve_inventory": args.output_dir / "curve_inventory.csv",
        "curve_memberships": args.output_dir / "curve_memberships.csv",
        "target_observations": args.output_dir / "target_observations.csv",
        "curve_target_coverage": args.output_dir / "curve_target_coverage.csv",
        "candidate_sources": args.output_dir / "candidate_sources.csv",
    }
    curve_inventory.to_csv(output_paths["curve_inventory"], index=False)
    memberships.to_csv(output_paths["curve_memberships"], index=False)
    observations.to_csv(output_paths["target_observations"], index=False)
    target_coverage.to_csv(output_paths["curve_target_coverage"], index=False)
    candidate_sources.to_csv(output_paths["candidate_sources"], index=False)
    report_path = write_report(args.output_dir, curve_inventory, candidate_sources, target_coverage, summary)

    summary["protocol_thresholds"] = {
        "minimum_unique_weights": MIN_UNIQUE_WEIGHTS,
        "minimum_weight_span": MIN_WEIGHT_SPAN,
    }
    summary["input_hashes"] = {
        source_id(path): file_sha256(path) for path in sorted(CANONICAL_SOURCE_ROLES, key=source_id)
    }
    summary["output_hashes"] = {path.name: file_sha256(path) for path in [*output_paths.values(), report_path]}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
