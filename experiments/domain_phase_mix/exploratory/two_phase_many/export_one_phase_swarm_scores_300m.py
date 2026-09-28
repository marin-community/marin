# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
# ]
# ///
"""Export one-phase qsplit240 300M swarm weights and eval scores.

This materializes a collaborator-facing CSV with one row per one-phase swarm
candidate, its bucket weights, uncheatable eval scores, and OLMoBaseEval Table-9
BPB scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
REFERENCE_OUTPUTS = Path(__file__).resolve().parent / "reference_outputs"

DEFAULT_MANIFEST = (
    REFERENCE_OUTPUTS
    / "single_phase_exposure_average_qsplit240_300m_6b"
    / "single_phase_exposure_average_qsplit240_300m_manifest.csv"
)
DEFAULT_UNCHEATABLE = (
    REFERENCE_OUTPUTS / "one_vs_two_phase_swarm_debug_20260630" / "single_phase_qsplit240_wandb_eval_scalars.csv"
)
DEFAULT_TABLE9 = (
    REFERENCE_OUTPUTS / "olmo_base_easy_one_phase_parity_panel_300m_20260628" / "single_phase_table9_wide.csv"
)
DEFAULT_AUGMENTED_PANEL = (
    REFERENCE_OUTPUTS / "olmo_base_easy_one_phase_parity_panel_300m_20260628" / "one_phase_augmented_fit_panel.csv"
)
DEFAULT_PROPORTIONAL_REFERENCE_TABLE9 = (
    REFERENCE_OUTPUTS / "olmo_base_easy_one_phase_parity_panel_300m_20260628" / "proportional_reference_table9.csv"
)
DEFAULT_QSPLIT_TRAINING_EVAL = (
    REFERENCE_OUTPUTS
    / "raw_metric_matrix_300m_training_eval_wandb_collect_20260623"
    / "pctrl_final_metric_matrix_with_training_eval.csv"
)
DEFAULT_PCTRL_TRAINING_EVAL = (
    REFERENCE_OUTPUTS / "pctrl_training_eval_wandb_collect_20260623" / "pctrl_final_metric_matrix_with_training_eval.csv"
)
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "one_phase_swarm_scores_export_300m_20260630"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "one_phase_swarm_uncheatable_table9_scores_300m.csv"
DEFAULT_AUGMENTED_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "one_phase_augmented_fit_panel_uncheatable_table9_scores_300m.csv"
DEFAULT_PROPORTIONAL_REFERENCE_OUTPUT_CSV = (
    DEFAULT_OUTPUT_DIR / "proportional_reference_uncheatable_table9_scores_300m.csv"
)
DEFAULT_SUMMARY_JSON = DEFAULT_OUTPUT_DIR / "summary.json"
DEFAULT_README = DEFAULT_OUTPUT_DIR / "README.md"
EXPECTED_ROWS = 240
EXPECTED_AUGMENTED_ROWS = 280
EXPECTED_DOMAIN_DELETION_ROWS = 39
EXPECTED_SHARED_STRATIFIED_ROWS = 1
EXPECTED_PROPORTIONAL_REFERENCE_ROWS = 11
PHASE_SUM_TOL = 1e-9
PHASE_TIE_TOL = 1e-12
UNCHEATABLE_COLUMNS = [
    "eval_uncheatable_eval_bpb",
    "eval_uncheatable_eval_macro_bpb",
    "eval_bpb",
    "eval_macro_bpb",
]
TRAINING_EVAL_COLUMN_RENAMES = {
    "eval/uncheatable_eval/bpb": "eval_uncheatable_eval_bpb",
    "eval/uncheatable_eval/macro_bpb": "eval_uncheatable_eval_macro_bpb",
    "eval/bpb": "eval_bpb",
    "eval/macro_bpb": "eval_macro_bpb",
}


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} input not found: {path}")
    return pd.read_csv(path)


def _assert_unique_run_names(df: pd.DataFrame, label: str) -> None:
    if "run_name" not in df.columns:
        raise ValueError(f"{label} is missing run_name")
    duplicated = df["run_name"][df["run_name"].duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"{label} has duplicate run_name values: {duplicated[:10]}")


def _weight_columns(manifest: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    phase_0 = sorted(col for col in manifest.columns if col.startswith("phase_0_"))
    phase_1 = sorted(col for col in manifest.columns if col.startswith("phase_1_"))
    if not phase_0 or not phase_1:
        raise ValueError("manifest must contain phase_0_* and phase_1_* weight columns")
    phase_0_domains = [col.removeprefix("phase_0_") for col in phase_0]
    phase_1_domains = [col.removeprefix("phase_1_") for col in phase_1]
    if phase_0_domains != phase_1_domains:
        raise ValueError("phase_0 and phase_1 domains differ")
    return phase_0, phase_1, phase_0_domains


def _validate_tied_one_phase(manifest: pd.DataFrame, phase_0: list[str], phase_1: list[str]) -> dict[str, float]:
    phase_0_values = manifest[phase_0].to_numpy(dtype=float)
    phase_1_values = manifest[phase_1].to_numpy(dtype=float)
    phase_0_sums = phase_0_values.sum(axis=1)
    phase_1_sums = phase_1_values.sum(axis=1)
    phase_delta = np.abs(phase_0_values - phase_1_values)

    max_phase_0_sum_error = float(np.max(np.abs(phase_0_sums - 1.0)))
    max_phase_1_sum_error = float(np.max(np.abs(phase_1_sums - 1.0)))
    max_phase_delta = float(np.max(phase_delta))
    min_weight = float(min(phase_0_values.min(), phase_1_values.min()))

    if max_phase_0_sum_error > PHASE_SUM_TOL:
        raise ValueError(f"phase 0 weights do not sum to 1; max error {max_phase_0_sum_error}")
    if max_phase_1_sum_error > PHASE_SUM_TOL:
        raise ValueError(f"phase 1 weights do not sum to 1; max error {max_phase_1_sum_error}")
    if max_phase_delta > PHASE_TIE_TOL:
        raise ValueError(f"one-phase manifest has untied phases; max abs delta {max_phase_delta}")
    if min_weight < -PHASE_SUM_TOL:
        raise ValueError(f"manifest contains negative weights; min {min_weight}")

    return {
        "max_phase_0_sum_error": max_phase_0_sum_error,
        "max_phase_1_sum_error": max_phase_1_sum_error,
        "max_phase_delta": max_phase_delta,
        "min_weight": min_weight,
    }


def _table9_component_columns(table9: pd.DataFrame) -> list[str]:
    metadata_cols = {
        "run_name",
        "eval_source_run_name",
        "eval_source_run_id",
        "eval_target_name",
        "wandb_run_id",
        "wandb_url",
        "wandb_state",
        "wandb_created_at",
        "wandb_updated_at",
        "native_table9_macro_bpb",
        "table9_macro_bpb",
    }
    component_cols = [col for col in table9.columns if col not in metadata_cols]
    if len(component_cols) != 51:
        raise ValueError(f"expected 51 Table-9 component columns, found {len(component_cols)}")
    return component_cols


def _rename_training_eval_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=TRAINING_EVAL_COLUMN_RENAMES)


def _proportional_uncheatable_reference(qsplit_training_eval_path: Path) -> pd.DataFrame:
    qsplit = _read_csv(qsplit_training_eval_path, "qsplit training/eval")
    required = {"run_name", "row_kind", *TRAINING_EVAL_COLUMN_RENAMES}
    missing = required.difference(qsplit.columns)
    if missing:
        raise ValueError(f"qsplit training/eval is missing columns: {sorted(missing)}")
    reference = qsplit[
        qsplit["run_name"].eq("baseline_proportional") | qsplit["row_kind"].eq("noise_variable_subset_proportional")
    ].copy()
    if len(reference) != EXPECTED_PROPORTIONAL_REFERENCE_ROWS:
        raise ValueError(
            f"expected {EXPECTED_PROPORTIONAL_REFERENCE_ROWS} proportional uncheatable rows, found {len(reference)}"
        )
    keep_cols = [
        col
        for col in [
            "run_name",
            "row_kind",
            "noise_trainer_seed",
            "noise_data_seed",
            "noise_simulated_epoch_subset_seed",
            "wandb_run_id",
            *TRAINING_EVAL_COLUMN_RENAMES,
        ]
        if col in reference.columns
    ]
    return _rename_training_eval_columns(reference[keep_cols])


def build_proportional_reference_export(
    proportional_reference_table9_path: Path,
    qsplit_training_eval_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    table9_reference = _read_csv(proportional_reference_table9_path, "proportional Table-9 reference")
    uncheatable_reference = _proportional_uncheatable_reference(qsplit_training_eval_path)
    if len(table9_reference) != EXPECTED_PROPORTIONAL_REFERENCE_ROWS:
        raise ValueError(
            f"expected {EXPECTED_PROPORTIONAL_REFERENCE_ROWS} proportional Table-9 rows, found {len(table9_reference)}"
        )
    _assert_unique_run_names(table9_reference, "proportional Table-9 reference")
    _assert_unique_run_names(uncheatable_reference, "proportional uncheatable reference")

    merged = table9_reference.merge(uncheatable_reference, on="run_name", how="left", validate="one_to_one")
    missing_uncheatable = merged[UNCHEATABLE_COLUMNS].isna().sum()
    missing_uncheatable = missing_uncheatable[missing_uncheatable > 0]
    if not missing_uncheatable.empty:
        raise ValueError(f"missing proportional uncheatable reference scores: {missing_uncheatable.to_dict()}")

    summary = {
        "row_count": len(merged),
        "table9_macro_mean": float(merged["table9_macro_bpb"].mean()),
        "table9_macro_std": float(merged["table9_macro_bpb"].std(ddof=1)),
        "uncheatable_means": {col: float(merged[col].mean()) for col in UNCHEATABLE_COLUMNS},
        "uncheatable_stds": {col: float(merged[col].std(ddof=1)) for col in UNCHEATABLE_COLUMNS},
    }
    return merged, summary


def _pctrl_domain_deletion_uncheatable(pctrl_training_eval_path: Path) -> pd.DataFrame:
    pctrl = _read_csv(pctrl_training_eval_path, "pctrl training/eval")
    required = {"run_name", "intervention_type", *TRAINING_EVAL_COLUMN_RENAMES}
    missing = required.difference(pctrl.columns)
    if missing:
        raise ValueError(f"pctrl training/eval is missing columns: {sorted(missing)}")
    deletion = pctrl[pctrl["intervention_type"].eq("domain_deletion")].copy()
    if len(deletion) != EXPECTED_DOMAIN_DELETION_ROWS:
        raise ValueError(f"expected {EXPECTED_DOMAIN_DELETION_ROWS} pctrl deletion rows, found {len(deletion)}")
    keep_cols = [
        col
        for col in [
            "run_name",
            "intervention_type",
            "target_domain",
            "wandb_training_run_id",
            "wandb_training_run_url",
            *TRAINING_EVAL_COLUMN_RENAMES,
        ]
        if col in deletion.columns
    ]
    return _rename_training_eval_columns(deletion[keep_cols])


def _shared_stratified_uncheatable(qsplit_training_eval_path: Path) -> pd.DataFrame:
    qsplit = _read_csv(qsplit_training_eval_path, "qsplit training/eval")
    required = {"run_name", *TRAINING_EVAL_COLUMN_RENAMES}
    missing = required.difference(qsplit.columns)
    if missing:
        raise ValueError(f"qsplit training/eval is missing columns: {sorted(missing)}")
    shared = qsplit.loc[qsplit["run_name"].eq("baseline_stratified")].copy()
    if len(shared) != EXPECTED_SHARED_STRATIFIED_ROWS:
        raise ValueError(f"expected {EXPECTED_SHARED_STRATIFIED_ROWS} shared stratified row, found {len(shared)}")
    keep_cols = [
        col
        for col in [
            "run_name",
            "wandb_run_id",
            "wandb_training_run_url",
            *TRAINING_EVAL_COLUMN_RENAMES,
        ]
        if col in shared.columns
    ]
    shared = _rename_training_eval_columns(shared[keep_cols])
    shared["run_name"] = "singleavg_baseline_stratified"
    shared = shared.rename(
        columns={
            "wandb_run_id": "training_wandb_id",
            "wandb_training_run_url": "training_wandb_url",
        }
    )
    shared["training_wandb_name"] = "baseline_stratified"
    shared["training_wandb_state"] = "finished"
    shared["training_wandb_created_at"] = pd.NA
    shared["uncheatable_source"] = "shared_phase_tied_stratified_checkpoint"
    return shared


def build_augmented_export(
    augmented_panel_path: Path,
    pure_export: pd.DataFrame,
    component_cols: list[str],
    proportional_reference: pd.DataFrame,
    qsplit_training_eval_path: Path,
    pctrl_training_eval_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    panel = _read_csv(augmented_panel_path, "one-phase augmented panel")
    if len(panel) != EXPECTED_AUGMENTED_ROWS:
        raise ValueError(f"augmented panel has {len(panel)} rows, expected {EXPECTED_AUGMENTED_ROWS}")
    _assert_unique_run_names(panel, "one-phase augmented panel")

    phase_0, phase_1, domains = _weight_columns(panel)
    phase_stats = _validate_tied_one_phase(panel, phase_0, phase_1)
    weight_df = panel[["run_name", *phase_0]].rename(
        columns={col: f"weight_{col.removeprefix('phase_0_')}" for col in phase_0}
    )
    weight_cols = [f"weight_{domain}" for domain in domains]

    pure_metric_cols = [
        "run_name",
        "training_wandb_name",
        "training_wandb_id",
        "training_wandb_state",
        "training_wandb_created_at",
        "training_wandb_url",
        *UNCHEATABLE_COLUMNS,
    ]
    pure_metrics = pure_export[pure_metric_cols].copy()
    pure_metrics["uncheatable_source"] = "single_phase_qsplit_eval"

    pctrl_metrics = _pctrl_domain_deletion_uncheatable(pctrl_training_eval_path)
    pctrl_metrics = pctrl_metrics.rename(
        columns={
            "wandb_training_run_id": "training_wandb_id",
            "wandb_training_run_url": "training_wandb_url",
        }
    )
    pctrl_metrics["training_wandb_name"] = pd.NA
    pctrl_metrics["training_wandb_state"] = pd.NA
    pctrl_metrics["training_wandb_created_at"] = pd.NA
    pctrl_metrics["uncheatable_source"] = "proportional_controllability_domain_deletion_eval"

    shared_stratified_metrics = _shared_stratified_uncheatable(qsplit_training_eval_path)

    metrics = pd.concat(
        [
            pure_metrics,
            pctrl_metrics[pure_metrics.columns],
            shared_stratified_metrics[pure_metrics.columns],
        ],
        ignore_index=True,
    )
    _assert_unique_run_names(metrics, "combined uncheatable metrics")

    proportional_means = {col: float(proportional_reference[col].mean()) for col in UNCHEATABLE_COLUMNS}
    proportional_stds = {
        f"{col}_proportional_reference_std": float(proportional_reference[col].std(ddof=1))
        for col in UNCHEATABLE_COLUMNS
    }
    prop_mask = metrics["run_name"].eq("singleavg_baseline_proportional")
    if int(prop_mask.sum()) != 1:
        raise ValueError("expected one singleavg_baseline_proportional metric row")
    for col, value in proportional_means.items():
        metrics.loc[prop_mask, col] = value
    metrics.loc[prop_mask, "uncheatable_source"] = "proportional_reference_mean_11"

    metadata_cols = [
        col
        for col in [
            "run_name",
            "source_experiment",
            "panel_source",
            "source_run_name",
            "source_panel",
            "is_shared_checkpoint_alias",
            "shared_checkpoint_run_name",
        ]
        if col in panel.columns
    ]
    output = panel[metadata_cols].merge(weight_df, on="run_name", how="left", validate="one_to_one")
    output = output.merge(metrics, on="run_name", how="left", validate="one_to_one")

    target_cols = ["table9_macro_bpb", *component_cols]
    output = output.merge(panel[["run_name", *target_cols]], on="run_name", how="left", validate="one_to_one")
    output["weight_sum"] = output[weight_cols].sum(axis=1)
    output["phase_tied"] = True
    output["phase_max_abs_delta"] = phase_stats["max_phase_delta"]
    output["target_uses_proportional_reference_mean"] = output["run_name"].eq("singleavg_baseline_proportional")
    output["proportional_reference_n"] = np.where(
        output["target_uses_proportional_reference_mean"],
        EXPECTED_PROPORTIONAL_REFERENCE_ROWS,
        np.nan,
    )
    for col, value in proportional_stds.items():
        output[col] = np.where(output["target_uses_proportional_reference_mean"], value, np.nan)

    required_non_null = [*UNCHEATABLE_COLUMNS, "table9_macro_bpb", *component_cols]
    missing_by_col = output[required_non_null].isna().sum()
    missing_by_col = missing_by_col[missing_by_col > 0]
    if not missing_by_col.empty:
        raise ValueError(f"augmented export has missing required scores: {missing_by_col.to_dict()}")

    output = output.sort_values(["source_panel", "run_name"], kind="stable").reset_index(drop=True)
    ordered_cols = [
        "run_name",
        "source_panel",
        "panel_source",
        "source_experiment",
        "source_run_name",
        "is_shared_checkpoint_alias",
        "shared_checkpoint_run_name",
        "uncheatable_source",
        "training_wandb_name",
        "training_wandb_id",
        "training_wandb_state",
        "training_wandb_created_at",
        "training_wandb_url",
        *UNCHEATABLE_COLUMNS,
        "table9_macro_bpb",
        "weight_sum",
        "phase_tied",
        "phase_max_abs_delta",
        "target_uses_proportional_reference_mean",
        "proportional_reference_n",
        *[f"{col}_proportional_reference_std" for col in UNCHEATABLE_COLUMNS],
        *component_cols,
        *weight_cols,
    ]
    ordered_cols = [col for col in ordered_cols if col in output.columns]
    remaining = [col for col in output.columns if col not in set(ordered_cols)]
    output = output[ordered_cols + remaining]

    source_counts = output["source_panel"].value_counts(dropna=False).to_dict()
    expected_counts = {
        "all": EXPECTED_ROWS,
        "proportional_domain_deletion": EXPECTED_DOMAIN_DELETION_ROWS,
        "shared_policy_intersection": EXPECTED_SHARED_STRATIFIED_ROWS,
    }
    if source_counts != expected_counts:
        raise ValueError(f"unexpected augmented source_panel counts: {source_counts}")

    summary = {
        "row_count": len(output),
        "source_panel_counts": {str(key): int(value) for key, value in source_counts.items()},
        "weight_column_count": len(weight_cols),
        "table9_component_count": len(component_cols),
        "phase_weight_checks": phase_stats,
        "proportional_reference_policy": (
            "singleavg_baseline_proportional uses the mean of 11 proportional observations for "
            "Table-9 and uncheatable score columns; the 11 observations are exported separately."
        ),
        "shared_checkpoint_alias_rows": int(output["is_shared_checkpoint_alias"].fillna(False).sum()),
        "score_summaries": {
            col: {
                "min": float(output[col].min()),
                "median": float(output[col].median()),
                "max": float(output[col].max()),
            }
            for col in [
                "eval_uncheatable_eval_bpb",
                "eval_uncheatable_eval_macro_bpb",
                "table9_macro_bpb",
            ]
        },
    }
    return output, summary


def build_export(
    manifest_path: Path, uncheatable_path: Path, table9_path: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    manifest = _read_csv(manifest_path, "manifest")
    uncheatable = _read_csv(uncheatable_path, "uncheatable")
    table9 = _read_csv(table9_path, "table9")

    for label, df in (
        ("manifest", manifest),
        ("uncheatable", uncheatable),
        ("table9", table9),
    ):
        if len(df) != EXPECTED_ROWS:
            raise ValueError(f"{label} has {len(df)} rows, expected {EXPECTED_ROWS}")
        _assert_unique_run_names(df, label)

    phase_0, phase_1, domains = _weight_columns(manifest)
    phase_stats = _validate_tied_one_phase(manifest, phase_0, phase_1)
    component_cols = _table9_component_columns(table9)

    weight_df = manifest[["run_name", *phase_0]].copy()
    weight_df = weight_df.rename(columns={col: f"weight_{col.removeprefix('phase_0_')}" for col in phase_0})

    manifest_meta_cols = [
        col
        for col in [
            "run_id",
            "run_name",
            "cohort",
            "model_family",
            "trainer_seed",
            "data_seed",
            "simulated_epoch_subset_seed",
            "source_run_id",
            "source_run_name",
            "source_two_phase_experiment",
            "candidate_run_id",
            "candidate_run_name",
            "candidate_source_experiment",
            "single_phase_strategy",
            "source_panel",
            "phase_tv",
            "scale",
            "scale_display_label",
            "experiment_budget",
            "realized_experiment_budget",
            "target_budget",
            "target_budget_multiplier",
            "num_train_steps",
            "target_final_checkpoint_step",
        ]
        if col in manifest.columns
    ]

    uncheatable_export = uncheatable.rename(
        columns={
            "wandb_name": "training_wandb_name",
            "wandb_id": "training_wandb_id",
            "state": "training_wandb_state",
            "created_at": "training_wandb_created_at",
            "wandb_url": "training_wandb_url",
        }
    )

    table9_export = table9.rename(
        columns={
            "eval_source_run_name": "table9_eval_source_run_name",
            "eval_source_run_id": "table9_eval_source_run_id",
            "eval_target_name": "table9_eval_target_name",
            "wandb_run_id": "table9_wandb_run_id",
            "wandb_url": "table9_wandb_url",
            "wandb_state": "table9_wandb_state",
            "wandb_created_at": "table9_wandb_created_at",
            "wandb_updated_at": "table9_wandb_updated_at",
        }
    )

    merged = manifest[manifest_meta_cols].merge(weight_df, on="run_name", how="left", validate="one_to_one")
    merged = merged.merge(uncheatable_export, on="run_name", how="left", validate="one_to_one")
    merged = merged.merge(table9_export, on="run_name", how="left", validate="one_to_one")

    required_non_null = [
        "eval_uncheatable_eval_bpb",
        "eval_uncheatable_eval_macro_bpb",
        "eval_bpb",
        "eval_macro_bpb",
        "native_table9_macro_bpb",
        "table9_macro_bpb",
        *component_cols,
    ]
    missing_by_col = merged[required_non_null].isna().sum()
    missing_by_col = missing_by_col[missing_by_col > 0]
    if not missing_by_col.empty:
        raise ValueError(f"missing required scores: {missing_by_col.to_dict()}")

    native_delta = float((merged["native_table9_macro_bpb"] - merged["table9_macro_bpb"]).abs().max())
    if native_delta > 1e-10:
        raise ValueError(f"native_table9_macro_bpb and table9_macro_bpb differ; max delta {native_delta}")

    weight_cols = [f"weight_{domain}" for domain in domains]
    merged["weight_sum"] = merged[weight_cols].sum(axis=1)
    merged["phase_tied"] = True
    merged["phase_max_abs_delta"] = phase_stats["max_phase_delta"]

    leading_cols = [
        *manifest_meta_cols,
        "training_wandb_name",
        "training_wandb_id",
        "training_wandb_state",
        "training_wandb_created_at",
        "training_wandb_url",
        "table9_eval_source_run_name",
        "table9_eval_source_run_id",
        "table9_eval_target_name",
        "table9_wandb_run_id",
        "table9_wandb_url",
        "table9_wandb_state",
        "table9_wandb_created_at",
        "table9_wandb_updated_at",
        "eval_uncheatable_eval_bpb",
        "eval_uncheatable_eval_macro_bpb",
        "eval_bpb",
        "eval_macro_bpb",
        "native_table9_macro_bpb",
        "table9_macro_bpb",
        "weight_sum",
        "phase_tied",
        "phase_max_abs_delta",
    ]
    ordered_cols = [col for col in leading_cols if col in merged.columns]
    ordered_cols.extend(component_cols)
    ordered_cols.extend(weight_cols)
    remaining = [col for col in merged.columns if col not in set(ordered_cols)]
    merged = merged[ordered_cols + remaining].sort_values("run_id", kind="stable").reset_index(drop=True)

    summary = {
        "row_count": len(merged),
        "expected_rows": EXPECTED_ROWS,
        "weight_column_count": len(weight_cols),
        "table9_component_count": len(component_cols),
        "inputs": {
            "manifest": str(manifest_path),
            "uncheatable": str(uncheatable_path),
            "table9": str(table9_path),
        },
        "score_columns": {
            "uncheatable": [
                "eval_uncheatable_eval_bpb",
                "eval_uncheatable_eval_macro_bpb",
                "eval_bpb",
                "eval_macro_bpb",
            ],
            "table9_macro": ["native_table9_macro_bpb", "table9_macro_bpb"],
            "table9_components": component_cols,
        },
        "phase_weight_checks": phase_stats | {"native_table9_macro_max_delta": native_delta},
        "score_summaries": {
            col: {
                "min": float(merged[col].min()),
                "median": float(merged[col].median()),
                "max": float(merged[col].max()),
            }
            for col in [
                "eval_uncheatable_eval_bpb",
                "eval_uncheatable_eval_macro_bpb",
                "table9_macro_bpb",
            ]
        },
    }
    return merged, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--uncheatable", type=Path, default=DEFAULT_UNCHEATABLE)
    parser.add_argument("--table9", type=Path, default=DEFAULT_TABLE9)
    parser.add_argument("--augmented-panel", type=Path, default=DEFAULT_AUGMENTED_PANEL)
    parser.add_argument("--proportional-reference-table9", type=Path, default=DEFAULT_PROPORTIONAL_REFERENCE_TABLE9)
    parser.add_argument("--qsplit-training-eval", type=Path, default=DEFAULT_QSPLIT_TRAINING_EVAL)
    parser.add_argument("--pctrl-training-eval", type=Path, default=DEFAULT_PCTRL_TRAINING_EVAL)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--augmented-output-csv", type=Path, default=DEFAULT_AUGMENTED_OUTPUT_CSV)
    parser.add_argument(
        "--proportional-reference-output-csv", type=Path, default=DEFAULT_PROPORTIONAL_REFERENCE_OUTPUT_CSV
    )
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    args = parser.parse_args()

    export, summary = build_export(args.manifest, args.uncheatable, args.table9)
    component_cols = summary["score_columns"]["table9_components"]
    proportional_reference, proportional_reference_summary = build_proportional_reference_export(
        args.proportional_reference_table9,
        args.qsplit_training_eval,
    )
    augmented_export, augmented_summary = build_augmented_export(
        args.augmented_panel,
        export,
        component_cols,
        proportional_reference,
        args.qsplit_training_eval,
        args.pctrl_training_eval,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(args.output_csv, index=False)
    augmented_export.to_csv(args.augmented_output_csv, index=False)
    proportional_reference.to_csv(args.proportional_reference_output_csv, index=False)
    combined_summary = {
        "pure_swarm_export": summary,
        "augmented_fit_export": augmented_summary,
        "proportional_reference_export": proportional_reference_summary,
        "outputs": {
            "pure_swarm_csv": str(args.output_csv),
            "augmented_fit_csv": str(args.augmented_output_csv),
            "proportional_reference_csv": str(args.proportional_reference_output_csv),
            "summary_json": str(args.summary_json),
            "readme": str(args.readme),
        },
    }
    args.summary_json.write_text(json.dumps(combined_summary, indent=2, sort_keys=True) + "\n")
    args.readme.write_text(
        "\n".join(
            [
                "# One-phase qsplit240 300M swarm score export",
                "",
                "This directory contains collaborator-facing exports for the one-phase exposure-average qsplit240 300M/6B swarm and its augmented model-fitting panel.",
                "",
                "## Files",
                "",
                f"- `{args.output_csv.name}`: 240 pure qsplit candidate rows with provenance, W&B links, uncheatable eval scores, native OLMoBaseEval Table-9 macro/component BPBs, and 39 one-phase bucket weights.",
                f"- `{args.augmented_output_csv.name}`: 280 model-fit rows: 240 qsplit rows, the shared phase-tied stratified baseline, and 39 proportional domain-deletion controls. This matches the two-phase fit-panel row count.",
                f"- `{args.proportional_reference_output_csv.name}`: 11 proportional reference observations used to replace the proportional fit row target mean and estimate repeat noise.",
                f"- `{args.summary_json.name}`: export schema, input paths, score summaries, and validation checks.",
                "",
                "## Key columns",
                "",
                "- `run_name`: one-phase candidate identifier.",
                "- `eval_uncheatable_eval_bpb`: training/eval scalar for uncheatable eval BPB; lower is better.",
                "- `eval_uncheatable_eval_macro_bpb`: macro version of the uncheatable eval BPB scalar; lower is better.",
                "- `table9_macro_bpb`: native OLMoBaseEval Table-9 51-component macro BPB; lower is better.",
                "- `olmo_base_eval/easy_bpb/.../bpb` and `mmlu_*`: individual Table-9 component BPBs.",
                "- `weight_*`: one-phase mixture weights for the 39 Dolma3/Dolmino buckets. The source manifest had tied `phase_0 == phase_1`; this export stores only one weight vector.",
                "- `source_panel`: `all` for qsplit rows and `proportional_domain_deletion` for deletion controls in the augmented export.",
                "- `source_panel=shared_policy_intersection`: the existing phase-tied stratified checkpoint, reused as the exact intersection of the one- and two-phase policy classes rather than redundantly retrained.",
                "- `is_shared_checkpoint_alias`: true only for `singleavg_baseline_stratified`; exclude that alias from statistics requiring an independent heldout checkpoint because it points to the same physical run as the two-phase `baseline_stratified` row.",
                "- `target_uses_proportional_reference_mean`: true only for the proportional fit row in the augmented export; its Table-9 and uncheatable targets are the 11-run proportional mean.",
                "",
                "## Validation",
                "",
                f"- Pure swarm row count: {summary['row_count']} / {summary['expected_rows']}.",
                f"- Augmented fit row count: {augmented_summary['row_count']} with source counts `{augmented_summary['source_panel_counts']}`.",
                f"- Proportional reference row count: {proportional_reference_summary['row_count']}.",
                f"- Weight columns: {summary['weight_column_count']}.",
                f"- Table-9 component columns: {summary['table9_component_count']}.",
                f"- Pure max phase tie delta: {summary['phase_weight_checks']['max_phase_delta']}.",
                f"- Augmented max phase tie delta: {augmented_summary['phase_weight_checks']['max_phase_delta']}.",
                f"- Max Table-9 native-vs-export macro delta: {summary['phase_weight_checks']['native_table9_macro_max_delta']}.",
                "",
                "All score columns required by the exporter were non-null at generation time.",
                "",
            ]
        )
    )

    print(
        json.dumps(
            combined_summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
