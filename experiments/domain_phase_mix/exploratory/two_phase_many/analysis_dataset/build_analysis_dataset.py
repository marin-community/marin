# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas"]
# ///
"""Build the canonical modeling dataset for two-phase-many mixture/scale work."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.launch_two_phase_many_genericfamily_penalty_raw_optima_300m_6b import (
    NAME as PENALTY_300M_SOURCE_EXPERIMENT,
)
from experiments.domain_phase_mix.launch_two_phase_many_genericfamily_penalty_raw_optima_300m_6b import (
    build_run_specs as build_penalty_300m_run_specs,
)
from experiments.domain_phase_mix.launch_two_phase_many_genericfamily_penalty_raw_optima_1_2b_24b import (
    NAME as PENALTY_1_2B_SOURCE_EXPERIMENT,
)
from experiments.domain_phase_mix.launch_two_phase_many_genericfamily_penalty_raw_optima_1_2b_24b import (
    build_run_specs as build_penalty_1_2b_run_specs,
)
from experiments.domain_phase_mix.launch_two_phase_many_genericfamily_penalty_raw_optima_520m_10p4b import (
    NAME as PENALTY_520M_SOURCE_EXPERIMENT,
)
from experiments.domain_phase_mix.launch_two_phase_many_genericfamily_penalty_raw_optima_520m_10p4b import (
    build_run_specs as build_penalty_520m_run_specs,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.build_metric_registry import (
    METRICS_WIDE_CSV,
)
from experiments.domain_phase_mix.proxy_sweep import compute_llama_non_embedding_params
from experiments.domain_phase_mix.scaling_study_recipes import SCALE_SPECS

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY_DIR = SCRIPT_DIR.parent
SOURCE_PACKET_ROOT = TWO_PHASE_MANY_DIR / "chatgpt_pro_hybrid_data_mixing_packet_v28"
SOURCE_RUNS_CSV = SOURCE_PACKET_ROOT / "data" / "nd_scale_runs.csv"
SOURCE_PACKET_NPZ = SOURCE_PACKET_ROOT / "data" / "nd_scale_packet.npz"
RUN_REGISTRY_DIR = TWO_PHASE_MANY_DIR / "run_registry"
LOGICAL_RUNS_CSV = RUN_REGISTRY_DIR / "logical_runs.csv"
STRONG_READY_CSV = RUN_REGISTRY_DIR / "strong_tier_perplexity_ready.csv"
OUTPUT_RUNS_CSV = SCRIPT_DIR / "nd_scale_runs.csv"
OUTPUT_PACKET_NPZ = SCRIPT_DIR / "nd_scale_packet.npz"
OUTPUT_SUMMARY_JSON = SCRIPT_DIR / "summary.json"

PRIMARY_METRIC = "eval/uncheatable_eval/bpb"
SECONDARY_MACRO_METRIC = "eval/uncheatable_eval/macro_bpb"
PALOMA_MACRO_METRIC = "eval/paloma/macro_bpb"
MARIN_VOCAB_SIZE = 128_256
STRONG_KEY_COLUMNS = ("scale", "source_experiment", "run_name", "target_budget_multiplier")
CANONICAL_KEY_COLUMNS = ("scale", "path", "source_experiment", "run_name", "target_budget_multiplier")
RAW_OPTIMUM_VALIDATION_PATH = "grp_penalty_raw_optima_validation"
STRONG_TIER_PATHS = {
    "qsplit_representative12",
    "stratified",
    "qsplit_baselines3_holdout",
    "stratified_holdout",
}

NOMINAL_MODEL_SIZES = {
    "60m_1p2b": 60_000_000,
    "130m_2p6b": 130_000_000,
    "300m_6b": 300_000_000,
    "520m_10p4b": 520_000_000,
    "1_2b_24b": 1_200_000_000,
}

SCALE_DISPLAY_LABELS = {
    "130m_2p6b": "20M/2.6B",
    "60m_1p2b": "60M/1.2B",
    "300m_6b": "100M/6B",
    "520m_10p4b": "340M/10.4B",
    "1_2b_24b": "900M/24B",
}

RAW_OPTIMUM_FAMILIES = {
    "grp_penalty_raw_optima_300m_6b": {
        "source_experiment": PENALTY_300M_SOURCE_EXPERIMENT,
        "scale": "300m_6b",
        "fit_role": "fit_region",
        "model_family": "regmix_300m_proxy",
        "specs": build_penalty_300m_run_specs,
    },
    "grp_penalty_raw_optima_520m_10p4b": {
        "source_experiment": PENALTY_520M_SOURCE_EXPERIMENT,
        "scale": "520m_10p4b",
        "fit_role": "external_eval_520m",
        "model_family": "regmix_520m_proxy",
        "specs": build_penalty_520m_run_specs,
    },
    "grp_penalty_raw_optima_1_2b_24b": {
        "source_experiment": PENALTY_1_2B_SOURCE_EXPERIMENT,
        "scale": "1_2b_24b",
        "fit_role": "external_eval_1_2b",
        "model_family": "regmix_1_2b_proxy",
        "specs": build_penalty_1_2b_run_specs,
    },
}


@dataclass(frozen=True)
class BuildResult:
    """In-memory build outputs plus audit metadata."""

    frame: pd.DataFrame
    packet_arrays: dict[str, np.ndarray]
    summary: dict[str, Any]


def _path_timestamp(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()


def _read_source_packet() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    if not SOURCE_RUNS_CSV.exists():
        raise FileNotFoundError(f"Missing source runs CSV: {SOURCE_RUNS_CSV}")
    if not SOURCE_PACKET_NPZ.exists():
        raise FileNotFoundError(f"Missing source packet NPZ: {SOURCE_PACKET_NPZ}")

    frame = pd.read_csv(SOURCE_RUNS_CSV, low_memory=False)
    payload = dict(np.load(SOURCE_PACKET_NPZ, allow_pickle=True))
    expected_rows = len(payload["registry_run_keys"])
    if len(frame) != expected_rows:
        raise ValueError(f"Source CSV/NPZ row mismatch: csv={len(frame)} npz={expected_rows}")
    frame = frame.copy()
    frame["_source_row_index"] = np.arange(len(frame), dtype=np.int64)
    frame["_epoch_source_row_index"] = frame["_source_row_index"].astype(np.int64)
    frame["_is_registry_raw_optimum_append"] = False
    return frame, payload


def _scale_metadata() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scale, spec in SCALE_SPECS.items():
        config = spec.model_config
        scale_key = scale.value
        non_embedding_params = int(compute_llama_non_embedding_params(config))
        input_embedding_params = int(MARIN_VOCAB_SIZE * config.hidden_dim)
        output_head_params = 0 if bool(config.tie_word_embeddings) else input_embedding_params
        rows.append(
            {
                "scale": scale_key,
                "nominal_model_size": NOMINAL_MODEL_SIZES[scale_key],
                "non_embedding_params": non_embedding_params,
                "input_embedding_params": input_embedding_params,
                "output_head_params": output_head_params,
                "tied_total_params": non_embedding_params + input_embedding_params + output_head_params,
                "tie_word_embeddings": bool(config.tie_word_embeddings),
                "scale_display_label": SCALE_DISPLAY_LABELS[scale_key],
            }
        )
    return pd.DataFrame(rows)


def _normalize_phase_weights(frame: pd.DataFrame, domain_names: np.ndarray) -> np.ndarray:
    weights = np.zeros((len(frame), 2, len(domain_names)), dtype=float)
    for phase_index in (0, 1):
        columns = [f"phase_{phase_index}_{domain_name}" for domain_name in domain_names.astype(str)]
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing phase {phase_index} weight columns: {missing[:8]}")

        phase_weights = frame[columns].fillna(0.0).to_numpy(dtype=float)
        totals = phase_weights.sum(axis=1)
        bad_rows = np.flatnonzero(totals <= 0.0)
        if len(bad_rows):
            raise ValueError(f"Non-positive phase {phase_index} weight totals at rows: {bad_rows[:8].tolist()}")

        normalized = phase_weights / totals[:, None]
        weights[:, phase_index, :] = normalized
        frame.loc[:, columns] = normalized
    return weights


def _epoch_template_indices(frame: pd.DataFrame) -> dict[tuple[str, int, int, float], int]:
    output: dict[tuple[str, int, int, float], int] = {}
    for index, row in frame.iterrows():
        key = (
            str(row["scale"]),
            int(row["experiment_budget"]),
            int(row["target_budget"]),
            float(row["target_budget_multiplier"]),
        )
        output.setdefault(key, int(index))
    return output


def _append_registry_raw_optimum_rows(
    frame: pd.DataFrame,
    domain_names: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    logical = pd.read_csv(LOGICAL_RUNS_CSV, low_memory=False)
    existing_keys = set(frame["registry_run_key"].astype(str))
    epoch_templates = _epoch_template_indices(frame)
    appended_rows: list[dict[str, Any]] = []
    skipped_existing = 0
    skipped_unlabeled = 0
    skipped_incomplete = 0
    skipped_missing_template = 0

    for family, metadata in RAW_OPTIMUM_FAMILIES.items():
        family_rows = logical[logical["family"].astype(str).eq(family)]
        specs_by_run_name = {spec.run_name: spec for spec in metadata["specs"]()}
        for _, logical_row in family_rows.iterrows():
            registry_key = str(logical_row["registry_id"])
            if registry_key in existing_keys:
                skipped_existing += 1
                continue
            if str(logical_row["logical_status"]) != "completed":
                skipped_incomplete += 1
                continue
            if pd.isna(logical_row["objective_metric_value"]):
                skipped_unlabeled += 1
                continue

            run_name = str(logical_row["run_name"])
            if run_name not in specs_by_run_name:
                raise ValueError(f"Missing raw-optimum run spec for {family}:{run_name}")
            spec = specs_by_run_name[run_name]
            template_key = (
                str(metadata["scale"]),
                int(spec.experiment_budget),
                int(spec.target_budget),
                float(spec.target_budget_multiplier),
            )
            if template_key not in epoch_templates:
                skipped_missing_template += 1
                continue

            row = {column: np.nan for column in frame.columns}
            row.update(
                {
                    "registry_run_key": registry_key,
                    "mixture_id": run_name,
                    "run_id": int(spec.run_id),
                    "run_name": run_name,
                    "scale": str(metadata["scale"]),
                    "path": RAW_OPTIMUM_VALIDATION_PATH,
                    "cohort": str(spec.cohort),
                    "fit_role": str(metadata["fit_role"]),
                    "status": str(logical_row["logical_status"]),
                    "study_cell_status": "registry_objective_metric",
                    "study_panel": "grp_penalty_raw_optima",
                    "model_family": str(metadata["model_family"]),
                    "experiment_budget": int(spec.experiment_budget),
                    "target_budget": int(spec.target_budget),
                    "target_budget_multiplier": float(spec.target_budget_multiplier),
                    "source_experiment": str(logical_row["source_experiment"]),
                    "source_name_prefix": str(metadata["source_experiment"]),
                    "candidate_source_experiment": str(spec.candidate_source_experiment),
                    "candidate_run_id": int(spec.candidate_run_id),
                    "candidate_run_name": str(spec.candidate_run_name),
                    "wandb_run_id": str(logical_row["wandb_run_id"]),
                    "checkpoint_root": str(logical_row["checkpoint_root"]),
                    "objective_metric_value": float(logical_row["objective_metric_value"]),
                    "is_perplexity_ready": True,
                    "final_checkpoint_step": float(logical_row["max_checkpoint_step"]),
                    "final_checkpoint_path": str(logical_row["checkpoint_root"]),
                    "has_primary_label": True,
                    "has_macro_label": False,
                    "has_paloma_label": False,
                    "is_complete_checkpoint": True,
                    PRIMARY_METRIC: float(logical_row["objective_metric_value"]),
                    "_source_row_index": -1,
                    "_epoch_source_row_index": int(epoch_templates[template_key]),
                    "_is_registry_raw_optimum_append": True,
                    "target_eval_step": int(logical_row["max_checkpoint_step"]),
                    "target_eval_objective_metric_value": float(logical_row["objective_metric_value"]),
                }
            )
            template_row = frame.iloc[int(epoch_templates[template_key])]
            if "realized_train_tokens" in frame.columns:
                row["realized_train_tokens"] = int(template_row["realized_train_tokens"])

            for phase_name, phase_weights in spec.phase_weights.items():
                phase_index = int(str(phase_name).split("_")[1])
                for domain_name in domain_names.astype(str):
                    row[f"phase_{phase_index}_{domain_name}"] = float(phase_weights[domain_name])

            appended_rows.append(row)
            existing_keys.add(registry_key)

    if not appended_rows:
        return frame, {
            "raw_optimum_rows_appended": 0,
            "raw_optimum_skipped_existing": skipped_existing,
            "raw_optimum_skipped_unlabeled": skipped_unlabeled,
            "raw_optimum_skipped_incomplete": skipped_incomplete,
            "raw_optimum_skipped_missing_epoch_template": skipped_missing_template,
        }

    output = pd.concat([frame, pd.DataFrame(appended_rows, columns=frame.columns)], ignore_index=True, sort=False)
    return output, {
        "raw_optimum_rows_appended": len(appended_rows),
        "raw_optimum_skipped_existing": skipped_existing,
        "raw_optimum_skipped_unlabeled": skipped_unlabeled,
        "raw_optimum_skipped_incomplete": skipped_incomplete,
        "raw_optimum_skipped_missing_epoch_template": skipped_missing_template,
    }


def _row_key(row: pd.Series, columns: tuple[str, ...]) -> tuple[object, ...]:
    values: list[object] = []
    for column in columns:
        value = row[column]
        if column == "target_budget_multiplier":
            value = float(value)
        values.append(value)
    return tuple(values)


def _canonical_modeling_key(frame: pd.DataFrame) -> pd.Series:
    parts = []
    for column in CANONICAL_KEY_COLUMNS:
        if column == "target_budget_multiplier":
            parts.append(frame[column].astype(float).map(lambda value: f"{value:g}"))
        else:
            parts.append(frame[column].astype(str))
    key = parts[0]
    for part in parts[1:]:
        key = key + ":" + part
    return key


def _overlay_strong_ready_labels(frame: pd.DataFrame) -> dict[str, Any]:
    ready = pd.read_csv(STRONG_READY_CSV, low_memory=False)
    duplicate_ready = int(ready.duplicated(list(STRONG_KEY_COLUMNS)).sum())
    if duplicate_ready:
        raise ValueError(f"Duplicate strong-ready label keys: {duplicate_ready}")

    frame["label_source"] = np.where(frame[PRIMARY_METRIC].notna(), "packet_historical_metric", "unlabeled")
    frame["is_target_step_label"] = frame[PRIMARY_METRIC].notna()
    raw_append_mask = frame.get("_is_registry_raw_optimum_append", pd.Series(False, index=frame.index)).fillna(False)
    frame.loc[raw_append_mask, "label_source"] = "run_registry_objective_metric"
    frame.loc[raw_append_mask, "is_target_step_label"] = True
    frame["is_registry_strong_tier_row"] = frame["path"].astype(str).isin(STRONG_TIER_PATHS)
    frame["has_strong_ready_target_eval"] = False
    if "target_eval_step" not in frame.columns:
        frame["target_eval_step"] = pd.NA
    if "target_eval_objective_metric_value" not in frame.columns:
        frame["target_eval_objective_metric_value"] = pd.NA

    index_by_key = {_row_key(row, STRONG_KEY_COLUMNS): int(index) for index, row in frame.iterrows()}
    matched = 0
    updated = 0
    newly_enabled = 0
    max_abs_delta = 0.0

    for _, ready_row in ready.iterrows():
        key = _row_key(ready_row, STRONG_KEY_COLUMNS)
        if key not in index_by_key:
            raise ValueError(f"Strong-ready row missing from source packet: {key}")

        row_index = index_by_key[key]
        value = float(ready_row["target_eval_objective_metric_value"])
        old_value = frame.at[row_index, PRIMARY_METRIC]
        had_label = pd.notna(old_value)
        if had_label:
            delta = abs(float(old_value) - value)
            max_abs_delta = max(max_abs_delta, delta)
            if not np.isclose(float(old_value), value, rtol=0.0, atol=1e-12):
                updated += 1
        else:
            newly_enabled += 1

        matched += 1
        frame.at[row_index, PRIMARY_METRIC] = value
        frame.at[row_index, "objective_metric_value"] = value
        frame.at[row_index, "is_perplexity_ready"] = True
        frame.at[row_index, "has_primary_label"] = True
        frame.at[row_index, "target_eval_step"] = ready_row["target_eval_step"]
        frame.at[row_index, "target_eval_objective_metric_value"] = value
        frame.at[row_index, "label_source"] = "run_registry_target_eval"
        frame.at[row_index, "is_target_step_label"] = True
        frame.at[row_index, "has_strong_ready_target_eval"] = True

    stale_strong_mask = frame["is_registry_strong_tier_row"] & ~frame["has_strong_ready_target_eval"]
    return {
        "strong_ready_rows": len(ready),
        "strong_ready_rows_matched": int(matched),
        "strong_ready_existing_labels_updated": int(updated),
        "strong_ready_new_labels_enabled": int(newly_enabled),
        "strong_ready_max_abs_existing_label_delta": float(max_abs_delta),
        "stale_or_unready_strong_tier_rows": int(stale_strong_mask.sum()),
    }


def _apply_scale_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    metadata = _scale_metadata()
    output = frame.merge(metadata, on="scale", how="left", suffixes=("_source", ""))
    missing_scale = output["non_embedding_params"].isna()
    if missing_scale.any():
        scales = sorted(output.loc[missing_scale, "scale"].astype(str).unique())
        raise ValueError(f"Missing scale metadata for scales: {scales}")

    if "model_size_source" in output.columns:
        source_nominal = output["model_size_source"]
    else:
        source_nominal = output["model_size"]
    if "nominal_model_size_source" in output.columns:
        output = output.drop(columns=["nominal_model_size_source"])
    output["source_model_size"] = source_nominal.astype("Int64")
    output["model_size"] = output["non_embedding_params"].astype("int64")
    output["nominal_model_size"] = output["nominal_model_size"].astype("int64")
    output["non_embedding_params"] = output["non_embedding_params"].astype("int64")
    output["tied_total_params"] = output["tied_total_params"].astype("int64")
    output["input_embedding_params"] = output["input_embedding_params"].astype("int64")
    output["output_head_params"] = output["output_head_params"].astype("int64")
    return output


def _factorize(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    names = pd.unique(values.astype(str)).astype(object)
    index = pd.Index(names)
    return names, index.get_indexer(values.astype(str)).astype(np.int64)


def _metric_values(frame: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    if metric not in frame.columns:
        return np.full(len(frame), np.nan, dtype=float), np.zeros(len(frame), dtype=bool)
    values = frame[metric].to_numpy(dtype=float)
    return values, np.isfinite(values)


def _build_packet_arrays(
    frame: pd.DataFrame,
    payload: dict[str, np.ndarray],
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    scale_names, scale_index = _factorize(frame["scale"])
    path_names, path_index = _factorize(frame["path"])
    cohort_names, cohort_index = _factorize(frame["cohort"])
    fit_role_names, fit_role_index = _factorize(frame["fit_role"])

    primary_y, primary_mask = _metric_values(frame, PRIMARY_METRIC)
    macro_y, macro_mask = _metric_values(frame, SECONDARY_MACRO_METRIC)
    paloma_y, paloma_mask = _metric_values(frame, PALOMA_MACRO_METRIC)
    epoch_source_indices = frame["_epoch_source_row_index"].to_numpy(dtype=np.int64)

    return {
        "weights": weights,
        "primary_y": primary_y,
        "primary_y_mask": primary_mask,
        "macro_y": macro_y,
        "macro_y_mask": macro_mask,
        "paloma_y": paloma_y,
        "paloma_y_mask": paloma_mask,
        "scale_names": scale_names,
        "path_names": path_names,
        "cohort_names": cohort_names,
        "fit_role_names": fit_role_names,
        "scale_index": scale_index,
        "path_index": path_index,
        "cohort_index": cohort_index,
        "fit_role_index": fit_role_index,
        "mixture_ids": frame["mixture_id"].astype(str).to_numpy(dtype=object),
        "run_ids": frame["run_id"].astype(int).to_numpy(dtype=np.int64),
        "run_names": frame["run_name"].astype(str).to_numpy(dtype=object),
        "registry_run_keys": frame["registry_run_key"].astype(str).to_numpy(dtype=object),
        "model_sizes": frame["model_size"].astype(int).to_numpy(dtype=np.int64),
        "experiment_budgets": frame["experiment_budget"].astype(int).to_numpy(dtype=np.int64),
        "target_budgets": frame["target_budget"].astype(int).to_numpy(dtype=np.int64),
        "target_budget_multipliers": frame["target_budget_multiplier"].astype(float).to_numpy(dtype=float),
        "realized_train_tokens": frame["realized_train_tokens"].astype(int).to_numpy(dtype=np.int64),
        "phase_fractions": np.asarray(payload["phase_fractions"], dtype=float),
        "domain_names": np.asarray(payload["domain_names"], dtype=object),
        "simulated_epoch_multipliers": np.asarray(payload["simulated_epoch_multipliers"], dtype=float)[
            epoch_source_indices
        ],
        "raw_full_corpus_epoch_multipliers": np.asarray(payload["raw_full_corpus_epoch_multipliers"], dtype=float)[
            epoch_source_indices
        ],
        "primary_metric": np.asarray([PRIMARY_METRIC], dtype=object),
        "secondary_macro_metric": np.asarray([SECONDARY_MACRO_METRIC], dtype=object),
        "paloma_macro_metric": np.asarray([PALOMA_MACRO_METRIC], dtype=object),
    }


def _validate_frame(frame: pd.DataFrame, weights: np.ndarray) -> dict[str, Any]:
    required_columns = [
        "canonical_modeling_key",
        "registry_run_key",
        "scale",
        "path",
        "cohort",
        "fit_role",
        "run_name",
        "source_experiment",
        "model_size",
        "nominal_model_size",
        "non_embedding_params",
        "tied_total_params",
        "experiment_budget",
        "target_budget",
        "target_budget_multiplier",
        "realized_train_tokens",
        "scale_display_label",
        "is_target_step_label",
        "label_source",
        PRIMARY_METRIC,
    ]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    missing_required = {
        column: int(frame[column].isna().sum())
        for column in required_columns
        if column != PRIMARY_METRIC and int(frame[column].isna().sum())
    }
    if missing_required:
        raise ValueError(f"Missing required metadata: {missing_required}")

    duplicate_keys = int(frame["canonical_modeling_key"].duplicated().sum())
    if duplicate_keys:
        examples = frame.loc[frame["canonical_modeling_key"].duplicated(keep=False), "canonical_modeling_key"].head(8)
        raise ValueError(f"Duplicate canonical modeling keys: {duplicate_keys}; examples={examples.tolist()}")

    if not frame["model_size"].astype(int).equals(frame["non_embedding_params"].astype(int)):
        raise ValueError("model_size must equal non_embedding_params in the analysis dataset")

    primary_missing = int(frame[PRIMARY_METRIC].isna().sum())
    if primary_missing:
        raise ValueError(f"Included rows without primary labels: {primary_missing}")

    phase_sums = weights.sum(axis=2)
    max_phase_sum_error = float(np.max(np.abs(phase_sums - 1.0)))
    if max_phase_sum_error > 1e-10:
        raise ValueError(f"Phase weights do not sum to 1; max error={max_phase_sum_error}")

    return {
        "duplicate_canonical_modeling_keys": duplicate_keys,
        "primary_missing_rows": primary_missing,
        "max_phase_sum_error": max_phase_sum_error,
        "model_size_equals_non_embedding_params": True,
    }


def build_analysis_dataset() -> BuildResult:
    source_frame, payload = _read_source_packet()
    domain_names = np.asarray(payload["domain_names"], dtype=object)

    frame = source_frame.copy()
    frame, raw_optimum_summary = _append_registry_raw_optimum_rows(frame, domain_names)
    weights = _normalize_phase_weights(frame, domain_names)
    overlay_summary = _overlay_strong_ready_labels(frame)
    frame = _apply_scale_metadata(frame)
    frame["canonical_modeling_key"] = _canonical_modeling_key(frame)

    source_rows = len(frame)
    stale_strong_mask = frame["is_registry_strong_tier_row"] & ~frame["has_strong_ready_target_eval"]
    unlabeled_mask = frame[PRIMARY_METRIC].isna()
    drop_mask = stale_strong_mask | unlabeled_mask
    dropped_unlabeled_rows = int(unlabeled_mask.sum())
    dropped_stale_strong_rows = int(stale_strong_mask.sum())
    dropped_total_rows = int(drop_mask.sum())
    frame = frame.loc[~drop_mask].reset_index(drop=True)
    kept_source_indices = frame["_source_row_index"].to_numpy(dtype=np.int64)
    weights = weights[kept_source_indices]

    validation_summary = _validate_frame(frame, weights)
    packet_arrays = _build_packet_arrays(frame, payload, weights)
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "source_runs_csv": str(SOURCE_RUNS_CSV),
            "source_runs_csv_mtime": _path_timestamp(SOURCE_RUNS_CSV),
            "source_packet_npz": str(SOURCE_PACKET_NPZ),
            "source_packet_npz_mtime": _path_timestamp(SOURCE_PACKET_NPZ),
            "run_registry_logical_runs_csv": str(LOGICAL_RUNS_CSV),
            "run_registry_logical_runs_csv_mtime": _path_timestamp(LOGICAL_RUNS_CSV),
            "strong_ready_csv": str(STRONG_READY_CSV),
            "strong_ready_csv_mtime": _path_timestamp(STRONG_READY_CSV),
            "metric_registry_metrics_wide_csv": str(METRICS_WIDE_CSV),
            "metric_registry_metrics_wide_csv_mtime": _path_timestamp(METRICS_WIDE_CSV),
        },
        "rows": {
            "source_rows": int(source_rows),
            "output_rows": len(frame),
            "dropped_unlabeled_rows": dropped_unlabeled_rows,
            "dropped_stale_or_unready_strong_tier_rows": dropped_stale_strong_rows,
            "dropped_total_rows": dropped_total_rows,
        },
        "by_scale": {str(key): int(value) for key, value in frame.groupby("scale").size().sort_index().items()},
        "by_label_source": {
            str(key): int(value) for key, value in frame.groupby("label_source").size().sort_index().items()
        },
        "overlay": overlay_summary,
        "registry_appends": raw_optimum_summary,
        "validation": validation_summary,
        "scale_metadata": _scale_metadata().sort_values("non_embedding_params").to_dict(orient="records"),
        "outputs": {
            "nd_scale_runs_csv": str(OUTPUT_RUNS_CSV),
            "nd_scale_packet_npz": str(OUTPUT_PACKET_NPZ),
            "summary_json": str(OUTPUT_SUMMARY_JSON),
        },
    }
    internal_columns = ["_source_row_index", "_epoch_source_row_index", "_is_registry_raw_optimum_append"]
    return BuildResult(frame=frame.drop(columns=internal_columns), packet_arrays=packet_arrays, summary=summary)


def write_outputs(result: BuildResult) -> None:
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    result.frame.to_csv(OUTPUT_RUNS_CSV, index=False)
    np.savez(OUTPUT_PACKET_NPZ, **result.packet_arrays)
    OUTPUT_SUMMARY_JSON.write_text(json.dumps(result.summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    result = build_analysis_dataset()
    write_outputs(result)
    print(f"Wrote {len(result.frame)} rows to {OUTPUT_RUNS_CSV}")
    print(f"Wrote packet arrays to {OUTPUT_PACKET_NPZ}")
    print(f"Wrote summary to {OUTPUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
