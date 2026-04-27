# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "gcsfs", "kaleido==0.2.1", "numpy", "pandas", "plotly"]
# ///
"""Render the central 1x baseline scaling trajectory plot.

The plot tracks five fixed mixtures across the corrected scale ladder:
20M/2.6B, 60M/1.2B, 100M/6B, 340M/10.4B, and 900M/24B. Solid markers are
reserved for rows with a committed checkpoint-backed perplexity metric. Hollow
markers show non-target-ready rows that are useful for auditing but should not
be treated as final paper points.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from experiments.domain_phase_mix.exploratory.paper_plots.paper_plot_style import (
    configure_interactive_layout,
    configure_static_layout,
    method_color,
    method_dash,
    write_static_images,
)

SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR = SCRIPT_DIR / "img"
TWO_PHASE_MANY_DIR = SCRIPT_DIR.parent / "two_phase_many"
ANALYSIS_CSV = TWO_PHASE_MANY_DIR / "analysis_dataset" / "nd_scale_runs.csv"
REGISTRY_CSV = TWO_PHASE_MANY_DIR / "run_registry" / "logical_runs.csv"

PRIMARY_METRIC = "eval/uncheatable_eval/bpb"
MACRO_METRIC = "eval/uncheatable_eval/macro_bpb"
EVAL_BPB_METRIC = "eval/bpb"
PREFERRED_METRICS = (PRIMARY_METRIC, MACRO_METRIC, EVAL_BPB_METRIC)
TARGET_MULTIPLIER = 1.0
LEGACY_60M_TARGET_STEP = 4576
METRIC_DISPLAY_NAMES = {
    PRIMARY_METRIC: "Uncheatable BPB",
    MACRO_METRIC: "Uncheatable macro BPB",
    EVAL_BPB_METRIC: "Validation BPB",
}

LEGACY_GRP_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_genericfamily_penalty_raw_optima_uncheatable_bpb"
LEGACY_GRP_RUN_NAME = "baseline_genericfamily_power_family_penalty_no_l2_raw_optimum"
DIRECT_TARGET_OVERRIDES = {
    ("olmix", "300m_6b"): {
        "source_experiment": "pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b",
        "run_name": "baseline_olmix_loglinear_uncheatable_bpb",
        "checkpoint_root": (
            "gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/"
            "ngd3dm2_qsplit240_300m_6b/baseline_olmix_loglinear_uncheatable_bpb-7ac5e9"
        ),
        "target_step": 22887,
    },
    ("uniform", "520m_10p4b"): {
        "source_experiment": "pinlin_calvin_xu/data_mixture/ngd3dm2_baseline_scaling_uniform_520m_10p4b",
        "run_name": "baseline_stratified",
        "checkpoint_root": (
            "gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/"
            "ngd3dm2_baseline_scaling_uniform_520m_10p4b/baseline_stratified-0fab44"
        ),
        "target_step": 19835,
    },
    ("uniform", "1_2b_24b"): {
        "source_experiment": "pinlin_calvin_xu/data_mixture/ngd3dm2_baseline_scaling_uniform_1_2b_24b",
        "run_name": "baseline_stratified",
        "checkpoint_root": (
            "gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/"
            "ngd3dm2_baseline_scaling_uniform_1_2b_24b/baseline_stratified-b504ac"
        ),
        "target_step": 45775,
    },
}


@dataclass(frozen=True)
class ScaleSpec:
    """Corrected metadata for one scale rung."""

    scale: str
    label: str
    non_embedding_params: int
    realized_train_tokens: int

    @property
    def axis_label(self) -> str:
        """Return the multiline x-axis label for paper plots."""
        return self.label


@dataclass(frozen=True)
class MethodSpec:
    """One baseline mixture tracked in the central plot."""

    method_id: str
    label: str
    run_names: tuple[str, ...]


SCALES = (
    ScaleSpec("130m_2p6b", "20M/2.6B", 22_813_184, 2_599_944_192),
    ScaleSpec("60m_1p2b", "60M/1.2B", 58_998_528, 1_199_833_088),
    ScaleSpec("300m_6b", "100M/6B", 102_648_576, 5_999_951_872),
    ScaleSpec("520m_10p4b", "340M/10.4B", 339_788_800, 10_399_776_768),
    ScaleSpec("1_2b_24b", "900M/24B", 906_037_248, 23_999_807_488),
)

METHODS = (
    MethodSpec(
        "grp_no_l2",
        "GRP no-L2",
        (
            LEGACY_GRP_RUN_NAME,
            "baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_130m_2p6b",
            "baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_300m_6b",
            "baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_520m_10p4b",
            "baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_1_2b_24b",
        ),
    ),
    MethodSpec("proportional", "Proportional", ("baseline_proportional",)),
    MethodSpec("olmix", "Olmix", ("baseline_olmix_loglinear_uncheatable_bpb",)),
    MethodSpec("uniform", "Uniform", ("baseline_stratified",)),
    MethodSpec("unimax", "UniMax", ("baseline_unimax",)),
)

SCALE_BY_KEY = {scale.scale: scale for scale in SCALES}
SCALE_ORDER = {scale.scale: index for index, scale in enumerate(SCALES)}
METHOD_BY_ID = {method.method_id: method for method in METHODS}


def _nonnull_string(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _float_or_nan(value: object) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else float("nan")


def _bool_value(value: object) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _is_perplexity_metric_column(column: str) -> bool:
    """Return whether an `eval/...` column is a perplexity-style metric."""
    if not column.startswith("eval/"):
        return False
    metric_name = column.rsplit("/", maxsplit=1)[-1]
    return metric_name in {"bpb", "loss", "macro_bpb", "macro_loss", "micro_bpb", "micro_loss"}


def _metric_values_from_mapping(values: dict[str, object]) -> dict[str, float]:
    """Extract finite perplexity metrics from a row or eval JSON payload."""
    metrics: dict[str, float] = {}
    for key, value in values.items():
        column = str(key)
        if not _is_perplexity_metric_column(column):
            continue
        number = _float_or_nan(value)
        if np.isfinite(number):
            metrics[column] = number
    return metrics


def _metric_label(metric_column: str) -> str:
    if metric_column in METRIC_DISPLAY_NAMES:
        return METRIC_DISPLAY_NAMES[metric_column]
    return metric_column.removeprefix("eval/").replace("/", " / ")


def _available_metric_columns(points: pd.DataFrame) -> list[str]:
    """Return selectable metric columns, keeping the headline metrics first."""
    available = [
        column
        for column in points.columns
        if _is_perplexity_metric_column(str(column)) and pd.to_numeric(points[column], errors="coerce").notna().any()
    ]
    preferred = [column for column in PREFERRED_METRICS if column in available]
    remaining = sorted(column for column in available if column not in preferred)
    return [*preferred, *remaining]


def _is_target_multiplier(frame: pd.DataFrame) -> pd.Series:
    if "target_budget_multiplier" not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = pd.to_numeric(frame["target_budget_multiplier"], errors="coerce")
    return np.isclose(values.fillna(-999.0), TARGET_MULTIPLIER)


def _read_last_eval_metrics(path: str) -> dict[str, float] | None:
    payload: dict[str, float] | None = None
    fs = fsspec.filesystem("gs")
    try:
        with fs.open(path, "r") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
    except FileNotFoundError:
        return None
    return payload


def _read_eval_metrics_at_step(path: str, *, step: int) -> dict[str, float] | None:
    fs = fsspec.filesystem("gs")
    try:
        with fs.open(path, "r") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if int(payload.get("step", -1)) == step:
                    return payload
    except FileNotFoundError:
        return None
    return None


def _metric_values_from_checkpoint_root(checkpoint_root: str, *, step: float) -> dict[str, float]:
    """Read full eval metrics for a checkpoint root at an exact target step."""
    if not checkpoint_root.startswith("gs://") or not np.isfinite(step):
        return {}
    metrics = _read_eval_metrics_at_step(
        checkpoint_root.rstrip("/") + "/checkpoints/eval_metrics.jsonl",
        step=int(step),
    )
    if metrics is None:
        return {}
    return _metric_values_from_mapping(metrics)


def _legacy_grp_60m_diagnostic() -> dict[str, object] | None:
    pattern = (
        "marin-us-east5/checkpoints/"
        f"{LEGACY_GRP_SOURCE_EXPERIMENT}/{LEGACY_GRP_RUN_NAME}-*/checkpoints/eval_metrics.jsonl"
    )
    fs = fsspec.filesystem("gs")
    matches = sorted(fs.glob(pattern))
    if not matches:
        return None

    metrics = _read_last_eval_metrics(matches[-1])
    if metrics is None or PRIMARY_METRIC not in metrics:
        return None

    checkpoint_root = "gs://" + matches[-1].removesuffix("/checkpoints/eval_metrics.jsonl")
    metric_values = _metric_values_from_mapping(metrics)
    return {
        "method_id": "grp_no_l2",
        "method": METHOD_BY_ID["grp_no_l2"].label,
        "scale": "60m_1p2b",
        "scale_label": SCALE_BY_KEY["60m_1p2b"].label,
        "x_order": SCALE_ORDER["60m_1p2b"],
        "non_embedding_params": SCALE_BY_KEY["60m_1p2b"].non_embedding_params,
        "realized_train_tokens": SCALE_BY_KEY["60m_1p2b"].realized_train_tokens,
        "run_name": LEGACY_GRP_RUN_NAME,
        "source_experiment": LEGACY_GRP_SOURCE_EXPERIMENT,
        "status": "legacy_complete",
        "target_budget_multiplier": TARGET_MULTIPLIER,
        "target_ready": True,
        "point_kind": "target_ready",
        "cell_status": "target_ready",
        "recommended_action": "none",
        "metric_source": "legacy_gcs_target_eval",
        "metric_value": float(metrics[PRIMARY_METRIC]),
        **metric_values,
        "checkpoint_root": checkpoint_root,
        "gcs_checkpoint_path": checkpoint_root,
        "target_eval_step": LEGACY_60M_TARGET_STEP,
        "target_final_checkpoint_step": LEGACY_60M_TARGET_STEP,
        "max_checkpoint_step": LEGACY_60M_TARGET_STEP,
    }


def _direct_target_override(method: MethodSpec, scale: ScaleSpec) -> dict[str, object] | None:
    spec = DIRECT_TARGET_OVERRIDES.get((method.method_id, scale.scale))
    if spec is None:
        return None
    checkpoint_root = str(spec["checkpoint_root"])
    target_step = int(spec["target_step"])
    metrics = _read_eval_metrics_at_step(
        checkpoint_root.rstrip("/") + "/checkpoints/eval_metrics.jsonl",
        step=target_step,
    )
    if metrics is None or PRIMARY_METRIC not in metrics:
        return None
    metric_values = _metric_values_from_mapping(metrics)
    return {
        "method_id": method.method_id,
        "method": method.label,
        "scale": scale.scale,
        "scale_label": scale.label,
        "x_order": SCALE_ORDER[scale.scale],
        "non_embedding_params": scale.non_embedding_params,
        "realized_train_tokens": scale.realized_train_tokens,
        "run_name": str(spec["run_name"]),
        "source_experiment": str(spec["source_experiment"]),
        "status": "artifact_success",
        "target_budget_multiplier": TARGET_MULTIPLIER,
        "target_ready": True,
        "point_kind": "target_ready",
        "cell_status": "target_ready",
        "recommended_action": "none",
        "metric_source": "direct_gcs_target_eval",
        "metric_value": float(metrics[PRIMARY_METRIC]),
        **metric_values,
        "checkpoint_root": checkpoint_root,
        "gcs_checkpoint_path": checkpoint_root,
        "target_eval_step": target_step,
        "target_final_checkpoint_step": target_step,
        "max_checkpoint_step": target_step,
    }


def _analysis_candidates(analysis: pd.DataFrame, method: MethodSpec, scale: ScaleSpec) -> pd.DataFrame:
    mask = (
        analysis["scale"].astype(str).eq(scale.scale)
        & analysis["run_name"].astype(str).isin(method.run_names)
        & _is_target_multiplier(analysis)
        & pd.to_numeric(analysis.get(PRIMARY_METRIC), errors="coerce").notna()
    )
    return analysis.loc[mask].copy()


def _registry_candidates(registry: pd.DataFrame, method: MethodSpec, scale: ScaleSpec) -> pd.DataFrame:
    mask = registry["scale"].astype(str).eq(scale.scale) & registry["run_name"].astype(str).isin(method.run_names)
    if "target_budget_multiplier" in registry.columns:
        multipliers = pd.to_numeric(registry["target_budget_multiplier"], errors="coerce")
        mask &= np.isclose(multipliers.fillna(TARGET_MULTIPLIER), TARGET_MULTIPLIER)
    return registry.loc[mask].copy()


def _analysis_row_is_ready(row: pd.Series) -> bool:
    metric = _float_or_nan(row.get(PRIMARY_METRIC))
    checkpoint_root = _nonnull_string(row.get("checkpoint_root") or row.get("final_checkpoint_path"))
    label_source = _nonnull_string(row.get("label_source"))
    if not np.isfinite(metric) or not checkpoint_root or not _bool_value(row.get("is_perplexity_ready")):
        return False
    return label_source in {"run_registry_target_eval", "run_registry_objective_metric"}


def _is_exact_legacy_60m_target_row(scale: ScaleSpec, row: pd.Series) -> bool:
    """Return whether a legacy 60M packet row has an exact target-step label.

    The old 60M export predates the run-registry artifact scanner. Its CSV has
    final perplexity metrics and the analysis dataset carries audited checkpoint
    roots, but the registry row itself lacks target-eval metadata. Treat these
    exact step-4576 rows as target-ready in the central plot instead of calling
    them diagnostic.
    """
    if scale.scale != "60m_1p2b":
        return False
    label_source = _nonnull_string(row.get("label_source"))
    if label_source != "packet_historical_metric":
        return False
    metric = _float_or_nan(row.get(PRIMARY_METRIC))
    checkpoint_root = _nonnull_string(row.get("checkpoint_root") or row.get("final_checkpoint_path"))
    final_checkpoint_step = _float_or_nan(row.get("final_checkpoint_step"))
    return bool(np.isfinite(metric) and checkpoint_root and int(final_checkpoint_step) == LEGACY_60M_TARGET_STEP)


def _registry_row_is_ready(row: pd.Series) -> bool:
    metric = _float_or_nan(row.get("target_eval_objective_metric_value"))
    checkpoint_root = _nonnull_string(row.get("checkpoint_root"))
    return bool(
        np.isfinite(metric)
        and checkpoint_root
        and _bool_value(row.get("is_perplexity_ready"))
        and _bool_value(row.get("has_target_eval"))
    )


def _ready_sort_key(scale: ScaleSpec, row: pd.Series) -> tuple[int, int, float]:
    return (
        int(_analysis_row_is_ready(row) or _is_exact_legacy_60m_target_row(scale, row)),
        int(_bool_value(row.get("has_strong_ready_target_eval"))),
        _float_or_nan(row.get("target_eval_step")),
    )


def _best_analysis_row(scale: ScaleSpec, candidates: pd.DataFrame) -> pd.Series | None:
    if candidates.empty:
        return None
    sorted_candidates = candidates.assign(
        _ready=[_ready_sort_key(scale, row) for _, row in candidates.iterrows()],
    ).sort_values("_ready", ascending=False)
    return sorted_candidates.iloc[0]


def _best_registry_row(candidates: pd.DataFrame) -> pd.Series | None:
    if candidates.empty:
        return None
    ranked = candidates.assign(
        _ready=[int(_registry_row_is_ready(row)) for _, row in candidates.iterrows()],
        _metric_present=pd.to_numeric(candidates.get("objective_metric_value"), errors="coerce").notna().astype(int),
        _max_checkpoint=pd.to_numeric(candidates.get("max_checkpoint_step"), errors="coerce").fillna(-1),
    ).sort_values(["_ready", "_metric_present", "_max_checkpoint"], ascending=False)
    return ranked.iloc[0]


def _point_from_analysis_row(method: MethodSpec, scale: ScaleSpec, row: pd.Series) -> dict[str, object]:
    ready = _analysis_row_is_ready(row) or _is_exact_legacy_60m_target_row(scale, row)
    checkpoint_root = _nonnull_string(row.get("checkpoint_root") or row.get("final_checkpoint_path"))
    metric = float(row[PRIMARY_METRIC])
    label_source = _nonnull_string(row.get("label_source"))
    target_eval_step = _float_or_nan(row.get("target_eval_step"))
    target_final_checkpoint_step = _float_or_nan(row.get("target_final_checkpoint_step"))
    if ready and scale.scale == "60m_1p2b" and not np.isfinite(target_eval_step):
        target_eval_step = float(LEGACY_60M_TARGET_STEP)
    if ready and scale.scale == "60m_1p2b" and not np.isfinite(target_final_checkpoint_step):
        target_final_checkpoint_step = float(LEGACY_60M_TARGET_STEP)
    full_metric_step = target_eval_step
    if not np.isfinite(full_metric_step):
        full_metric_step = target_final_checkpoint_step
    if not np.isfinite(full_metric_step):
        full_metric_step = _float_or_nan(row.get("final_checkpoint_step"))
    metric_values = _metric_values_from_mapping(row.to_dict())
    metric_values.update(_metric_values_from_checkpoint_root(checkpoint_root, step=full_metric_step))
    if PRIMARY_METRIC not in metric_values:
        metric_values[PRIMARY_METRIC] = metric
    return {
        "method_id": method.method_id,
        "method": method.label,
        "scale": scale.scale,
        "scale_label": scale.label,
        "x_order": SCALE_ORDER[scale.scale],
        "non_embedding_params": scale.non_embedding_params,
        "realized_train_tokens": scale.realized_train_tokens,
        "run_name": _nonnull_string(row.get("run_name")),
        "source_experiment": _nonnull_string(row.get("source_experiment")),
        "status": _nonnull_string(row.get("status") or row.get("logical_status") or "analysis_dataset"),
        "target_budget_multiplier": TARGET_MULTIPLIER,
        "target_ready": ready,
        "point_kind": "target_ready" if ready else "diagnostic",
        "cell_status": "target_ready" if ready else "diagnostic_only",
        "recommended_action": "none" if ready else "relaunch_or_recover_target_step_metric",
        "metric_source": "legacy_60m_target_eval" if _is_exact_legacy_60m_target_row(scale, row) else label_source,
        "metric_value": metric,
        **metric_values,
        "checkpoint_root": checkpoint_root,
        "gcs_checkpoint_path": checkpoint_root,
        "target_eval_step": target_eval_step,
        "target_final_checkpoint_step": target_final_checkpoint_step,
        "max_checkpoint_step": _float_or_nan(row.get("final_checkpoint_step")),
    }


def _point_from_registry_row(method: MethodSpec, scale: ScaleSpec, row: pd.Series) -> dict[str, object]:
    ready = _registry_row_is_ready(row)
    metric = (
        _float_or_nan(row.get("target_eval_objective_metric_value"))
        if ready
        else _float_or_nan(row.get("objective_metric_value"))
    )
    checkpoint_root = _nonnull_string(row.get("checkpoint_root"))
    logical_status = _nonnull_string(row.get("logical_status")) or "registry"
    reached_target = _bool_value(row.get("reached_target_step"))
    target_step = _float_or_nan(row.get("target_final_checkpoint_step"))
    max_step = _float_or_nan(row.get("max_checkpoint_step"))
    overshot_without_target_eval = (
        not ready and np.isfinite(target_step) and np.isfinite(max_step) and int(max_step) > int(target_step)
    )
    if ready:
        cell_status = "target_ready"
        recommended_action = "none"
    elif overshot_without_target_eval:
        metric = np.nan
        cell_status = "needs_relaunch"
        recommended_action = "relaunch_without_resume_latest_checkpoint"
    elif reached_target and np.isfinite(metric):
        cell_status = "diagnostic_only"
        recommended_action = "recover_or_write_exact_target_step_eval_metric"
    elif logical_status == "failed":
        cell_status = "needs_relaunch"
        recommended_action = _nonnull_string(row.get("resubmit_hint")) or "relaunch_perplexity_only"
    else:
        cell_status = "needs_relaunch"
        recommended_action = _nonnull_string(row.get("resubmit_hint")) or "launch_perplexity_only"

    metric_values: dict[str, float] = {}
    if ready and checkpoint_root:
        step = _float_or_nan(row.get("target_eval_step"))
        if not np.isfinite(step):
            step = target_step
        if np.isfinite(step):
            payload = _read_eval_metrics_at_step(
                checkpoint_root.rstrip("/") + "/checkpoints/eval_metrics.jsonl",
                step=int(step),
            )
            if payload is not None:
                metric_values = _metric_values_from_mapping(payload)
    if PRIMARY_METRIC not in metric_values and np.isfinite(metric):
        metric_values[PRIMARY_METRIC] = metric

    return {
        "method_id": method.method_id,
        "method": method.label,
        "scale": scale.scale,
        "scale_label": scale.label,
        "x_order": SCALE_ORDER[scale.scale],
        "non_embedding_params": scale.non_embedding_params,
        "realized_train_tokens": scale.realized_train_tokens,
        "run_name": _nonnull_string(row.get("run_name")),
        "source_experiment": _nonnull_string(row.get("source_experiment")),
        "status": logical_status,
        "target_budget_multiplier": TARGET_MULTIPLIER,
        "target_ready": ready,
        "point_kind": "target_ready" if ready else "diagnostic",
        "cell_status": cell_status,
        "recommended_action": recommended_action,
        "metric_source": "run_registry_target_eval" if ready else "run_registry_latest_metric",
        "metric_value": metric,
        **metric_values,
        "checkpoint_root": checkpoint_root,
        "gcs_checkpoint_path": checkpoint_root,
        "target_eval_step": _float_or_nan(row.get("target_eval_step")),
        "target_final_checkpoint_step": target_step,
        "max_checkpoint_step": max_step,
    }


def _missing_cell(method: MethodSpec, scale: ScaleSpec) -> dict[str, object]:
    return {
        "method_id": method.method_id,
        "method": method.label,
        "scale": scale.scale,
        "scale_label": scale.label,
        "x_order": SCALE_ORDER[scale.scale],
        "non_embedding_params": scale.non_embedding_params,
        "realized_train_tokens": scale.realized_train_tokens,
        "run_name": "",
        "source_experiment": "",
        "status": "missing",
        "target_budget_multiplier": TARGET_MULTIPLIER,
        "target_ready": False,
        "point_kind": "missing",
        "cell_status": "needs_launch",
        "recommended_action": "create_perplexity_only_launch_recipe",
        "metric_source": "",
        "metric_value": np.nan,
        PRIMARY_METRIC: np.nan,
        MACRO_METRIC: np.nan,
        EVAL_BPB_METRIC: np.nan,
        "checkpoint_root": "",
        "gcs_checkpoint_path": "",
        "target_eval_step": np.nan,
        "target_final_checkpoint_step": np.nan,
        "max_checkpoint_step": np.nan,
    }


def build_manifest(analysis: pd.DataFrame, registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one 25-row manifest plus plotted points."""
    manifest_rows: list[dict[str, object]] = []
    plotted_rows: list[dict[str, object]] = []

    for method in METHODS:
        for scale in SCALES:
            analysis_row = _best_analysis_row(scale, _analysis_candidates(analysis, method, scale))
            registry_row = _best_registry_row(_registry_candidates(registry, method, scale))
            row: dict[str, object]
            if registry_row is not None and _registry_row_is_ready(registry_row):
                row = _point_from_registry_row(method, scale, registry_row)
            elif analysis_row is not None:
                row = _point_from_analysis_row(method, scale, analysis_row)
            elif registry_row is not None:
                row = _point_from_registry_row(method, scale, registry_row)
            else:
                row = _missing_cell(method, scale)

            manifest_rows.append(row)
            if np.isfinite(_float_or_nan(row["metric_value"])):
                plotted_rows.append(row)

            direct_target = _direct_target_override(method, scale)
            if direct_target is not None and not bool(row["target_ready"]):
                manifest_rows[-1] = direct_target
                if np.isfinite(_float_or_nan(direct_target["metric_value"])):
                    if plotted_rows and plotted_rows[-1] == row:
                        plotted_rows[-1] = direct_target
                    else:
                        plotted_rows.append(direct_target)

    legacy_grp = _legacy_grp_60m_diagnostic()
    if legacy_grp is not None:
        is_missing_grp_60m = (
            (pd.DataFrame(manifest_rows)["method_id"] == "grp_no_l2")
            & (pd.DataFrame(manifest_rows)["scale"] == "60m_1p2b")
            & (pd.DataFrame(manifest_rows)["point_kind"] == "missing")
        )
        if bool(is_missing_grp_60m.any()):
            index = int(np.flatnonzero(is_missing_grp_60m.to_numpy())[0])
            manifest_rows[index] = legacy_grp
            plotted_rows.append(legacy_grp)

    manifest = pd.DataFrame(manifest_rows)
    if manifest.duplicated(["method_id", "scale", "target_budget_multiplier"]).any():
        duplicates = manifest.loc[
            manifest.duplicated(["method_id", "scale", "target_budget_multiplier"], keep=False),
            ["method_id", "scale", "target_budget_multiplier"],
        ]
        raise ValueError(f"Duplicate baseline scaling cells: {duplicates.to_dict(orient='records')}")
    if len(manifest) != len(METHODS) * len(SCALES):
        raise ValueError(f"Expected 25 manifest rows, found {len(manifest)}")

    plotted = pd.DataFrame(plotted_rows)
    return manifest.sort_values(["method_id", "x_order"]), plotted.sort_values(["method_id", "x_order"])


def _hover_text(row: pd.Series, metric_column: str) -> str:
    metric_value = _float_or_nan(row.get(metric_column))
    primary_value = _float_or_nan(row.get(PRIMARY_METRIC))
    checkpoint = _nonnull_string(row.get("gcs_checkpoint_path"))
    checkpoint_line = checkpoint if checkpoint else "n/a"
    primary_line = (
        ""
        if metric_column == PRIMARY_METRIC or not np.isfinite(primary_value)
        else f"{PRIMARY_METRIC}: {primary_value:.6f}<br>"
    )
    return (
        f"<b>{row['method']}</b><br>"
        f"Run: {row.get('run_name', '')}<br>"
        f"Scale key: {row['scale']}<br>"
        f"Corrected scale: {row['scale_label']}<br>"
        f"N: {int(row['non_embedding_params']):,}<br>"
        f"D: {int(row['realized_train_tokens']):,}<br>"
        f"{_metric_label(metric_column)}: {metric_value:.6f}<br>"
        f"{primary_line}"
        f"Status: {row.get('status', '')}<br>"
        f"Cell status: {row.get('cell_status', '')}<br>"
        f"Metric source: {row.get('metric_source', '')}<br>"
        f"Target eval step: {row.get('target_eval_step', '')}<br>"
        f"Max checkpoint step: {row.get('max_checkpoint_step', '')}<br>"
        f"Checkpoint: {checkpoint_line}<extra></extra>"
    )


def render_plot(points: pd.DataFrame, *, include_diagnostics: bool) -> go.Figure:
    """Create the Plotly figure for the central paper plot."""
    x_values = [scale.axis_label for scale in SCALES]
    metric_columns = _available_metric_columns(points)
    if not metric_columns:
        raise ValueError("No available perplexity metrics to plot")
    default_metric = PRIMARY_METRIC if PRIMARY_METRIC in metric_columns else metric_columns[0]

    fig = go.Figure()
    trace_metric_columns: list[str] = []
    trace_showlegend: list[bool] = []
    for metric_column in metric_columns:
        points_for_metric = points.loc[pd.to_numeric(points[metric_column], errors="coerce").notna()].copy()
        points_for_metric[metric_column] = pd.to_numeric(points_for_metric[metric_column], errors="coerce")
        for method in METHODS:
            method_points = points_for_metric.loc[points_for_metric["method_id"] == method.method_id].sort_values(
                "x_order"
            )
            ready = method_points.loc[method_points["target_ready"].astype(bool)]
            diagnostic = method_points.loc[~method_points["target_ready"].astype(bool)]
            trajectory = method_points if include_diagnostics else ready
            visible = metric_column == default_metric

            if not trajectory.empty:
                fig.add_trace(
                    go.Scatter(
                        x=[SCALE_BY_KEY[str(scale)].axis_label for scale in trajectory["scale"]],
                        y=trajectory[metric_column],
                        mode="lines",
                        name=method.label,
                        line={
                            "color": method_color(method.method_id),
                            "width": 3.6,
                            "dash": method_dash(method.method_id),
                        },
                        hoverinfo="skip",
                        visible=visible,
                        showlegend=visible,
                    )
                )
                trace_metric_columns.append(metric_column)
                trace_showlegend.append(True)

            if not ready.empty:
                fig.add_trace(
                    go.Scatter(
                        x=[SCALE_BY_KEY[str(scale)].axis_label for scale in ready["scale"]],
                        y=ready[metric_column],
                        mode="markers",
                        name=f"{method.label} target-ready",
                        marker={
                            "color": method_color(method.method_id),
                            "size": 9,
                            "symbol": "circle",
                            "line": {"color": "white", "width": 1.1},
                        },
                        hovertext=[_hover_text(row, metric_column) for _, row in ready.iterrows()],
                        hoverinfo="text",
                        visible=visible,
                        showlegend=False,
                    )
                )
                trace_metric_columns.append(metric_column)
                trace_showlegend.append(False)

            if include_diagnostics and not diagnostic.empty:
                fig.add_trace(
                    go.Scatter(
                        x=[SCALE_BY_KEY[str(scale)].axis_label for scale in diagnostic["scale"]],
                        y=diagnostic[metric_column],
                        mode="markers",
                        name=f"{method.label} diagnostic",
                        marker={
                            "color": method_color(method.method_id),
                            "size": 8,
                            "symbol": "diamond-open",
                            "line": {"width": 2.0},
                        },
                        hovertext=[_hover_text(row, metric_column) for _, row in diagnostic.iterrows()],
                        hoverinfo="text",
                        visible=visible,
                        showlegend=visible,
                    )
                )
                trace_metric_columns.append(metric_column)
                trace_showlegend.append(True)

    buttons = []
    for metric_column in metric_columns:
        visible = [trace_metric_column == metric_column for trace_metric_column in trace_metric_columns]
        buttons.append(
            {
                "label": _metric_label(metric_column),
                "method": "update",
                "args": [
                    {
                        "visible": visible,
                        "showlegend": [
                            should_show and is_visible
                            for should_show, is_visible in zip(trace_showlegend, visible, strict=True)
                        ],
                    },
                    {
                        "title": {
                            "text": (
                                "1x Chinchilla baseline scaling trajectories"
                                f"<br><sup>{_metric_label(metric_column)}</sup>"
                            ),
                            "x": 0.04,
                            "xanchor": "left",
                            "font": {"size": 24},
                        },
                        "yaxis": {
                            "title": _metric_label(metric_column),
                        },
                    },
                ],
            }
        )

    configure_interactive_layout(
        fig,
        title=f"1x Chinchilla baseline scaling trajectories<br><sup>{_metric_label(default_metric)}</sup>",
        y_title=_metric_label(default_metric),
        x_title="Scale (non-embedding params / training tokens)",
    )
    fig.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 1.0,
                "xanchor": "right",
                "y": 1.22,
                "yanchor": "top",
                "showactive": True,
                "pad": {"r": 0, "t": 0},
            }
        ],
    )
    fig.update_xaxes(categoryorder="array", categoryarray=x_values)
    return fig


def write_outputs(manifest: pd.DataFrame, points: pd.DataFrame, *, include_diagnostics: bool) -> None:
    """Write the central plot, CSV inputs, and JSON summary."""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(IMG_DIR / "baseline_scaling_trajectories_manifest.csv", index=False)
    points.to_csv(IMG_DIR / "baseline_scaling_trajectories_points.csv", index=False)
    metric_columns = _available_metric_columns(points)
    metric_cell_counts = {
        column: int(pd.to_numeric(points[column], errors="coerce").notna().sum()) for column in metric_columns
    }

    summary = {
        "expected_cells": len(METHODS) * len(SCALES),
        "manifest_rows": len(manifest),
        "target_ready_cells": int(manifest["target_ready"].astype(bool).sum()),
        "diagnostic_cells": int((manifest["point_kind"] == "diagnostic").sum()),
        "missing_cells": int((manifest["point_kind"] == "missing").sum()),
        "cell_status_counts": {
            str(key): int(value) for key, value in manifest.groupby("cell_status").size().sort_index().items()
        },
        "default_metric": PRIMARY_METRIC if PRIMARY_METRIC in metric_columns else metric_columns[0],
        "available_metrics": metric_columns,
        "metric_cell_counts": metric_cell_counts,
        "include_diagnostics": include_diagnostics,
    }
    (IMG_DIR / "baseline_scaling_trajectories_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fig = render_plot(points, include_diagnostics=include_diagnostics)
    fig.write_html(IMG_DIR / "baseline_scaling_trajectories.html", include_plotlyjs="cdn")
    static_fig = go.Figure(fig)
    configure_static_layout(
        static_fig,
        y_title=_metric_label(summary["default_metric"]),
        x_title="Scale (non-embedding params / training tokens)",
    )
    write_static_images(static_fig, IMG_DIR / "baseline_scaling_trajectories")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Hide diagnostic hollow markers. This should only be used once every cell is target-ready.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    analysis = pd.read_csv(ANALYSIS_CSV, low_memory=False)
    registry = pd.read_csv(REGISTRY_CSV, low_memory=False)
    manifest, points = build_manifest(analysis, registry)
    write_outputs(manifest, points, include_diagnostics=not args.final_only)
    display_columns = ["method", "scale_label", "cell_status", "metric_value", "recommended_action"]
    print(manifest[display_columns].to_string(index=False))
    print(f"Wrote {IMG_DIR / 'baseline_scaling_trajectories.html'}")
    print(f"Wrote {IMG_DIR / 'baseline_scaling_trajectories.png'}")


if __name__ == "__main__":
    main()
