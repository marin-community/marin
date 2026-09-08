# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a self-contained packet for ChatGPT Pro mixture-objective review."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

SCRIPT_DIR = Path("experiments/domain_phase_mix/exploratory/two_phase_many")
PACKET_ROOT = SCRIPT_DIR / "chatgpt_pro_mixture_optimization_packet_20260517"
MATRIX_DIR = SCRIPT_DIR / "metric_registry" / "raw_metric_matrix_300m"
MOE_DIR = SCRIPT_DIR / "reference_outputs" / "grug_moe_mix_dashboard_20260517"
DSP_DIR = SCRIPT_DIR / "reference_outputs" / "dsp_issue5416_aggregate_300m_20260510"
PERTURBATION_DIR = SCRIPT_DIR / "reference_outputs" / "proportional_perturbation_scale_transfer_20260507"
STANDALONE_DSP = SCRIPT_DIR / "standalone_code" / "dsp_exact.py"
ISSUE5416_CODE = SCRIPT_DIR / "metric_registry" / "issue5416_aggregate.py"

SIGNAL_MATRIX = MATRIX_DIR / "raw_metric_matrix_300m.csv"
FIXED_NOISE_MATRIX = MATRIX_DIR / "noise_baseline_run00097_fixed_subset_300m.csv"
VARIABLE_NOISE_MATRIX = MATRIX_DIR / "noise_baseline_run00097_variable_subset_300m.csv"
ISSUE5416_PROJECTION = DSP_DIR / "issue5416_projection.json"

PHASE0_PREFIX = "phase_0_"
PHASE1_PREFIX = "phase_1_"
EXPOSURE_PREFIX = "exposure_80_20_"
MIXTURE_PREFIXES = (PHASE0_PREFIX, PHASE1_PREFIX, EXPOSURE_PREFIX)
METADATA_COLUMNS = {
    "registry_run_key",
    "run_name",
    "run_id",
    "scale",
    "cohort",
    "source_cohort",
    "source_experiment",
    "checkpoint_root",
    "wandb_run_id",
    "status",
    "row_kind",
    "is_qsplit240_core",
    "noise_subset_mode",
    "noise_anchor_run_name",
    "noise_source_run_name",
    "noise_trainer_seed",
    "noise_data_seed",
    "noise_simulated_epoch_subset_seed",
    "trainer_seed",
    "data_seed",
    "simulated_epoch_subset_seed",
}
EXCLUDED_PACKET_FILENAMES = {".DS_Store", "__MACOSX", "ASK.md"}


def metric_group(metric: str) -> str:
    """Return a coarse group label for a metric path."""
    if metric.startswith("eval/uncheatable_eval/"):
        return "uncheatable"
    if metric.startswith("eval/paloma/"):
        return "paloma"
    if metric.startswith("eval/agentic_coding/"):
        return "agentic_coding"
    if metric.startswith("raw_ppl/"):
        return "raw_ppl"
    if metric.startswith("teacher_forced/gsm8k"):
        return "gsm8k_smooth"
    if metric.startswith("teacher_forced/humaneval"):
        return "humaneval_smooth"
    if metric.startswith("mcq_smooth/"):
        return "mcq_smooth"
    if metric.startswith("lm_eval/mmlu"):
        return "mmlu"
    if metric.startswith("lm_eval/gsm8k"):
        return "gsm8k_hard"
    if metric.startswith("lm_eval/humaneval"):
        return "humaneval_hard"
    if metric.startswith("lm_eval/"):
        return "lm_eval_other"
    return metric.split("/", maxsplit=1)[0]


def task_family(metric: str) -> str:
    """Return a more semantic task family for eval metrics."""
    if "arc_" in metric or "openbookqa" in metric or "sciq" in metric:
        return "arc_openbook_sciq"
    if "hellaswag" in metric or "swag" in metric:
        return "hellaswag_swag"
    if any(name in metric for name in ("boolq", "copa", "csqa", "piqa", "winogrande", "wsc273", "socialiqa")):
        return "commonsense"
    if "truthfulqa" in metric:
        return "truthfulqa"
    if "mmlu" in metric:
        return "mmlu"
    if "gsm8k" in metric:
        return "gsm8k"
    if "humaneval" in metric:
        return "humaneval"
    if "agentic_coding" in metric:
        return "agentic_coding"
    if "uncheatable_eval" in metric:
        return "uncheatable"
    if "paloma" in metric:
        return "paloma"
    return metric_group(metric)


def domain_family(domain: str) -> str:
    """Return a coarse source family for a data domain."""
    if domain.startswith("dolma3_cc/"):
        return "dolma3_cc"
    if domain.startswith("dolmino_synth_"):
        return "dolmino_synth"
    if domain.startswith("dolmino_"):
        return "dolmino_other"
    if domain.startswith("dolma3_"):
        return "dolma3_other"
    if domain.startswith("paloma/"):
        return "paloma"
    if domain.startswith("uncheatable_eval/"):
        return "uncheatable_eval"
    return domain.split("/", maxsplit=1)[0]


def ensure_clean_dir(path: Path) -> None:
    """Remove and recreate a generated directory."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def copy_file(src: Path, dst: Path) -> None:
    """Copy one file, creating the parent directory."""
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def read_projection() -> dict[str, Any]:
    return json.loads(ISSUE5416_PROJECTION.read_text())


def phase_domains(frame: pd.DataFrame) -> list[str]:
    domains = sorted(column.removeprefix(PHASE0_PREFIX) for column in frame.columns if column.startswith(PHASE0_PREFIX))
    missing = [domain for domain in domains if f"{PHASE1_PREFIX}{domain}" not in frame.columns]
    if missing:
        raise ValueError(f"Missing phase_1 columns for domains: {missing[:5]}")
    return domains


def metric_columns(frame: pd.DataFrame) -> list[str]:
    """Return raw metric columns, excluding metadata and mixture features."""
    metrics: list[str] = []
    for column in frame.columns:
        if column in METADATA_COLUMNS:
            continue
        if column.startswith(MIXTURE_PREFIXES):
            continue
        if "/" not in column:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            metrics.append(column)
    return metrics


def write_matrix_copies() -> None:
    data_dir = PACKET_ROOT / "data"
    copy_file(SIGNAL_MATRIX, data_dir / "signal_metrics_300m.csv")
    copy_file(FIXED_NOISE_MATRIX, data_dir / "noise_fixed_300m.csv")
    copy_file(VARIABLE_NOISE_MATRIX, data_dir / "noise_variable_300m.csv")


def write_domain_metadata(signal: pd.DataFrame, domains: list[str]) -> None:
    rows = []
    for domain in domains:
        exposure_col = f"{EXPOSURE_PREFIX}{domain}"
        rows.append(
            {
                "domain": domain,
                "family": domain_family(domain),
                "phase_0_column": f"{PHASE0_PREFIX}{domain}",
                "phase_1_column": f"{PHASE1_PREFIX}{domain}",
                "exposure_80_20_column": exposure_col if exposure_col in signal.columns else "",
                "signal_exposure_min": (
                    float(signal[exposure_col].min()) if exposure_col in signal.columns else float("nan")
                ),
                "signal_exposure_max": (
                    float(signal[exposure_col].max()) if exposure_col in signal.columns else float("nan")
                ),
                "signal_exposure_std": (
                    float(signal[exposure_col].std(ddof=1)) if exposure_col in signal.columns else float("nan")
                ),
            }
        )
    pd.DataFrame(rows).to_csv(PACKET_ROOT / "data" / "domain_metadata_300m.csv", index=False)


def write_mixtures_long(signal: pd.DataFrame, fixed: pd.DataFrame, variable: pd.DataFrame, domains: list[str]) -> None:
    frames = []
    for name, frame in (("signal", signal), ("noise_fixed", fixed), ("noise_variable", variable)):
        id_columns = [
            column
            for column in [
                "registry_run_key",
                "run_name",
                "run_id",
                "scale",
                "row_kind",
                "cohort",
                "source_experiment",
                "is_qsplit240_core",
                "noise_subset_mode",
            ]
            if column in frame.columns
        ]
        for phase, prefix in (
            ("phase_0", PHASE0_PREFIX),
            ("phase_1", PHASE1_PREFIX),
            ("exposure_80_20", EXPOSURE_PREFIX),
        ):
            cols = [f"{prefix}{domain}" for domain in domains if f"{prefix}{domain}" in frame.columns]
            if not cols:
                continue
            renamed = frame[id_columns + cols].melt(id_vars=id_columns, var_name="domain", value_name="weight")
            renamed["domain"] = renamed["domain"].str.removeprefix(prefix)
            renamed["phase"] = phase
            renamed["source_table"] = name
            frames.append(renamed)
    pd.concat(frames, ignore_index=True).to_csv(PACKET_ROOT / "data" / "mixtures_long_300m.csv", index=False)


def projection_frame(projection: dict[str, Any]) -> pd.DataFrame:
    columns = projection["task_columns"]
    return pd.DataFrame(
        {
            "metric": columns,
            "issue5416_sign": projection["task_signs"],
            "issue5416_projection_weight": projection["projection_vector"],
            "issue5416_mean": projection["means"],
            "issue5416_std": projection["stds"],
            "issue5416_uniqueness": projection["uniquenesses"],
            "issue5416_item_noise_share": projection["item_noise_share"],
            "metric_group": [metric_group(metric) for metric in columns],
            "task_family": [task_family(metric) for metric in columns],
        }
    )


def numeric_std(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() <= 1:
        return float("nan")
    return float(values.std(ddof=1))


def signal_stats(signal: pd.DataFrame, variable: pd.DataFrame, fixed: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        signal_values = pd.to_numeric(signal[metric], errors="coerce")
        variable_values = (
            pd.to_numeric(variable[metric], errors="coerce") if metric in variable.columns else pd.Series(dtype=float)
        )
        fixed_values = (
            pd.to_numeric(fixed[metric], errors="coerce") if metric in fixed.columns else pd.Series(dtype=float)
        )
        signal_std = numeric_std(signal_values)
        variable_std = numeric_std(variable_values)
        fixed_std = numeric_std(fixed_values)
        rows.append(
            {
                "metric": metric,
                "metric_group": metric_group(metric),
                "task_family": task_family(metric),
                "signal_n": int(signal_values.notna().sum()),
                "variable_noise_n": int(variable_values.notna().sum()),
                "fixed_noise_n": int(fixed_values.notna().sum()),
                "signal_mean": float(signal_values.mean(skipna=True)),
                "signal_std": signal_std,
                "signal_min": float(signal_values.min(skipna=True)),
                "signal_max": float(signal_values.max(skipna=True)),
                "signal_range": float(signal_values.max(skipna=True) - signal_values.min(skipna=True)),
                "variable_noise_std": variable_std,
                "fixed_noise_std": fixed_std,
                "snr_variable": (
                    signal_std / variable_std if variable_std and math.isfinite(variable_std) else float("nan")
                ),
                "snr_fixed": signal_std / fixed_std if fixed_std and math.isfinite(fixed_std) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def exposure_matrix(signal: pd.DataFrame, domains: list[str]) -> pd.DataFrame:
    exposure_cols = [f"{EXPOSURE_PREFIX}{domain}" for domain in domains]
    if all(column in signal.columns for column in exposure_cols):
        values = signal[exposure_cols].copy()
        values.columns = domains
        return values
    values = pd.DataFrame(index=signal.index)
    for domain in domains:
        values[domain] = 0.8 * signal[f"{PHASE0_PREFIX}{domain}"] + 0.2 * signal[f"{PHASE1_PREFIX}{domain}"]
    return values


def controllability_tables(
    signal: pd.DataFrame,
    exposure: pd.DataFrame,
    metric_stats: pd.DataFrame,
    metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    x_values = exposure.to_numpy(dtype=float)
    x_ranges = np.nanmax(x_values, axis=0) - np.nanmin(x_values, axis=0)
    for metric in metrics:
        y = pd.to_numeric(signal[metric], errors="coerce").to_numpy(dtype=float)
        valid_y = np.isfinite(y)
        if valid_y.sum() < 5:
            continue
        y_center = y[valid_y] - np.nanmean(y[valid_y])
        y_var = float(np.dot(y_center, y_center))
        for domain_idx, domain in enumerate(exposure.columns):
            x = x_values[:, domain_idx]
            valid = valid_y & np.isfinite(x)
            if valid.sum() < 5:
                continue
            x_center = x[valid] - np.nanmean(x[valid])
            denom = float(np.dot(x_center, x_center))
            beta = float(np.dot(x_center, y[valid] - np.nanmean(y[valid])) / denom) if denom > 0 else float("nan")
            effect = beta * float(x_ranges[domain_idx]) if math.isfinite(beta) else float("nan")
            rows.append(
                {
                    "metric": metric,
                    "domain": domain,
                    "domain_family": domain_family(domain),
                    "signal_n": int(valid.sum()),
                    "domain_exposure_range": float(x_ranges[domain_idx]),
                    "beta_metric_per_exposure": beta,
                    "effect_over_observed_exposure_range": effect,
                    "abs_effect_over_observed_exposure_range": abs(effect) if math.isfinite(effect) else float("nan"),
                    "univariate_r2": (
                        float((effect * effect) / y_var) if y_var > 0 and math.isfinite(effect) else float("nan")
                    ),
                }
            )
    per_domain = pd.DataFrame(rows)
    summary = metric_stats.copy()
    if per_domain.empty:
        return summary, per_domain
    best = (
        per_domain.sort_values("abs_effect_over_observed_exposure_range", ascending=False)
        .groupby("metric", as_index=False)
        .head(1)
        .rename(
            columns={
                "domain": "best_controllability_domain",
                "domain_family": "best_controllability_domain_family",
                "effect_over_observed_exposure_range": "max_abs_domain_effect_signed",
                "abs_effect_over_observed_exposure_range": "max_abs_domain_effect",
            }
        )
    )
    summary = summary.merge(
        best[
            [
                "metric",
                "best_controllability_domain",
                "best_controllability_domain_family",
                "max_abs_domain_effect_signed",
                "max_abs_domain_effect",
                "univariate_r2",
            ]
        ].rename(columns={"univariate_r2": "best_domain_univariate_r2"}),
        on="metric",
        how="left",
    )
    summary["controllability_variable_noise_units"] = summary["max_abs_domain_effect"] / summary["variable_noise_std"]
    summary["controllability_signal_std_units"] = summary["max_abs_domain_effect"] / summary["signal_std"]
    return summary, per_domain


def aggregate_metric_correlations(signal: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    target_path = DSP_DIR / "target_scores.csv"
    if not target_path.exists():
        return pd.DataFrame()
    target = pd.read_csv(target_path)[["run_name", "issue5416_aggregate"]]
    frame = signal.merge(target, on="run_name", how="inner")
    rows = []
    target_values = pd.to_numeric(frame["issue5416_aggregate"], errors="coerce")
    for metric in metrics:
        values = pd.to_numeric(frame[metric], errors="coerce")
        valid = target_values.notna() & values.notna()
        if valid.sum() < 5:
            continue
        rows.append(
            {
                "metric": metric,
                "metric_group": metric_group(metric),
                "task_family": task_family(metric),
                "n": int(valid.sum()),
                "pearson_with_issue5416_aggregate": float(values[valid].corr(target_values[valid], method="pearson")),
                "spearman_with_issue5416_aggregate": float(values[valid].corr(target_values[valid], method="spearman")),
            }
        )
    return pd.DataFrame(rows)


def design_matrix_diagnostics(exposure: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    x = exposure.to_numpy(dtype=float)
    x_centered = x - x.mean(axis=0, keepdims=True)
    x_std = x_centered / np.where(x_centered.std(axis=0, ddof=1) == 0, 1.0, x_centered.std(axis=0, ddof=1))
    # The full matrix is simplex-constrained, so also report a reference-dropped condition number.
    dropped = x_std[:, :-1]
    singular_full = np.linalg.svd(x_std, compute_uv=False)
    singular_dropped = np.linalg.svd(dropped, compute_uv=False)
    corr = pd.DataFrame(x_std, columns=exposure.columns).corr()
    pairs = []
    domains = list(exposure.columns)
    for i, left in enumerate(domains):
        for j in range(i + 1, len(domains)):
            right = domains[j]
            value = float(corr.loc[left, right])
            if abs(value) >= 0.95:
                pairs.append({"domain_a": left, "domain_b": right, "pearson_corr": value, "abs_corr": abs(value)})
    vif_rows = []
    for i, domain in enumerate(domains[:-1]):
        y = dropped[:, i]
        others = np.delete(dropped, i, axis=1)
        coef, *_ = np.linalg.lstsq(others, y, rcond=None)
        pred = others @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        vif_rows.append(
            {"domain": domain, "reference_dropped_vif": 1.0 / max(1.0 - r2, 1e-12), "r2_on_other_domains": r2}
        )
    diagnostics = {
        "n_rows": int(x.shape[0]),
        "n_domains": int(x.shape[1]),
        "simplex_collinearity_warning": "All domain exposures sum to one; full design matrix is rank-deficient.",
        "full_singular_values": singular_full.tolist(),
        "reference_dropped_singular_values": singular_dropped.tolist(),
        "full_condition_number": float(singular_full.max() / max(singular_full.min(), 1e-12)),
        "reference_dropped_condition_number": float(singular_dropped.max() / max(singular_dropped.min(), 1e-12)),
        "vif_reference_domain_dropped": domains[-1],
        "vif_top10": sorted(vif_rows, key=lambda row: row["reference_dropped_vif"], reverse=True)[:10],
        "num_abs_corr_ge_0p95_pairs": len(pairs),
    }
    return diagnostics, pd.DataFrame(pairs).sort_values("abs_corr", ascending=False) if pairs else pd.DataFrame()


def write_moe_artifacts() -> None:
    data_dir = PACKET_ROOT / "data"
    diagnostics_dir = PACKET_ROOT / "diagnostics"
    for src_name, dst_name in [
        ("grug_moe_mix_weights_long.csv", "recovered_moe_4track_mixtures_long.csv"),
        ("grug_moe_mix_eval_metrics_long.csv", "moe_4track_eval_metrics_long.csv"),
        ("grug_moe_mix_task_loss_like_metrics.csv", "moe_4track_task_loss_like_metrics.csv"),
        ("grug_moe_mix_cell_summary.csv", "moe_4track_cell_summary.csv"),
        ("grug_moe_mix_runs.csv", "moe_4track_runs.csv"),
    ]:
        copy_file(MOE_DIR / src_name, data_dir / dst_name)
    for src_name in [
        "grug_moe_mix_task_powerlaw_fits.csv",
        "grug_moe_mix_accuracy_summary.csv",
        "grug_moe_mix_preferred_task_metrics.csv",
    ]:
        copy_file(MOE_DIR / src_name, diagnostics_dir / src_name)
    if (MOE_DIR / "dashboard.html").exists():
        copy_file(MOE_DIR / "dashboard.html", PACKET_ROOT / "moe_dashboard" / "dashboard.html")
    for plot in [
        "task_loss_scaling_loglog.html",
        "aggregate_scaling.html",
        "mixture_full_domain_heatmap.html",
        "task_metric_facets.html",
    ]:
        src = MOE_DIR / "img" / plot
        if src.exists():
            copy_file(src, PACKET_ROOT / "moe_dashboard" / "img" / plot)


def moe_exposure_table() -> pd.DataFrame:
    weights = pd.read_csv(MOE_DIR / "grug_moe_mix_weights_long.csv")
    runs = pd.read_csv(MOE_DIR / "grug_moe_mix_runs.csv")
    rows = []
    for run_id, group in weights.groupby("run_id"):
        run = runs[runs["run_id"].eq(run_id)].iloc[0].to_dict()
        phase_names = set(group["phase"])
        domains = sorted(group["domain"].unique())
        if phase_names == {"constant"}:
            exposure = group.set_index("domain")["weight"].to_dict()
        else:
            phase1_boundary = float(group[group["phase"].eq("phase_1")]["boundary"].iloc[0])
            target_steps = float(run["target_steps"])
            phase0_frac = phase1_boundary / target_steps
            phase1_frac = 1.0 - phase0_frac
            p0 = group[group["phase"].eq("phase_0")].set_index("domain")["weight"].to_dict()
            p1 = group[group["phase"].eq("phase_1")].set_index("domain")["weight"].to_dict()
            exposure = {
                domain: phase0_frac * p0.get(domain, 0.0) + phase1_frac * p1.get(domain, 0.0) for domain in domains
            }
        for domain, value in exposure.items():
            rows.append(
                {
                    "run_id": run_id,
                    "track": run["track"],
                    "track_label": run["track_label"],
                    "hidden_dim": run["hidden_dim"],
                    "budget": run["budget"],
                    "domain": domain,
                    "family": domain_family(domain),
                    "exposure_weight": value,
                }
            )
    return pd.DataFrame(rows)


def write_moe_coverage() -> None:
    exposure = moe_exposure_table()
    exposure.to_csv(PACKET_ROOT / "data" / "recovered_moe_4track_exposure_approx_long.csv", index=False)
    wide = exposure.pivot_table(
        index=["track", "track_label", "hidden_dim", "budget", "run_id"],
        columns="domain",
        values="exposure_weight",
        fill_value=0.0,
        observed=False,
    )
    rows = []
    keys = list(wide.index)
    values = wide.to_numpy(dtype=float)
    for i, key_a in enumerate(keys):
        for j in range(i + 1, len(keys)):
            key_b = keys[j]
            tv = 0.5 * float(np.abs(values[i] - values[j]).sum())
            rows.append(
                {
                    "run_id_a": key_a[4],
                    "track_label_a": key_a[1],
                    "hidden_dim_a": key_a[2],
                    "run_id_b": key_b[4],
                    "track_label_b": key_b[1],
                    "hidden_dim_b": key_b[2],
                    "approx_exposure_tv": tv,
                }
            )
    pd.DataFrame(rows).to_csv(PACKET_ROOT / "diagnostics" / "moe_track_pairwise_tv.csv", index=False)


def fit_r2(y: np.ndarray, x: np.ndarray) -> float:
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    if valid.sum() <= x.shape[1]:
        return float("nan")
    coef, *_ = np.linalg.lstsq(x[valid], y[valid], rcond=None)
    pred = x[valid] @ coef
    ss_res = float(np.sum((y[valid] - pred) ** 2))
    ss_tot = float(np.sum((y[valid] - y[valid].mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def write_moe_scale_transfer_diagnostics() -> None:
    path = MOE_DIR / "grug_moe_mix_task_loss_like_metrics.csv"
    if not path.exists():
        return
    data = pd.read_csv(path)
    data = data[data["common_task"] & data["loss_value"].gt(0)].copy()
    rows = []
    for task, group in data.groupby("task_alias"):
        group = group.copy()
        group["log_budget"] = np.log(pd.to_numeric(group["budget"], errors="coerce"))
        group["log_loss"] = np.log(pd.to_numeric(group["loss_value"], errors="coerce"))
        track_dummies = pd.get_dummies(group["track_label"], drop_first=True, dtype=float)
        scale_x = np.column_stack([np.ones(len(group)), group["log_budget"].to_numpy(dtype=float)])
        track_x = np.column_stack([scale_x, track_dummies.to_numpy(dtype=float)])
        interaction_terms = track_dummies.to_numpy(dtype=float) * group["log_budget"].to_numpy(dtype=float)[:, None]
        interaction_x = np.column_stack([track_x, interaction_terms])
        y = group["log_loss"].to_numpy(dtype=float)
        scale_r2 = fit_r2(y, scale_x)
        track_r2 = fit_r2(y, track_x)
        interaction_r2 = fit_r2(y, interaction_x)
        largest_hidden = group["hidden_dim"].max()
        train = group["hidden_dim"].lt(largest_hidden)
        test = group["hidden_dim"].eq(largest_hidden)
        leave_largest_mae_log = float("nan")
        leave_largest_n = int(test.sum())
        if train.sum() > track_x.shape[1] and test.sum() > 0:
            train_tracks = pd.get_dummies(group.loc[train, "track_label"], drop_first=True, dtype=float)
            train_cols = list(train_tracks.columns)
            x_train = np.column_stack(
                [
                    np.ones(train.sum()),
                    group.loc[train, "log_budget"].to_numpy(dtype=float),
                    train_tracks.to_numpy(dtype=float),
                ]
            )
            coef, *_ = np.linalg.lstsq(x_train, group.loc[train, "log_loss"].to_numpy(dtype=float), rcond=None)
            test_tracks = pd.get_dummies(group.loc[test, "track_label"], dtype=float)
            for col in train_cols:
                if col not in test_tracks.columns:
                    test_tracks[col] = 0.0
            x_test = np.column_stack(
                [
                    np.ones(test.sum()),
                    group.loc[test, "log_budget"].to_numpy(dtype=float),
                    test_tracks[train_cols].to_numpy(dtype=float),
                ]
            )
            pred = x_test @ coef
            leave_largest_mae_log = float(np.mean(np.abs(group.loc[test, "log_loss"].to_numpy(dtype=float) - pred)))
        taus = []
        for left, right in zip(
            sorted(group["hidden_dim"].unique())[:-1], sorted(group["hidden_dim"].unique())[1:], strict=False
        ):
            left_rank = group[group["hidden_dim"].eq(left)].set_index("track_label")["loss_value"].rank(ascending=True)
            right_rank = group[group["hidden_dim"].eq(right)].set_index("track_label")["loss_value"].rank(ascending=True)
            common = left_rank.index.intersection(right_rank.index)
            if len(common) >= 3:
                tau = kendalltau(left_rank.loc[common], right_rank.loc[common]).statistic
                if math.isfinite(tau):
                    taus.append(float(tau))
        rows.append(
            {
                "task_alias": task,
                "loss_metric": group["loss_metric"].mode().iloc[0],
                "n_cells": len(group),
                "n_tracks": int(group["track_label"].nunique()),
                "n_scales": int(group["hidden_dim"].nunique()),
                "scale_only_log_r2": scale_r2,
                "scale_plus_track_intercept_log_r2": track_r2,
                "track_partial_r2_after_scale": (
                    (track_r2 - scale_r2) / max(1.0 - scale_r2, 1e-12)
                    if math.isfinite(scale_r2) and math.isfinite(track_r2)
                    else float("nan")
                ),
                "scale_track_interaction_log_r2": interaction_r2,
                "interaction_partial_r2_after_track": (
                    (interaction_r2 - track_r2) / max(1.0 - track_r2, 1e-12)
                    if math.isfinite(track_r2) and math.isfinite(interaction_r2)
                    else float("nan")
                ),
                "leave_largest_scale_mae_log": leave_largest_mae_log,
                "leave_largest_scale_n": leave_largest_n,
                "adjacent_scale_kendall_tau_mean": float(np.mean(taus)) if taus else float("nan"),
            }
        )
    pd.DataFrame(rows).to_csv(PACKET_ROOT / "diagnostics" / "moe_scale_transfer_per_metric.csv", index=False)


def write_current_aggregate_artifacts(projection: dict[str, Any]) -> None:
    aggregate_dir = PACKET_ROOT / "current_aggregate"
    copy_file(ISSUE5416_PROJECTION, aggregate_dir / "issue5416_projection.json")
    copy_file(ISSUE5416_CODE, aggregate_dir / "issue5416_aggregate.py")
    copy_file(DSP_DIR / "target_scores.csv", aggregate_dir / "target_scores.csv")
    projection_frame(projection).to_csv(aggregate_dir / "issue5416_projection_items.csv", index=False)
    (aggregate_dir / "known_issues.md").write_text(
        """# Known Issues: Issue #5416 Aggregate

- The current projection is learned from the same 300M signal matrix later used by DSP fits. Treat this as a known co-adaptation risk unless you implement metric-row or mixture-row split validation.
- The projection is a signed factor score, not a direct hard-accuracy mean. It is intended to denoise smooth task proxies and BPB-style metrics.
- The Grug-MoE dashboard aggregate is a separate diagnostic equal-weight z-score aggregate and should not be conflated with this issue #5416 projection.
- Hard accuracy metrics are mostly reporting/validation signals, not primary optimization targets.
""",
    )


def write_current_dsp_artifacts() -> None:
    dsp_packet = PACKET_ROOT / "current_dsp"
    copy_file(STANDALONE_DSP, dsp_packet / "dsp_exact.py")
    copy_file(SCRIPT_DIR / "fit_dsp_issue5416_aggregate_300m.py", dsp_packet / "fit_dsp_issue5416_aggregate_300m.py")
    for name in ["summary.csv", "report.md", "raw_optima_variant_weights_long.csv", "raw_optima_variant_grid.png"]:
        src = DSP_DIR / name
        if src.exists():
            copy_file(src, dsp_packet / name)
    for variant_dir in sorted(path for path in DSP_DIR.iterdir() if path.is_dir() and path.name.startswith("dsp_")):
        for file_name in [
            "params.json",
            "raw_optimum_weights.csv",
            "fit_trace.csv",
            "predicted_vs_actual_issue5416.png",
            "raw_optimum_mixture.png",
        ]:
            src = variant_dir / file_name
            if src.exists():
                copy_file(src, dsp_packet / "variants" / variant_dir.name / file_name)
    (dsp_packet / "known_issues.md").write_text(
        """# Known Issues: DSP Fits

- Raw simplex optima are diagnostic until validated; several variants find off-manifold or sparse optima.
- The canonical collaborator-facing implementation is included as `dsp_exact.py`, but packet reviewers should compare variants rather than assume one form is settled.
- Nonnegative benefit and penalty heads encode a substantive assumption: domains help until overexposed. Reviewers should test whether signed or stronger-regularized variants change conclusions.
- Fit summaries should be read with parameter count, row count, and nearest-observed TV diagnostics.
""",
    )


def write_perturbation_artifacts() -> None:
    if not PERTURBATION_DIR.exists():
        return
    dst = PACKET_ROOT / "diagnostics" / "proportional_perturbation"
    for name in [
        "dsp_domain_perturbation_agreement.csv",
        "dsp_domain_perturbation_agreement_summary.csv",
        "dsp_scale_specific_domain_perturbation_predictions.csv",
        "dsp_three_vector_alignment_100m.csv",
        "dsp_three_vector_alignment_100m_long.csv",
    ]:
        src = PERTURBATION_DIR / name
        if src.exists():
            copy_file(src, dst / name)


def stable_hash_bucket(value: str, modulo: int = 100) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def write_splits(signal: pd.DataFrame, projection: dict[str, Any]) -> None:
    splits_dir = PACKET_ROOT / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    qsplit = signal[signal.get("is_qsplit240_core", False).astype(bool)].copy()
    holdout = sorted(name for name in qsplit["run_name"].astype(str) if stable_hash_bucket(name) < 15)
    train = sorted(name for name in qsplit["run_name"].astype(str) if name not in holdout)
    baselines = sorted(signal[~signal["run_name"].astype(str).str.startswith("run_")]["run_name"].astype(str).unique())
    (splits_dir / "mixture_train_vs_holdout.json").write_text(
        json.dumps(
            {
                "description": "Deterministic proposal only; not used by current DSP summaries.",
                "hash_rule": "sha256(run_name) first 32 bits modulo 100; <15 goes to holdout",
                "train_qsplit_core_run_names": train,
                "holdout_qsplit_core_run_names": holdout,
                "baseline_eval_only_run_names": baselines,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    metric_rows = []
    for metric in projection["task_columns"]:
        family = task_family(metric)
        split = "validation" if family in {"truthfulqa", "gsm8k", "humaneval"} else "calibration"
        metric_rows.append({"metric": metric, "task_family": family, "proposed_split": split})
    pd.DataFrame(metric_rows).to_csv(splits_dir / "metric_calibration_vs_validation.csv", index=False)
    (splits_dir / "README.md").write_text(
        """# Proposed Splits

These are proposed validation splits for ChatGPT Pro to critique, not splits already used in the published summaries.

- `mixture_train_vs_holdout.json`: deterministic 85/15 qsplit-core split plus baselines kept as evaluation-only anchors.
- `metric_calibration_vs_validation.csv`: a first-pass metric split that holds out TruthfulQA, GSM8K, and HumanEval families from aggregate calibration.
- Scale validation is represented by `diagnostics/moe_scale_transfer_per_metric.csv` using the Grug-MoE scaling tracks.
""",
    )


def write_docs(signal: pd.DataFrame, fixed: pd.DataFrame, variable: pd.DataFrame, metrics: list[str]) -> None:
    readme = f"""# ChatGPT Pro Mixture Optimization Packet

Generated: {datetime.now(UTC).isoformat()}

This packet is for reviewing how to construct a reliable aggregate objective for data-mixture optimization.

## Goal

Develop a data mixture that improves as many benchmark/eval metrics as possible, as much as possible, while accounting for:

- metric noise from trainer/data-subset randomness,
- weak controllability under the current partition,
- task tradeoffs against broad-language proportional baselines,
- extrapolation risk from optimizing a fitted DSP surrogate.

## Primary Estimand

Let \\(w = (w^{{(0)}}, w^{{(1)}})\\) be two-phase domain weights. We want an objective \\(A(w)\\) such that optimizing a DSP-style surrogate for \\(A\\) tends to improve broad downstream performance:

\\[
\\max_w \\; \\mathbb{{E}}[A(w) \\mid \\text{{training scale and budget}}]
\\]

The current candidate \\(A\\) is the issue #5416 signed factor aggregate. This packet asks you to critique and improve that objective.

## Row Counts

- 300M signal matrix rows: {len(signal)}
- fixed-subset noise rows: {len(fixed)}
- variable-subset noise rows: {len(variable)}
- candidate raw metric columns: {len(metrics)}

## Read First

The prompt to paste into ChatGPT Pro is generated next to this archive as `ASK.md`.
It is intentionally excluded from the zip so the archive remains a data/code attachment.

Inside the archive, read:

1. `PROVENANCE.md`
2. `diagnostics/per_metric_summary.csv`
3. `current_aggregate/known_issues.md`
4. `current_dsp/known_issues.md`
"""
    ask = """# Prompt for ChatGPT Pro

You are helping us reason about data-mixture optimization for language-model pretraining. The attached zip is a self-contained packet with the current 300M/6B mixture-swarm matrix, noise baselines, current aggregate target, current DSP surrogate implementation, local perturbation diagnostics, and a recovered 4-track MoE scaling dashboard.

## Background

Our practical goal is to choose data mixtures that improve broad downstream behavior, not just one perplexity metric. The training setup is a two-phase domain mixture: each run has phase-0 and phase-1 weights over 39 data domains, and the current main panel is a 242-row randomized 300M/6B swarm. We also have 10 fixed-subset and 10 variable-subset noise repeats around one reference mixture, so we can estimate how noisy each metric is under trainer/data-subset randomness.

The core modeling question is whether we can build an aggregate objective that is smooth, controllable under the current domain partition, and aligned with broad task performance. We have had strong results fitting single BPB targets with GRP/DSP-style functional forms, but direct fitting to noisy task metrics is much less stable. The current candidate target is the issue #5416 signed low-rank/factor aggregate, intended to denoise a basket of task-relevant metrics before fitting DSP.

Several difficulties matter for your analysis:

- Many hard task metrics are noisy enough that a true improvement over proportional may require repeated measurements to verify.
- Some metrics may be weakly controllable under the current partition, because the sampler only controls domain-level mass and not within-domain selection.
- Some metrics trade off against each other, so a single aggregate can hide regressions.
- Raw surrogate optima can be off the observed mixture manifold, so trust-region, top-hull, or sampled-mixture optimization may be safer than unconstrained argmax.
- Scaling evidence is limited. The recovered Grug-MoE 4-track ladder is useful, but observational rather than a randomized design.

## What Is In The Zip

- `data/signal_metrics_300m.csv`: 242 signal rows with mixtures and metrics.
- `data/noise_fixed_300m.csv` and `data/noise_variable_300m.csv`: noise repeat panels.
- `data/mixtures_long_300m.csv` and `data/domain_metadata_300m.csv`: domain weights and metadata.
- `diagnostics/per_metric_summary.csv`: per-metric noise, range, SNR, controllability, and issue #5416 membership.
- `diagnostics/controllability_per_metric_per_domain.csv`: per-domain local linear diagnostic table.
- `diagnostics/moe_scale_transfer_per_metric.csv`: scaling regularity diagnostics from recovered Grug-MoE tracks.
- `current_aggregate/`: current issue #5416 aggregate projection and code.
- `current_dsp/`: standalone DSP implementation, current DSP fitting script, fit summaries, and raw optimum artifacts.
- `moe_dashboard/`: recovered Grug-MoE scaling dashboard artifacts.

## Deliverables

Please produce six concrete deliverables.

1. Aggregate critique: evaluate the issue #5416 projection, including sign choices, factor count, noise anchoring, factor weighting, and whether the current metric basket is appropriate.
2. Revised estimator: propose a robust aggregate plus DSP fitting procedure with explicit metric validation, mixture validation, and scale validation.
3. Controllability and noise report: classify metric groups as optimize, guardrail, report-only, or exclude. Use both variable-subset noise and empirical controllability.
4. Scale-transfer report: use the Grug-MoE tracks to identify which metrics scale predictably, which are noisy/floor-dominated, and which show possible mixture-scale interaction.
5. DSP/optimization report: critique the current DSP fit and optimum uncertainty. Recommend whether raw argmax, trust-region optimization, top-hull optimization, sampled-mixture search, or another method should be used.
6. Validation recommendations: propose up to three mixtures to train next, each with a falsifiable predicted improvement, expected failure mode, and measurement plan.

## Constraints

- Treat Grug-MoE 4-track mixtures as observational recovered mixtures, not a randomized design.
- Do not propose more than three candidate mixtures.
- If the data are insufficient for a deliverable, state the minimum additional data needed.
- Prefer theoretically motivated objectives over generic black-box regression.
- Do not assume hard accuracy metrics are directly optimizable; account for noise and controllability.
- Separate what can be concluded from the existing data from what would require a new randomized experiment.
"""
    provenance = """# Provenance

## Reproducible Sources

- 300M raw metric matrix and noise rows come from `metric_registry/raw_metric_matrix_300m/`.
- Issue #5416 projection comes from `reference_outputs/dsp_issue5416_aggregate_300m_20260510/issue5416_projection.json`.
- DSP code is included as standalone `current_dsp/dsp_exact.py`.
- Proportional perturbation diagnostics come from `reference_outputs/proportional_perturbation_scale_transfer_20260507/`.

## Observational / Recovered Sources

- Grug-MoE 4-track mixture weights were recovered from GCS executor metadata and are included in `data/recovered_moe_4track_mixtures_long.csv`.
- The branch code does not currently reproduce the non-proportional `v2/v3/v4` mixture generation procedure.
- As of this packet, `v4 d1536` training is complete but the logprob eval was missing from dashboard inputs; a new eval was submitted separately. Rerun the packet after that eval completes for full 20/20 eval-cell coverage.

## Aggregate Definitions

- `current_aggregate/issue5416_projection.json`: current optimization target candidate.
- `moe_dashboard` aggregate: diagnostic equal-weight common-task z-score aggregate, not the same as issue #5416.
"""
    known = """# Known Methodological Risks

- The current issue #5416 projection and DSP fit use overlapping rows, which can induce target/model co-adaptation.
- Some metrics are low-SNR, weakly controllable, or both. These should not be optimized directly without denoising or guardrail logic.
- Grug-MoE tracks are observational and may be selected by an unknown prior optimizer/generator.
- Raw DSP optima can be off-manifold; nearest-observed TV and bootstrap/trust-region diagnostics matter.
- Scaling diagnostics with only four tracks have low statistical power. Treat them as evidence, not proof.
"""
    (PACKET_ROOT / "README.md").write_text(readme)
    (PACKET_ROOT / "ASK.md").write_text(ask)
    (PACKET_ROOT / "PROVENANCE.md").write_text(provenance)
    (PACKET_ROOT / "KNOWN_METHODOLOGICAL_RISKS.md").write_text(known)


def copy_required_artifacts() -> None:
    write_moe_artifacts()
    write_perturbation_artifacts()


def write_manifest_and_zip() -> None:
    files = []
    for path in sorted(PACKET_ROOT.rglob("*")):
        if not path.is_file() or path.suffix == ".zip":
            continue
        if path.name in EXCLUDED_PACKET_FILENAMES:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(
            {
                "path": path.relative_to(PACKET_ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    (PACKET_ROOT / "MANIFEST.json").write_text(
        json.dumps(
            {"packet": PACKET_ROOT.name, "generated_at_utc": datetime.now(UTC).isoformat(), "files": files}, indent=2
        )
        + "\n"
    )
    archive_path = PACKET_ROOT.with_suffix(".zip")
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(PACKET_ROOT.rglob("*")):
            if path.name in EXCLUDED_PACKET_FILENAMES:
                continue
            if path.is_file():
                archive.write(path, path.relative_to(PACKET_ROOT.parent))
    print(f"Wrote archive: {archive_path}")


def main() -> None:
    ensure_clean_dir(PACKET_ROOT)
    for child in ["data", "diagnostics", "current_aggregate", "current_dsp", "splits", "moe_dashboard"]:
        (PACKET_ROOT / child).mkdir(parents=True, exist_ok=True)

    signal = pd.read_csv(SIGNAL_MATRIX)
    fixed = pd.read_csv(FIXED_NOISE_MATRIX)
    variable = pd.read_csv(VARIABLE_NOISE_MATRIX)
    domains = phase_domains(signal)
    metrics = metric_columns(signal)
    projection = read_projection()

    write_matrix_copies()
    write_domain_metadata(signal, domains)
    write_mixtures_long(signal, fixed, variable, domains)
    exposure = exposure_matrix(signal, domains)

    stats = signal_stats(signal, variable, fixed, metrics)
    stats, per_domain = controllability_tables(signal, exposure, stats, metrics)
    projection_items = projection_frame(projection)
    stats = stats.merge(
        projection_items[["metric", "issue5416_sign", "issue5416_projection_weight"]],
        on="metric",
        how="left",
    )
    stats["in_issue5416_projection"] = stats["issue5416_projection_weight"].notna()
    stats.to_csv(PACKET_ROOT / "diagnostics" / "per_metric_summary.csv", index=False)
    per_domain.to_csv(PACKET_ROOT / "diagnostics" / "controllability_per_metric_per_domain.csv", index=False)
    aggregate_metric_correlations(signal, metrics).to_csv(
        PACKET_ROOT / "diagnostics" / "aggregate_vs_metric_correlations.csv", index=False
    )
    design_diagnostics, confounded_pairs = design_matrix_diagnostics(exposure)
    (PACKET_ROOT / "diagnostics" / "design_matrix_diagnostics.json").write_text(
        json.dumps(design_diagnostics, indent=2, sort_keys=True) + "\n"
    )
    confounded_pairs.to_csv(
        PACKET_ROOT / "diagnostics" / "design_confounded_domain_pairs_abs_corr_ge_0p95.csv", index=False
    )

    copy_required_artifacts()
    write_moe_coverage()
    write_moe_scale_transfer_diagnostics()
    write_current_aggregate_artifacts(projection)
    write_current_dsp_artifacts()
    write_splits(signal, projection)
    write_docs(signal, fixed, variable, metrics)
    write_manifest_and_zip()
    print(f"Built packet: {PACKET_ROOT}")


if __name__ == "__main__":
    main()
