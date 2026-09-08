#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test domain-aware scaling-law synergy terms on the ND mixture panel.

This is a local analysis script. It adapts the "Domain-Aware Scaling Laws
Uncover Data Synergy" first- and second-order terms to our anchored
mixture/scale setup and tests whether they explain residual errors in
`eval/uncheatable_eval/bpb`.

The paper's first-order term uses z_k = u_k log(u_k D). In an anchored law,
holding mixture fixed and subtracting the D0 anchor makes this equivalent to a
mixture-specific data-scaling head, u_k log(D/D0), up to the paper's
sum-to-zero gamma identifiability constraint. The second-order term uses a
soft-min interaction between log(1 + u_k D) and log(1 + u_l D), which vanishes
when either source has zero mass. We test these terms at domain and source-group
levels, both as standalone anchored models and as residual corrections on top
of the current canonical MCT-LRQ predictions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import run_threshold_gate_joint_law_sprint_20260507 as base

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "analysis_dataset" / "nd_scale_runs.csv"
CANONICAL_SPLIT_FILE = (
    ROOT
    / "reference_outputs"
    / "joint_model_refreshed_20260426"
    / "mct_lrq_no_barrier_canonical"
    / "csv"
    / "row_predictions.csv"
)
OUTPUT_DIR = ROOT / "reference_outputs" / "domain_aware_synergy_joint_law_20260510"
TARGET = "eval/uncheatable_eval/bpb"

RIDGE_ANCHOR = 1e-4
RIDGE_GRID = tuple(10.0**power for power in range(-8, 5))
EPS = 1e-12


@dataclass(frozen=True)
class SynergySpec:
    name: str
    description: str
    feature_builder: str
    group_fn_name: str | None = None
    token_transform: str = "power"
    include_entropy: bool = False
    include_first_order: bool = False
    include_second_order: bool = False
    second_order_tau: float = 0.1


@dataclass
class FittedSynergyModel:
    spec: SynergySpec
    domains: list[str]
    anchor_feature_names: list[str]
    anchor_coef: np.ndarray
    anchor_mean: np.ndarray
    anchor_std: np.ndarray
    scale_feature_names: list[str]
    feature_scale: np.ndarray
    scale_coef: np.ndarray
    ridge: float


def _stable_hash_fraction(value: str) -> float:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _effective_exposure(w0: np.ndarray, w1: np.ndarray) -> np.ndarray:
    return 0.8 * w0 + 0.2 * w1


def _token_factor(d: np.ndarray, transform: str) -> np.ndarray:
    if transform == "power":
        return (d / base.D0) ** (-base.BETA) - 1.0
    if transform == "log":
        return np.log(np.clip(d, EPS, None) / base.D0)
    raise ValueError(f"Unknown token transform: {transform}")


def _softmin(a: np.ndarray, b: np.ndarray, tau: float) -> np.ndarray:
    values = np.stack([-a / tau, -b / tau], axis=0)
    max_value = np.max(values, axis=0)
    return -tau * (max_value + np.log(np.exp(values[0] - max_value) + np.exp(values[1] - max_value)))


def _row_center_features(features: np.ndarray) -> np.ndarray:
    if features.shape[1] == 0:
        return features
    return features - features.mean(axis=1, keepdims=True)


def _entropy(exposure: np.ndarray) -> np.ndarray:
    clipped = np.clip(exposure, EPS, None)
    return -(clipped * np.log(clipped)).sum(axis=1, keepdims=True)


def _group_shares(
    exposure: np.ndarray,
    domains: list[str],
    group_fn_name: str,
) -> tuple[list[str], np.ndarray]:
    groups, group_map = base.group_matrix(domains, group_fn_name)
    return groups, exposure @ group_map


def _first_order_features(
    exposure: np.ndarray,
    d: np.ndarray,
    names: list[str],
    token_transform: str,
) -> tuple[list[str], np.ndarray]:
    token = _token_factor(d, token_transform)
    centered_share = exposure - 1.0 / exposure.shape[1]
    features = token[:, None] * centered_share
    return [f"paper_first_order_{token_transform}:{name}" for name in names], features


def _entropy_feature(exposure: np.ndarray, d: np.ndarray, token_transform: str) -> tuple[list[str], np.ndarray]:
    token = _token_factor(d, token_transform)
    entropy = _entropy(exposure)
    return [f"paper_entropy_{token_transform}"], token[:, None] * entropy


def _second_order_features(
    exposure: np.ndarray,
    d: np.ndarray,
    names: list[str],
    tau: float,
) -> tuple[list[str], np.ndarray]:
    feature_names: list[str] = []
    columns: list[np.ndarray] = []
    z = np.log1p(np.clip(exposure, 0.0, None) * d[:, None])
    z_anchor = np.log1p(np.clip(exposure, 0.0, None) * base.D0)
    for left in range(exposure.shape[1]):
        for right in range(left + 1, exposure.shape[1]):
            pair = _softmin(z[:, left], z[:, right], tau) - _softmin(z_anchor[:, left], z_anchor[:, right], tau)
            columns.append(pair)
            feature_names.append(f"paper_second_order_tau{tau:g}:{names[left]}__{names[right]}")
    if not columns:
        return [], np.zeros((len(d), 0), dtype=float)
    return feature_names, _row_center_features(np.column_stack(columns))


def _synergy_features(
    spec: SynergySpec,
    w0: np.ndarray,
    w1: np.ndarray,
    n: np.ndarray,
    d: np.ndarray,
    domains: list[str],
) -> tuple[list[str], np.ndarray]:
    _, canonical_family_map = base.group_matrix(domains, "canonical")
    scale_names, scale_x = base.scale_features(w0, w1, n, d, domains, canonical_family_map)
    names = list(scale_names)
    blocks = [scale_x]
    exposure = _effective_exposure(w0, w1)

    if spec.feature_builder == "baseline":
        return names, np.hstack(blocks)

    if spec.group_fn_name is None:
        share_names = domains
        shares = exposure
    else:
        share_names, shares = _group_shares(exposure, domains, spec.group_fn_name)

    if spec.include_first_order:
        first_names, first_x = _first_order_features(shares, d, share_names, spec.token_transform)
        names.extend(first_names)
        blocks.append(first_x)

    if spec.include_entropy:
        entropy_names, entropy_x = _entropy_feature(shares, d, spec.token_transform)
        names.extend(entropy_names)
        blocks.append(entropy_x)

    if spec.include_second_order:
        second_names, second_x = _second_order_features(shares, d, share_names, spec.second_order_tau)
        names.extend(second_names)
        blocks.append(second_x)

    return names, np.hstack(blocks)


def build_specs() -> list[SynergySpec]:
    specs = [
        SynergySpec(
            name="smooth_mct_lrq_like",
            description="Existing anchored LRQ mixture body plus global N, canonical family D, and ND cross terms.",
            feature_builder="baseline",
        ),
        SynergySpec(
            name="paper_first_order_domain_power",
            description="Paper first-order domain synergy adapted as centered domain-specific D-scaling heads.",
            feature_builder="paper",
            token_transform="power",
            include_first_order=True,
        ),
        SynergySpec(
            name="paper_first_order_domain_log",
            description="Same as first-order domain, using log(D/D0) exactly from z_k(D)-z_k(D0).",
            feature_builder="paper",
            token_transform="log",
            include_first_order=True,
        ),
        SynergySpec(
            name="paper_first_order_current_source_power",
            description="First-order synergy on source-aware groups instead of all 39 domains.",
            feature_builder="paper",
            group_fn_name="current_source",
            token_transform="power",
            include_first_order=True,
        ),
        SynergySpec(
            name="paper_first_order_current_source_log",
            description="Source-aware first-order synergy using log(D/D0).",
            feature_builder="paper",
            group_fn_name="current_source",
            token_transform="log",
            include_first_order=True,
        ),
        SynergySpec(
            name="paper_first_order_canonical_power",
            description="First-order synergy on the canonical MCT-LRQ family partition.",
            feature_builder="paper",
            group_fn_name="canonical",
            token_transform="power",
            include_first_order=True,
        ),
        SynergySpec(
            name="paper_entropy_current_source_power",
            description="Paper log-D decomposition entropy term on source-aware shares.",
            feature_builder="paper",
            group_fn_name="current_source",
            token_transform="power",
            include_entropy=True,
        ),
        SynergySpec(
            name="paper_second_order_current_source_tau0.1",
            description="Paper second-order soft-min co-occurrence terms on source-aware groups, tau=0.1.",
            feature_builder="paper",
            group_fn_name="current_source",
            include_second_order=True,
            second_order_tau=0.1,
        ),
        SynergySpec(
            name="paper_second_order_current_source_tau1",
            description="Paper second-order soft-min co-occurrence terms on source-aware groups, tau=1.",
            feature_builder="paper",
            group_fn_name="current_source",
            include_second_order=True,
            second_order_tau=1.0,
        ),
        SynergySpec(
            name="paper_first_second_current_source_power_tau0.1",
            description="Source-aware first-order plus second-order soft-min synergy.",
            feature_builder="paper",
            group_fn_name="current_source",
            token_transform="power",
            include_first_order=True,
            include_second_order=True,
            second_order_tau=0.1,
        ),
        SynergySpec(
            name="paper_full_current_source_power_tau0.1",
            description="Source-aware first-order, entropy, and second-order synergy terms.",
            feature_builder="paper",
            group_fn_name="current_source",
            token_transform="power",
            include_first_order=True,
            include_entropy=True,
            include_second_order=True,
            second_order_tau=0.1,
        ),
    ]
    return specs


def _fit_anchor(
    df: pd.DataFrame,
    w0: np.ndarray,
    w1: np.ndarray,
    y: np.ndarray,
    domains: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anchor_names, anchor_x = base.lrq_anchor_features(w0, w1, domains)
    anchor_mask = (
        (df["scale_display_label"] == "100M/6B")
        & np.isclose(df["target_budget_multiplier"].to_numpy(dtype=float), 1.0)
        & (df["fit_role"] == "fit_region")
    ).to_numpy()
    if anchor_mask.sum() < 50:
        raise ValueError(f"Anchor mask too small: {anchor_mask.sum()}")
    anchor_x_std, anchor_mean, anchor_std = base.standardize_fit(anchor_x[anchor_mask])
    anchor_coef = base.ridge_fit(anchor_x_std, y[anchor_mask], RIDGE_ANCHOR, penalize_intercept=False)
    anchor_pred = base.standardize_apply(anchor_x, anchor_mean, anchor_std) @ anchor_coef
    return anchor_names, anchor_pred, anchor_coef, anchor_mean, anchor_std


def _feature_scale_from_train(x: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    scale = np.std(x[fit_mask], axis=0)
    return np.where(scale < 1e-12, 1.0, scale)


def _ridge_no_intercept(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    penalty = ridge * np.eye(x.shape[1])
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def _inner_train_val_masks(df: pd.DataFrame, fit_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys = df["registry_run_key"].fillna(df["run_name"]).astype(str)
    fractions = keys.map(_stable_hash_fraction).to_numpy(dtype=float)
    inner_val = fit_mask & (fractions < 0.2)
    inner_train = fit_mask & ~inner_val
    if inner_val.sum() < 10 or inner_train.sum() < 10:
        return fit_mask, fit_mask
    return inner_train, inner_val


def _select_ridge(
    df: pd.DataFrame,
    x: np.ndarray,
    residual: np.ndarray,
    fit_mask: np.ndarray,
) -> float:
    inner_train, inner_val = _inner_train_val_masks(df, fit_mask)
    best_ridge = RIDGE_GRID[0]
    best_rmse = float("inf")
    for ridge in RIDGE_GRID:
        coef = _ridge_no_intercept(x[inner_train], residual[inner_train], ridge)
        pred = x[inner_val] @ coef
        rmse = float(np.sqrt(np.mean((pred - residual[inner_val]) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_ridge = ridge
    return best_ridge


def fit_model(
    df: pd.DataFrame,
    w0: np.ndarray,
    w1: np.ndarray,
    n: np.ndarray,
    d: np.ndarray,
    y: np.ndarray,
    domains: list[str],
    spec: SynergySpec,
) -> tuple[FittedSynergyModel, np.ndarray]:
    anchor_names, anchor_pred, anchor_coef, anchor_mean, anchor_std = _fit_anchor(df, w0, w1, y, domains)
    feature_names, feature_x = _synergy_features(spec, w0, w1, n, d, domains)
    fit_mask = base.split_column(df, "seed7_train")
    if not np.any(fit_mask):
        fit_mask = (df["fit_role"] == "fit_region").to_numpy()
    feature_scale = _feature_scale_from_train(feature_x, fit_mask)
    feature_x_scaled = feature_x / feature_scale
    residual = y - anchor_pred
    ridge = _select_ridge(df, feature_x_scaled, residual, fit_mask)
    coef = _ridge_no_intercept(feature_x_scaled[fit_mask], residual[fit_mask], ridge)
    pred = anchor_pred + feature_x_scaled @ coef
    model = FittedSynergyModel(
        spec=spec,
        domains=domains,
        anchor_feature_names=anchor_names,
        anchor_coef=anchor_coef,
        anchor_mean=anchor_mean,
        anchor_std=anchor_std,
        scale_feature_names=feature_names,
        feature_scale=feature_scale,
        scale_coef=coef,
        ridge=ridge,
    )
    return model, pred


def predict_model(model: FittedSynergyModel, w0: np.ndarray, w1: np.ndarray, n: np.ndarray, d: np.ndarray) -> np.ndarray:
    _, anchor_x = base.lrq_anchor_features(w0, w1, model.domains)
    anchor_pred = base.standardize_apply(anchor_x, model.anchor_mean, model.anchor_std) @ model.anchor_coef
    _, feature_x = _synergy_features(model.spec, w0, w1, n, d, model.domains)
    return anchor_pred + (feature_x / model.feature_scale) @ model.scale_coef


def _load_panel() -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(DATASET)
    if CANONICAL_SPLIT_FILE.exists():
        splits = pd.read_csv(CANONICAL_SPLIT_FILE)
        splits = splits[(splits["model"] == "mct_lrq69_drop_no_barrier") & (splits["fit_protocol"] == "seed7")]
        split_columns = [
            "registry_run_key",
            "seed7_train",
            "seed7_holdout",
            "fixed340_holdout",
            "random_supplement",
            "all900_holdout",
        ]
        df = df.merge(splits[split_columns], on="registry_run_key", how="left")
        for column in split_columns[1:]:
            df[column] = df[column].where(df[column].notna(), False).astype(bool)
    domains = base.phase_domains(list(df.columns))
    valid = pd.to_numeric(df[TARGET], errors="coerce").notna()
    df = df.loc[valid].reset_index(drop=True)
    w0, w1 = base.normalized_phase_arrays(df, domains)
    y = pd.to_numeric(df[TARGET], errors="coerce").to_numpy(dtype=float)
    n = pd.to_numeric(df["non_embedding_params"], errors="coerce").to_numpy(dtype=float)
    d = pd.to_numeric(df["target_budget"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(y) & np.isfinite(n) & np.isfinite(d) & (n > 0) & (d > 0)
    df = df.loc[finite].reset_index(drop=True)
    return df, domains, w0[finite], w1[finite], y[finite], n[finite], d[finite]


def _prediction_frame(
    df: pd.DataFrame,
    model_name: str,
    spec: SynergySpec,
    y: np.ndarray,
    pred: np.ndarray,
    ridge: float,
    feature_count: int,
) -> pd.DataFrame:
    columns = [
        column
        for column in [
            "registry_run_key",
            "run_name",
            "mixture_id",
            "scale",
            "scale_display_label",
            "target_budget_multiplier",
            "fit_role",
            "seed7_train",
            "seed7_holdout",
            "fixed340_holdout",
            "random_supplement",
            "all900_holdout",
        ]
        if column in df
    ]
    out = df[columns].copy()
    out["model"] = model_name
    out["feature_builder"] = spec.feature_builder
    out["group_fn_name"] = spec.group_fn_name or "domain"
    out["ridge"] = ridge
    out["feature_count"] = feature_count
    out["actual_bpb"] = y
    out["pred_bpb"] = pred
    out["residual_pred_minus_actual"] = pred - y
    return out


def summarize_predictions(
    df: pd.DataFrame, y: np.ndarray, pred: np.ndarray, model_name: str
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for split, mask in base.split_masks(df).items():
        if not np.any(mask):
            continue
        metrics = base.regression_metrics(y[mask], pred[mask])
        rows.append({"model": model_name, "split": split, **metrics})
    return rows


def _canonical_base_prediction(df: pd.DataFrame) -> np.ndarray:
    canonical = pd.read_csv(CANONICAL_SPLIT_FILE)
    canonical = canonical[(canonical["model"] == "mct_lrq69_drop_no_barrier") & (canonical["fit_protocol"] == "seed7")]
    pred_map = canonical.drop_duplicates("registry_run_key").set_index("registry_run_key")["pred_bpb"]
    return df["registry_run_key"].map(pred_map).to_numpy(dtype=float)


def run_canonical_residual_diagnostic(
    df: pd.DataFrame,
    w0: np.ndarray,
    w1: np.ndarray,
    y: np.ndarray,
    n: np.ndarray,
    d: np.ndarray,
    domains: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_pred = _canonical_base_prediction(df)
    finite = np.isfinite(base_pred)
    work = df.loc[finite].reset_index(drop=True)
    local_w0 = w0[finite]
    local_w1 = w1[finite]
    local_y = y[finite]
    local_n = n[finite]
    local_d = d[finite]
    base_pred = base_pred[finite]
    residual = local_y - base_pred
    fit_mask = base.split_column(work, "seed7_train")
    if not np.any(fit_mask):
        fit_mask = (work["fit_role"] == "fit_region").to_numpy()

    rows: list[dict[str, float | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    baseline_spec = SynergySpec("canonical_mct_lrq69_drop", "Current canonical MCT-LRQ model.", "baseline")
    baseline_row: dict[str, float | str] = {
        "model": baseline_spec.name,
        "description": baseline_spec.description,
        "ridge": float("nan"),
        "feature_count": 0.0,
    }
    for split, mask in base.split_masks(work).items():
        if split not in {
            "seed7_train",
            "seed7_holdout",
            "fixed340_holdout",
            "random_supplement",
            "all900_holdout",
            "fixed340_all",
        }:
            continue
        if np.any(mask):
            metrics = base.regression_metrics(local_y[mask], base_pred[mask])
            baseline_row[f"{split}_rmse"] = metrics["rmse"]
            baseline_row[f"{split}_spearman"] = metrics["spearman"]
    rows.append(baseline_row)
    prediction_frames.append(
        _prediction_frame(work, baseline_spec.name, baseline_spec, local_y, base_pred, float("nan"), 0)
    )

    for spec in build_specs()[1:]:
        feature_names, feature_x = _synergy_features(spec, local_w0, local_w1, local_n, local_d, domains)
        feature_scale = _feature_scale_from_train(feature_x, fit_mask)
        feature_x_scaled = feature_x / feature_scale
        ridge = _select_ridge(work, feature_x_scaled, residual, fit_mask)
        coef = _ridge_no_intercept(feature_x_scaled[fit_mask], residual[fit_mask], ridge)
        pred = base_pred + feature_x_scaled @ coef
        row: dict[str, float | str] = {
            "model": f"canon_resid_{spec.name}",
            "description": spec.description,
            "ridge": ridge,
            "feature_count": float(len(feature_names)),
        }
        for split, mask in base.split_masks(work).items():
            if split not in {
                "seed7_train",
                "seed7_holdout",
                "fixed340_holdout",
                "random_supplement",
                "all900_holdout",
                "fixed340_all",
            }:
                continue
            if np.any(mask):
                metrics = base.regression_metrics(local_y[mask], pred[mask])
                row[f"{split}_rmse"] = metrics["rmse"]
                row[f"{split}_spearman"] = metrics["spearman"]
        rows.append(row)
        prediction_frames.append(_prediction_frame(work, row["model"], spec, local_y, pred, ridge, len(feature_names)))

    summary = pd.DataFrame(rows)
    summary["score"] = (
        summary["seed7_holdout_rmse"]
        + summary["fixed340_holdout_rmse"]
        + summary["random_supplement_rmse"]
        + summary["all900_holdout_rmse"]
    )
    return summary.sort_values("score").reset_index(drop=True), pd.concat(prediction_frames, ignore_index=True)


def selected_models(summary: pd.DataFrame, count: int = 8) -> list[str]:
    pivot = summary.pivot_table(index="model", columns="split", values="rmse", aggfunc="first")
    score = (
        pivot.get("seed7_holdout", pivot.get("fit_region", pd.Series(index=pivot.index, dtype=float)))
        + pivot.get("fixed340_holdout", pivot.get("fixed340_all", pd.Series(index=pivot.index, dtype=float)))
        + pivot.get("random_supplement", pivot.get("external_60m", pd.Series(index=pivot.index, dtype=float)))
        + pivot.get("all900_holdout", pivot.get("external_900m", pd.Series(index=pivot.index, dtype=float)))
    )
    best = score.sort_values().head(count).index.tolist()
    if "smooth_mct_lrq_like" in pivot.index and "smooth_mct_lrq_like" not in best:
        best = ["smooth_mct_lrq_like", *best[: count - 1]]
    return best


def fixed340_drop_summary(df: pd.DataFrame, pred: np.ndarray, model_name: str) -> dict[str, float | str]:
    return base.fixed340_drop_summary(df, pred, model_name)


def write_plot(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path)
    try:
        fig.write_image(path.with_suffix(".png"), scale=2)
    except ValueError:
        pass


def make_plots(
    out_dir: Path,
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    canonical_predictions: pd.DataFrame,
    canonical_summary: pd.DataFrame,
) -> None:
    plot_dir = out_dir / "plots"
    selected = selected_models(summary)
    pred_plot = predictions[predictions["model"].isin(selected)].copy()
    fig = px.scatter(
        pred_plot,
        x="actual_bpb",
        y="pred_bpb",
        color="scale_display_label",
        facet_col="model",
        facet_col_wrap=3,
        hover_data=["run_name", "target_budget_multiplier", "fit_role", "residual_pred_minus_actual"],
        title="Domain-aware synergy full local models: predicted vs actual BPB",
    )
    lo = min(pred_plot["actual_bpb"].min(), pred_plot["pred_bpb"].min())
    hi = max(pred_plot["actual_bpb"].max(), pred_plot["pred_bpb"].max())
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line={"dash": "dash", "color": "black"})
    write_plot(fig, plot_dir / "domain_aware_synergy_pred_actual.html")

    external = summary[
        summary["split"].isin(["seed7_holdout", "fixed340_holdout", "random_supplement", "all900_holdout"])
    ].copy()
    external = external[external["model"].isin(selected)]
    fig = px.bar(
        external,
        x="model",
        y="rmse",
        color="split",
        barmode="group",
        title="Holdout RMSE for selected domain-aware synergy variants",
    )
    fig.update_layout(xaxis_tickangle=-35)
    write_plot(fig, plot_dir / "domain_aware_synergy_rmse_selected.html")

    canonical_selected = canonical_summary.sort_values("score").head(8)["model"].tolist()
    canon_plot = canonical_predictions[canonical_predictions["model"].isin(canonical_selected)].copy()
    fig = px.scatter(
        canon_plot,
        x="actual_bpb",
        y="pred_bpb",
        color="scale_display_label",
        facet_col="model",
        facet_col_wrap=3,
        hover_data=["run_name", "target_budget_multiplier", "fit_role", "residual_pred_minus_actual"],
        title="Canonical MCT-LRQ plus paper-style synergy residual corrections",
    )
    lo = min(canon_plot["actual_bpb"].min(), canon_plot["pred_bpb"].min())
    hi = max(canon_plot["actual_bpb"].max(), canon_plot["pred_bpb"].max())
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line={"dash": "dash", "color": "black"})
    write_plot(fig, plot_dir / "canonical_residual_domain_aware_synergy_pred_actual.html")


def markdown_table(df: pd.DataFrame, columns: list[str], float_digits: int = 5) -> str:
    work = df[columns].copy()
    for column in work.columns:
        if pd.api.types.is_float_dtype(work[column]):
            work[column] = work[column].map(lambda value: "" if pd.isna(value) else f"{value:.{float_digits}f}")
    return work.to_markdown(index=False)


def _summary_flat(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot_table(index="model", columns="split", values=["rmse", "spearman"], aggfunc="first")
    flat = pd.DataFrame({"model": pivot.index})
    for split in [
        "seed7_train",
        "seed7_holdout",
        "fixed340_holdout",
        "fixed340_all",
        "random_supplement",
        "all900_holdout",
        "external_60m",
        "external_900m",
    ]:
        if ("rmse", split) in pivot:
            flat[f"{split}_rmse"] = pivot[("rmse", split)].to_numpy()
        if ("spearman", split) in pivot:
            flat[f"{split}_spearman"] = pivot[("spearman", split)].to_numpy()
    for column in ["seed7_holdout_rmse", "fixed340_holdout_rmse", "random_supplement_rmse", "all900_holdout_rmse"]:
        if column not in flat:
            flat[column] = np.nan
    flat["selection_score"] = (
        flat["seed7_holdout_rmse"]
        + flat["fixed340_holdout_rmse"]
        + flat["random_supplement_rmse"]
        + flat["all900_holdout_rmse"]
    )
    return flat.sort_values("selection_score").reset_index(drop=True)


def write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    drop_summary: pd.DataFrame,
    canonical_summary: pd.DataFrame,
    specs: list[SynergySpec],
) -> None:
    flat = _summary_flat(summary)
    top = flat.head(12).merge(
        drop_summary[["model", "drop_0p5_to_1_ratio", "drop_0p5_to_2_ratio", "drop_1_to_2_ratio"]],
        on="model",
        how="left",
    )
    canonical_top = canonical_summary.sort_values("score").head(12).copy()
    canonical_baseline = canonical_summary[canonical_summary["model"] == "canonical_mct_lrq69_drop"].iloc[0]
    canonical_best_nonbaseline = canonical_summary[canonical_summary["model"] != "canonical_mct_lrq69_drop"].iloc[0]
    full_local_baseline = flat[flat["model"] == "smooth_mct_lrq_like"].iloc[0]
    full_local_best = flat.iloc[0]
    spec_rows = pd.DataFrame(
        [
            {
                "model": spec.name,
                "group": spec.group_fn_name or "domain/all",
                "transform": spec.token_transform,
                "first_order": spec.include_first_order,
                "entropy": spec.include_entropy,
                "second_order": spec.include_second_order,
                "description": spec.description,
            }
            for spec in specs
        ]
    )
    report = [
        "# Domain-Aware Synergy Joint-Law Sprint",
        "",
        "## Goal",
        "",
        (
            "Test whether the domain-aware scaling-law features from `Domain-Aware Scaling Laws "
            "Uncover Data Synergy` improve our joint mixture/scale law on the canonical ND panel."
        ),
        "",
        "The paper proposes two relevant ideas:",
        "",
        "- First-order domain-benchmark synergy: domain-specific deviations from the global data scaling exponent.",
        (
            "- Second-order pretraining synergy: pairwise domain co-occurrence terms based on a soft-min "
            "of domain token exposures."
        ),
        "",
        "## Headline Result",
        "",
        (
            "The reliable canonical-residual diagnostic is negative: the unchanged canonical MCT-LRQ "
            f"baseline remains best with score `{canonical_baseline['score']:.5f}`. The best paper-style "
            f"residual correction is `{canonical_best_nonbaseline['model']}` with score "
            f"`{canonical_best_nonbaseline['score']:.5f}`."
        ),
        (
            "The full local anchored ablation is more encouraging but less decisive: "
            f"`{full_local_best['model']}` improves the local selection score from "
            f"`{full_local_baseline['selection_score']:.5f}` to `{full_local_best['selection_score']:.5f}`. "
            "This should be interpreted as evidence that current-source/entropy scale features are useful, "
            "not as a promotion candidate."
        ),
        "",
        "In our anchored single-target setting, the first-order feature reduces to:",
        "",
        "```latex",
        r"\Delta z_k(w,D) = u_k(w)\log(D/D_0), \quad \sum_k \gamma_k = 0.",
        "```",
        "",
        "The sum-to-zero constraint is implemented by centering the share vector before multiplying by the scale term.",
        "",
        "The source-group second-order feature is:",
        "",
        "```latex",
        (
            r"\Delta q_{ab}(w,D)=\operatorname{softmin}_\tau(\log(1+u_aD),\log(1+u_bD))"
            r"-\operatorname{softmin}_\tau(\log(1+u_aD_0),\log(1+u_bD_0))."
        ),
        "```",
        "",
        "All scale features are zero at the corrected `100M/6B` anchor, preserving the anchor mixture regression.",
        "",
        "## Variants Tested",
        "",
        markdown_table(
            spec_rows,
            ["model", "group", "transform", "first_order", "entropy", "second_order", "description"],
            float_digits=4,
        ),
        "",
        "## Canonical MCT-LRQ Residual Diagnostic",
        "",
        (
            "This is the highest-signal diagnostic: start from current canonical `mct_lrq69_drop_no_barrier` "
            "row predictions, then fit only paper-style residual corrections."
        ),
        "",
        markdown_table(
            canonical_top,
            [
                "model",
                "feature_count",
                "ridge",
                "seed7_holdout_rmse",
                "fixed340_holdout_rmse",
                "random_supplement_rmse",
                "all900_holdout_rmse",
                "score",
                "seed7_holdout_spearman",
            ],
            float_digits=5,
        ),
        "",
        "## Full Local Anchored Model Fit",
        "",
        (
            "These models refit the LRQ anchor plus each scale-feature family. They are ablations, not exact "
            "replacements for canonical MCT-LRQ."
        ),
        "",
        markdown_table(
            top,
            [
                "model",
                "seed7_train_rmse",
                "seed7_holdout_rmse",
                "fixed340_holdout_rmse",
                "random_supplement_rmse",
                "all900_holdout_rmse",
                "external_60m_rmse",
                "external_900m_rmse",
                "seed7_holdout_spearman",
                "drop_0p5_to_1_ratio",
                "drop_0p5_to_2_ratio",
                "drop_1_to_2_ratio",
                "selection_score",
            ],
            float_digits=5,
        ),
        "",
        "## Interpretation",
        "",
        (
            "- The paper's first-order term is a real structural direction for our setting, but after "
            "anchoring it mostly becomes domain- or group-specific data-scaling heads."
        ),
        (
            "- On top of canonical MCT-LRQ, raw domain first-order heads slightly improve the tiny 900M "
            "diagnostic and fixed-340M-all RMSE, but they worsen seed7 holdout, fixed-340M holdout, "
            "and random supplement RMSE."
        ),
        (
            "- Source-group/entropy terms help the weaker full local anchored model, which suggests useful "
            "missing structure in the compact local ablation."
        ),
        (
            "- Second-order soft-min terms are interpretable, but with only one primary BPB target they "
            "are weakly identified unless restricted to coarse source groups."
        ),
        (
            "- Recommendation: do not replace the current law with these terms yet. If we pursue this "
            "direction, use a constrained source-group entropy/first-order correction inside the exact "
            "canonical implementation and validate raw optima."
        ),
        "",
        "## Artifacts",
        "",
        "- `csv/model_summary.csv`: split metrics for full local anchored variants.",
        "- `csv/row_predictions.csv`: row-level predictions for full local variants.",
        "- `csv/canonical_residual_synergy_summary.csv`: residual corrections on top of canonical MCT-LRQ.",
        "- `csv/canonical_residual_synergy_predictions.csv`: row-level canonical residual predictions.",
        "- `csv/fixed340_drop_summary.csv`: fixed-340M target-budget drop ratios.",
        "- `plots/domain_aware_synergy_pred_actual.html`: full local prediction scatter.",
        "- `plots/canonical_residual_domain_aware_synergy_pred_actual.html`: canonical residual prediction scatter.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    out_dir = OUTPUT_DIR
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    df, domains, w0, w1, y, n, d = _load_panel()
    specs = build_specs()

    summary_rows: list[dict[str, float | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    drop_rows: list[dict[str, float | str]] = []
    model_metadata: list[dict[str, float | str]] = []
    for spec in specs:
        model, pred = fit_model(df, w0, w1, n, d, y, domains, spec)
        summary_rows.extend(summarize_predictions(df, y, pred, spec.name))
        prediction_frames.append(
            _prediction_frame(df, spec.name, spec, y, pred, model.ridge, len(model.scale_feature_names))
        )
        drop_rows.append(fixed340_drop_summary(df, pred, spec.name))
        model_metadata.append(
            {
                "model": spec.name,
                "description": spec.description,
                "feature_count": len(model.scale_feature_names),
                "ridge": model.ridge,
                "coef_l2": float(np.linalg.norm(model.scale_coef)),
            }
        )

    summary = pd.DataFrame(summary_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    drops = pd.DataFrame(drop_rows)
    metadata = pd.DataFrame(model_metadata)
    summary = summary.merge(metadata, on="model", how="left")
    canonical_summary, canonical_predictions = run_canonical_residual_diagnostic(df, w0, w1, y, n, d, domains)

    summary.to_csv(csv_dir / "model_summary.csv", index=False)
    predictions.to_csv(csv_dir / "row_predictions.csv", index=False)
    drops.to_csv(csv_dir / "fixed340_drop_summary.csv", index=False)
    metadata.to_csv(csv_dir / "model_metadata.csv", index=False)
    canonical_summary.to_csv(csv_dir / "canonical_residual_synergy_summary.csv", index=False)
    canonical_predictions.to_csv(csv_dir / "canonical_residual_synergy_predictions.csv", index=False)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "target": TARGET,
                "rows": len(df),
                "domains": len(domains),
                "model_count": len(specs),
                "ridge_grid": list(RIDGE_GRID),
                "constants": {
                    "n0": base.N0,
                    "d0": base.D0,
                    "alpha": base.ALPHA,
                    "beta": base.BETA,
                    "gamma": base.GAMMA,
                    "delta": base.DELTA,
                },
            },
            f,
            indent=2,
        )
    make_plots(out_dir, predictions, summary, canonical_predictions, canonical_summary)
    write_report(out_dir, summary, drops, canonical_summary, specs)
    print(f"Wrote domain-aware synergy sprint outputs to {out_dir}")


if __name__ == "__main__":
    main()
