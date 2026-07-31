# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "plotly", "scikit-learn"]
# ///
"""Diagnose post-selection optimism for validated Table-9 DSP candidates.

The Table-9 scaling validation showed that moderate-KL DSP candidates can beat
OLMix, while low-KL DSP candidates validate badly. This script joins the saved
300M DSP KL-sweep predictions with the observed 3e18 Table-9 validations and
tests simple calibration/trust-region diagnostics:

* residuals versus KL/TV/max-epoch,
* leave-one-out calibration models on the small validated panel,
* a nearest-observed pessimism sweep using only 300M prediction metadata.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "table9_dsp_postselection_calibration_20260630"
DEFAULT_VALIDATION_RANKING = (
    REFERENCE_OUTPUTS
    / "delphi_table9_dsp_validation_mixtures_3e18_20260628"
    / "table9_3e18_observed_ranking_20260628.csv"
)
DEFAULT_PER_COMPONENT_SWEEP = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_per_component_dsp_kl_sweep_300m_20260628"
    / "dsp_olmix_overlay"
    / "per_component_dsp_kl_sweep_summary.csv"
)
DEFAULT_AGGREGATE_SWEEP = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_table9_macro_dsp_300m_20260625"
    / "effective_exposure_table9_macro_kl_sweep_linear_reg_0p0001"
    / "effective_exposure_table9_macro_kl_sweep_summary.csv"
)
DEFAULT_MATERIALIZED_SUMMARY = (
    REFERENCE_OUTPUTS
    / "table9_dsp_validation_mixtures_300m_20260628"
    / "materialized_mixture_summary.csv"
)

PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
KL_SUFFIX_RE = re.compile(r"kl(?P<slug>[0-9]+(?:p[0-9]+)?)$")


@dataclass(frozen=True)
class ParsedEvalName:
    model_family: str
    kl_reg: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-ranking", type=Path, default=DEFAULT_VALIDATION_RANKING)
    parser.add_argument("--per-component-sweep", type=Path, default=DEFAULT_PER_COMPONENT_SWEEP)
    parser.add_argument("--aggregate-sweep", type=Path, default=DEFAULT_AGGREGATE_SWEEP)
    parser.add_argument("--materialized-summary", type=Path, default=DEFAULT_MATERIALIZED_SUMMARY)
    return parser.parse_args()


def parse_slug(slug: str) -> float:
    return float(slug.replace("p", "."))


def kl_slug(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def parse_eval_name(eval_run_name: str) -> ParsedEvalName | None:
    if eval_run_name.startswith("t9_3e18_dsp_percomp_"):
        match = KL_SUFFIX_RE.search(eval_run_name)
        if match is None:
            return None
        return ParsedEvalName("per_component", parse_slug(match.group("slug")))
    if eval_run_name.startswith("t9_3e18_dsp_table9_"):
        match = KL_SUFFIX_RE.search(eval_run_name)
        if match is None:
            return None
        return ParsedEvalName("aggregate", parse_slug(match.group("slug")))
    if eval_run_name.startswith("t9_3e18_dsp_effexp_"):
        match = KL_SUFFIX_RE.search(eval_run_name)
        if match is None:
            return None
        return ParsedEvalName("aggregate", parse_slug(match.group("slug")))
    if eval_run_name == "t9_3e18_dsp_table9":
        # Historical first Table-9 DSP validation name for the KL=0.025
        # aggregate effective-exposure mixture.
        return ParsedEvalName("aggregate", 0.025)
    return None


def nearest_metadata_row(frame: pd.DataFrame, *, model_family: str, kl_reg: float) -> pd.Series:
    family_frame = frame[frame["model_family"].eq(model_family)].copy()
    if family_frame.empty:
        raise ValueError(f"No sweep rows for model family {model_family}")
    family_frame["kl_distance"] = (family_frame["kl_reg"] - float(kl_reg)).abs()
    return family_frame.sort_values(["kl_distance", "kl_reg"]).iloc[0]


def overlay_materialized_summary(predictions: pd.DataFrame, materialized_path: Path) -> pd.DataFrame:
    if not materialized_path.exists():
        return predictions
    materialized = pd.read_csv(materialized_path)
    family_map = {
        "aggregate_effective_exposure_dsp": "aggregate",
        "per_component_effective_exposure_dsp": "per_component",
    }
    materialized["model_family"] = materialized["family"].map(family_map)
    materialized = materialized[materialized["model_family"].notna()].copy()
    out = predictions.copy()
    rows_to_append: list[dict[str, object]] = []
    for row in materialized.itertuples(index=False):
        mask = out["model_family"].eq(row.model_family) & np.isclose(out["kl_reg"], float(row.kl_reg))
        if mask.any():
            idx = out.index[mask][0]
            out.loc[idx, "predicted_objective"] = float(row.predicted_bpb)
            out.loc[idx, "regularized_objective"] = float(row.regularized_objective)
            out.loc[idx, "mean_phase_tv_to_proportional"] = float(row.mean_phase_tv_to_proportional)
            out.loc[idx, "max_simulated_epoch"] = float(row.max_simulated_epoch)
            out.loc[idx, "q95_simulated_epoch"] = float(row.q95_simulated_epoch)
            out.loc[idx, "optimizer_status"] = str(row.optimizer_status)
            out.loc[idx, "metadata_source"] = "materialized_summary_overlay"
            continue
        nearest = nearest_metadata_row(out, model_family=str(row.model_family), kl_reg=float(row.kl_reg))
        new_row = nearest.to_dict()
        new_row.update(
            {
                "model_family": str(row.model_family),
                "variant": str(row.key),
                "kl_reg": float(row.kl_reg),
                "predicted_objective": float(row.predicted_bpb),
                "regularized_objective": float(row.regularized_objective),
                "mean_phase_tv_to_proportional": float(row.mean_phase_tv_to_proportional),
                "max_simulated_epoch": float(row.max_simulated_epoch),
                "q95_simulated_epoch": float(row.q95_simulated_epoch),
                "optimizer_status": str(row.optimizer_status),
                "metadata_source": "materialized_summary_with_nearest_sweep_metadata",
            }
        )
        rows_to_append.append(new_row)
    if rows_to_append:
        out = pd.concat([out, pd.DataFrame(rows_to_append)], ignore_index=True)
    out["kl_key"] = out["model_family"] + "_kl" + out["kl_reg"].map(kl_slug)
    return out


def load_sweep_predictions(per_component_path: Path, aggregate_path: Path, materialized_path: Path) -> pd.DataFrame:
    per_component = pd.read_csv(per_component_path)
    aggregate = pd.read_csv(aggregate_path)
    rows: list[pd.DataFrame] = []
    per_component = per_component.rename(
        columns={
            "macro_oof_rmse": "oof_rmse",
            "macro_oof_spearman": "oof_spearman",
            "macro_fold_mean_regret_at_1": "fold_mean_regret_at_1",
        }
    )
    per_component["model_family"] = "per_component"
    aggregate["model_family"] = "aggregate"
    per_component["metadata_source"] = "per_component_kl_sweep"
    aggregate["metadata_source"] = "aggregate_kl_sweep"
    keep = [
        "model_family",
        "variant",
        "kl_reg",
        "predicted_objective",
        "regularized_objective",
        "proportional_predicted",
        "best_observed_run_name",
        "best_observed_value",
        "nearest_observed_run_name",
        "nearest_observed_value",
        "nearest_observed_mean_phase_tv",
        "mean_phase_tv_to_proportional",
        "max_simulated_epoch",
        "q95_simulated_epoch",
        "max_weight",
        "optimizer_status",
        "oof_rmse",
        "oof_spearman",
        "fold_mean_regret_at_1",
        "metadata_source",
    ]
    for frame in [per_component, aggregate]:
        missing = sorted(set(keep).difference(frame.columns))
        if missing:
            raise ValueError(f"Missing expected sweep columns: {missing}")
        rows.append(frame[keep].copy())
    out = pd.concat(rows, ignore_index=True)
    out["kl_key"] = out["model_family"] + "_kl" + out["kl_reg"].map(kl_slug)
    return overlay_materialized_summary(out, materialized_path)


def load_validated_candidates(validation_path: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    validation = pd.read_csv(validation_path)
    parsed_rows: list[dict[str, object]] = []
    for row in validation.itertuples(index=False):
        parsed = parse_eval_name(str(row.eval_run_name))
        if parsed is None:
            continue
        parsed_rows.append(
            {
                "eval_run_name": str(row.eval_run_name),
                "observed_rank": int(row.rank),
                "observed_bpb": float(row.table9_macro_bpb),
                "model_family": parsed.model_family,
                "kl_reg": parsed.kl_reg,
                "wandb_url": str(row.wandb_url),
                "wandb_id": str(row.wandb_id),
            }
        )
    parsed = pd.DataFrame(parsed_rows)
    merged = parsed.merge(predictions, on=["model_family", "kl_reg"], how="left", validate="one_to_one")
    if merged["predicted_objective"].isna().any():
        missing = merged[merged["predicted_objective"].isna()][["eval_run_name", "model_family", "kl_reg"]]
        raise ValueError(f"Missing prediction metadata for validated rows:\n{missing}")
    best_observed = float(merged["observed_bpb"].min())
    merged["candidate_key"] = merged["model_family"] + "_kl" + merged["kl_reg"].map(kl_slug)
    merged["observed_regret"] = merged["observed_bpb"] - best_observed
    merged["postselection_optimism"] = merged["observed_bpb"] - merged["predicted_objective"]
    merged["regularized_optimism"] = merged["observed_bpb"] - merged["regularized_objective"]
    merged["nearest_gap"] = merged["nearest_observed_value"] - merged["predicted_objective"]
    merged["nearest_gap_oof_units"] = merged["nearest_gap"] / merged["oof_rmse"]
    merged["postselection_optimism_oof_units"] = merged["postselection_optimism"] / merged["oof_rmse"]
    merged["log_max_simulated_epoch"] = np.log(merged["max_simulated_epoch"].clip(lower=1e-12))
    return merged.sort_values(["model_family", "kl_reg"]).reset_index(drop=True)


def regression_metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = pred - actual
    return {
        "rmse": float(math.sqrt(mean_squared_error(actual, pred))),
        "mae": float(np.mean(np.abs(residual))),
        "spearman": float(pd.Series(actual).corr(pd.Series(pred), method="spearman")),
        "pearson": float(pd.Series(actual).corr(pd.Series(pred), method="pearson")),
    }


def selected_regret(frame: pd.DataFrame, score: np.ndarray) -> dict[str, object]:
    idx = int(np.argmin(score))
    best_idx = int(frame["observed_bpb"].to_numpy().argmin())
    return {
        "selected_eval_run_name": str(frame.iloc[idx]["eval_run_name"]),
        "selected_candidate_key": str(frame.iloc[idx]["candidate_key"]),
        "selected_observed_bpb": float(frame.iloc[idx]["observed_bpb"]),
        "selected_score": float(score[idx]),
        "best_eval_run_name": str(frame.iloc[best_idx]["eval_run_name"]),
        "best_observed_bpb": float(frame.iloc[best_idx]["observed_bpb"]),
        "observed_regret": float(frame.iloc[idx]["observed_bpb"] - frame.iloc[best_idx]["observed_bpb"]),
    }


def baseline_selector_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selectors = {
        "raw_predicted_objective": frame["predicted_objective"].to_numpy(dtype=float),
        "regularized_objective": frame["regularized_objective"].to_numpy(dtype=float),
        "nearest_observed_value": frame["nearest_observed_value"].to_numpy(dtype=float),
        "tv_only": frame["mean_phase_tv_to_proportional"].to_numpy(dtype=float),
    }
    for subset_name, subset in [("all_dsp", frame), *[(family, group) for family, group in frame.groupby("model_family")]]:
        for selector_name, score in selectors.items():
            subset_score = score[subset.index.to_numpy(dtype=int)]
            rows.append({"subset": subset_name, "selector": selector_name, **selected_regret(subset, subset_score)})
    return pd.DataFrame(rows)


def make_feature_pipeline(numeric_features: list[str], categorical_features: list[str], *, ridge_alpha: float) -> Pipeline:
    transformer = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop=None, handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )
    regressor = LinearRegression() if ridge_alpha == 0 else Ridge(alpha=ridge_alpha)
    return Pipeline([("features", transformer), ("regressor", regressor)])


def calibration_model_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_specs = [
        ("identity_uncalibrated", [], [], None),
        ("affine_predicted", ["predicted_objective"], [], 0.0),
        ("nearest_gap_linear", ["predicted_objective", "nearest_gap"], [], 1.0),
        (
            "trust_region_linear",
            ["predicted_objective", "nearest_gap", "mean_phase_tv_to_proportional", "log_max_simulated_epoch"],
            ["model_family"],
            1.0,
        ),
        (
            "kl_tv_linear",
            ["predicted_objective", "kl_reg", "mean_phase_tv_to_proportional", "log_max_simulated_epoch"],
            ["model_family"],
            1.0,
        ),
    ]
    y = frame["observed_bpb"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    loo = LeaveOneOut()
    for model_name, numeric_features, categorical_features, ridge_alpha in model_specs:
        if model_name == "identity_uncalibrated":
            pred = frame["predicted_objective"].to_numpy(dtype=float)
        else:
            pred = np.zeros(len(frame), dtype=float)
            for train_idx, test_idx in loo.split(frame):
                pipeline = make_feature_pipeline(numeric_features, categorical_features, ridge_alpha=float(ridge_alpha))
                pipeline.fit(frame.iloc[train_idx], y[train_idx])
                pred[test_idx] = pipeline.predict(frame.iloc[test_idx])
        metrics = regression_metrics(y, pred)
        rows.append({"calibration_model": model_name, **metrics, **selected_regret(frame, pred)})
        for idx, value in enumerate(pred):
            pred_rows.append(
                {
                    "calibration_model": model_name,
                    "eval_run_name": frame.iloc[idx]["eval_run_name"],
                    "candidate_key": frame.iloc[idx]["candidate_key"],
                    "observed_bpb": y[idx],
                    "calibrated_prediction": float(value),
                    "calibration_residual_pred_minus_actual": float(value - y[idx]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def pessimism_sweep(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    alpha_grid = np.round(np.arange(0.0, 1.501, 0.025), 3)
    for subset_name, subset in [("all_dsp", frame), *[(family, group) for family, group in frame.groupby("model_family")]]:
        for alpha in alpha_grid:
            score = subset["predicted_objective"].to_numpy(dtype=float) + alpha * subset["nearest_gap"].to_numpy(dtype=float)
            rows.append(
                {
                    "subset": subset_name,
                    "alpha": float(alpha),
                    "score_formula": "predicted_objective + alpha * (nearest_observed_value - predicted_objective)",
                    **selected_regret(subset, score),
                }
            )
    return pd.DataFrame(rows)


def trust_region_gate_sweep(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sweep_specs = [
        ("max_tv", "mean_phase_tv_to_proportional", "<="),
        ("min_kl", "kl_reg", ">="),
        ("max_epoch", "max_simulated_epoch", "<="),
    ]
    subsets = [("all_dsp", frame), *[(family, group) for family, group in frame.groupby("model_family")]]
    for subset_name, subset in subsets:
        for gate_name, column, direction in sweep_specs:
            thresholds = sorted(float(value) for value in subset[column].dropna().unique())
            for threshold in thresholds:
                if direction == "<=":
                    eligible = subset[subset[column].le(threshold)]
                elif direction == ">=":
                    eligible = subset[subset[column].ge(threshold)]
                else:
                    raise ValueError(f"Unknown gate direction {direction}")
                if eligible.empty:
                    continue
                rows.append(
                    {
                        "subset": subset_name,
                        "gate": gate_name,
                        "column": column,
                        "direction": direction,
                        "threshold": float(threshold),
                        "n_eligible": int(len(eligible)),
                        "selector": "min_predicted_objective_inside_gate",
                        **selected_regret(eligible, eligible["predicted_objective"].to_numpy(dtype=float)),
                    }
                )
    return pd.DataFrame(rows)


def write_residual_plot(path: Path, frame: pd.DataFrame) -> None:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Optimism vs KL", "Optimism vs TV", "Optimism vs max simulated epoch"],
    )
    x_specs = [
        ("kl_reg", "KL regularization"),
        ("mean_phase_tv_to_proportional", "Mean phase TV to proportional"),
        ("max_simulated_epoch", "Max simulated epoch"),
    ]
    colors = {"aggregate": "#2b6cb0", "per_component": "#dd6b20"}
    for col, (x_col, x_title) in enumerate(x_specs, start=1):
        for family, group in frame.groupby("model_family"):
            fig.add_trace(
                go.Scatter(
                    x=group[x_col],
                    y=group["postselection_optimism"],
                    mode="markers+text",
                    text=group["kl_reg"].map(lambda value: f"KL={value:g}"),
                    textposition="top center",
                    marker={"size": 10, "color": colors.get(family, "#444")},
                    name=family,
                    legendgroup=family,
                    showlegend=col == 1,
                    customdata=np.stack([group["eval_run_name"], group["observed_bpb"], group["predicted_objective"]], axis=1),
                    hovertemplate=(
                        "%{customdata[0]}<br>x=%{x:.4f}<br>optimism=%{y:.4f}"
                        "<br>observed=%{customdata[1]:.6f}<br>predicted=%{customdata[2]:.6f}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text=x_title, row=1, col=col)
        fig.update_yaxes(title_text="Observed - predicted BPB", row=1, col=col)
    fig.update_layout(
        title="3e18 Table-9 DSP validation: post-selection optimism is large across the good KL region",
        template="plotly_white",
        width=1450,
        height=560,
    )
    fig.write_html(path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_observed_predicted_plot(path: Path, frame: pd.DataFrame, calibrated_predictions: pd.DataFrame) -> None:
    selected_models = ["identity_uncalibrated", "nearest_gap_linear", "trust_region_linear"]
    plot_frame = calibrated_predictions[calibrated_predictions["calibration_model"].isin(selected_models)].merge(
        frame[["eval_run_name", "model_family", "kl_reg", "observed_rank"]],
        on="eval_run_name",
        how="left",
        validate="many_to_one",
    )
    fig = make_subplots(rows=1, cols=len(selected_models), subplot_titles=selected_models)
    lo = float(min(plot_frame["observed_bpb"].min(), plot_frame["calibrated_prediction"].min()))
    hi = float(max(plot_frame["observed_bpb"].max(), plot_frame["calibrated_prediction"].max()))
    for col, model_name in enumerate(selected_models, start=1):
        group = plot_frame[plot_frame["calibration_model"].eq(model_name)]
        fig.add_trace(
            go.Scatter(
                x=group["observed_bpb"],
                y=group["calibrated_prediction"],
                mode="markers+text",
                text=group["kl_reg"].map(lambda value: f"{value:g}"),
                textposition="top center",
                marker={
                    "size": 10,
                    "color": group["model_family"].map({"aggregate": 0, "per_component": 1}),
                    "colorscale": "RdYlGn_r",
                    "showscale": col == len(selected_models),
                    "colorbar": {"title": "family"},
                },
                customdata=np.stack([group["eval_run_name"], group["model_family"]], axis=1),
                hovertemplate=(
                    "%{customdata[0]}<br>family=%{customdata[1]}<br>observed=%{x:.6f}"
                    "<br>predicted/calibrated=%{y:.6f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"dash": "dash", "color": "#555"}, showlegend=False),
            row=1,
            col=col,
        )
        fig.update_xaxes(title_text="Observed 3e18 Table-9 BPB", row=1, col=col)
        fig.update_yaxes(title_text="Predicted/calibrated BPB", row=1, col=col)
    fig.update_layout(
        title="3e18 validated DSP candidates: raw prediction vs simple LOOCV calibration",
        template="plotly_white",
        width=1450,
        height=560,
    )
    fig.write_html(path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_pessimism_plot(path: Path, sweep: pd.DataFrame) -> None:
    fig = go.Figure()
    colors = {"all_dsp": "#2b6cb0", "aggregate": "#805ad5", "per_component": "#dd6b20"}
    for subset, group in sweep.groupby("subset", sort=False):
        fig.add_trace(
            go.Scatter(
                x=group["alpha"],
                y=group["observed_regret"],
                mode="lines+markers",
                name=subset,
                line={"color": colors.get(subset)},
                customdata=np.stack([group["selected_eval_run_name"], group["selected_candidate_key"]], axis=1),
                hovertemplate=(
                    "alpha=%{x:.3f}<br>observed regret=%{y:.6f}"
                    "<br>selected=%{customdata[0]}<br>key=%{customdata[1]}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Nearest-observed pessimism sweep for validated DSP candidates",
        xaxis_title="alpha in predicted + alpha * nearest-gap",
        yaxis_title="Observed regret of selected candidate at 3e18",
        template="plotly_white",
        width=1050,
        height=600,
    )
    fig.write_html(path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_gate_plot(path: Path, sweep: pd.DataFrame) -> None:
    gates = ["max_tv", "min_kl", "max_epoch"]
    fig = make_subplots(rows=1, cols=len(gates), subplot_titles=gates)
    colors = {"all_dsp": "#2b6cb0", "aggregate": "#805ad5", "per_component": "#dd6b20"}
    for col, gate in enumerate(gates, start=1):
        gate_frame = sweep[sweep["gate"].eq(gate)]
        for subset, group in gate_frame.groupby("subset", sort=False):
            fig.add_trace(
                go.Scatter(
                    x=group["threshold"],
                    y=group["observed_regret"],
                    mode="lines+markers",
                    name=subset,
                    legendgroup=subset,
                    showlegend=col == 1,
                    line={"color": colors.get(subset)},
                    customdata=np.stack([group["selected_eval_run_name"], group["n_eligible"]], axis=1),
                    hovertemplate=(
                        "threshold=%{x:.4f}<br>observed regret=%{y:.6f}<br>"
                        "selected=%{customdata[0]}<br>n eligible=%{customdata[1]}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text=gate, row=1, col=col)
        fig.update_yaxes(title_text="Observed regret", row=1, col=col)
    fig.update_layout(
        title="Hard trust-region gates, then choose lowest predicted BPB inside gate",
        template="plotly_white",
        width=1450,
        height=560,
    )
    fig.write_html(path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_sweep_predictions(args.per_component_sweep, args.aggregate_sweep, args.materialized_summary)
    validated = load_validated_candidates(args.validation_ranking, predictions)
    baseline_selectors = baseline_selector_summary(validated)
    calibration_summary, calibration_predictions = calibration_model_summary(validated)
    alpha_sweep = pessimism_sweep(validated)
    gate_sweep = trust_region_gate_sweep(validated)

    validated.to_csv(args.output_dir / "validated_dsp_candidates_with_predictions.csv", index=False)
    baseline_selectors.to_csv(args.output_dir / "baseline_selector_summary.csv", index=False)
    calibration_summary.to_csv(args.output_dir / "calibration_model_summary.csv", index=False)
    calibration_predictions.to_csv(args.output_dir / "calibration_model_predictions.csv", index=False)
    alpha_sweep.to_csv(args.output_dir / "nearest_observed_pessimism_sweep.csv", index=False)
    gate_sweep.to_csv(args.output_dir / "trust_region_gate_sweep.csv", index=False)
    write_residual_plot(args.output_dir / "validated_residuals_vs_trust_region.html", validated)
    write_observed_predicted_plot(args.output_dir / "validated_observed_vs_calibrated.html", validated, calibration_predictions)
    write_pessimism_plot(args.output_dir / "nearest_observed_pessimism_selector_sweep.html", alpha_sweep)
    write_gate_plot(args.output_dir / "trust_region_gate_selector_sweep.html", gate_sweep)

    print("Wrote", args.output_dir)
    print("\nValidated DSP rows:")
    print(
        validated[
            [
                "eval_run_name",
                "model_family",
                "kl_reg",
                "observed_bpb",
                "observed_rank",
                "predicted_objective",
                "regularized_objective",
                "postselection_optimism",
                "nearest_gap_oof_units",
                "observed_regret",
            ]
        ]
        .sort_values("observed_bpb")
        .to_string(index=False)
    )
    print("\nBaseline selectors:")
    print(baseline_selectors.to_string(index=False))
    print("\nCalibration summary:")
    print(calibration_summary.to_string(index=False))
    print("\nBest hard trust-region gates by subset/gate:")
    best_gate = gate_sweep.sort_values(["subset", "gate", "observed_regret", "n_eligible"], ascending=[True, True, True, False])
    print(best_gate.groupby(["subset", "gate"], as_index=False).head(1).to_string(index=False))


if __name__ == "__main__":
    main()
