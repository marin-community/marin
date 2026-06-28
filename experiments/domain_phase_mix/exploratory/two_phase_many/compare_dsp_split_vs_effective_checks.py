# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "plotly", "scipy", "kaleido", "matplotlib", "scikit-learn", "tabulate"]
# ///
"""Compare split-phase DSP against effective-exposure DSP on perturbation checks.

This is a lightweight post-hoc check. It does not refit models; it loads cached
DSP fits and tests whether their local/finite perturbation geometry agrees with
the proportional domain-bump experiment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import pearsonr, spearmanr

from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dsp_canonical_variants_60m import (
    _load_packet as load_60m_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dsp_canonical_variants_300m import (
    VARIANTS,
    FittedDSPModel,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dsp_canonical_variants_300m import (
    _load_packet as load_300m_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dsp_canonical_variants_300m import (
    _predict as predict_dsp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PERTURBATION_DIR = SCRIPT_DIR / "reference_outputs" / "proportional_perturbation_scale_transfer_20260507"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "dsp_split_vs_effective_checks_20260514"
IMG_DIR = OUTPUT_DIR / "img"
SUMMARY_60M = SCRIPT_DIR / "reference_outputs" / "dsp_canonical_variants_60m_20260510" / "summary.csv"
SUMMARY_300M = SCRIPT_DIR / "reference_outputs" / "dsp_canonical_variants_300m_20260510" / "summary.csv"
MODEL_DIR_BY_SCALE = {
    "60m_1p2b": SCRIPT_DIR / "reference_outputs" / "dsp_canonical_variants_60m_20260510",
    "300m_6b": SCRIPT_DIR / "reference_outputs" / "dsp_canonical_variants_300m_20260510",
}
SCALE_LABEL_BY_KEY = {"60m_1p2b": "60M/1.2B", "300m_6b": "100M/6B"}
ACTUAL_EFFECT_COLUMN_BY_SCALE = {"60m_1p2b": "effect_60_bpb", "300m_6b": "effect_100_bpb"}
VARIANT_NAMES = (
    "dsp_effective_exposure_penalty_nnls",
    "dsp_phase_benefit_saturation_penalty_nnls",
)
DERIVATIVE_STEP = 1e-4


def load_model(variant_name: str, scale_key: str) -> FittedDSPModel:
    """Load a cached DSP model."""

    variant_by_name = {variant.name: variant for variant in VARIANTS}
    model_path = MODEL_DIR_BY_SCALE[scale_key] / variant_name / "model.json"
    payload = json.loads(model_path.read_text())
    params = {
        key: np.asarray(value, dtype=float) if isinstance(value, list) else value
        for key, value in payload["params"].items()
    }
    return FittedDSPModel(
        variant=variant_by_name[variant_name],
        params=params,
        intercept=float(payload["intercept"]),
        benefit_coef=np.asarray(payload["benefit_coef"], dtype=float),
        penalty_coef=np.asarray(payload["penalty_coef"], dtype=float),
    )


def safe_pearson(x: pd.Series, y: pd.Series) -> float:
    """Return Pearson correlation, or nan for degenerate inputs."""

    if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return float("nan")
    return float(pearsonr(x, y).statistic)


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    """Return Spearman correlation, or nan for degenerate inputs."""

    if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def calibration_slope(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Slope of observed effect on predicted effect."""

    variance = float(np.var(predicted))
    if variance <= 0.0:
        return float("nan")
    return float(np.cov(predicted, observed, ddof=0)[0, 1] / variance)


def sign_agreement(observed: pd.Series, predicted: pd.Series) -> float:
    """Fraction of rows where predicted and observed perturbation signs agree."""

    return float((np.sign(observed) == np.sign(predicted)).mean())


def top_overlap(frame: pd.DataFrame, actual_column: str, predicted_column: str, helpful: bool, k: int = 8) -> float:
    """Overlap between observed and predicted top-k helpful or harmful domain bumps."""

    selector = pd.DataFrame.nsmallest if helpful else pd.DataFrame.nlargest
    actual = set(selector(frame, k, actual_column)["intervention_id"])
    predicted = set(selector(frame, k, predicted_column)["intervention_id"])
    return len(actual & predicted) / float(k)


def fit_summary() -> pd.DataFrame:
    """Load compact OOF/raw-optimum summaries for both variants and scales."""

    rows = []
    for scale_key, summary_path in [("60m_1p2b", SUMMARY_60M), ("300m_6b", SUMMARY_300M)]:
        summary = pd.read_csv(summary_path)
        keep = summary.loc[summary["variant"].isin(VARIANT_NAMES)].copy()
        keep.insert(0, "scale", scale_key)
        keep.insert(1, "scale_label", SCALE_LABEL_BY_KEY[scale_key])
        rows.append(keep)
    combined = pd.concat(rows, ignore_index=True)
    columns = [
        "scale",
        "scale_label",
        "variant",
        "total_param_count",
        "fitted_gamma",
        "fitted_gamma_benefit",
        "fitted_gamma_saturation",
        "fitted_gamma_penalty",
        "train_rmse",
        "cv_rmse",
        "oof_spearman",
        "oof_pearson",
        "cv_foldmean_regret_at_1",
        "lower_tail_optimism",
        "raw_nearest_observed_tv",
        "raw_nearest_observed_run_name",
        "phase0_max_weight",
        "phase1_max_weight",
    ]
    return combined[columns]


def perturbation_predictions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Predict all domain perturbations using every fit-scale/target-scale pair."""

    manifest = pd.read_csv(PERTURBATION_DIR / "intervention_manifest.csv", low_memory=False)
    paired = pd.read_csv(PERTURBATION_DIR / "paired_bpb_effects.csv")
    domain_manifest = manifest.loc[manifest["intervention_type"].eq("domain_bump")].copy()
    domain_effects = paired.loc[paired["intervention_type"].eq("domain_bump")].copy()
    packets = {"60m_1p2b": load_60m_packet(), "300m_6b": load_300m_packet()}

    prediction_rows: list[dict[str, Any]] = []
    for model_fit_scale, packet in packets.items():
        phase_columns = [
            [f"phase_{phase_idx}_{domain_name}" for domain_name in packet.domain_names] for phase_idx in range(2)
        ]
        base_weights = (
            domain_manifest.set_index("target_domain")["target_mass_before"].reindex(packet.domain_names).to_numpy()
        )
        if np.isnan(base_weights).any():
            missing = [name for name, value in zip(packet.domain_names, base_weights, strict=True) if np.isnan(value)]
            raise ValueError(f"Missing base weights for {model_fit_scale}: {missing}")
        base_weights = base_weights / base_weights.sum()
        proportional_weights = np.stack([base_weights, base_weights], axis=0)

        for variant_name in VARIANT_NAMES:
            model = load_model(variant_name, model_fit_scale)
            base_pred = float(predict_dsp(model, proportional_weights[None, :, :], packet)[0])
            for _, row in domain_manifest.iterrows():
                bumped_weights = np.stack(
                    [
                        row[phase_columns[0]].to_numpy(dtype=float),
                        row[phase_columns[1]].to_numpy(dtype=float),
                    ]
                )
                bump_pred = float(predict_dsp(model, bumped_weights[None, :, :], packet)[0])
                local_weights = proportional_weights + DERIVATIVE_STEP * (bumped_weights - proportional_weights)
                local_pred = float(predict_dsp(model, local_weights[None, :, :], packet)[0])
                effect_row = domain_effects.loc[domain_effects["intervention_id"].eq(row["intervention_id"])].iloc[0]
                finite_effect = bump_pred - base_pred
                local_effect = (local_pred - base_pred) / DERIVATIVE_STEP
                for target_scale, actual_column in ACTUAL_EFFECT_COLUMN_BY_SCALE.items():
                    actual_effect = float(effect_row[actual_column])
                    prediction_rows.append(
                        {
                            "variant": variant_name,
                            "model_fit_scale": model_fit_scale,
                            "model_fit_scale_label": SCALE_LABEL_BY_KEY[model_fit_scale],
                            "target_scale": target_scale,
                            "target_scale_label": SCALE_LABEL_BY_KEY[target_scale],
                            "intervention_id": row["intervention_id"],
                            "target_domain": row["target_domain"],
                            "actual_effect_bpb": actual_effect,
                            "finite_predicted_effect_bpb": finite_effect,
                            "local_predicted_effect_bpb": local_effect,
                            "finite_residual_bpb": actual_effect - finite_effect,
                            "local_residual_bpb": actual_effect - local_effect,
                            "finite_sign_agrees": bool(np.sign(actual_effect) == np.sign(finite_effect)),
                            "local_sign_agrees": bool(np.sign(actual_effect) == np.sign(local_effect)),
                        }
                    )

    predictions = pd.DataFrame.from_records(prediction_rows)
    summary_rows = []
    for keys, group in predictions.groupby(["variant", "model_fit_scale", "target_scale"], sort=False):
        variant, model_fit_scale, target_scale = keys
        summary_rows.append(
            {
                "variant": variant,
                "model_fit_scale": model_fit_scale,
                "target_scale": target_scale,
                "n_domain_bumps": len(group),
                "finite_rmse": float(np.sqrt(np.mean(np.square(group["finite_residual_bpb"])))),
                "finite_mae": float(np.mean(np.abs(group["finite_residual_bpb"]))),
                "finite_pearson": safe_pearson(group["actual_effect_bpb"], group["finite_predicted_effect_bpb"]),
                "finite_spearman": safe_spearman(group["actual_effect_bpb"], group["finite_predicted_effect_bpb"]),
                "finite_sign_agreement": float(group["finite_sign_agrees"].mean()),
                "finite_calibration_slope_obs_on_pred": calibration_slope(
                    group["actual_effect_bpb"].to_numpy(), group["finite_predicted_effect_bpb"].to_numpy()
                ),
                "finite_top8_helpful_overlap": top_overlap(
                    group, "actual_effect_bpb", "finite_predicted_effect_bpb", helpful=True
                ),
                "finite_top8_harmful_overlap": top_overlap(
                    group, "actual_effect_bpb", "finite_predicted_effect_bpb", helpful=False
                ),
                "local_rmse": float(np.sqrt(np.mean(np.square(group["local_residual_bpb"])))),
                "local_mae": float(np.mean(np.abs(group["local_residual_bpb"]))),
                "local_pearson": safe_pearson(group["actual_effect_bpb"], group["local_predicted_effect_bpb"]),
                "local_spearman": safe_spearman(group["actual_effect_bpb"], group["local_predicted_effect_bpb"]),
                "local_sign_agreement": float(group["local_sign_agrees"].mean()),
                "local_calibration_slope_obs_on_pred": calibration_slope(
                    group["actual_effect_bpb"].to_numpy(), group["local_predicted_effect_bpb"].to_numpy()
                ),
                "local_top8_helpful_overlap": top_overlap(
                    group, "actual_effect_bpb", "local_predicted_effect_bpb", helpful=True
                ),
                "local_top8_harmful_overlap": top_overlap(
                    group, "actual_effect_bpb", "local_predicted_effect_bpb", helpful=False
                ),
            }
        )
    summary = pd.DataFrame.from_records(summary_rows)

    interaction_rows = []
    within = predictions.loc[predictions["model_fit_scale"].eq(predictions["target_scale"])].copy()
    for (variant, intervention_id), group in within.groupby(["variant", "intervention_id"], sort=False):
        if set(group["target_scale"]) != set(ACTUAL_EFFECT_COLUMN_BY_SCALE):
            raise ValueError(f"Incomplete within-scale predictions for {variant}/{intervention_id}")
        by_scale = group.set_index("target_scale")
        effect_row = domain_effects.loc[domain_effects["intervention_id"].eq(intervention_id)].iloc[0]
        interaction_rows.append(
            {
                "variant": variant,
                "intervention_id": intervention_id,
                "target_domain": effect_row["target_domain"],
                "actual_scale_interaction_bpb": float(effect_row["scale_interaction_bpb"]),
                "finite_predicted_scale_interaction_bpb": float(
                    by_scale.loc["300m_6b", "finite_predicted_effect_bpb"]
                    - by_scale.loc["60m_1p2b", "finite_predicted_effect_bpb"]
                ),
                "local_predicted_scale_interaction_bpb": float(
                    by_scale.loc["300m_6b", "local_predicted_effect_bpb"]
                    - by_scale.loc["60m_1p2b", "local_predicted_effect_bpb"]
                ),
            }
        )
    interactions = pd.DataFrame.from_records(interaction_rows)
    interactions["finite_scale_interaction_residual_bpb"] = (
        interactions["actual_scale_interaction_bpb"] - interactions["finite_predicted_scale_interaction_bpb"]
    )
    interactions["local_scale_interaction_residual_bpb"] = (
        interactions["actual_scale_interaction_bpb"] - interactions["local_predicted_scale_interaction_bpb"]
    )
    return predictions, summary, interactions


def interaction_summary(interactions: pd.DataFrame) -> pd.DataFrame:
    """Summarize scale-interaction prediction quality."""

    rows = []
    for variant, group in interactions.groupby("variant", sort=False):
        rows.append(
            {
                "variant": variant,
                "finite_interaction_rmse": float(
                    np.sqrt(np.mean(np.square(group["finite_scale_interaction_residual_bpb"])))
                ),
                "finite_interaction_pearson": safe_pearson(
                    group["actual_scale_interaction_bpb"], group["finite_predicted_scale_interaction_bpb"]
                ),
                "finite_interaction_spearman": safe_spearman(
                    group["actual_scale_interaction_bpb"], group["finite_predicted_scale_interaction_bpb"]
                ),
                "local_interaction_rmse": float(
                    np.sqrt(np.mean(np.square(group["local_scale_interaction_residual_bpb"])))
                ),
                "local_interaction_pearson": safe_pearson(
                    group["actual_scale_interaction_bpb"], group["local_predicted_scale_interaction_bpb"]
                ),
                "local_interaction_spearman": safe_spearman(
                    group["actual_scale_interaction_bpb"], group["local_predicted_scale_interaction_bpb"]
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def write_plots(fit: pd.DataFrame, summary: pd.DataFrame, predictions: pd.DataFrame, interactions: pd.DataFrame) -> None:
    """Write compact comparison plots."""

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    fit_long = fit.melt(
        id_vars=["scale_label", "variant"],
        value_vars=["cv_rmse", "oof_spearman", "raw_nearest_observed_tv", "phase0_max_weight", "phase1_max_weight"],
        var_name="metric",
        value_name="value",
    )
    fig = px.bar(
        fit_long,
        x="scale_label",
        y="value",
        color="variant",
        facet_col="metric",
        barmode="group",
        title="DSP fit and raw-optimum diagnostics",
        height=430,
    )
    fig.update_layout(template="plotly_white")
    fig.write_html(IMG_DIR / "fit_and_raw_optimum_diagnostics.html", include_plotlyjs="cdn")
    try:
        fig.write_image(IMG_DIR / "fit_and_raw_optimum_diagnostics.png", scale=2)
    except ValueError:
        pass

    within = summary.loc[summary["model_fit_scale"].eq(summary["target_scale"])].copy()
    check_long = within.melt(
        id_vars=["variant", "target_scale"],
        value_vars=[
            "finite_pearson",
            "finite_spearman",
            "finite_sign_agreement",
            "local_pearson",
            "local_spearman",
            "local_sign_agreement",
        ],
        var_name="metric",
        value_name="value",
    )
    check_long["target_scale_label"] = check_long["target_scale"].map(SCALE_LABEL_BY_KEY)
    fig = px.bar(
        check_long,
        x="target_scale_label",
        y="value",
        color="variant",
        facet_col="metric",
        barmode="group",
        title="Within-scale domain-perturbation agreement",
        height=520,
    )
    fig.update_layout(template="plotly_white")
    fig.write_html(IMG_DIR / "within_scale_perturbation_agreement.html", include_plotlyjs="cdn")
    try:
        fig.write_image(IMG_DIR / "within_scale_perturbation_agreement.png", scale=2)
    except ValueError:
        pass

    scatter_frame = predictions.loc[predictions["model_fit_scale"].eq(predictions["target_scale"])].copy()
    scatter_frame["target_scale_label"] = scatter_frame["target_scale"].map(SCALE_LABEL_BY_KEY)
    fig = px.scatter(
        scatter_frame,
        x="actual_effect_bpb",
        y="finite_predicted_effect_bpb",
        color="variant",
        facet_col="target_scale_label",
        hover_name="target_domain",
        hover_data=["local_predicted_effect_bpb", "finite_sign_agrees", "local_sign_agrees"],
        title="Finite domain-bump predictions vs observed effects",
        labels={
            "actual_effect_bpb": "Observed BPB effect; negative helps",
            "finite_predicted_effect_bpb": "DSP finite predicted effect",
        },
        height=520,
    )
    lo = min(scatter_frame["actual_effect_bpb"].min(), scatter_frame["finite_predicted_effect_bpb"].min())
    hi = max(scatter_frame["actual_effect_bpb"].max(), scatter_frame["finite_predicted_effect_bpb"].max())
    fig.add_shape(type="line", x0=lo, x1=hi, y0=lo, y1=hi, line={"dash": "dot", "color": "#333333"})
    fig.add_hline(y=0, line_dash="dash", line_color="#777777")
    fig.add_vline(x=0, line_dash="dash", line_color="#777777")
    fig.update_layout(template="plotly_white")
    fig.write_html(IMG_DIR / "finite_prediction_vs_actual.html", include_plotlyjs="cdn")
    try:
        fig.write_image(IMG_DIR / "finite_prediction_vs_actual.png", scale=2)
    except ValueError:
        pass

    fig = px.scatter(
        interactions,
        x="actual_scale_interaction_bpb",
        y="finite_predicted_scale_interaction_bpb",
        color="variant",
        hover_name="target_domain",
        title="Predicted vs observed 60M-to-100M scale-interaction effects",
        labels={
            "actual_scale_interaction_bpb": "Observed effect_100 - effect_60 BPB",
            "finite_predicted_scale_interaction_bpb": "Predicted effect_100 - effect_60 BPB",
        },
        height=520,
    )
    lo = min(
        interactions["actual_scale_interaction_bpb"].min(),
        interactions["finite_predicted_scale_interaction_bpb"].min(),
    )
    hi = max(
        interactions["actual_scale_interaction_bpb"].max(),
        interactions["finite_predicted_scale_interaction_bpb"].max(),
    )
    fig.add_shape(type="line", x0=lo, x1=hi, y0=lo, y1=hi, line={"dash": "dot", "color": "#333333"})
    fig.add_hline(y=0, line_dash="dash", line_color="#777777")
    fig.add_vline(x=0, line_dash="dash", line_color="#777777")
    fig.update_layout(template="plotly_white")
    fig.write_html(IMG_DIR / "scale_interaction_prediction_vs_actual.html", include_plotlyjs="cdn")
    try:
        fig.write_image(IMG_DIR / "scale_interaction_prediction_vs_actual.png", scale=2)
    except ValueError:
        pass


def report(
    fit: pd.DataFrame,
    perturbation_summary: pd.DataFrame,
    scale_interaction: pd.DataFrame,
    three_vector: pd.DataFrame,
) -> str:
    """Render a Markdown report."""

    within = perturbation_summary.loc[
        perturbation_summary["model_fit_scale"].eq(perturbation_summary["target_scale"])
    ].copy()
    cross = perturbation_summary.loc[
        ~perturbation_summary["model_fit_scale"].eq(perturbation_summary["target_scale"])
    ].copy()
    fit_table = fit[
        [
            "scale_label",
            "variant",
            "total_param_count",
            "cv_rmse",
            "oof_spearman",
            "raw_nearest_observed_tv",
            "phase0_max_weight",
            "phase1_max_weight",
        ]
    ]
    within_table = within[
        [
            "variant",
            "target_scale",
            "finite_rmse",
            "finite_pearson",
            "finite_spearman",
            "finite_sign_agreement",
            "local_rmse",
            "local_pearson",
            "local_spearman",
            "local_sign_agreement",
        ]
    ]
    cross_table = cross[
        [
            "variant",
            "model_fit_scale",
            "target_scale",
            "finite_rmse",
            "finite_pearson",
            "finite_spearman",
            "finite_sign_agreement",
        ]
    ]
    interaction_table = scale_interaction[
        [
            "variant",
            "finite_interaction_rmse",
            "finite_interaction_pearson",
            "finite_interaction_spearman",
            "local_interaction_rmse",
            "local_interaction_pearson",
            "local_interaction_spearman",
        ]
    ]
    three_vector_keep = three_vector.loc[
        three_vector["model_variant"].isin(VARIANT_NAMES),
        ["model_variant", "comparison", "pearson", "spearman", "cosine", "sign_agreement"],
    ].copy()
    lines = [
        "# Split DSP vs Effective-Exposure DSP Checks",
        "",
        "This report compares the promoted effective-exposure DSP against the split benefit/saturation/penalty DSP.",
        "All perturbation checks use the proportional domain-bump experiment and cached DSP fits; no models are refit here.",
        "",
        "## Fit And Raw-Optimum Diagnostics",
        "",
        fit_table.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Within-Scale Domain-Bump Agreement",
        "",
        "Finite predictions compare `DSP(w_bump) - DSP(w_prop)` against the observed 0.05-domain bump.",
        "Local predictions compare the directional derivative at proportional against that same finite intervention.",
        "",
        within_table.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Cross-Scale Transfer",
        "",
        "Rows below evaluate a model fit at one scale against observed perturbation effects at the other scale.",
        "",
        cross_table.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Scale-Interaction Prediction",
        "",
        "Interaction is `effect_100_bpb - effect_60_bpb`; negative means a perturbation helps more at 100M/6B.",
        "",
        interaction_table.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Existing Three-Vector Alignment At 100M",
        "",
        "This uses the previously generated vectors: measured perturbation gradient, DSP-predicted perturbation gradient, and direction from proportional to the raw DSP optimum.",
        "",
        three_vector_keep.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        "- Split benefit/saturation/penalty DSP improves OOF fit on both scales, but its raw optimum is still extrapolative.",
        "- Effective-exposure DSP has slightly weaker OOF fit but cleaner local-gradient behavior near proportional.",
        "- Split DSP is competitive for finite 0.05 perturbation prediction; it is less convincing as a true local-gradient oracle.",
        "- For validation candidates, treat split DSP as a strong fit/finite-bump surrogate and effective-exposure DSP as the safer local-geometry comparator.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    fit = fit_summary()
    predictions, perturbation_summary, interactions = perturbation_predictions()
    scale_interaction = interaction_summary(interactions)
    three_vector_path = PERTURBATION_DIR / "dsp_three_vector_alignment_100m.csv"
    if three_vector_path.exists():
        three_vector = pd.read_csv(three_vector_path)
    else:
        three_vector = pd.DataFrame(
            columns=["model_variant", "comparison", "pearson", "spearman", "cosine", "sign_agreement"]
        )

    fit.to_csv(OUTPUT_DIR / "fit_summary.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "domain_perturbation_predictions.csv", index=False)
    perturbation_summary.to_csv(OUTPUT_DIR / "domain_perturbation_prediction_summary.csv", index=False)
    interactions.to_csv(OUTPUT_DIR / "scale_interaction_predictions.csv", index=False)
    scale_interaction.to_csv(OUTPUT_DIR / "scale_interaction_prediction_summary.csv", index=False)
    three_vector.to_csv(OUTPUT_DIR / "three_vector_alignment_100m.csv", index=False)
    write_plots(fit, perturbation_summary, predictions, interactions)
    (OUTPUT_DIR / "report.md").write_text(report(fit, perturbation_summary, scale_interaction, three_vector))

    print(fit.to_string(index=False))
    print()
    print(perturbation_summary.to_string(index=False))
    print()
    print(scale_interaction.to_string(index=False))
    print(f"Wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
