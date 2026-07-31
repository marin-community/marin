# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: B018,E501

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "plotly",
#     "scikit-learn",
#     "scipy",
# ]
# ///
"""Marimo notebook for proportional perturbation scale-transfer analysis."""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px

    from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dsp_canonical_variants_60m import (
        _load_packet as load_60m_dsp_packet,
    )
    from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dsp_canonical_variants_300m import (
        VARIANTS as DSP_VARIANTS,
    )
    from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dsp_canonical_variants_300m import (
        FittedDSPModel,
    )
    from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dsp_canonical_variants_300m import (
        _load_packet as load_dsp_packet,
    )
    from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dsp_canonical_variants_300m import (
        _predict as predict_dsp,
    )
    from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.issue5416_aggregate import (
        fit_issue5416_projection,
        score_issue5416_aggregate,
        write_issue5416_projection,
    )

    return (
        DSP_VARIANTS,
        FittedDSPModel,
        Path,
        fit_issue5416_projection,
        json,
        load_60m_dsp_packet,
        load_dsp_packet,
        mo,
        np,
        pd,
        predict_dsp,
        px,
        score_issue5416_aggregate,
        write_issue5416_projection,
    )


@app.cell
def _(Path):
    TWO_PHASE_ROOT = Path(__file__).resolve().parent
    OUTPUT_DIR = TWO_PHASE_ROOT / "reference_outputs" / "proportional_perturbation_scale_transfer_20260507"
    IMG_DIR = OUTPUT_DIR / "img"
    RUN_REGISTRY_CSV = TWO_PHASE_ROOT / "run_registry" / "logical_runs.csv"
    METRICS_WIDE_CSV = TWO_PHASE_ROOT / "metric_registry" / "metrics_wide.csv"
    RAW_MATRIX_DIR = TWO_PHASE_ROOT / "metric_registry" / "raw_metric_matrix_300m"
    RAW_MATRIX_CSV = RAW_MATRIX_DIR / "raw_metric_matrix_300m.csv"
    VARIABLE_NOISE_CSV = RAW_MATRIX_DIR / "noise_baseline_run00097_variable_subset_300m.csv"
    DSP_OUTPUT_DIR = TWO_PHASE_ROOT / "reference_outputs" / "dsp_canonical_variants_300m_20260510"
    DSP_OUTPUT_60M_DIR = TWO_PHASE_ROOT / "reference_outputs" / "dsp_canonical_variants_60m_20260510"
    DSP_MODEL_VARIANTS = [
        "dsp_effective_exposure_penalty_nnls",
        "dsp_phase_benefit_saturation_penalty_nnls",
    ]
    DSP_SCALE_MODEL_DIRS = {
        "60m_1p2b": DSP_OUTPUT_60M_DIR,
        "300m_6b": DSP_OUTPUT_DIR,
    }
    INTERVENTION_MANIFEST_CSV = OUTPUT_DIR / "intervention_manifest.csv"
    BPB_METRIC = "eval/uncheatable_eval/bpb"
    SCALE_ORDER = ["60m_1p2b", "300m_6b"]
    PERTURBATION_FAMILIES = {"proportional_perturbation_60m_1p2b", "proportional_perturbation_300m_6b"}
    COLOR_MAP = {
        "domain_bump": "#4C78A8",
        "family_bump": "#F58518",
        "quality_swap": "#54A24B",
        "baseline": "#222222",
    }
    return (
        BPB_METRIC,
        COLOR_MAP,
        DSP_MODEL_VARIANTS,
        DSP_OUTPUT_DIR,
        DSP_SCALE_MODEL_DIRS,
        IMG_DIR,
        INTERVENTION_MANIFEST_CSV,
        METRICS_WIDE_CSV,
        OUTPUT_DIR,
        PERTURBATION_FAMILIES,
        RAW_MATRIX_CSV,
        RUN_REGISTRY_CSV,
        SCALE_ORDER,
        VARIABLE_NOISE_CSV,
    )


@app.cell
def _(mo):
    mo.md(
        """
    # Proportional perturbation scale-transfer analysis

    This notebook analyzes the Stage 1 interventions around the proportional mixture.
    Each intervention is trained at `60M/1.2B` and `100M/6B`, with phase-constant
    weights. The primary completed metric is `eval/uncheatable_eval/bpb`.

    Sign convention:

    | Quantity | Definition | Helpful Direction |
    | :--- | :--- | :--- |
    | `effect_60_bpb` | `BPB_60(intervention) - BPB_60(proportional)` | negative |
    | `effect_100_bpb` | `BPB_100(intervention) - BPB_100(proportional)` | negative |
    | `scale_interaction_bpb` | `effect_100_bpb - effect_60_bpb` | negative means the perturbation helps more at 100M |
    | `effect_*_issue5416_aggregate` | aggregate intervention minus aggregate proportional | positive |
    """
    )
    return


@app.cell
def _(METRICS_WIDE_CSV, PERTURBATION_FAMILIES, RUN_REGISTRY_CSV, mo, pd):
    logical = pd.read_csv(RUN_REGISTRY_CSV, low_memory=False)
    metrics = pd.read_csv(METRICS_WIDE_CSV, low_memory=False)
    perturbation_registry = logical.loc[logical["family"].isin(PERTURBATION_FAMILIES)].copy()
    perturbation_metrics = metrics.loc[metrics["family"].isin(PERTURBATION_FAMILIES)].copy()
    registry_count = len(perturbation_registry)
    metric_count = len(perturbation_metrics)
    intervention_count = perturbation_registry["intervention_id"].nunique()
    mo.md(
        f"""
        ## Data coverage

        | Source | Rows | Unique interventions |
        | :--- | ---: | ---: |
        | run registry perturbation rows | {registry_count} | {intervention_count} |
        | metric registry perturbation rows | {metric_count} | {perturbation_metrics["intervention_id"].nunique()} |
        """
    )
    return metrics, perturbation_metrics


@app.cell
def _(BPB_METRIC, SCALE_ORDER, metrics, pd):
    def _proportional_anchor(scale: str) -> pd.Series:
        frame = metrics.loc[(metrics["run_name"].eq("baseline_proportional")) & (metrics["scale"].eq(scale))].copy()
        frame = frame.loc[frame[BPB_METRIC].notna()]
        if scale == "60m_1p2b":
            preferred = frame.loc[frame["source_experiment"].eq("pinlin_calvin_xu/data_mixture/ngd3dm2_hybrid_canary")]
        else:
            preferred = frame.loc[
                frame["source_experiment"].eq("pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b")
            ]
        if not preferred.empty:
            frame = preferred
        if frame.empty:
            raise ValueError(f"Missing proportional anchor for {scale}")
        return frame.iloc[0]

    anchors = pd.DataFrame([_proportional_anchor(scale).to_dict() for scale in SCALE_ORDER])
    return (anchors,)


@app.cell
def _(BPB_METRIC, anchors, perturbation_metrics):
    required_metric_rows = 110
    if len(perturbation_metrics) != required_metric_rows:
        raise ValueError(f"Expected {required_metric_rows} perturbation metric rows, found {len(perturbation_metrics)}")
    if perturbation_metrics.duplicated(["scale", "intervention_id"]).any():
        raise ValueError("Duplicate perturbation rows by scale and intervention_id")
    if perturbation_metrics[BPB_METRIC].isna().any():
        missing = perturbation_metrics.loc[
            perturbation_metrics[BPB_METRIC].isna(), ["scale", "intervention_id", "run_name"]
        ]
        raise ValueError(f"Perturbation BPB is incomplete:\n{missing.to_string(index=False)}")

    anchor_by_scale = anchors.set_index("scale")[BPB_METRIC].to_dict()
    effect_rows = perturbation_metrics.copy()
    effect_rows["proportional_bpb"] = effect_rows["scale"].map(anchor_by_scale)
    effect_rows["effect_bpb"] = effect_rows[BPB_METRIC] - effect_rows["proportional_bpb"]
    paired = (
        effect_rows.pivot(
            index=[
                "intervention_id",
                "intervention_type",
                "target_unit",
                "target_domain",
                "target_family",
                "quality_high_domain",
                "quality_low_domain",
                "tv_distance",
                "target_mass_before",
                "target_mass_after",
            ],
            columns="scale",
            values="effect_bpb",
        )
        .reset_index()
        .rename(columns={"60m_1p2b": "effect_60_bpb", "300m_6b": "effect_100_bpb"})
    )
    paired["scale_interaction_bpb"] = paired["effect_100_bpb"] - paired["effect_60_bpb"]
    if len(paired) != 55:
        raise ValueError(f"Expected 55 paired interventions, found {len(paired)}")
    return (paired,)


@app.cell
def _(IMG_DIR, OUTPUT_DIR, paired):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    paired.to_csv(OUTPUT_DIR / "paired_bpb_effects.csv", index=False)
    return


@app.cell
def _(COLOR_MAP, IMG_DIR, paired, px):
    def _save_plot(fig, stem: str) -> None:
        fig.write_html(IMG_DIR / f"{stem}.html", include_plotlyjs="cdn")
        try:
            fig.write_image(IMG_DIR / f"{stem}.png", scale=2)
        except ValueError:
            pass

    plot_frame = paired.sort_values("effect_60_bpb")
    fig60 = px.bar(
        plot_frame,
        x="effect_60_bpb",
        y="intervention_id",
        color="intervention_type",
        color_discrete_map=COLOR_MAP,
        orientation="h",
        title="60M/1.2B BPB effect vs proportional",
        labels={"effect_60_bpb": "BPB effect; negative helps", "intervention_id": "intervention"},
        height=1100,
    )
    fig60.add_vline(x=0, line_dash="dash", line_color="#333333")
    fig60.update_layout(template="plotly_white", legend_title_text="Intervention type")
    _save_plot(fig60, "ranked_60m_bpb_effects")

    plot_frame = paired.sort_values("effect_100_bpb")
    fig100 = px.bar(
        plot_frame,
        x="effect_100_bpb",
        y="intervention_id",
        color="intervention_type",
        color_discrete_map=COLOR_MAP,
        orientation="h",
        title="100M/6B BPB effect vs proportional",
        labels={"effect_100_bpb": "BPB effect; negative helps", "intervention_id": "intervention"},
        height=1100,
    )
    fig100.add_vline(x=0, line_dash="dash", line_color="#333333")
    fig100.update_layout(template="plotly_white", legend_title_text="Intervention type")
    _save_plot(fig100, "ranked_100m_bpb_effects")

    scatter = px.scatter(
        paired,
        x="effect_60_bpb",
        y="effect_100_bpb",
        color="intervention_type",
        color_discrete_map=COLOR_MAP,
        hover_name="intervention_id",
        hover_data=["target_unit", "tv_distance", "scale_interaction_bpb"],
        title="Perturbation effect at 60M vs 100M",
        labels={"effect_60_bpb": "60M BPB effect", "effect_100_bpb": "100M BPB effect"},
        height=650,
    )
    lo = min(float(paired["effect_60_bpb"].min()), float(paired["effect_100_bpb"].min()))
    hi = max(float(paired["effect_60_bpb"].max()), float(paired["effect_100_bpb"].max()))
    scatter.add_trace(
        dict(
            type="scatter",
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line={"dash": "dot", "color": "#333333"},
            name="equal effect",
        )
    )
    scatter.add_hline(y=0, line_dash="dash", line_color="#777777")
    scatter.add_vline(x=0, line_dash="dash", line_color="#777777")
    scatter.update_layout(template="plotly_white")
    _save_plot(scatter, "bpb_effect_60m_vs_100m_scatter")

    interaction = paired.sort_values("scale_interaction_bpb")
    fig_interaction = px.scatter(
        interaction,
        x="scale_interaction_bpb",
        y="intervention_id",
        color="intervention_type",
        color_discrete_map=COLOR_MAP,
        hover_data=["effect_60_bpb", "effect_100_bpb", "target_unit", "tv_distance"],
        title="Scale interaction: 100M effect minus 60M effect",
        labels={
            "scale_interaction_bpb": "BPB interaction; negative helps more at 100M",
            "intervention_id": "intervention",
        },
        height=1100,
    )
    for _, row in interaction.iterrows():
        fig_interaction.add_shape(
            type="line",
            x0=0,
            y0=row["intervention_id"],
            x1=row["scale_interaction_bpb"],
            y1=row["intervention_id"],
            line={"color": "rgba(120,120,120,0.35)", "width": 1},
        )
    fig_interaction.add_vline(x=0, line_dash="dash", line_color="#333333")
    fig_interaction.update_layout(template="plotly_white", legend_title_text="Intervention type")
    _save_plot(fig_interaction, "scale_interaction_bpb_lollipop")

    tv_fig = px.scatter(
        paired,
        x="tv_distance",
        y="scale_interaction_bpb",
        color="intervention_type",
        color_discrete_map=COLOR_MAP,
        hover_name="intervention_id",
        hover_data=["effect_60_bpb", "effect_100_bpb", "target_unit"],
        title="Scale interaction vs TV distance from proportional",
        labels={"tv_distance": "TV distance", "scale_interaction_bpb": "BPB interaction"},
        height=600,
    )
    tv_fig.add_hline(y=0, line_dash="dash", line_color="#333333")
    tv_fig.update_layout(template="plotly_white")
    _save_plot(tv_fig, "scale_interaction_vs_tv_distance")
    return fig100, fig60, fig_interaction, scatter


@app.cell
def _(fig60):
    fig60
    return


@app.cell
def _(fig100):
    fig100
    return


@app.cell
def _(scatter):
    scatter
    return


@app.cell
def _(fig_interaction):
    fig_interaction
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## DSP agreement with domain perturbations

    The next diagnostics test whether the effective-exposure DSP surrogate agrees with the actual
    one-at-a-time domain perturbation experiment around proportional. We use only the
    39 domain bumps here, because family bumps and quality swaps are composed directions.

    Two comparisons are useful:

    | Diagnostic | Definition | Interpretation |
    | :--- | :--- | :--- |
    | finite prediction | `DSP(w_bump) - DSP(w_proportional)` | Does DSP predict the measured 0.05-TV intervention effect? |
    | local directional derivative | derivative of DSP at proportional in the exact bump direction | Does DSP's local geometry point in the measured helpful/harmful directions? |
    """
    )
    return


@app.cell
def _(
    DSP_MODEL_VARIANTS,
    DSP_OUTPUT_DIR,
    DSP_VARIANTS,
    FittedDSPModel,
    INTERVENTION_MANIFEST_CSV,
    OUTPUT_DIR,
    json,
    load_60m_dsp_packet,
    load_dsp_packet,
    np,
    paired,
    pd,
    predict_dsp,
):
    dsp_packet = load_dsp_packet()
    dsp_agreement_variant_by_name = {variant.name: variant for variant in DSP_VARIANTS}

    def _load_dsp_model(dsp_variant_name: str) -> FittedDSPModel:
        model_path = DSP_OUTPUT_DIR / dsp_variant_name / "model.json"
        model_payload = json.loads(model_path.read_text())
        params = {
            key: np.asarray(value, dtype=float) if isinstance(value, list) else value
            for key, value in model_payload["params"].items()
        }
        return FittedDSPModel(
            variant=dsp_agreement_variant_by_name[dsp_variant_name],
            params=params,
            intercept=float(model_payload["intercept"]),
            benefit_coef=np.asarray(model_payload["benefit_coef"], dtype=float),
            penalty_coef=np.asarray(model_payload["penalty_coef"], dtype=float),
        )

    manifest = pd.read_csv(INTERVENTION_MANIFEST_CSV, low_memory=False)
    domain_manifest = manifest.loc[manifest["intervention_type"].eq("domain_bump")].copy()
    domain_effects = paired.loc[paired["intervention_type"].eq("domain_bump")].copy()
    if len(domain_manifest) != len(dsp_packet.domain_names):
        raise ValueError(f"Expected {len(dsp_packet.domain_names)} domain-bump rows, found {len(domain_manifest)}")

    base_weights = (
        domain_manifest.set_index("target_domain")["target_mass_before"].reindex(dsp_packet.domain_names).to_numpy()
    )
    if np.isnan(base_weights).any():
        dsp_missing_base_domains = [
            name for name, value in zip(dsp_packet.domain_names, base_weights, strict=True) if np.isnan(value)
        ]
        raise ValueError(f"Missing proportional base weights for DSP domains: {dsp_missing_base_domains}")
    base_weights = base_weights / base_weights.sum()
    proportional_weights = np.stack([base_weights, base_weights], axis=0)

    phase_columns = [
        [f"phase_{phase_idx}_{domain_name}" for domain_name in dsp_packet.domain_names] for phase_idx in range(2)
    ]

    def _weights_from_manifest_row(row: pd.Series) -> np.ndarray:
        return np.stack([row[phase_columns[0]].to_numpy(dtype=float), row[phase_columns[1]].to_numpy(dtype=float)])

    def _safe_corr(frame: pd.DataFrame, left: str, right: str, method: str) -> float:
        return float(frame[left].corr(frame[right], method=method))

    agreement_rows = []
    derivative_step = 1e-4
    for dsp_agreement_variant_name in DSP_MODEL_VARIANTS:
        model = _load_dsp_model(dsp_agreement_variant_name)
        base_pred = float(predict_dsp(model, proportional_weights[None, :, :], dsp_packet)[0])
        for _, dsp_manifest_row in domain_manifest.iterrows():
            bumped_weights = _weights_from_manifest_row(dsp_manifest_row)
            bump_pred = float(predict_dsp(model, bumped_weights[None, :, :], dsp_packet)[0])
            local_weights = proportional_weights + derivative_step * (bumped_weights - proportional_weights)
            local_pred = float(predict_dsp(model, local_weights[None, :, :], dsp_packet)[0])
            agreement_rows.append(
                {
                    "model_variant": dsp_agreement_variant_name,
                    "intervention_id": dsp_manifest_row["intervention_id"],
                    "target_domain": dsp_manifest_row["target_domain"],
                    "tv_distance": float(dsp_manifest_row["tv_distance"]),
                    "dsp_base_pred_bpb": base_pred,
                    "dsp_finite_effect_bpb": bump_pred - base_pred,
                    "dsp_local_directional_effect_bpb": (local_pred - base_pred) / derivative_step,
                    "dsp_finite_effect_per_tv": (bump_pred - base_pred) / float(dsp_manifest_row["tv_distance"]),
                    "dsp_local_directional_effect_per_tv": (
                        ((local_pred - base_pred) / derivative_step) / float(dsp_manifest_row["tv_distance"])
                    ),
                }
            )

    dsp_domain_agreement = pd.DataFrame.from_records(agreement_rows).merge(
        domain_effects[
            [
                "intervention_id",
                "target_unit",
                "effect_60_bpb",
                "effect_100_bpb",
                "scale_interaction_bpb",
            ]
        ],
        on="intervention_id",
        how="left",
        validate="many_to_one",
    )
    dsp_domain_agreement["observed_100_effect_per_tv"] = (
        dsp_domain_agreement["effect_100_bpb"] / dsp_domain_agreement["tv_distance"]
    )
    dsp_domain_agreement["observed_60_effect_per_tv"] = (
        dsp_domain_agreement["effect_60_bpb"] / dsp_domain_agreement["tv_distance"]
    )
    dsp_domain_agreement["finite_sign_agrees_100"] = np.sign(dsp_domain_agreement["dsp_finite_effect_bpb"]) == np.sign(
        dsp_domain_agreement["effect_100_bpb"]
    )
    dsp_domain_agreement["local_sign_agrees_100"] = np.sign(
        dsp_domain_agreement["dsp_local_directional_effect_bpb"]
    ) == np.sign(dsp_domain_agreement["effect_100_bpb"])

    summary_rows = []
    for dsp_summary_variant_name, group in dsp_domain_agreement.groupby("model_variant", sort=False):
        observed_helpful = set(group.nsmallest(8, "effect_100_bpb")["intervention_id"])
        observed_harmful = set(group.nlargest(8, "effect_100_bpb")["intervention_id"])
        finite_helpful = set(group.nsmallest(8, "dsp_finite_effect_bpb")["intervention_id"])
        finite_harmful = set(group.nlargest(8, "dsp_finite_effect_bpb")["intervention_id"])
        local_helpful = set(group.nsmallest(8, "dsp_local_directional_effect_bpb")["intervention_id"])
        local_harmful = set(group.nlargest(8, "dsp_local_directional_effect_bpb")["intervention_id"])
        finite_slope = float(
            np.cov(group["dsp_finite_effect_bpb"], group["effect_100_bpb"], ddof=0)[0, 1]
            / np.var(group["dsp_finite_effect_bpb"])
        )
        local_slope = float(
            np.cov(group["dsp_local_directional_effect_bpb"], group["effect_100_bpb"], ddof=0)[0, 1]
            / np.var(group["dsp_local_directional_effect_bpb"])
        )
        summary_rows.append(
            {
                "model_variant": dsp_summary_variant_name,
                "n_domain_bumps": len(group),
                "finite_pearson_100": _safe_corr(group, "dsp_finite_effect_bpb", "effect_100_bpb", "pearson"),
                "finite_spearman_100": _safe_corr(group, "dsp_finite_effect_bpb", "effect_100_bpb", "spearman"),
                "finite_sign_agreement_100": float(group["finite_sign_agrees_100"].mean()),
                "finite_calibration_slope_obs_on_pred": finite_slope,
                "finite_top8_helpful_overlap": len(observed_helpful & finite_helpful) / 8.0,
                "finite_top8_harmful_overlap": len(observed_harmful & finite_harmful) / 8.0,
                "local_pearson_100": _safe_corr(group, "dsp_local_directional_effect_bpb", "effect_100_bpb", "pearson"),
                "local_spearman_100": _safe_corr(
                    group, "dsp_local_directional_effect_bpb", "effect_100_bpb", "spearman"
                ),
                "local_sign_agreement_100": float(group["local_sign_agrees_100"].mean()),
                "local_calibration_slope_obs_on_pred": local_slope,
                "local_top8_helpful_overlap": len(observed_helpful & local_helpful) / 8.0,
                "local_top8_harmful_overlap": len(observed_harmful & local_harmful) / 8.0,
            }
        )

    dsp_domain_agreement_summary = pd.DataFrame.from_records(summary_rows)
    dsp_domain_agreement.to_csv(OUTPUT_DIR / "dsp_domain_perturbation_agreement.csv", index=False)
    dsp_domain_agreement_summary.to_csv(OUTPUT_DIR / "dsp_domain_perturbation_agreement_summary.csv", index=False)
    return dsp_domain_agreement, dsp_domain_agreement_summary


@app.cell
def _(IMG_DIR, dsp_domain_agreement, px):
    finite_fig = px.scatter(
        dsp_domain_agreement,
        x="effect_100_bpb",
        y="dsp_finite_effect_bpb",
        color="model_variant",
        hover_name="target_domain",
        hover_data=[
            "effect_60_bpb",
            "effect_100_bpb",
            "dsp_finite_effect_bpb",
            "dsp_local_directional_effect_bpb",
            "scale_interaction_bpb",
        ],
        title="DSP finite predictions vs observed 100M domain-bump effects",
        labels={
            "effect_100_bpb": "Observed 100M BPB effect; negative helps",
            "dsp_finite_effect_bpb": "DSP predicted finite effect",
        },
        height=650,
    )
    dsp_finite_lo = min(
        float(dsp_domain_agreement["effect_100_bpb"].min()),
        float(dsp_domain_agreement["dsp_finite_effect_bpb"].min()),
    )
    dsp_finite_hi = max(
        float(dsp_domain_agreement["effect_100_bpb"].max()),
        float(dsp_domain_agreement["dsp_finite_effect_bpb"].max()),
    )
    finite_fig.add_trace(
        dict(
            type="scatter",
            x=[dsp_finite_lo, dsp_finite_hi],
            y=[dsp_finite_lo, dsp_finite_hi],
            mode="lines",
            line={"dash": "dot", "color": "#333333"},
            name="perfect agreement",
        )
    )
    finite_fig.add_hline(y=0, line_dash="dash", line_color="#777777")
    finite_fig.add_vline(x=0, line_dash="dash", line_color="#777777")
    finite_fig.update_layout(template="plotly_white")
    finite_fig.write_html(IMG_DIR / "dsp_domain_finite_prediction_agreement.html", include_plotlyjs="cdn")
    try:
        finite_fig.write_image(IMG_DIR / "dsp_domain_finite_prediction_agreement.png", scale=2)
    except ValueError:
        pass

    local_fig = px.scatter(
        dsp_domain_agreement,
        x="effect_100_bpb",
        y="dsp_local_directional_effect_bpb",
        color="model_variant",
        hover_name="target_domain",
        hover_data=[
            "effect_60_bpb",
            "effect_100_bpb",
            "dsp_finite_effect_bpb",
            "dsp_local_directional_effect_bpb",
            "scale_interaction_bpb",
        ],
        title="DSP local directional effects vs observed 100M domain-bump effects",
        labels={
            "effect_100_bpb": "Observed 100M BPB effect; negative helps",
            "dsp_local_directional_effect_bpb": "DSP local effect along bump direction",
        },
        height=650,
    )
    dsp_local_lo = min(
        float(dsp_domain_agreement["effect_100_bpb"].min()),
        float(dsp_domain_agreement["dsp_local_directional_effect_bpb"].min()),
    )
    dsp_local_hi = max(
        float(dsp_domain_agreement["effect_100_bpb"].max()),
        float(dsp_domain_agreement["dsp_local_directional_effect_bpb"].max()),
    )
    local_fig.add_trace(
        dict(
            type="scatter",
            x=[dsp_local_lo, dsp_local_hi],
            y=[dsp_local_lo, dsp_local_hi],
            mode="lines",
            line={"dash": "dot", "color": "#333333"},
            name="perfect agreement",
        )
    )
    local_fig.add_hline(y=0, line_dash="dash", line_color="#777777")
    local_fig.add_vline(x=0, line_dash="dash", line_color="#777777")
    local_fig.update_layout(template="plotly_white")
    local_fig.write_html(IMG_DIR / "dsp_domain_local_gradient_agreement.html", include_plotlyjs="cdn")
    try:
        local_fig.write_image(IMG_DIR / "dsp_domain_local_gradient_agreement.png", scale=2)
    except ValueError:
        pass
    return finite_fig, local_fig


@app.cell
def _(dsp_domain_agreement_summary, finite_fig, local_fig, mo):
    mo.vstack(
        [
            mo.md("### DSP agreement summary"),
            mo.ui.table(dsp_domain_agreement_summary, page_size=5),
            finite_fig,
            local_fig,
        ]
    )
    return


@app.cell
def _(
    DSP_OUTPUT_DIR,
    DSP_MODEL_VARIANTS,
    DSP_SCALE_MODEL_DIRS,
    DSP_VARIANTS,
    FittedDSPModel,
    INTERVENTION_MANIFEST_CSV,
    OUTPUT_DIR,
    json,
    load_60m_dsp_packet,
    load_dsp_packet,
    np,
    paired,
    pd,
    predict_dsp,
):
    scale_label_by_key = {"60m_1p2b": "60M/1.2B", "300m_6b": "100M/6B"}
    actual_effect_column_by_scale = {"60m_1p2b": "effect_60_bpb", "300m_6b": "effect_100_bpb"}
    scale_plot_dsp_variant_by_name = {variant.name: variant for variant in DSP_VARIANTS}
    dsp_packet_by_scale = {
        "60m_1p2b": load_60m_dsp_packet(),
        "300m_6b": load_dsp_packet(),
    }

    def _load_scale_specific_dsp_model(scale_key: str, scale_variant_name: str) -> FittedDSPModel:
        scale_model_dir = DSP_SCALE_MODEL_DIRS.get(scale_key, DSP_OUTPUT_DIR)
        model_path = scale_model_dir / scale_variant_name / "model.json"
        model_payload = json.loads(model_path.read_text())
        params = {
            key: np.asarray(value, dtype=float) if isinstance(value, list) else value
            for key, value in model_payload["params"].items()
        }
        return FittedDSPModel(
            variant=scale_plot_dsp_variant_by_name[scale_variant_name],
            params=params,
            intercept=float(model_payload["intercept"]),
            benefit_coef=np.asarray(model_payload["benefit_coef"], dtype=float),
            penalty_coef=np.asarray(model_payload["penalty_coef"], dtype=float),
        )

    scale_plot_manifest = pd.read_csv(INTERVENTION_MANIFEST_CSV, low_memory=False)
    scale_plot_domain_manifest = scale_plot_manifest.loc[
        scale_plot_manifest["intervention_type"].eq("domain_bump")
    ].copy()

    scale_specific_rows = []
    domain_effects_for_scale_plot = paired.loc[paired["intervention_type"].eq("domain_bump")].copy()
    for scale_key, actual_effect_column in actual_effect_column_by_scale.items():
        scale_packet = dsp_packet_by_scale[scale_key]
        scale_plot_base_weights = (
            scale_plot_domain_manifest.set_index("target_domain")["target_mass_before"]
            .reindex(scale_packet.domain_names)
            .to_numpy()
        )
        scale_plot_base_weights = scale_plot_base_weights / scale_plot_base_weights.sum()
        scale_plot_proportional_weights = np.stack([scale_plot_base_weights, scale_plot_base_weights], axis=0)
        scale_plot_phase_columns = [
            [f"phase_{phase_idx}_{domain_name}" for domain_name in scale_packet.domain_names] for phase_idx in range(2)
        ]
        for scale_dsp_variant_name in DSP_MODEL_VARIANTS:
            scale_model = _load_scale_specific_dsp_model(scale_key, scale_dsp_variant_name)
            scale_base_pred = float(
                predict_dsp(scale_model, scale_plot_proportional_weights[None, :, :], scale_packet)[0]
            )
            for _, scale_plot_row in scale_plot_domain_manifest.iterrows():
                scale_weights = np.stack(
                    [
                        scale_plot_row[scale_plot_phase_columns[0]].to_numpy(dtype=float),
                        scale_plot_row[scale_plot_phase_columns[1]].to_numpy(dtype=float),
                    ]
                )
                scale_pred = float(predict_dsp(scale_model, scale_weights[None, :, :], scale_packet)[0])
                observed_row = domain_effects_for_scale_plot.loc[
                    domain_effects_for_scale_plot["intervention_id"].eq(scale_plot_row["intervention_id"])
                ].iloc[0]
                actual_effect = float(observed_row[actual_effect_column])
                predicted_effect = scale_pred - scale_base_pred
                sign_agrees = bool(np.sign(actual_effect) == np.sign(predicted_effect))
                scale_specific_rows.append(
                    {
                        "scale": scale_key,
                        "scale_label": scale_label_by_key[scale_key],
                        "model_variant": scale_dsp_variant_name,
                        "model_fit_scale": scale_key,
                        "intervention_id": scale_plot_row["intervention_id"],
                        "target_domain": scale_plot_row["target_domain"],
                        "actual_effect_bpb": actual_effect,
                        "dsp_predicted_effect_bpb": predicted_effect,
                        "prediction_residual_bpb": actual_effect - predicted_effect,
                        "sign_agrees": sign_agrees,
                        "sign_status": "agree" if sign_agrees else "SIGN DISAGREE",
                    }
                )

    dsp_scale_specific_perturbation_predictions = pd.DataFrame.from_records(scale_specific_rows)
    dsp_scale_specific_perturbation_predictions.to_csv(
        OUTPUT_DIR / "dsp_scale_specific_domain_perturbation_predictions.csv", index=False
    )
    return (dsp_scale_specific_perturbation_predictions,)


@app.cell
def _(IMG_DIR, dsp_scale_specific_perturbation_predictions, px):
    dsp_scale_plot_frame = dsp_scale_specific_perturbation_predictions.copy()
    dsp_scale_plot_frame["abs_residual_bpb"] = dsp_scale_plot_frame["prediction_residual_bpb"].abs()
    dsp_scale_prediction_fig = px.scatter(
        dsp_scale_plot_frame,
        x="actual_effect_bpb",
        y="dsp_predicted_effect_bpb",
        facet_col="scale_label",
        color="sign_status",
        symbol="sign_status",
        size="abs_residual_bpb",
        color_discrete_map={"agree": "#2563eb", "SIGN DISAGREE": "#dc2626"},
        symbol_map={"agree": "circle", "SIGN DISAGREE": "x"},
        hover_name="target_domain",
        hover_data=[
            "intervention_id",
            "actual_effect_bpb",
            "dsp_predicted_effect_bpb",
            "prediction_residual_bpb",
        ],
        title="Scale-specific effective-exposure DSP predictions vs actual domain perturbation effects",
        labels={
            "actual_effect_bpb": "Actual BPB effect vs proportional; negative helps",
            "dsp_predicted_effect_bpb": "DSP predicted BPB effect vs proportional",
            "sign_status": "Sign",
            "scale_label": "Scale",
        },
        height=650,
    )
    dsp_scale_axis_lo = min(
        float(dsp_scale_plot_frame["actual_effect_bpb"].min()),
        float(dsp_scale_plot_frame["dsp_predicted_effect_bpb"].min()),
    )
    dsp_scale_axis_hi = max(
        float(dsp_scale_plot_frame["actual_effect_bpb"].max()),
        float(dsp_scale_plot_frame["dsp_predicted_effect_bpb"].max()),
    )
    for dsp_scale_col in range(1, 3):
        dsp_scale_prediction_fig.add_shape(
            type="line",
            x0=dsp_scale_axis_lo,
            x1=dsp_scale_axis_hi,
            y0=dsp_scale_axis_lo,
            y1=dsp_scale_axis_hi,
            line={"dash": "dot", "color": "#333333"},
            row=1,
            col=dsp_scale_col,
        )
        dsp_scale_prediction_fig.add_hline(y=0, line_dash="dash", line_color="#777777", row=1, col=dsp_scale_col)
        dsp_scale_prediction_fig.add_vline(x=0, line_dash="dash", line_color="#777777", row=1, col=dsp_scale_col)
    dsp_scale_prediction_fig.update_traces(marker={"line": {"width": 1, "color": "white"}})
    dsp_scale_prediction_fig.update_layout(template="plotly_white")
    dsp_scale_prediction_fig.write_html(
        IMG_DIR / "dsp_scale_specific_domain_prediction_vs_actual.html", include_plotlyjs="cdn"
    )
    try:
        dsp_scale_prediction_fig.write_image(IMG_DIR / "dsp_scale_specific_domain_prediction_vs_actual.png", scale=2)
    except ValueError:
        pass
    dsp_scale_prediction_fig
    return


@app.cell
def _(COLOR_MAP, IMG_DIR, OUTPUT_DIR, paired, px):
    quality_swaps = paired.loc[paired["intervention_type"].eq("quality_swap")].sort_values("scale_interaction_bpb")
    quality_swaps.to_csv(OUTPUT_DIR / "quality_swap_bpb_effects.csv", index=False)
    quality_fig = px.bar(
        quality_swaps,
        x="scale_interaction_bpb",
        y="intervention_id",
        color="intervention_type",
        color_discrete_map=COLOR_MAP,
        orientation="h",
        title="Quality swap scale interactions",
        labels={
            "scale_interaction_bpb": "BPB interaction; negative helps more at 100M",
            "intervention_id": "quality swap",
        },
        height=520,
    )
    quality_fig.add_vline(x=0, line_dash="dash", line_color="#333333")
    quality_fig.update_layout(template="plotly_white", showlegend=False)
    quality_fig.write_html(IMG_DIR / "quality_swap_bpb_scale_interactions.html", include_plotlyjs="cdn")
    try:
        quality_fig.write_image(IMG_DIR / "quality_swap_bpb_scale_interactions.png", scale=2)
    except ValueError:
        pass
    quality_fig
    return (quality_swaps,)


@app.cell
def _(
    OUTPUT_DIR,
    RAW_MATRIX_CSV,
    VARIABLE_NOISE_CSV,
    fit_issue5416_projection,
    metrics,
    pd,
    score_issue5416_aggregate,
    write_issue5416_projection,
):
    aggregate_ready = False
    aggregate_message = "Issue #5416 aggregate not scored yet."
    aggregate_effect_rows = pd.DataFrame()
    aggregate_paired = pd.DataFrame()
    if RAW_MATRIX_CSV.exists() and VARIABLE_NOISE_CSV.exists():
        raw_matrix = pd.read_csv(RAW_MATRIX_CSV, low_memory=False)
        variable_noise = pd.read_csv(VARIABLE_NOISE_CSV, low_memory=False)
        projection = fit_issue5416_projection(signal_frame=raw_matrix, noise_frame=variable_noise)
        write_issue5416_projection(projection, OUTPUT_DIR / "issue5416_projection.json")
        score_frame = metrics.copy()
        required_columns = list(projection.task_columns)
        candidate_rows = score_frame.loc[
            score_frame["family"].isin({"proportional_perturbation_60m_1p2b", "proportional_perturbation_300m_6b"})
            | (score_frame["run_name"].eq("baseline_proportional") & score_frame["scale"].isin(["60m_1p2b", "300m_6b"]))
        ].copy()
        missing_columns = [column for column in required_columns if column not in candidate_rows.columns]
        incomplete_rows = 0
        if not missing_columns:
            scores = score_issue5416_aggregate(candidate_rows, projection, fail_missing=True)
            candidate_rows["issue5416_aggregate"] = scores
            incomplete_rows = int(candidate_rows["issue5416_aggregate"].isna().sum())
            if incomplete_rows == 0:
                _aggregate_anchors = candidate_rows.loc[
                    candidate_rows["run_name"].eq("baseline_proportional")
                    & candidate_rows["scale"].isin(["60m_1p2b", "300m_6b"])
                ].copy()
                _aggregate_anchors = _aggregate_anchors.sort_values(["scale", "source_experiment"]).drop_duplicates(
                    "scale", keep="last"
                )
                _aggregate_anchor_by_scale = _aggregate_anchors.set_index("scale")["issue5416_aggregate"].to_dict()
                aggregate_effect_rows = candidate_rows.loc[
                    candidate_rows["family"].isin(
                        {"proportional_perturbation_60m_1p2b", "proportional_perturbation_300m_6b"}
                    )
                ].copy()
                aggregate_effect_rows["proportional_issue5416_aggregate"] = aggregate_effect_rows["scale"].map(
                    _aggregate_anchor_by_scale
                )
                aggregate_effect_rows["effect_issue5416_aggregate"] = (
                    aggregate_effect_rows["issue5416_aggregate"]
                    - aggregate_effect_rows["proportional_issue5416_aggregate"]
                )
                aggregate_paired = (
                    aggregate_effect_rows.pivot(
                        index=[
                            "intervention_id",
                            "intervention_type",
                            "target_unit",
                            "target_domain",
                            "target_family",
                            "quality_high_domain",
                            "quality_low_domain",
                            "tv_distance",
                        ],
                        columns="scale",
                        values="effect_issue5416_aggregate",
                    )
                    .reset_index()
                    .rename(
                        columns={
                            "60m_1p2b": "effect_60_issue5416_aggregate",
                            "300m_6b": "effect_100_issue5416_aggregate",
                        }
                    )
                )
                aggregate_paired["scale_interaction_issue5416_aggregate"] = (
                    aggregate_paired["effect_100_issue5416_aggregate"]
                    - aggregate_paired["effect_60_issue5416_aggregate"]
                )
                aggregate_effect_rows.to_csv(OUTPUT_DIR / "issue5416_aggregate_effect_rows.csv", index=False)
                aggregate_paired.to_csv(OUTPUT_DIR / "paired_issue5416_aggregate_effects.csv", index=False)
                aggregate_ready = True
                aggregate_message = "Issue #5416 aggregate effects are complete."
            else:
                aggregate_message = (
                    f"Aggregate projection fit, but {incomplete_rows} perturbation/baseline rows lack required columns."
                )
        else:
            aggregate_message = (
                f"Aggregate projection fit, but perturbation rows are missing {len(missing_columns)} required columns."
            )
    return aggregate_message, aggregate_paired, aggregate_ready


@app.cell
def _(aggregate_message, mo):
    mo.md(
        f"""
    ## Issue #5416 aggregate status\n\n{aggregate_message}
    """
    )
    return


@app.cell
def _(COLOR_MAP, IMG_DIR, aggregate_paired, aggregate_ready, paired, px):
    if aggregate_ready:
        merged = paired.merge(
            aggregate_paired[
                [
                    "intervention_id",
                    "effect_60_issue5416_aggregate",
                    "effect_100_issue5416_aggregate",
                    "scale_interaction_issue5416_aggregate",
                ]
            ],
            on="intervention_id",
            how="inner",
        )
        tradeoff = px.scatter(
            merged,
            x="scale_interaction_bpb",
            y="scale_interaction_issue5416_aggregate",
            color="intervention_type",
            color_discrete_map=COLOR_MAP,
            hover_name="intervention_id",
            hover_data=["effect_60_bpb", "effect_100_bpb", "target_unit", "tv_distance"],
            title="Scale-transfer tradeoff: BPB interaction vs issue #5416 aggregate interaction",
            labels={
                "scale_interaction_bpb": "BPB interaction; negative helps more at 100M",
                "scale_interaction_issue5416_aggregate": "Aggregate interaction; positive helps more at 100M",
            },
            height=650,
        )
        tradeoff.add_hline(y=0, line_dash="dash", line_color="#777777")
        tradeoff.add_vline(x=0, line_dash="dash", line_color="#777777")
        tradeoff.update_layout(template="plotly_white")
        tradeoff.write_html(IMG_DIR / "bpb_vs_issue5416_aggregate_scale_interaction.html", include_plotlyjs="cdn")
        try:
            tradeoff.write_image(IMG_DIR / "bpb_vs_issue5416_aggregate_scale_interaction.png", scale=2)
        except ValueError:
            pass
    else:
        tradeoff = None
    if tradeoff is not None:
        tradeoff
    return


@app.cell
def _(mo, paired, quality_swaps):
    top_100 = paired.nsmallest(8, "effect_100_bpb")[
        ["intervention_id", "intervention_type", "effect_60_bpb", "effect_100_bpb", "scale_interaction_bpb"]
    ]
    top_interaction = paired.nsmallest(8, "scale_interaction_bpb")[
        ["intervention_id", "intervention_type", "effect_60_bpb", "effect_100_bpb", "scale_interaction_bpb"]
    ]
    mo.vstack(
        [
            mo.md("## Top BPB findings"),
            mo.md("Best 100M BPB effects (negative helps):"),
            mo.ui.table(top_100, page_size=8),
            mo.md("Most negative scale interactions (helps more at 100M than at 60M):"),
            mo.ui.table(top_interaction, page_size=8),
            mo.md("Quality swap effects:"),
            mo.ui.table(
                quality_swaps[
                    [
                        "intervention_id",
                        "effect_60_bpb",
                        "effect_100_bpb",
                        "scale_interaction_bpb",
                    ]
                ],
                page_size=13,
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
