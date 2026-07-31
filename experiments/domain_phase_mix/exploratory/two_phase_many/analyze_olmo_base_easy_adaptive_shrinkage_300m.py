# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "plotly", "scipy", "tabulate"]
# ///
"""Explore adaptive proportional-prior shrinkage for Table-9 DSP predictions.

This is a zero-new-training diagnostic. It uses the existing 300M
OLMoBaseEval Easy Table-9 fit panel and per-component DSP OOF predictions to
ask whether reliability-derived shrinkage would have improved held-out
selection among already-observed mixtures.

The diagnostic deliberately does not refit DSP or launch jobs. Its main purpose
is to separate three cases:

1. fixed component reliabilities, which are algebraically just component
   reweighting plus KL in continuous optimization;
2. candidate-dependent shrinkage, which is not algebraically equivalent but may
   suppress extrapolative optimism;
3. uncertainty penalties, which use reliability as uncertainty rather than as a
   redefined objective.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PACKET_DIR = SCRIPT_DIR / "chatgpt_pro_olmo_base_easy_dsp_packet_20260626" / "data"
DEFAULT_PANEL = DEFAULT_PACKET_DIR / "fit_panel_table9_macro.csv"
DEFAULT_COMPONENTS = DEFAULT_PACKET_DIR / "table9_macro_components.csv"
DEFAULT_OOF = (
    SCRIPT_DIR
    / "reference_outputs"
    / "olmo_base_easy_per_component_dsp_kl_sweep_300m_20260628"
    / "selected_component_oof_predictions.csv"
)
DEFAULT_COMPONENT_SUMMARY = (
    SCRIPT_DIR
    / "reference_outputs"
    / "olmo_base_easy_per_component_dsp_kl_sweep_300m_20260628"
    / "selected_component_l2_summary.csv"
)
DEFAULT_RELIABILITY = (
    SCRIPT_DIR / "reference_outputs" / "olmo_base_easy_reliability_weighting_20260625" / "component_reliability.csv"
)
DEFAULT_VALIDATION = (
    SCRIPT_DIR
    / "reference_outputs"
    / "delphi_table9_dsp_validation_mixtures_3e18_20260628"
    / "table9_3e18_observed_ranking_20260628.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "olmo_base_easy_adaptive_shrinkage_300m_20260628"
MACRO_TARGET = "table9_macro_bpb"
PROPORTIONAL_RUN = "baseline_proportional"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class MethodRecord:
    candidate_set: str
    method: str
    reliability: str
    alpha: float
    beta: float
    gamma: float
    rmse: float
    spearman: float
    regret_at_1: float
    regret_at_3: float
    regret_at_5: float
    selected_run_name: str
    selected_panel_source: str
    selected_actual: float
    selected_score: float
    selected_tv_to_proportional: float
    selected_actual_rank: int
    best_run_name: str
    best_actual: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--components", type=Path, default=DEFAULT_COMPONENTS)
    parser.add_argument("--oof-predictions", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--component-summary", type=Path, default=DEFAULT_COMPONENT_SUMMARY)
    parser.add_argument("--reliability", type=Path, default=DEFAULT_RELIABILITY)
    parser.add_argument("--validation-ranking", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def clipped_unit(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.clip(np.asarray(values, dtype=float), 0.0, 1.0), nan=0.0)


def regression_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def regression_spearman(actual: np.ndarray, predicted: np.ndarray) -> float:
    result = spearmanr(actual, predicted)
    return float(result.statistic)


def phase_tv_to_proportional(panel: pd.DataFrame, proportional_idx: int) -> np.ndarray:
    domains = [column.removeprefix("phase_0_") for column in panel.columns if column.startswith("phase_0_")]
    phase_0 = panel[[f"phase_0_{domain}" for domain in domains]].to_numpy(dtype=float)
    phase_1 = panel[[f"phase_1_{domain}" for domain in domains]].to_numpy(dtype=float)
    prop_0 = phase_0[proportional_idx]
    prop_1 = phase_1[proportional_idx]
    return 0.25 * (np.abs(phase_0 - prop_0).sum(axis=1) + np.abs(phase_1 - prop_1).sum(axis=1))


def component_quality(actual: np.ndarray, predicted: np.ndarray, components: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for index, component in enumerate(components):
        y = actual[:, index]
        y_hat = predicted[:, index]
        total = float(np.sum((y - y.mean()) ** 2))
        residual = float(np.sum((y - y_hat) ** 2))
        rows.append(
            {
                "component": component,
                "oof_r2": float(1.0 - residual / total) if total > 0.0 else np.nan,
                "oof_rmse": regression_rmse(y, y_hat),
                "oof_spearman_calc": regression_spearman(y, y_hat),
            }
        )
    return pd.DataFrame(rows)


def reliability_arrays(quality: pd.DataFrame) -> dict[str, np.ndarray]:
    r2 = clipped_unit(quality["oof_r2"])
    spearman = clipped_unit(quality["selected_oof_spearman"])
    harm_t = clipped_unit(quality["harm_t_excess"])
    two_sided_t = clipped_unit(quality["two_sided_t_excess"])
    harm_bonferroni = clipped_unit(quality["harm_bonferroni"])
    return {
        "uniform": np.ones(len(quality), dtype=float),
        "oof_r2_pos": r2,
        "oof_spearman_pos": spearman,
        "harm_t_excess": harm_t,
        "two_sided_t_excess": two_sided_t,
        "harm_bonferroni": harm_bonferroni,
        "oof_r2_x_harm_t": r2 * harm_t,
    }


def evaluate_score(
    *,
    panel: pd.DataFrame,
    actual_macro: np.ndarray,
    tv: np.ndarray,
    score: np.ndarray,
    mask: np.ndarray,
    candidate_set: str,
    method: str,
    reliability: str,
    alpha: float,
    beta: float,
    gamma: float,
) -> MethodRecord:
    eligible = np.where(mask)[0]
    if len(eligible) == 0:
        raise ValueError(f"Candidate set {candidate_set} is empty")
    order = eligible[np.argsort(score[eligible])]
    actual_order = eligible[np.argsort(actual_macro[eligible])]
    best_idx = int(actual_order[0])
    selected_idx = int(order[0])
    actual_rank_lookup = {int(idx): rank + 1 for rank, idx in enumerate(actual_order)}
    selected_rank = actual_rank_lookup[selected_idx]
    return MethodRecord(
        candidate_set=candidate_set,
        method=method,
        reliability=reliability,
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
        rmse=regression_rmse(actual_macro[eligible], score[eligible]),
        spearman=regression_spearman(actual_macro[eligible], score[eligible]),
        regret_at_1=float(actual_macro[selected_idx] - actual_macro[best_idx]),
        regret_at_3=float(np.min(actual_macro[order[:3]]) - actual_macro[best_idx]),
        regret_at_5=float(np.min(actual_macro[order[:5]]) - actual_macro[best_idx]),
        selected_run_name=str(panel.iloc[selected_idx]["run_name"]),
        selected_panel_source=str(panel.iloc[selected_idx]["panel_source"]),
        selected_actual=float(actual_macro[selected_idx]),
        selected_score=float(score[selected_idx]),
        selected_tv_to_proportional=float(tv[selected_idx]),
        selected_actual_rank=int(selected_rank),
        best_run_name=str(panel.iloc[best_idx]["run_name"]),
        best_actual=float(actual_macro[best_idx]),
    )


def candidate_masks(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    non_proportional = ~panel["run_name"].eq(PROPORTIONAL_RUN).to_numpy()
    qsplit = panel["panel_source"].eq("qsplit_signal").to_numpy() & non_proportional
    deletion = panel["panel_source"].eq("domain_deletion").to_numpy()
    return {
        "qsplit_non_proportional": qsplit,
        "all_non_proportional": non_proportional,
        "domain_deletions": deletion,
    }


def method_grid(
    *,
    panel: pd.DataFrame,
    actual_components: np.ndarray,
    predicted_components: np.ndarray,
    actual_macro: np.ndarray,
    prop_idx: int,
    tv: np.ndarray,
    reliability: dict[str, np.ndarray],
) -> pd.DataFrame:
    prop_pred = predicted_components[prop_idx]
    tv_scale = float(np.median(tv[tv > 0.0]))
    component_sigma = np.sqrt(np.mean((actual_components - predicted_components) ** 2, axis=0))
    macro_sigma = float(np.sqrt(np.mean(component_sigma**2)) / np.sqrt(actual_components.shape[1]))
    component_delta_scale = np.median(np.abs(predicted_components - prop_pred), axis=0) + 1e-12
    records: list[MethodRecord] = []
    masks = candidate_masks(panel)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    betas = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    gammas = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    for reliability_name, base_r in reliability.items():
        for alpha in alphas:
            r = alpha * base_r + (1.0 - alpha) * float(np.mean(base_r))
            fixed_predictions = prop_pred + (predicted_components - prop_pred) * r[None, :]
            fixed_score = fixed_predictions.mean(axis=1)
            for candidate_set, mask in masks.items():
                records.append(
                    evaluate_score(
                        panel=panel,
                        actual_macro=actual_macro,
                        tv=tv,
                        score=fixed_score,
                        mask=mask,
                        candidate_set=candidate_set,
                        method="fixed_component_shrinkage",
                        reliability=reliability_name,
                        alpha=alpha,
                        beta=0.0,
                        gamma=0.0,
                    )
                )
            for beta in betas:
                tv_r = r[None, :] / (1.0 + beta * (tv[:, None] / tv_scale))
                tv_score = (prop_pred + (predicted_components - prop_pred) * tv_r).mean(axis=1)
                delta_r = r[None, :] / (
                    1.0 + beta * np.abs(predicted_components - prop_pred) / component_delta_scale[None, :]
                )
                delta_score = (prop_pred + (predicted_components - prop_pred) * delta_r).mean(axis=1)
                for candidate_set, mask in masks.items():
                    records.append(
                        evaluate_score(
                            panel=panel,
                            actual_macro=actual_macro,
                            tv=tv,
                            score=tv_score,
                            mask=mask,
                            candidate_set=candidate_set,
                            method="tv_adaptive_shrinkage",
                            reliability=reliability_name,
                            alpha=alpha,
                            beta=beta,
                            gamma=0.0,
                        )
                    )
                    records.append(
                        evaluate_score(
                            panel=panel,
                            actual_macro=actual_macro,
                            tv=tv,
                            score=delta_score,
                            mask=mask,
                            candidate_set=candidate_set,
                            method="delta_adaptive_shrinkage",
                            reliability=reliability_name,
                            alpha=alpha,
                            beta=beta,
                            gamma=0.0,
                        )
                    )
            for gamma in gammas:
                uncertainty = macro_sigma * (1.0 + tv / tv_scale)
                uncertainty_score = fixed_score + gamma * uncertainty
                for candidate_set, mask in masks.items():
                    records.append(
                        evaluate_score(
                            panel=panel,
                            actual_macro=actual_macro,
                            tv=tv,
                            score=uncertainty_score,
                            mask=mask,
                            candidate_set=candidate_set,
                            method="tv_uncertainty_penalty",
                            reliability=reliability_name,
                            alpha=alpha,
                            beta=0.0,
                            gamma=gamma,
                        )
                    )
    return pd.DataFrame([asdict(record) for record in records])


def write_plots(output_dir: Path, summary: pd.DataFrame, quality: pd.DataFrame) -> None:
    qsplit = summary.loc[summary["candidate_set"].eq("qsplit_non_proportional")].copy()
    fig = px.scatter(
        qsplit,
        x="rmse",
        y="regret_at_1",
        color="regret_at_3",
        color_continuous_scale="RdYlGn_r",
        hover_data=[
            "method",
            "reliability",
            "alpha",
            "beta",
            "gamma",
            "spearman",
            "selected_run_name",
            "selected_actual_rank",
        ],
        title="Adaptive shrinkage observed-row diagnostics, qsplit candidate set",
        labels={
            "rmse": "OOF macro RMSE on candidate set",
            "regret_at_1": "Observed Regret@1 from predicted selected row",
            "regret_at_3": "Observed Regret@3",
        },
        width=1000,
        height=650,
    )
    fig.write_html(output_dir / "adaptive_shrinkage_qsplit_rmse_regret.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    best_by_method = (
        qsplit.sort_values(["regret_at_1", "regret_at_3", "rmse"])
        .groupby(["method", "reliability"], as_index=False)
        .head(1)
    )
    fig2 = px.bar(
        best_by_method.sort_values(["regret_at_1", "regret_at_3", "rmse"]),
        x="selected_run_name",
        y="selected_actual",
        color="regret_at_3",
        color_continuous_scale="RdYlGn_r",
        hover_data=["method", "reliability", "alpha", "beta", "gamma", "regret_at_1", "regret_at_3"],
        title="Selected observed rows for best shrinkage settings by family",
        labels={"selected_actual": "Observed Table-9 macro BPB (lower is better)"},
        width=1200,
        height=650,
    )
    fig2.write_html(output_dir / "adaptive_shrinkage_selected_rows.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    fig3 = go.Figure()
    fig3.add_trace(
        go.Histogram(
            x=quality["oof_r2"],
            name="OOF R2",
            opacity=0.75,
            marker_color="#2563eb",
        )
    )
    fig3.add_trace(
        go.Histogram(
            x=quality["harm_t_excess"],
            name="Deletion harm t-excess",
            opacity=0.75,
            marker_color="#ef4444",
        )
    )
    fig3.update_layout(
        title="Component reliability signals used by shrinkage diagnostics",
        xaxis_title="Reliability weight",
        yaxis_title="Number of Table-9 components",
        barmode="overlay",
        template="plotly_white",
        width=1000,
        height=600,
    )
    fig3.write_html(output_dir / "component_reliability_weight_histograms.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def report_lines(summary: pd.DataFrame, quality: pd.DataFrame, validation: pd.DataFrame | None) -> list[str]:
    qsplit = summary.loc[summary["candidate_set"].eq("qsplit_non_proportional")].copy()
    fixed_uniform = qsplit[
        qsplit["method"].eq("fixed_component_shrinkage")
        & qsplit["reliability"].eq("uniform")
        & np.isclose(qsplit["alpha"], 1.0)
    ].iloc[0]
    best_regret = qsplit.sort_values(["regret_at_1", "regret_at_3", "rmse"]).head(8)
    best_rmse = qsplit.sort_values(["rmse", "regret_at_1"]).head(8)
    lines = [
        "# Adaptive proportional-prior shrinkage diagnostic",
        "",
        "This diagnostic uses existing 300M OLMoBaseEval Easy Table-9 data only. It evaluates whether reliability-derived shrinkage improves out-of-fold selection among observed mixtures, before proposing any new continuous optimization or validation run.",
        "",
        "## Algebraic point",
        "",
        "For fixed component reliabilities `r_c`, optimizing `mean_c[f_c(p) + r_c (f_hat_c(w) - f_hat_c(p))] + lambda KL(w || p)` is equivalent to optimizing a reweighted component objective plus the same KL term, because all proportional terms are constants in `w`. Fixed shrinkage is therefore not a distinct decision rule from the reliability weighting we already tried.",
        "",
        "The only genuinely new variants tested here are candidate-dependent shrinkage and uncertainty penalties, where the shrinkage/penalty depends on distance from proportional or on predicted component displacement.",
        "",
        "## Component reliability inputs",
        "",
        f"Components: {len(quality)}.",
        f"OOF R2 mean/median: {quality['oof_r2'].mean():.3f} / {quality['oof_r2'].median():.3f}.",
        f"OOF Spearman mean/median: {quality['selected_oof_spearman'].mean():.3f} / {quality['selected_oof_spearman'].median():.3f}.",
        f"Deletion harm t-excess mean/median: {quality['harm_t_excess'].mean():.3f} / {quality['harm_t_excess'].median():.3f}.",
        "",
        "## Baseline observed-row diagnostic",
        "",
        f"Uniform per-component OOF DSP selects `{fixed_uniform['selected_run_name']}` on the qsplit candidate set.",
        f"Selected observed Table-9 macro BPB: {fixed_uniform['selected_actual']:.6f}; best observed qsplit BPB: {fixed_uniform['best_actual']:.6f}; Regret@1: {fixed_uniform['regret_at_1']:.6f}.",
        "",
        "## Best exploratory settings by observed qsplit regret",
        "",
        best_regret[
            [
                "method",
                "reliability",
                "alpha",
                "beta",
                "gamma",
                "rmse",
                "spearman",
                "regret_at_1",
                "regret_at_3",
                "regret_at_5",
                "selected_run_name",
                "selected_actual",
                "selected_actual_rank",
            ]
        ].to_markdown(index=False),
        "",
        "## Best exploratory settings by observed qsplit RMSE",
        "",
        best_rmse[
            [
                "method",
                "reliability",
                "alpha",
                "beta",
                "gamma",
                "rmse",
                "spearman",
                "regret_at_1",
                "regret_at_3",
                "regret_at_5",
                "selected_run_name",
                "selected_actual",
                "selected_actual_rank",
            ]
        ].to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- Fixed component reliabilities do not improve top-1 observed selection; they mostly alter calibration and sometimes worsen RMSE.",
        "- The best rank-correlation settings are Spearman-derived shrinkage or a small TV uncertainty penalty, but improvements are small and do not change the top-1 selected row.",
        "- Some conservative settings make the true best observed qsplit mixture appear in Top-3 or Top-5, which supports using reliability as uncertainty for candidate shortlisting rather than as a replacement objective.",
        "- This is not yet a justification to launch a new adaptive-shrinkage optimizer. The strongest validated result so far remains the KL-regularized per-component DSP sweep, especially the mid-KL proposals.",
    ]
    if validation is not None and not validation.empty:
        lines.extend(
            [
                "",
                "## Current 3e18 validation context",
                "",
                validation.head(8).to_markdown(index=False),
            ]
        )
    return lines


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(args.panel)
    components = pd.read_csv(args.components)["component"].tolist()
    oof = pd.read_csv(args.oof_predictions)
    component_summary = pd.read_csv(args.component_summary)
    reliability = pd.read_csv(args.reliability)

    if panel["run_name"].tolist() != oof["run_name"].tolist():
        raise ValueError("Fit panel and OOF prediction rows do not align by run_name")
    prop_rows = np.where(panel["run_name"].eq(PROPORTIONAL_RUN).to_numpy())[0]
    if len(prop_rows) != 1:
        raise ValueError(f"Expected exactly one {PROPORTIONAL_RUN} row")
    prop_idx = int(prop_rows[0])

    actual_components = panel[components].to_numpy(dtype=float)
    predicted_components = oof[[f"pred::{component}" for component in components]].to_numpy(dtype=float)
    actual_macro = actual_components.mean(axis=1)
    tv = phase_tv_to_proportional(panel, prop_idx)
    quality = (
        component_quality(actual_components, predicted_components, components)
        .merge(component_summary, on="component", how="left")
        .merge(reliability, on="component", how="left")
    )
    quality.to_csv(args.output_dir / "component_reliability_weights.csv", index=False)

    summary = method_grid(
        panel=panel,
        actual_components=actual_components,
        predicted_components=predicted_components,
        actual_macro=actual_macro,
        prop_idx=prop_idx,
        tv=tv,
        reliability=reliability_arrays(quality),
    )
    summary.to_csv(args.output_dir / "adaptive_shrinkage_observed_row_summary.csv", index=False)
    (
        summary.loc[summary["candidate_set"].eq("qsplit_non_proportional")]
        .sort_values(["regret_at_1", "regret_at_3", "rmse"])
        .head(50)
        .to_csv(args.output_dir / "adaptive_shrinkage_top_qsplit_methods.csv", index=False)
    )

    validation = pd.read_csv(args.validation_ranking) if args.validation_ranking.exists() else None
    write_plots(args.output_dir, summary, quality)
    (args.output_dir / "adaptive_shrinkage_report.md").write_text(
        "\n".join(report_lines(summary, quality, validation)) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote adaptive shrinkage diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
