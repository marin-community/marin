# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "plotly", "scipy"]
# ///
"""Diagnose noise shape and winner's-curse risk for mixture optimization.

This script tests whether existing repeated-anchor data support moving beyond
the current symmetric heteroskedastic noise approximation. It does not fit a
new production model. Instead it emits:

- within-anchor skew/tail diagnostics for proportional 300M repeats;
- within-anchor skew/tail diagnostics for StarCoder repeated anchors;
- raw-vs-logit checks for bounded metrics;
- winner's-curse corrections for selecting among many noisy candidates;
- proportional-to-readiness-optimum path diagnostics for `broad_screened` and
  `steerable_guardrail_stabilized`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp


SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_DIR / "noise_shape_winner_curse_20260616"
DEFAULT_RAW_WITH_PROP_NOISE = (
    REFERENCE_DIR
    / "raw_metric_matrix_300m_dclm_updated_20260615"
    / "raw_metric_matrix_300m_with_proportional_noise.csv"
)
DEFAULT_STARCODER_REPEATS = (
    SCRIPT_DIR.parent / "reference_outputs" / "starcoder_heteroskedastic_snr_20260523" / "collected_train_only_metrics_live.csv"
)
DEFAULT_READINESS_OUTPUT = REFERENCE_DIR / "readiness_weighted_aggregate_dsp_20260616"
TO_IMAGE_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


METRIC_PREFIXES = ("eval/", "lm_eval/", "mcq_smooth/", "raw_ppl/", "teacher_forced/", "dclm_core_v2/")
COUNT_SUFFIXES = ("/bytes", "/documents", "/example_count", "/loading_time", "/total_time")
LOWER_IS_BETTER_TOKENS = ("/bpb", "/loss", "/nll", "/bits", "perplexity", "/ppl")
KEY_METRICS = [
    "eval/bpb",
    "eval/loss",
    "eval/paloma/dolma_100_programing_languages/bpb",
    "eval/uncheatable_eval/github_python/bpb",
    "eval/uncheatable_eval/github_cpp/bpb",
    "eval/uncheatable_eval/bpb",
    "lm_eval/hellaswag_0shot/acc",
    "lm_eval/hellaswag_0shot/acc_norm",
    "lm_eval/hellaswag_10shot/acc",
    "lm_eval/hellaswag_10shot/acc_norm",
    "teacher_forced/humaneval_10shot_canonical_solution/bpb",
]


@dataclass(frozen=True)
class NoiseStats:
    """Distributional diagnostics for one metric in one repeated panel."""

    metric: str
    panel: str
    group: str
    n: int
    lower_is_better: bool
    bounded_raw: bool
    mean: float
    median: float
    sd: float
    utility_mean: float
    utility_median: float
    utility_sd: float
    utility_skew: float
    utility_tail_asymmetry: float
    utility_median_minus_mean: float
    raw_skew: float
    logit_utility_skew: float | None
    logit_tail_asymmetry: float | None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-with-prop-noise", type=Path, default=DEFAULT_RAW_WITH_PROP_NOISE)
    parser.add_argument("--starcoder-repeats", type=Path, default=DEFAULT_STARCODER_REPEATS)
    parser.add_argument("--readiness-output", type=Path, default=DEFAULT_READINESS_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def is_metric_column(column: str) -> bool:
    """Return whether a column is a metric value rather than metadata/weights."""
    return column.startswith(METRIC_PREFIXES) and not column.endswith(COUNT_SUFFIXES)


def lower_is_better(metric: str) -> bool:
    """Return whether smaller raw values are better."""
    lowered = metric.lower()
    return any(token in lowered for token in LOWER_IS_BETTER_TOKENS)


def sample_skew(values: np.ndarray) -> float:
    """Compute a stable moment skewness estimate."""
    finite = values[np.isfinite(values)]
    if len(finite) < 3:
        return float("nan")
    centered = finite - float(np.mean(finite))
    m2 = float(np.mean(centered**2))
    if m2 <= 0.0:
        return float("nan")
    m3 = float(np.mean(centered**3))
    return m3 / (m2 ** 1.5)


def tail_asymmetry(values: np.ndarray) -> float:
    """Return normalized upper-minus-lower tail width around the median."""
    finite = values[np.isfinite(values)]
    if len(finite) < 4:
        return float("nan")
    q10, q50, q90 = np.quantile(finite, [0.1, 0.5, 0.9])
    scale = q90 - q10
    if not np.isfinite(scale) or scale <= 0.0:
        return float("nan")
    return float(((q90 - q50) - (q50 - q10)) / scale)


def bounded_probability(values: np.ndarray, metric: str) -> bool:
    """Return whether metric values look like bounded probabilities/scores."""
    if lower_is_better(metric):
        return False
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return False
    if metric.endswith(("/example_count", "/documents", "/bytes")):
        return False
    return bool(np.nanmin(finite) >= -1e-9 and np.nanmax(finite) <= 1.0 + 1e-9)


def logit(values: np.ndarray) -> np.ndarray:
    """Clipped logit transform."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.full_like(values, np.nan, dtype=float)
    # Use a sample-size-aware clipping floor when values are exact 0/1.
    eps = max(1e-6, 0.5 / max(len(finite), 1))
    clipped = np.clip(values.astype(float), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def metric_stats(metric: str, values: np.ndarray, *, panel: str, group: str) -> NoiseStats | None:
    """Compute distribution diagnostics for one metric/group."""
    raw = values.astype(float)
    raw = raw[np.isfinite(raw)]
    if len(raw) < 4:
        return None
    if float(np.nanstd(raw, ddof=1)) <= 0.0:
        return None
    is_lower = lower_is_better(metric)
    utility = -raw if is_lower else raw
    bounded = bounded_probability(raw, metric)
    logit_utility_skew = None
    logit_tail_asymmetry = None
    if bounded:
        transformed = logit(raw)
        logit_utility_skew = sample_skew(transformed)
        logit_tail_asymmetry = tail_asymmetry(transformed)
    return NoiseStats(
        metric=metric,
        panel=panel,
        group=group,
        n=int(len(raw)),
        lower_is_better=is_lower,
        bounded_raw=bounded,
        mean=float(np.mean(raw)),
        median=float(np.median(raw)),
        sd=float(np.std(raw, ddof=1)),
        utility_mean=float(np.mean(utility)),
        utility_median=float(np.median(utility)),
        utility_sd=float(np.std(utility, ddof=1)),
        utility_skew=sample_skew(utility),
        utility_tail_asymmetry=tail_asymmetry(utility),
        utility_median_minus_mean=float(np.median(utility) - np.mean(utility)),
        raw_skew=sample_skew(raw),
        logit_utility_skew=logit_utility_skew,
        logit_tail_asymmetry=logit_tail_asymmetry,
    )


def stats_to_frame(rows: list[NoiseStats]) -> pd.DataFrame:
    """Convert dataclass rows to a frame."""
    return pd.DataFrame([row.__dict__ for row in rows])


def proportional_noise_stats(raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute noise-shape stats for proportional variable-subset repeats."""
    frame = pd.read_csv(raw_path, low_memory=False)
    noise = frame[frame["run_name"].astype(str).str.startswith("propvar_300m_6b_trainer_seed_")].copy()
    if noise.empty:
        raise ValueError(f"No proportional variable-subset rows found in {raw_path}")
    rows: list[NoiseStats] = []
    for metric in [column for column in noise.columns if is_metric_column(column)]:
        values = pd.to_numeric(noise[metric], errors="coerce").to_numpy(dtype=float)
        stat = metric_stats(metric, values, panel="proportional_300m_variable_subset", group="proportional")
        if stat is not None:
            rows.append(stat)
    return stats_to_frame(rows), noise


def starcoder_noise_stats(starcoder_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute noise-shape stats for StarCoder repeated anchors."""
    frame = pd.read_csv(starcoder_path, low_memory=False)
    metric_columns = [column for column in frame.columns if is_metric_column(column)]
    rows: list[dict[str, Any]] = []
    for (anchor_id, anchor_index), group in frame.groupby(["anchor_id", "anchor_index"], dropna=False):
        group_label = str(anchor_id)
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            stat = metric_stats(metric, values, panel="starcoder_repeated_anchor", group=group_label)
            if stat is None:
                continue
            record = stat.__dict__.copy()
            record["anchor_index"] = int(anchor_index)
            record["phase_0_starcoder"] = float(group["phase_0_starcoder"].iloc[0])
            record["phase_1_starcoder"] = float(group["phase_1_starcoder"].iloc[0])
            record["total_starcoder_epochs"] = float(group["total_starcoder_epochs"].iloc[0])
            record["rarity_proxy"] = -np.log1p(record["total_starcoder_epochs"])
            rows.append(record)
    return pd.DataFrame(rows), frame


def summarize_starcoder_rarity(starcoder_stats: pd.DataFrame) -> pd.DataFrame:
    """Summarize skew/variance association with StarCoder exposure."""
    rows = []
    for metric, group in starcoder_stats.groupby("metric"):
        finite = group[["rarity_proxy", "utility_skew", "utility_sd", "utility_tail_asymmetry"]].replace(
            [np.inf, -np.inf], np.nan
        )
        finite = finite.dropna(subset=["rarity_proxy"])
        if len(finite) < 5:
            continue
        row: dict[str, Any] = {"metric": metric, "anchor_count": int(len(finite))}
        for column in ["utility_skew", "utility_sd", "utility_tail_asymmetry"]:
            valid = finite[["rarity_proxy", column]].dropna()
            if len(valid) < 5 or valid[column].nunique() <= 1:
                row[f"{column}_vs_rarity_spearman"] = float("nan")
                row[f"{column}_vs_rarity_p_value"] = float("nan")
                continue
            stat = spearmanr(valid["rarity_proxy"], valid[column])
            row[f"{column}_vs_rarity_spearman"] = float(stat.statistic)
            row[f"{column}_vs_rarity_p_value"] = float(stat.pvalue)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_max_residual(residuals: np.ndarray, candidate_count: int, reps: int, rng: np.random.Generator) -> dict[str, float]:
    """Estimate selection optimism when taking max over candidate_count noisy predictions."""
    finite = residuals[np.isfinite(residuals)]
    if len(finite) == 0:
        return {"candidate_count": candidate_count, "p50": float("nan"), "p90": float("nan"), "p95": float("nan")}
    draws = rng.choice(finite, size=(reps, candidate_count), replace=True)
    max_draws = np.max(draws, axis=1)
    return {
        "candidate_count": int(candidate_count),
        "p50": float(np.quantile(max_draws, 0.5)),
        "p90": float(np.quantile(max_draws, 0.9)),
        "p95": float(np.quantile(max_draws, 0.95)),
        "mean": float(np.mean(max_draws)),
    }


def load_label_weights(weights_path: Path, label: str, domain_order: list[str]) -> np.ndarray:
    """Load a two-phase weight matrix from a long mixture-weight CSV."""
    frame = pd.read_csv(weights_path)
    subset = frame[frame["label"].eq(label)].copy()
    if subset.empty:
        raise ValueError(f"Missing label={label} in {weights_path}")
    subset = subset.set_index("domain").reindex(domain_order)
    missing_domains = subset[subset["phase_0_weight"].isna() | subset["phase_1_weight"].isna()].index.tolist()
    if missing_domains:
        raise ValueError(f"Missing {len(missing_domains)} domains for label={label}: {missing_domains[:5]}")
    weights = np.stack(
        [
            subset["phase_0_weight"].to_numpy(dtype=float),
            subset["phase_1_weight"].to_numpy(dtype=float),
        ],
        axis=0,
    )
    sums = weights.sum(axis=1, keepdims=True)
    if np.any(sums <= 0.0):
        raise ValueError(f"Empty phase for label={label} in {weights_path}")
    return weights / sums


def path_candidate_diagnostics(readiness_output: Path, *, reps: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build path candidate and winner's-curse diagnostics for readiness targets."""
    all_predictions = pd.read_csv(readiness_output / "all_observed_predictions.csv")
    rows: list[dict[str, Any]] = []
    correction_rows: list[dict[str, Any]] = []
    t_values = np.array([0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.33, 0.5, 0.67, 0.75, 1.0])
    for target_name in ["broad_screened", "steerable_guardrail_stabilized"]:
        target_dir = readiness_output / target_name
        model = dsp.model_from_json(json.loads((target_dir / "model.json").read_text()))
        weights_path = target_dir / "mixture_weights.csv"
        proportional = load_label_weights(weights_path, "proportional", model.domain_names)
        raw_optimum = load_label_weights(weights_path, "raw_dsp_optimum", model.domain_names)
        target_predictions = all_predictions[all_predictions["target_name"].eq(target_name)].copy()
        # oof_residual is saved as predicted - actual; positive means overprediction.
        residuals = pd.to_numeric(target_predictions["oof_residual"], errors="coerce").to_numpy(dtype=float)
        for candidate_count in [1, len(t_values), len(target_predictions)]:
            correction = bootstrap_max_residual(residuals, int(candidate_count), reps, rng)
            correction["target_name"] = target_name
            correction_rows.append(correction)
        path_count_correction = next(row for row in correction_rows if row["target_name"] == target_name and row["candidate_count"] == len(t_values))
        proportional_pred = -float(dsp.predict(model, proportional[None, :, :])[0])
        for t in t_values:
            weights = (1.0 - t) * proportional + t * raw_optimum
            weights = weights / weights.sum(axis=1, keepdims=True)
            pred = -float(dsp.predict(model, weights[None, :, :])[0])
            tv = float(dsp.average_phase_tv_distance(proportional[None, :, :], weights[None, :, :])[0])
            rows.append(
                {
                    "target_name": target_name,
                    "t": float(t),
                    "pred_target": pred,
                    "pred_gain_vs_proportional": pred - proportional_pred,
                    "tv_vs_proportional": tv,
                    "phase0_max_weight": float(np.max(weights[0])),
                    "phase1_max_weight": float(np.max(weights[1])),
                    "phase0_effective_support": float(np.exp(dsp.entropy(weights[0]))),
                    "phase1_effective_support": float(np.exp(dsp.entropy(weights[1]))),
                    "winner_curse_path_p50": path_count_correction["p50"],
                    "winner_curse_path_p90": path_count_correction["p90"],
                    "winner_curse_path_p95": path_count_correction["p95"],
                    "p50_adjusted_gain": pred - proportional_pred - path_count_correction["p50"],
                    "p90_adjusted_gain": pred - proportional_pred - path_count_correction["p90"],
                    "p95_adjusted_gain": pred - proportional_pred - path_count_correction["p95"],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(correction_rows)


def write_plots(
    output_dir: Path,
    prop_stats: pd.DataFrame,
    starcoder_stats: pd.DataFrame,
    starcoder_rarity: pd.DataFrame,
    path_candidates: pd.DataFrame,
    winner_corrections: pd.DataFrame,
) -> None:
    """Write HTML plots."""
    fig = px.histogram(
        prop_stats,
        x="utility_skew",
        nbins=80,
        color="bounded_raw",
        marginal="rug",
        title="Proportional 300M repeated rows: utility skew distribution",
        labels={"utility_skew": "within-anchor utility skew; positive = good-tail skew"},
    )
    fig.update_layout(width=1100, height=650)
    fig.write_html(output_dir / "proportional_noise_skew_distribution.html", config=TO_IMAGE_CONFIG)

    bounded = prop_stats[prop_stats["bounded_raw"] & prop_stats["logit_utility_skew"].notna()].copy()
    if not bounded.empty:
        fig = px.scatter(
            bounded,
            x="utility_skew",
            y="logit_utility_skew",
            hover_name="metric",
            color="utility_sd",
            color_continuous_scale="RdYlGn_r",
            title="Bounded proportional metrics: raw-scale skew vs logit-scale skew",
            labels={"utility_skew": "raw utility skew", "logit_utility_skew": "logit-scale skew"},
        )
        fig.add_hline(y=0, line_dash="dot")
        fig.add_vline(x=0, line_dash="dot")
        fig.update_layout(width=950, height=750)
        fig.write_html(output_dir / "bounded_metric_raw_vs_logit_skew.html", config=TO_IMAGE_CONFIG)

    key_starcoder = starcoder_stats[
        starcoder_stats["metric"].isin([metric for metric in KEY_METRICS if metric in set(starcoder_stats["metric"])])
    ].copy()
    if not key_starcoder.empty:
        fig = px.scatter(
            key_starcoder,
            x="total_starcoder_epochs",
            y="utility_skew",
            color="metric",
            hover_name="group",
            size="utility_sd",
            log_x=True,
            title="StarCoder repeated anchors: utility skew vs StarCoder exposure",
            labels={"utility_skew": "utility skew; positive = good-tail skew"},
        )
        fig.add_hline(y=0, line_dash="dot")
        fig.update_layout(width=1150, height=750)
        fig.write_html(output_dir / "starcoder_skew_vs_exposure_key_metrics.html", config=TO_IMAGE_CONFIG)

        fig = px.scatter(
            key_starcoder,
            x="total_starcoder_epochs",
            y="utility_sd",
            color="metric",
            hover_name="group",
            log_x=True,
            title="StarCoder repeated anchors: noise SD vs StarCoder exposure",
            labels={"utility_sd": "within-anchor utility SD"},
        )
        fig.update_layout(width=1150, height=750)
        fig.write_html(output_dir / "starcoder_sd_vs_exposure_key_metrics.html", config=TO_IMAGE_CONFIG)

    top_rarity = starcoder_rarity.reindex(
        starcoder_rarity["utility_skew_vs_rarity_spearman"].abs().sort_values(ascending=False).index
    ).head(40)
    if not top_rarity.empty:
        fig = px.bar(
            top_rarity,
            x="utility_skew_vs_rarity_spearman",
            y="metric",
            orientation="h",
            color="utility_skew_vs_rarity_spearman",
            color_continuous_scale="RdYlGn_r",
            title="StarCoder anchors: strongest skew-vs-rarity associations",
        )
        fig.update_layout(width=1200, height=1000, yaxis={"categoryorder": "total ascending"})
        fig.write_html(output_dir / "starcoder_skew_rarity_association_top.html", config=TO_IMAGE_CONFIG)

    fig = px.line(
        path_candidates,
        x="t",
        y="pred_gain_vs_proportional",
        color="target_name",
        markers=True,
        hover_data=[
            "tv_vs_proportional",
            "phase0_max_weight",
            "phase1_max_weight",
            "p50_adjusted_gain",
            "p90_adjusted_gain",
            "p95_adjusted_gain",
        ],
        title="Readiness-optimum paths: predicted gain before winner's-curse correction",
        labels={"t": "interpolation from proportional to raw optimum", "pred_gain_vs_proportional": "target-SD gain"},
    )
    for target_name, group in path_candidates.groupby("target_name"):
        correction = winner_corrections[
            winner_corrections["target_name"].eq(target_name)
            & winner_corrections["candidate_count"].eq(path_candidates["t"].nunique())
        ].iloc[0]
        fig.add_hline(
            y=float(correction["p50"]),
            line_dash="dot",
            annotation_text=f"{target_name} path p50 max-residual correction",
            annotation_position="top left",
        )
    fig.update_layout(width=1100, height=700)
    fig.write_html(output_dir / "readiness_path_gain_vs_winner_curse.html", config=TO_IMAGE_CONFIG)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Path candidate correction", "Observed-library correction"))
    for target_name, group in winner_corrections.groupby("target_name"):
        fig.add_trace(
            go.Bar(
                x=group["candidate_count"].astype(str),
                y=group["p50"],
                name=f"{target_name} p50",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=group["candidate_count"].astype(str),
                y=group["p95"],
                name=f"{target_name} p95",
            ),
            row=1,
            col=2,
        )
    fig.update_layout(width=1150, height=600, barmode="group", title="Bootstrap max-residual winner's-curse correction")
    fig.write_html(output_dir / "winner_curse_corrections.html", config=TO_IMAGE_CONFIG)


def write_report(
    output_dir: Path,
    prop_stats: pd.DataFrame,
    starcoder_stats: pd.DataFrame,
    starcoder_rarity: pd.DataFrame,
    path_candidates: pd.DataFrame,
    winner_corrections: pd.DataFrame,
) -> None:
    """Write a concise report."""
    prop_summary = {
        "metric_count": int(len(prop_stats)),
        "median_utility_skew": float(prop_stats["utility_skew"].median()),
        "share_positive_utility_skew": float((prop_stats["utility_skew"] > 0).mean()),
        "share_abs_skew_gt_1": float((prop_stats["utility_skew"].abs() > 1).mean()),
    }
    bounded = prop_stats[prop_stats["bounded_raw"] & prop_stats["logit_utility_skew"].notna()]
    if not bounded.empty:
        prop_summary["bounded_metric_count"] = int(len(bounded))
        prop_summary["bounded_raw_logit_skew_corr"] = float(
            bounded[["utility_skew", "logit_utility_skew"]].corr(method="spearman").iloc[0, 1]
        )
        prop_summary["bounded_share_sign_changed_after_logit"] = float(
            (np.sign(bounded["utility_skew"]) != np.sign(bounded["logit_utility_skew"])).mean()
        )
    key_starcoder_metrics = [metric for metric in KEY_METRICS if metric in set(starcoder_stats["metric"])]
    starcoder_key = starcoder_rarity[starcoder_rarity["metric"].isin(key_starcoder_metrics)].copy()
    path_best = (
        path_candidates.sort_values(["target_name", "p50_adjusted_gain"], ascending=[True, False])
        .groupby("target_name")
        .head(3)
    )
    correction_pivot = winner_corrections.pivot(index="target_name", columns="candidate_count", values="p50")
    lines = [
        "# Noise Shape and Winner's-Curse Diagnostics",
        "",
        "This analysis tests whether existing repeated-anchor data justify replacing symmetric heteroskedastic noise with an asymmetric/spike-prone model. It also quantifies selection-risk corrections for readiness-weighted aggregate optimization.",
        "",
        "## Proportional Repeats",
        "",
        pd.DataFrame([prop_summary]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "Interpretation: with 10 proportional repeats, skew diagnostics are useful as screens but not as stable third-moment estimates. A large share of metrics with positive utility skew would be consistent with good-tail spikes, but bounded metrics require logit-scale checks before attributing this to rare data inclusion.",
        "",
        "## StarCoder Rarity Association",
        "",
    ]
    if not starcoder_key.empty:
        lines.extend(
            [
                starcoder_key[
                    [
                        "metric",
                        "anchor_count",
                        "utility_skew_vs_rarity_spearman",
                        "utility_sd_vs_rarity_spearman",
                        "utility_tail_asymmetry_vs_rarity_spearman",
                    ]
                ].to_markdown(index=False, floatfmt=".4f"),
                "",
            ]
        )
    else:
        lines.extend(["No key StarCoder metrics were available for rarity association.", ""])
    lines.extend(
        [
            "Interpretation: the rare-content spike story predicts positive utility-skew and larger utility-SD as the relevant content becomes rarer. The StarCoder panel has only 5 repeats per anchor, so these associations should be treated as qualitative.",
            "",
            "## Winner's-Curse / Path Diagnostics",
            "",
            "Best path points after median max-residual correction:",
            "",
            path_best[
                [
                    "target_name",
                    "t",
                    "pred_gain_vs_proportional",
                    "tv_vs_proportional",
                    "p50_adjusted_gain",
                    "p90_adjusted_gain",
                    "p95_adjusted_gain",
                    "phase0_max_weight",
                    "phase1_max_weight",
                ]
            ].to_markdown(index=False, floatfmt=".4f"),
            "",
            "Median max-residual correction by candidate count:",
            "",
            correction_pivot.to_markdown(floatfmt=".4f"),
            "",
            "Interpretation: path candidates should be evaluated after subtracting a selection correction. If the adjusted gain remains positive at moderate TV, the objective is more credible. If gains survive only near the raw optimum, the result remains an extrapolative modeling artifact.",
            "",
            "## Modeling Implications",
            "",
            "- Keep symmetric heteroskedastic models as the default interior approximation; do not globally assume skewed noise.",
            "- Add robust decision layers: median/quantile or LCB objectives, sign stability, path/trust-region constraints, and fresh-seed validation.",
            "- Treat fixed-subset results as conditional-objective estimates, not marginal deployment estimates.",
            "- Correct winner's curse whenever selecting the best mixture from a candidate library or optimized path.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    """Run the diagnostics."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("computing proportional noise shape diagnostics", flush=True)
    prop_stats, prop_noise_rows = proportional_noise_stats(args.raw_with_prop_noise)
    print(f"proportional metrics={len(prop_stats)} rows={len(prop_noise_rows)}", flush=True)

    print("computing StarCoder repeated-anchor diagnostics", flush=True)
    starcoder_stats, starcoder_rows = starcoder_noise_stats(args.starcoder_repeats)
    starcoder_rarity = summarize_starcoder_rarity(starcoder_stats)
    print(f"starcoder stats={len(starcoder_stats)} rows={len(starcoder_rows)}", flush=True)

    print("computing winner's-curse and path diagnostics", flush=True)
    path_candidates, winner_corrections = path_candidate_diagnostics(
        args.readiness_output,
        reps=args.bootstrap_reps,
        rng=rng,
    )

    prop_stats.to_csv(args.output_dir / "proportional_noise_shape_stats.csv", index=False)
    starcoder_stats.to_csv(args.output_dir / "starcoder_anchor_noise_shape_stats.csv", index=False)
    starcoder_rarity.to_csv(args.output_dir / "starcoder_rarity_association.csv", index=False)
    path_candidates.to_csv(args.output_dir / "readiness_path_candidates_winner_corrected.csv", index=False)
    winner_corrections.to_csv(args.output_dir / "winner_curse_residual_corrections.csv", index=False)

    write_plots(args.output_dir, prop_stats, starcoder_stats, starcoder_rarity, path_candidates, winner_corrections)
    write_report(args.output_dir, prop_stats, starcoder_stats, starcoder_rarity, path_candidates, winner_corrections)

    summary = {
        "output_dir": str(args.output_dir),
        "proportional_repeat_rows": int(len(prop_noise_rows)),
        "proportional_metric_count": int(len(prop_stats)),
        "proportional_median_utility_skew": float(prop_stats["utility_skew"].median()),
        "proportional_share_positive_utility_skew": float((prop_stats["utility_skew"] > 0).mean()),
        "starcoder_repeat_rows": int(len(starcoder_rows)),
        "starcoder_anchor_metric_rows": int(len(starcoder_stats)),
        "path_targets": sorted(path_candidates["target_name"].unique().tolist()),
        "best_p50_adjusted_path": path_candidates.sort_values("p50_adjusted_gain", ascending=False)
        .head(1)
        .to_dict(orient="records")[0],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
