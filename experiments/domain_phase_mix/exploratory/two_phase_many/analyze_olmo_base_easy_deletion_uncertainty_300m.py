# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "plotly", "scipy", "tabulate"]
# ///
"""Uncertainty diagnostics for OLMoBaseEval Easy domain-deletion stress tests.

This is a post-hoc statistical diagnostic over row-level predictions from
`analyze_olmo_base_easy_training_regime_stability_300m.py`. It asks whether the
39 domain-deletion rows carry reliable rank signal for each Table-9 component.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_TRAINING_REGIME_DIR = REFERENCE_OUTPUTS / "olmo_base_easy_training_regime_stability_300m_20260626"
DEFAULT_RELIABILITY = (
    REFERENCE_OUTPUTS / "olmo_base_easy_reliability_weighting_20260625" / "component_reliability.csv"
)
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "olmo_base_easy_deletion_uncertainty_300m_20260626"
DEFAULT_FIT_PANEL = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_paper_faithful_olmix_300m_20260625"
    / "fit_panel_table9_macro.csv"
)
MACRO_TARGET = "table9_macro_bpb"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class SpearmanUncertainty:
    scope: str
    target: str
    method: str
    n_rows: int
    spearman: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    permutation_two_sided_p: float
    permutation_positive_p: float
    bh_q_value: float | None
    significant_bh_0p05: bool | None
    bias: float
    rmse: float
    two_sided_t_excess: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-regime-dir", type=Path, default=DEFAULT_TRAINING_REGIME_DIR)
    parser.add_argument("--fit-panel", type=Path, default=DEFAULT_FIT_PANEL)
    parser.add_argument("--component-reliability", type=Path, default=DEFAULT_RELIABILITY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--permutation-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def component_family(component: str) -> str:
    name = component.replace("olmo_base_eval/easy_bpb/", "")
    if name.startswith("minerva_math") or name.startswith("mmlu_stem"):
        return "math_stem"
    if (
        name.startswith("codex_humaneval")
        or name.startswith("mbpp")
        or name.startswith("mt_mbpp")
        or name.startswith("basic_skills_coding")
    ):
        return "code"
    if name.startswith("mmlu_"):
        return "mmlu_nonstem"
    if name.startswith("basic_skills"):
        return "basic_skills_noncode"
    if name in {"arc_easy/bpb", "arc_challenge/bpb", "sciq/bpb", "medmcqa/bpb"}:
        return "science_qa"
    if name in {"csqa/bpb", "hellaswag/bpb", "winogrande/bpb", "socialiqa/bpb", "piqa/bpb"}:
        return "commonsense"
    if name in {"coqa/bpb", "drop/bpb", "jeopardy/bpb", "naturalqs/bpb", "squad/bpb", "lambada/bpb"}:
        return "reading_qa"
    return "other"


def spearman_stat(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 3 or np.std(y) == 0.0 or np.std(pred) == 0.0:
        return float("nan")
    return float(spearmanr(y, pred).statistic)


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    residual = pred - y
    return float(np.sqrt(np.mean(residual * residual)))


def bootstrap_ci(
    y: np.ndarray,
    pred: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values: list[float] = []
    n = len(y)
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        value = spearman_stat(y[idx], pred[idx])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def permutation_pvalues(
    y: np.ndarray,
    pred: np.ndarray,
    observed: float,
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if not np.isfinite(observed):
        return float("nan"), float("nan")
    null_values = np.empty(samples, dtype=float)
    for idx in range(samples):
        null_values[idx] = spearman_stat(y, rng.permutation(pred))
    two_sided = (1.0 + np.sum(np.abs(null_values) >= abs(observed))) / (samples + 1.0)
    positive = (1.0 + np.sum(null_values >= observed)) / (samples + 1.0)
    return float(two_sided), float(positive)


def metric_rows(
    *,
    scope: str,
    target: str,
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    reliability_t_excess: float | None,
    samples: int,
    permutations: int,
    rng: np.random.Generator,
) -> list[SpearmanUncertainty]:
    rows: list[SpearmanUncertainty] = []
    for method, pred in predictions.items():
        observed = spearman_stat(y, pred)
        ci_low, ci_high = bootstrap_ci(y, pred, samples=samples, rng=rng)
        two_sided_p, positive_p = permutation_pvalues(y, pred, observed, samples=permutations, rng=rng)
        rows.append(
            SpearmanUncertainty(
                scope=scope,
                target=target,
                method=method,
                n_rows=int(len(y)),
                spearman=observed,
                bootstrap_ci_low=ci_low,
                bootstrap_ci_high=ci_high,
                permutation_two_sided_p=two_sided_p,
                permutation_positive_p=positive_p,
                bh_q_value=None,
                significant_bh_0p05=None,
                bias=float(np.mean(pred - y)),
                rmse=rmse(y, pred),
                two_sided_t_excess=reliability_t_excess,
            )
        )
    return rows


def add_bh_q_values(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["bh_q_value"] = np.nan
    out["significant_bh_0p05"] = False
    for (_scope, method), index in out.groupby(["scope", "method"]).groups.items():
        p = pd.to_numeric(out.loc[index, "permutation_two_sided_p"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(p)
        if not np.any(valid):
            continue
        valid_positions = np.flatnonzero(valid)
        p_valid = p[valid]
        order = np.argsort(p_valid)
        ranked = p_valid[order]
        m = len(ranked)
        q_ranked = np.empty(m, dtype=float)
        running = 1.0
        for reverse_rank in range(m - 1, -1, -1):
            rank = reverse_rank + 1
            running = min(running, ranked[reverse_rank] * m / rank)
            q_ranked[reverse_rank] = running
        q_valid = np.empty(m, dtype=float)
        q_valid[order] = q_ranked
        target_index = np.asarray(index)[valid_positions]
        out.loc[target_index, "bh_q_value"] = q_valid
        out.loc[target_index, "significant_bh_0p05"] = q_valid <= 0.05
    return out


def load_component_predictions(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    if "panel_source" not in data.columns:
        raise ValueError(f"{path} is missing panel_source")
    return data


def component_prediction_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in data.columns if column.startswith("pred::")]


def write_plots(output_dir: Path, rows: pd.DataFrame) -> None:
    component = rows[rows["scope"].eq("component")].copy()
    component["short_target"] = component["target"].str.replace("olmo_base_eval/easy_bpb/", "", regex=False)
    qsplit = component[component["method"].eq("per_component_mean_qsplit_only")].sort_values("spearman")
    full = component[component["method"].eq("per_component_mean_deletion_augmented")].sort_values("spearman")
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Qsplit-only held-out deletion Spearman CI", "Deletion-augmented deletion Spearman CI"),
    )
    for col, view, color in (
        (1, qsplit, "#2f5d8a"),
        (2, full, "#c75035"),
    ):
        fig.add_trace(
            go.Scatter(
                x=view["spearman"],
                y=view["short_target"],
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": view["bootstrap_ci_high"] - view["spearman"],
                    "arrayminus": view["spearman"] - view["bootstrap_ci_low"],
                },
                mode="markers",
                marker={"color": color, "size": 8},
                hovertemplate="%{y}<br>rho=%{x:.3f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
    fig.update_xaxes(title_text="Spearman on 39 deletion rows")
    fig.update_layout(
        title="Domain-deletion rank-signal uncertainty by component",
        template="plotly_white",
        width=1700,
        height=1200,
        showlegend=False,
    )
    fig.write_html(output_dir / "component_deletion_spearman_ci.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    family = rows[rows["scope"].eq("family")].copy()
    fig = go.Figure()
    for method, color in (
        ("per_component_mean_qsplit_only", "#2f5d8a"),
        ("per_component_mean_deletion_augmented", "#c75035"),
    ):
        view = family[family["method"].eq(method)].sort_values("spearman")
        fig.add_trace(
            go.Bar(
                x=view["spearman"],
                y=view["target"],
                orientation="h",
                name=method,
                marker_color=color,
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": view["bootstrap_ci_high"] - view["spearman"],
                    "arrayminus": view["spearman"] - view["bootstrap_ci_low"],
                },
            )
        )
    fig.update_layout(
        title="Family-pooled domain-deletion rank signal",
        template="plotly_white",
        width=1200,
        height=650,
        barmode="group",
    )
    fig.update_xaxes(title_text="Spearman on deletion rows")
    fig.write_html(output_dir / "family_deletion_spearman_ci.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(output_dir: Path, rows: pd.DataFrame) -> None:
    macro = rows[rows["scope"].eq("macro")].copy()
    component = rows[rows["scope"].eq("component")].copy()
    family = rows[rows["scope"].eq("family")].copy()
    significant = component[
        component["significant_bh_0p05"]
        & (component["bootstrap_ci_low"] > 0.0)
        & component["method"].eq("per_component_mean_qsplit_only")
    ].sort_values("spearman", ascending=False)
    weak = component[
        component["method"].eq("per_component_mean_qsplit_only")
        & (component["bootstrap_ci_low"] <= 0.0)
        & (component["bootstrap_ci_high"] >= 0.0)
    ].sort_values("spearman")
    lines = [
        "# OLMoBaseEval Easy deletion rank-signal uncertainty",
        "",
        "This analyzes the 39 domain-deletion rows as held-out stress tests. Component-level CIs are wide; this should be interpreted as diagnostic evidence, not as a final reliability oracle.",
        "",
        "## Macro deletion rank signal",
        "",
        macro[
            [
                "method",
                "spearman",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "permutation_two_sided_p",
                "bias",
                "rmse",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Components with positive held-out deletion rank signal after BH correction",
        "",
        significant[
            [
                "target",
                "spearman",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "permutation_two_sided_p",
                "bh_q_value",
                "bias",
                "rmse",
                "two_sided_t_excess",
            ]
        ].head(20).to_markdown(index=False, floatfmt=".6f")
        if not significant.empty
        else "(none under CI-low > 0 and BH q < 0.05)",
        "",
        "## Components whose deletion rank signal is CI-consistent with zero",
        "",
        weak[
            [
                "target",
                "spearman",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "permutation_two_sided_p",
                "bh_q_value",
                "bias",
                "two_sided_t_excess",
            ]
        ].head(20).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Family-pooled deletion rank signal",
        "",
        family[
            [
                "target",
                "method",
                "spearman",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "permutation_two_sided_p",
                "bh_q_value",
                "bias",
                "rmse",
            ]
        ].sort_values(["method", "spearman"], ascending=[True, False]).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Artifacts",
        "",
        "- `deletion_spearman_uncertainty.csv`",
        "- `component_deletion_spearman_ci.html`",
        "- `family_deletion_spearman_ci.html`",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    macro = pd.read_csv(args.training_regime_dir / "method_macro_predictions.csv")
    fit_panel = pd.read_csv(args.fit_panel)
    qsplit_components = load_component_predictions(args.training_regime_dir / "qsplit_only_component_predictions.csv")
    full_components = load_component_predictions(args.training_regime_dir / "deletion_augmented_component_predictions.csv")
    actual_components = qsplit_components[["run_name", "panel_source"]].merge(
        fit_panel,
        on="run_name",
        how="left",
        validate="one_to_one",
        suffixes=("", "_actual"),
    )
    reliability = pd.read_csv(args.component_reliability)
    reliability_by_component = dict(zip(reliability["component"], reliability["two_sided_t_excess"], strict=False))

    deletion_mask = macro["panel_source"].eq("domain_deletion").to_numpy(dtype=bool)
    rows: list[SpearmanUncertainty] = []
    macro_predictions = {
        "aggregate_dsp_effective_exposure_qsplit_only": macro.loc[
            deletion_mask, "aggregate_dsp_effective_exposure_qsplit_only"
        ].to_numpy(dtype=float),
        "aggregate_dsp_effective_exposure_deletion_augmented": macro.loc[
            deletion_mask, "aggregate_dsp_effective_exposure_deletion_augmented"
        ].to_numpy(dtype=float),
        "per_component_mean_qsplit_only": macro.loc[deletion_mask, "per_component_mean_qsplit_only"].to_numpy(dtype=float),
        "per_component_mean_deletion_augmented": macro.loc[
            deletion_mask, "per_component_mean_deletion_augmented"
        ].to_numpy(dtype=float),
    }
    rows.extend(
        metric_rows(
            scope="macro",
            target=MACRO_TARGET,
            y=macro.loc[deletion_mask, MACRO_TARGET].to_numpy(dtype=float),
            predictions=macro_predictions,
            reliability_t_excess=None,
            samples=int(args.bootstrap_samples),
            permutations=int(args.permutation_samples),
            rng=rng,
        )
    )

    pred_cols = component_prediction_columns(qsplit_components)
    for pred_col in pred_cols:
        component = pred_col.removeprefix("pred::")
        y = actual_components.loc[deletion_mask, component].to_numpy(dtype=float)
        predictions = {
            "per_component_mean_qsplit_only": qsplit_components.loc[deletion_mask, pred_col].to_numpy(dtype=float),
            "per_component_mean_deletion_augmented": full_components.loc[deletion_mask, pred_col].to_numpy(dtype=float),
        }
        rows.extend(
            metric_rows(
                scope="component",
                target=component,
                y=y,
                predictions=predictions,
                reliability_t_excess=float(reliability_by_component.get(component, float("nan"))),
                samples=int(args.bootstrap_samples),
                permutations=int(args.permutation_samples),
                rng=rng,
            )
        )

    family_map = {pred_col: component_family(pred_col.removeprefix("pred::")) for pred_col in pred_cols}
    for family in sorted(set(family_map.values())):
        family_cols = [pred_col for pred_col, mapped in family_map.items() if mapped == family]
        if not family_cols:
            continue
        actual_cols = [pred_col.removeprefix("pred::") for pred_col in family_cols]
        y = actual_components.loc[deletion_mask, actual_cols].mean(axis=1).to_numpy(dtype=float)
        predictions = {
            "per_component_mean_qsplit_only": qsplit_components.loc[deletion_mask, family_cols].mean(axis=1).to_numpy(
                dtype=float
            ),
            "per_component_mean_deletion_augmented": full_components.loc[deletion_mask, family_cols].mean(axis=1).to_numpy(
                dtype=float
            ),
        }
        rows.extend(
            metric_rows(
                scope="family",
                target=family,
                y=y,
                predictions=predictions,
                reliability_t_excess=None,
                samples=int(args.bootstrap_samples),
                permutations=int(args.permutation_samples),
                rng=rng,
            )
        )

    frame = add_bh_q_values(pd.DataFrame([asdict(row) for row in rows]))
    frame.to_csv(args.output_dir / "deletion_spearman_uncertainty.csv", index=False)
    write_plots(args.output_dir, frame)
    write_report(args.output_dir, frame)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "training_regime_dir": str(args.training_regime_dir),
                "fit_panel": str(args.fit_panel),
                "component_reliability": str(args.component_reliability),
                "bootstrap_samples": int(args.bootstrap_samples),
                "permutation_samples": int(args.permutation_samples),
                "seed": int(args.seed),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(frame[frame["scope"].eq("macro")].to_string(index=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
