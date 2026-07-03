# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scipy", "scikit-learn", "tabulate"]
# ///
"""Matched sample-count controls for OLMoBaseEval Easy deletion augmentation.

The training-regime diagnostic found that adding 39 domain-deletion rows to the
241-row qsplit panel slightly improved aggregate Effective-exposure DSP fit, but
did not change the selected qsplit mixture. This script checks whether that
small improvement is deletion-specific, mostly explained by regularization, or
merely an artifact of adding more rows.

We hold the DSP variant and linear regularization fixed, then repeatedly train on
241-row subsets of the 280-row qsplit-plus-deletion panel:

* random_241_from_full: random 241-row subsets from all 280 rows.
* all_deletion_plus_qsplit: all 39 deletion rows plus random 202 qsplit rows.

These same-size controls are substitutive: deletion rows displace qsplit rows.
To test additive value directly, the script also compares paired fixed-base
samples:

* base_202_qsplit: random 202 qsplit rows.
* additive_202_qsplit_plus_39_deletion: the same 202 qsplit rows plus all 39
  deletion rows.

Qsplit rows used for training get fold-refit linear-head OOF predictions; qsplit
rows not sampled for training get held-out predictions from the fitted model.
The headline target remains unweighted OLMoBaseEval Easy Table-9 macro BPB.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_olmo_base_easy_training_regime_stability_300m as tr,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as base,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "olmo_base_easy_matched_sample_count_control_300m_20260626"
DEFAULT_FIT_PANEL = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_paper_faithful_olmix_300m_20260625"
    / "fit_panel_table9_macro.csv"
)
DEFAULT_PRIOR_METHOD_PREDICTIONS = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_training_regime_stability_300m_20260626"
    / "method_macro_predictions.csv"
)
DEFAULT_FULL_AGGREGATE_DSP = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_table9_macro_dsp_300m_20260625"
    / "effective_exposure_table9_macro_predictions.csv"
)

MACRO_TARGET = "table9_macro_bpb"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class ControlFit:
    method: str
    replicate: int
    train_regime: str
    train_subset_kind: str
    n_train: int
    n_train_qsplit: int
    n_train_deletion: int
    linear_reg: float
    qsplit_rmse: float
    qsplit_spearman: float
    qsplit_regret_at_1: float
    qsplit_regret_at_3: float
    qsplit_regret_at_5: float
    selected_run_name: str
    selected_actual_bpb: float
    selected_prediction: float
    selected_actual_rank: int
    best_observed_run_name: str
    best_observed_bpb: float
    deletion_rmse: float
    deletion_spearman: float
    deletion_regret_at_3: float
    full_rmse: float
    full_spearman: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-panel", type=Path, default=DEFAULT_FIT_PANEL)
    parser.add_argument("--prior-method-predictions", type=Path, default=DEFAULT_PRIOR_METHOD_PREDICTIONS)
    parser.add_argument("--full-aggregate-dsp-predictions", type=Path, default=DEFAULT_FULL_AGGREGATE_DSP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--additive-replicates", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--linear-reg", type=float, default=0.001)
    parser.add_argument("--extra-qsplit-linear-reg-values", default="0.0001")
    parser.add_argument("--maxiter", type=int, default=24)
    parser.add_argument("--coarse-top-k", type=int, default=2)
    parser.add_argument("--basin-hopping-iters", type=int, default=0)
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def regression_rmse(y: np.ndarray, pred: np.ndarray) -> float:
    residual = pred - y
    return float(np.sqrt(np.mean(residual * residual)))


def regression_spearman(y: np.ndarray, pred: np.ndarray) -> float:
    if np.std(y) == 0.0 or np.std(pred) == 0.0:
        return float("nan")
    return float(spearmanr(y, pred).statistic)


def metric_summary(
    *,
    method: str,
    replicate: int,
    train_regime: str,
    train_subset_kind: str,
    train_indices: np.ndarray,
    qsplit_indices: np.ndarray,
    deletion_indices: np.ndarray,
    panel: pd.DataFrame,
    y: np.ndarray,
    pred: np.ndarray,
    linear_reg: float,
) -> ControlFit:
    qsplit_decision = tr.summarize_prediction(
        method=method,
        family="aggregate_dsp",
        train_regime=train_regime,
        eval_subset="qsplit_signal",
        prediction_convention="train_oof_else_heldout",
        panel=panel,
        y=y,
        pred=pred,
        indices=qsplit_indices,
    )
    deletion_decision = tr.summarize_prediction(
        method=method,
        family="aggregate_dsp",
        train_regime=train_regime,
        eval_subset="domain_deletion",
        prediction_convention="train_oof_else_heldout",
        panel=panel,
        y=y,
        pred=pred,
        indices=deletion_indices,
    )
    full_indices = np.arange(len(panel), dtype=int)
    return ControlFit(
        method=method,
        replicate=int(replicate),
        train_regime=train_regime,
        train_subset_kind=train_subset_kind,
        n_train=int(len(train_indices)),
        n_train_qsplit=int(np.intersect1d(train_indices, qsplit_indices).size),
        n_train_deletion=int(np.intersect1d(train_indices, deletion_indices).size),
        linear_reg=float(linear_reg),
        qsplit_rmse=float(qsplit_decision.rmse),
        qsplit_spearman=float(qsplit_decision.spearman),
        qsplit_regret_at_1=float(qsplit_decision.regret_at_1),
        qsplit_regret_at_3=float(qsplit_decision.regret_at_3),
        qsplit_regret_at_5=float(qsplit_decision.regret_at_5),
        selected_run_name=qsplit_decision.selected_run_name,
        selected_actual_bpb=float(qsplit_decision.selected_actual_bpb),
        selected_prediction=float(qsplit_decision.selected_prediction),
        selected_actual_rank=int(qsplit_decision.selected_actual_rank),
        best_observed_run_name=qsplit_decision.best_observed_run_name,
        best_observed_bpb=float(qsplit_decision.best_observed_bpb),
        deletion_rmse=float(deletion_decision.rmse),
        deletion_spearman=float(deletion_decision.spearman),
        deletion_regret_at_3=float(deletion_decision.regret_at_3),
        full_rmse=regression_rmse(y[full_indices], pred[full_indices]),
        full_spearman=regression_spearman(y[full_indices], pred[full_indices]),
    )


def load_prior_prediction(panel: pd.DataFrame, path: Path, column: str) -> np.ndarray:
    data = pd.read_csv(path)
    if column not in data.columns:
        raise ValueError(f"Missing prediction column {column!r} in {path}")
    merged = panel[["run_name"]].merge(data[["run_name", column]], on="run_name", how="left", validate="one_to_one")
    if merged[column].isna().any():
        raise ValueError(f"Missing rows while merging {column!r}")
    return merged[column].to_numpy(dtype=float)


def load_full_augmented_prediction(panel: pd.DataFrame, path: Path, *, linear_reg: float) -> np.ndarray:
    data = pd.read_csv(path)
    view = data[
        data["variant"].eq("effective_exposure") & np.isclose(data["hyperparameter_value"].to_numpy(dtype=float), linear_reg)
    ][["run_name", "oof_prediction"]].copy()
    if view.empty:
        raise ValueError(f"No full-panel aggregate prediction for linear_reg={linear_reg:g}")
    merged = panel[["run_name"]].merge(view, on="run_name", how="left", validate="one_to_one")
    if merged["oof_prediction"].isna().any():
        raise ValueError("Missing rows in full-panel aggregate prediction")
    return merged["oof_prediction"].to_numpy(dtype=float)


def fit_subset_prediction(
    *,
    packet_full: tr.dsp.PacketData,
    train_indices: np.ndarray,
    linear_reg: float,
    maxiter: int,
    coarse_top_k: int,
    basin_hopping_iters: int,
) -> np.ndarray:
    _model, all_pred, train_oof = tr.fit_effective_exposure_on_subset(
        packet_full=packet_full,
        train_indices=train_indices,
        linear_reg=linear_reg,
        maxiter=maxiter,
        coarse_top_k=coarse_top_k,
        basin_hopping_iters=basin_hopping_iters,
    )
    return tr.prediction_with_train_oof(all_pred=all_pred, train_indices=train_indices, train_oof=train_oof)


def random_full_subsets(
    *,
    rng: np.random.Generator,
    all_indices: np.ndarray,
    n_train: int,
    replicates: int,
) -> list[np.ndarray]:
    return [
        np.sort(rng.choice(all_indices, size=n_train, replace=False).astype(int))
        for _ in range(replicates)
    ]


def all_deletion_plus_qsplit_subsets(
    *,
    rng: np.random.Generator,
    qsplit_indices: np.ndarray,
    deletion_indices: np.ndarray,
    n_train: int,
    replicates: int,
) -> list[np.ndarray]:
    n_qsplit_needed = n_train - len(deletion_indices)
    if n_qsplit_needed <= 0:
        raise ValueError("Training count must exceed deletion count")
    return [
        np.sort(
            np.concatenate(
                [
                    deletion_indices,
                    rng.choice(qsplit_indices, size=n_qsplit_needed, replace=False).astype(int),
                ]
            )
        )
        for _ in range(replicates)
    ]


def paired_additive_subsets(
    *,
    rng: np.random.Generator,
    qsplit_indices: np.ndarray,
    deletion_indices: np.ndarray,
    base_qsplit_count: int,
    replicates: int,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    pairs: list[tuple[int, np.ndarray, np.ndarray]] = []
    for replicate in range(replicates):
        base_qsplit = np.sort(rng.choice(qsplit_indices, size=base_qsplit_count, replace=False).astype(int))
        additive = np.sort(np.concatenate([base_qsplit, deletion_indices]))
        pairs.append((replicate, base_qsplit, additive))
    return pairs


def summarize_by_method(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby("method", sort=True)
    summary = grouped.agg(
        n_replicates=("replicate", "count"),
        mean_train_deletion=("n_train_deletion", "mean"),
        qsplit_rmse_mean=("qsplit_rmse", "mean"),
        qsplit_rmse_sd=("qsplit_rmse", "std"),
        qsplit_spearman_mean=("qsplit_spearman", "mean"),
        qsplit_spearman_sd=("qsplit_spearman", "std"),
        qsplit_regret_at_3_mean=("qsplit_regret_at_3", "mean"),
        selected_rank_median=("selected_actual_rank", "median"),
        selected_rank_min=("selected_actual_rank", "min"),
        selected_rank_max=("selected_actual_rank", "max"),
        deletion_rmse_mean=("deletion_rmse", "mean"),
        deletion_spearman_mean=("deletion_spearman", "mean"),
        full_rmse_mean=("full_rmse", "mean"),
        full_spearman_mean=("full_spearman", "mean"),
    ).reset_index()
    return summary


def write_plots(output_dir: Path, rows: pd.DataFrame, reference_rows: pd.DataFrame) -> None:
    stochastic = rows[rows["replicate"] >= 0].copy()
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Qsplit RMSE", "Qsplit Spearman", "Selected actual rank"),
    )
    colors = {
        "random_241_from_full": "#2f5d8a",
        "all_deletion_plus_qsplit": "#c75035",
        "base_202_qsplit": "#7a7a7a",
        "additive_202_qsplit_plus_39_deletion": "#3d8f5f",
    }
    for method, group in stochastic.groupby("method", sort=True):
        fig.add_trace(
            go.Box(y=group["qsplit_rmse"], name=method, marker_color=colors.get(method, "#555555")),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Box(y=group["qsplit_spearman"], name=method, marker_color=colors.get(method, "#555555"), showlegend=False),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Box(y=group["selected_actual_rank"], name=method, marker_color=colors.get(method, "#555555"), showlegend=False),
            row=1,
            col=3,
        )
    for _, ref in reference_rows.iterrows():
        line_color = "#111111" if "qsplit_only" in ref["method"] else "#777777"
        fig.add_hline(y=float(ref["qsplit_rmse"]), line_dash="dash", line_color=line_color, row=1, col=1)
        fig.add_hline(y=float(ref["qsplit_spearman"]), line_dash="dash", line_color=line_color, row=1, col=2)
        fig.add_hline(y=float(ref["selected_actual_rank"]), line_dash="dash", line_color=line_color, row=1, col=3)
    fig.update_layout(
        title="Matched sample-count controls for deletion augmentation",
        template="plotly_white",
        width=1650,
        height=650,
        boxmode="group",
    )
    fig.write_html(output_dir / "matched_sample_count_control_boxplots.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    fig = go.Figure()
    for method, group in stochastic.groupby("method", sort=True):
        fig.add_trace(
            go.Scatter(
                x=group["n_train_deletion"],
                y=group["qsplit_rmse"],
                mode="markers",
                name=method,
                marker={"color": colors.get(method, "#555555"), "opacity": 0.75, "size": 8},
                hovertemplate="method=%{fullData.name}<br>deletion rows=%{x}<br>qsplit RMSE=%{y:.5f}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Qsplit RMSE versus number of deletion rows in same-size training subset",
        template="plotly_white",
        width=1050,
        height=650,
        xaxis_title="Deletion rows in 241-row training subset",
        yaxis_title="Qsplit RMSE",
    )
    fig.write_html(output_dir / "matched_sample_count_deletion_count_vs_rmse.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(output_dir: Path, rows: pd.DataFrame, summary: pd.DataFrame, reference_rows: pd.DataFrame) -> None:
    stochastic = rows[rows["replicate"] >= 0].copy()
    random_summary = stochastic[stochastic["method"].eq("random_241_from_full")]
    all_deletion_summary = stochastic[stochastic["method"].eq("all_deletion_plus_qsplit")]
    base_202 = stochastic[stochastic["method"].eq("base_202_qsplit")]
    additive = stochastic[stochastic["method"].eq("additive_202_qsplit_plus_39_deletion")]
    additive_pairs = pd.DataFrame()
    if not base_202.empty and not additive.empty:
        additive_pairs = base_202[
            ["replicate", "qsplit_rmse", "qsplit_spearman", "qsplit_regret_at_3", "selected_actual_rank"]
        ].merge(
            additive[
                ["replicate", "qsplit_rmse", "qsplit_spearman", "qsplit_regret_at_3", "selected_actual_rank"]
            ],
            on="replicate",
            suffixes=("_base_202", "_additive"),
            validate="one_to_one",
        )
        additive_pairs["delta_qsplit_rmse_additive_minus_base"] = (
            additive_pairs["qsplit_rmse_additive"] - additive_pairs["qsplit_rmse_base_202"]
        )
        additive_pairs["delta_qsplit_spearman_additive_minus_base"] = (
            additive_pairs["qsplit_spearman_additive"] - additive_pairs["qsplit_spearman_base_202"]
        )
    reference = reference_rows.set_index("method")
    qsplit_rmse = float(reference.loc["qsplit_only_reference", "qsplit_rmse"])
    full_same_reg_rmse = float(reference.loc["full_deletion_augmented_same_l2_reference", "qsplit_rmse"])
    full_best_rmse = float(reference.loc["full_deletion_augmented_prior_best_l2_reference", "qsplit_rmse"])

    lines = [
        "# OLMoBaseEval Easy matched sample-count deletion control",
        "",
        "This control asks whether the apparent deletion-augmented aggregate DSP gain survives when the training sample count is held fixed at the 241 qsplit rows.",
        "",
        "## Reference rows",
        "",
        reference_rows[
            [
                "method",
                "n_train",
                "n_train_qsplit",
                "n_train_deletion",
                "linear_reg",
                "qsplit_rmse",
                "qsplit_spearman",
                "qsplit_regret_at_3",
                "selected_run_name",
                "selected_actual_rank",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Matched-count control summary",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        f"- Qsplit-only reference RMSE: `{qsplit_rmse:.6f}`.",
        f"- Full 280-row deletion-augmented reference at the same L2 RMSE: `{full_same_reg_rmse:.6f}`.",
        f"- Prior best 280-row deletion-augmented reference RMSE: `{full_best_rmse:.6f}`.",
    ]
    if not random_summary.empty:
        lines.append(
            f"- Random same-size full-panel subsets have mean qsplit RMSE `{random_summary['qsplit_rmse'].mean():.6f}` "
            f"with SD `{random_summary['qsplit_rmse'].std(ddof=1):.6f}`."
        )
    if not all_deletion_summary.empty:
        lines.append(
            f"- Same-size subsets that force all deletion rows have mean qsplit RMSE `{all_deletion_summary['qsplit_rmse'].mean():.6f}` "
            f"with SD `{all_deletion_summary['qsplit_rmse'].std(ddof=1):.6f}`."
        )
    if not additive_pairs.empty:
        lines.extend(
            [
                f"- Paired additive controls change qsplit RMSE by mean "
                f"`{additive_pairs['delta_qsplit_rmse_additive_minus_base'].mean():.6f}` "
                f"(additive minus base; negative is better) with SD "
                f"`{additive_pairs['delta_qsplit_rmse_additive_minus_base'].std(ddof=1):.6f}`.",
                f"- Paired additive controls change qsplit Spearman by mean "
                f"`{additive_pairs['delta_qsplit_spearman_additive_minus_base'].mean():.6f}` "
                f"(positive is better) with SD "
                f"`{additive_pairs['delta_qsplit_spearman_additive_minus_base'].std(ddof=1):.6f}`.",
            ]
        )
    best_random = stochastic.sort_values("qsplit_rmse").head(5)[
        ["method", "replicate", "n_train_deletion", "qsplit_rmse", "qsplit_spearman", "selected_run_name", "selected_actual_rank"]
    ]
    lines.extend(
        [
            "",
            "## Best same-size replicates by qsplit RMSE",
            "",
            best_random.to_markdown(index=False, floatfmt=".6f"),
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    if not additive_pairs.empty:
        additive_pairs.to_csv(output_dir / "paired_additive_control_deltas.csv", index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.fit_panel)
    y = pd.to_numeric(panel[MACRO_TARGET], errors="raise").to_numpy(dtype=float)
    qsplit_indices = np.flatnonzero(panel["panel_source"].eq("qsplit_signal").to_numpy(dtype=bool))
    deletion_indices = np.flatnonzero(panel["panel_source"].eq("domain_deletion").to_numpy(dtype=bool))
    all_indices = np.arange(len(panel), dtype=int)
    n_train = int(len(qsplit_indices))

    _signal, columns, domains, _natural = base.load_raw_signal_panel()
    token_counts = base.load_domain_token_counts(domains)
    packet_full = tr.build_packet(panel, columns, domains, token_counts, MACRO_TARGET)

    qsplit_reference_pred = load_prior_prediction(
        panel,
        args.prior_method_predictions,
        "aggregate_dsp_effective_exposure_qsplit_only",
    )
    full_same_l2_pred = load_full_augmented_prediction(
        panel,
        args.full_aggregate_dsp_predictions,
        linear_reg=float(args.linear_reg),
    )
    full_prior_best_pred = load_prior_prediction(
        panel,
        args.prior_method_predictions,
        "aggregate_dsp_effective_exposure_deletion_augmented",
    )

    reference_specs = [
        ("qsplit_only_reference", "qsplit_only", qsplit_indices, qsplit_reference_pred, float(args.linear_reg)),
        (
            "full_deletion_augmented_same_l2_reference",
            "qsplit_plus_deletion",
            all_indices,
            full_same_l2_pred,
            float(args.linear_reg),
        ),
        (
            "full_deletion_augmented_prior_best_l2_reference",
            "qsplit_plus_deletion",
            all_indices,
            full_prior_best_pred,
            0.0001,
        ),
    ]
    reference_rows = pd.DataFrame(
        [
            asdict(
                metric_summary(
                    method=method,
                    replicate=-1,
                    train_regime=train_regime,
                    train_subset_kind="reference",
                    train_indices=train_indices,
                    qsplit_indices=qsplit_indices,
                    deletion_indices=deletion_indices,
                    panel=panel,
                    y=y,
                    pred=pred,
                    linear_reg=linear_reg,
                )
            )
            for method, train_regime, train_indices, pred, linear_reg in reference_specs
        ]
    )
    extra_qsplit_refs: list[dict[str, Any]] = []
    for extra_linear_reg in parse_float_list(str(args.extra_qsplit_linear_reg_values)):
        if np.isclose(extra_linear_reg, float(args.linear_reg)):
            continue
        print(f"Fitting qsplit-only reference at linear_reg={extra_linear_reg:g}", flush=True)
        extra_pred = fit_subset_prediction(
            packet_full=packet_full,
            train_indices=qsplit_indices,
            linear_reg=float(extra_linear_reg),
            maxiter=int(args.maxiter),
            coarse_top_k=int(args.coarse_top_k),
            basin_hopping_iters=int(args.basin_hopping_iters),
        )
        extra_qsplit_refs.append(
            asdict(
                metric_summary(
                    method=f"qsplit_only_l2_{extra_linear_reg:g}_reference",
                    replicate=-1,
                    train_regime="qsplit_only",
                    train_subset_kind="reference",
                    train_indices=qsplit_indices,
                    qsplit_indices=qsplit_indices,
                    deletion_indices=deletion_indices,
                    panel=panel,
                    y=y,
                    pred=extra_pred,
                    linear_reg=float(extra_linear_reg),
                )
            )
        )
    if extra_qsplit_refs:
        reference_rows = pd.concat([reference_rows, pd.DataFrame(extra_qsplit_refs)], ignore_index=True)

    rng = np.random.default_rng(int(args.seed))
    subsets: list[tuple[str, int, np.ndarray]] = []
    for replicate, indices in enumerate(
        random_full_subsets(
            rng=rng,
            all_indices=all_indices,
            n_train=n_train,
            replicates=int(args.replicates),
        )
    ):
        subsets.append(("random_241_from_full", replicate, indices))
    for replicate, indices in enumerate(
        all_deletion_plus_qsplit_subsets(
            rng=rng,
            qsplit_indices=qsplit_indices,
            deletion_indices=deletion_indices,
            n_train=n_train,
            replicates=int(args.replicates),
        )
    ):
        subsets.append(("all_deletion_plus_qsplit", replicate, indices))
    base_qsplit_count = n_train - len(deletion_indices)
    for replicate, base_qsplit, additive in paired_additive_subsets(
        rng=rng,
        qsplit_indices=qsplit_indices,
        deletion_indices=deletion_indices,
        base_qsplit_count=base_qsplit_count,
        replicates=int(args.additive_replicates),
    ):
        subsets.append(("base_202_qsplit", replicate, base_qsplit))
        subsets.append(("additive_202_qsplit_plus_39_deletion", replicate, additive))

    rows: list[ControlFit] = []
    for position, (method, replicate, train_indices) in enumerate(subsets, start=1):
        print(
            f"[{position}/{len(subsets)}] fitting {method} replicate={replicate} "
            f"n_deletion={np.intersect1d(train_indices, deletion_indices).size}",
            flush=True,
        )
        pred = fit_subset_prediction(
            packet_full=packet_full,
            train_indices=train_indices,
            linear_reg=float(args.linear_reg),
            maxiter=int(args.maxiter),
            coarse_top_k=int(args.coarse_top_k),
            basin_hopping_iters=int(args.basin_hopping_iters),
        )
        rows.append(
            metric_summary(
                method=method,
                replicate=replicate,
                train_regime="same_size_mixed_panel",
                train_subset_kind=method,
                train_indices=train_indices,
                qsplit_indices=qsplit_indices,
                deletion_indices=deletion_indices,
                panel=panel,
                y=y,
                pred=pred,
                linear_reg=float(args.linear_reg),
            )
        )

    rows_frame = pd.concat([reference_rows, pd.DataFrame([asdict(row) for row in rows])], ignore_index=True)
    stochastic = rows_frame[rows_frame["replicate"] >= 0].copy()
    summary = summarize_by_method(stochastic)
    rows_frame.to_csv(args.output_dir / "matched_sample_count_control_replicates.csv", index=False)
    reference_rows.to_csv(args.output_dir / "matched_sample_count_reference_rows.csv", index=False)
    summary.to_csv(args.output_dir / "matched_sample_count_control_summary.csv", index=False)
    write_plots(args.output_dir, rows_frame, reference_rows)
    write_report(args.output_dir, rows_frame, summary, reference_rows)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "fit_panel": str(args.fit_panel),
                "prior_method_predictions": str(args.prior_method_predictions),
                "full_aggregate_dsp_predictions": str(args.full_aggregate_dsp_predictions),
                "replicates_per_control": int(args.replicates),
                "additive_replicates": int(args.additive_replicates),
                "seed": int(args.seed),
                "linear_reg": float(args.linear_reg),
                "extra_qsplit_linear_reg_values": parse_float_list(str(args.extra_qsplit_linear_reg_values)),
                "maxiter": int(args.maxiter),
                "coarse_top_k": int(args.coarse_top_k),
                "basin_hopping_iters": int(args.basin_hopping_iters),
                "n_qsplit": int(len(qsplit_indices)),
                "n_deletion": int(len(deletion_indices)),
                "headline_objective": "unweighted 51-component Table-9 macro BPB",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(summary.to_string(index=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
