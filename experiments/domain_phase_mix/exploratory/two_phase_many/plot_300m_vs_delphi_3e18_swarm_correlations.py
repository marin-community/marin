# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "matplotlib", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Compare matched 300M and Delphi 3e18 one- and two-phase swarm outcomes."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
OUTPUT_DIR = REFERENCE_OUTPUTS / "300m_vs_delphi_3e18_swarm_correlations_20260719"
DELPHI_TWO_PHASE = REFERENCE_OUTPUTS / "delphi_augmented_swarm_3e18_20260714" / "delphi_augmented_swarm_3e18_wide.csv"
DELPHI_ONE_PHASE_MANIFEST = (
    REFERENCE_OUTPUTS / "delphi_one_phase_augmented_swarm_3e18_20260715" / "training_manifest.csv"
)
DELPHI_HELDOUTS = REFERENCE_OUTPUTS / "delphi_3e18_append_only_heldouts_20260714" / "heldout_current.csv"
ONE_PHASE_300M = (
    REFERENCE_OUTPUTS
    / "one_phase_swarm_scores_export_300m_20260630"
    / "one_phase_augmented_fit_panel_uncheatable_table9_scores_300m.csv"
)
ONE_PHASE_SERIES = "delphi_one_phase_augmented_swarm_3e18_20260715"
OUTPUT_CONFIG = {"dpi": 220, "bbox_inches": "tight"}
EXPECTED_ROWS = 280
EXPECTED_NEW_ONE_PHASE_ROWS = 238
EXPECTED_ALIAS_ROWS = 42
POLICY_LABELS = {"single_phase": "Single phase", "two_phase": "Two phase"}
TARGETS = {
    "uncheatable": {
        "label": "Uncheatable eval BPB",
        "column_300m": "eval_uncheatable_eval_bpb",
        "column_3e18": "uncheatable_bpb",
    },
    "table9": {
        "label": "OLMoBaseEval Table-9 macro BPB",
        "column_300m": "table9_macro_bpb",
        "column_3e18": "table9_macro_bpb",
    },
}


@dataclass(frozen=True)
class CorrelationSummary:
    objective: str
    policy_class: str
    n: int
    pearson_r: float
    pearson_pvalue: float
    spearman_rho: float
    spearman_pvalue: float
    kendall_tau: float
    kendall_pvalue: float
    regression_slope: float
    regression_intercept: float
    bpb_300m_min: float
    bpb_300m_max: float
    bpb_3e18_min: float
    bpb_3e18_max: float


def _load_two_phase_3e18() -> pd.DataFrame:
    frame = pd.read_csv(DELPHI_TWO_PHASE)
    required = {"run_name", "uncheatable_bpb", "table9_macro_bpb", "table9_macro_bpb.1"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing Delphi two-phase columns: {sorted(required - set(frame.columns))}")
    if len(frame) != EXPECTED_ROWS or frame["run_name"].nunique() != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} unique Delphi two-phase rows, found {len(frame)}")
    if not np.allclose(frame["table9_macro_bpb"], frame["table9_macro_bpb.1"], atol=1e-12):
        raise ValueError("The two independently exported Delphi Table-9 macro columns disagree")
    return frame


def _load_one_phase_3e18(two_phase: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.read_csv(DELPHI_ONE_PHASE_MANIFEST)
    heldouts = pd.read_csv(DELPHI_HELDOUTS)
    heldouts = heldouts.loc[heldouts["training_series"] == ONE_PHASE_SERIES].copy()
    heldouts["manifest_run_name"] = heldouts["wandb_run_name"].str.replace(
        re.compile(r"-[0-9a-f]{6}$"),
        "",
        regex=True,
    )

    dispositions = manifest["disposition"].value_counts().to_dict()
    expected_dispositions = {
        "scheduled_new_training": EXPECTED_NEW_ONE_PHASE_ROWS,
        "reused_exact_phase_tied_alias": EXPECTED_ALIAS_ROWS,
    }
    if dispositions != expected_dispositions:
        raise ValueError(f"Unexpected one-phase manifest dispositions: {dispositions}")
    if len(heldouts) != EXPECTED_NEW_ONE_PHASE_ROWS or heldouts["manifest_run_name"].nunique() != len(heldouts):
        raise ValueError(f"Expected {EXPECTED_NEW_ONE_PHASE_ROWS} unique newly trained one-phase rows")

    score_columns = ["uncheatable_bpb", "table9_macro_bpb"]
    scores = heldouts[["manifest_run_name", *score_columns]]
    complete = manifest.merge(
        scores,
        left_on="run_name",
        right_on="manifest_run_name",
        how="left",
        validate="one_to_one",
    )
    alias_mask = complete["disposition"] == "reused_exact_phase_tied_alias"
    two_phase_index = two_phase.set_index("run_name")
    for column in score_columns:
        complete.loc[alias_mask, column] = complete.loc[alias_mask, "source_run_name"].map(two_phase_index[column])

    if len(complete) != EXPECTED_ROWS or complete[score_columns].isna().any().any():
        raise ValueError("The reconstructed Delphi one-phase panel is incomplete")
    return complete


def _one_phase_300m_match_key(frame: pd.DataFrame) -> pd.Series:
    # The deletion export uses the deleted domain as source_run_name; its run_name is the shared panel identifier.
    return frame["source_run_name"].where(
        frame["source_panel"] != "proportional_domain_deletion",
        frame["run_name"],
    )


def _matched_frames(
    objective: str,
    two_phase_3e18: pd.DataFrame,
    one_phase_3e18: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    target = TARGETS[objective]
    two_phase_300m = pooled.load_300m_dataset(objective)
    two_300 = pd.DataFrame(
        {
            "logical_run_name": two_phase_300m.frame["run_name"].astype(str),
            "panel_source": two_phase_300m.frame["panel_source"].astype(str),
            "bpb_300m": np.asarray(two_phase_300m.y, dtype=float),
        }
    )
    two_3e18 = two_phase_3e18[["run_name", target["column_3e18"]]].rename(
        columns={"run_name": "logical_run_name", target["column_3e18"]: "bpb_3e18"}
    )
    two_phase = two_300.merge(two_3e18, on="logical_run_name", validate="one_to_one")

    one_phase_300m = pd.read_csv(
        ONE_PHASE_300M,
        usecols=["run_name", "source_panel", "panel_source", "source_run_name", target["column_300m"]],
    )
    one_phase_300m["logical_run_name"] = _one_phase_300m_match_key(one_phase_300m)
    one_300 = one_phase_300m[["logical_run_name", "panel_source", target["column_300m"]]].rename(
        columns={target["column_300m"]: "bpb_300m"}
    )
    one_3e18 = one_phase_3e18[["source_run_name", target["column_3e18"]]].rename(
        columns={"source_run_name": "logical_run_name", target["column_3e18"]: "bpb_3e18"}
    )
    single_phase = one_300.merge(one_3e18, on="logical_run_name", validate="one_to_one")

    expected_keys = set(two_phase["logical_run_name"])
    if set(single_phase["logical_run_name"]) != expected_keys:
        raise ValueError("The one- and two-phase panels do not cover the same logical designs")
    for policy_class, frame in (("single_phase", single_phase), ("two_phase", two_phase)):
        if len(frame) != EXPECTED_ROWS or frame[["bpb_300m", "bpb_3e18"]].isna().any().any():
            raise ValueError(f"Incomplete matched panel for {objective}/{policy_class}")
        frame["rank_300m"] = rankdata(frame["bpb_300m"], method="average")
        frame["rank_3e18"] = rankdata(frame["bpb_3e18"], method="average")
        frame["objective"] = objective
        frame["policy_class"] = policy_class
    return {"single_phase": single_phase, "two_phase": two_phase}


def _summary(objective: str, policy_class: str, frame: pd.DataFrame) -> CorrelationSummary:
    pearson = pearsonr(frame["bpb_300m"], frame["bpb_3e18"])
    spearman = spearmanr(frame["bpb_300m"], frame["bpb_3e18"])
    kendall = kendalltau(frame["bpb_300m"], frame["bpb_3e18"])
    slope, intercept = np.polyfit(frame["bpb_300m"], frame["bpb_3e18"], deg=1)
    return CorrelationSummary(
        objective=objective,
        policy_class=policy_class,
        n=len(frame),
        pearson_r=float(pearson.statistic),
        pearson_pvalue=float(pearson.pvalue),
        spearman_rho=float(spearman.statistic),
        spearman_pvalue=float(spearman.pvalue),
        kendall_tau=float(kendall.statistic),
        kendall_pvalue=float(kendall.pvalue),
        regression_slope=float(slope),
        regression_intercept=float(intercept),
        bpb_300m_min=float(frame["bpb_300m"].min()),
        bpb_300m_max=float(frame["bpb_300m"].max()),
        bpb_3e18_min=float(frame["bpb_3e18"].min()),
        bpb_3e18_max=float(frame["bpb_3e18"].max()),
    )


def _plot_bpb(
    ax: plt.Axes,
    frame: pd.DataFrame,
    summary: CorrelationSummary,
    norm: Normalize,
) -> plt.Collection:
    scatter = ax.scatter(
        frame["bpb_300m"],
        frame["bpb_3e18"],
        c=frame["bpb_3e18"],
        cmap="RdYlGn_r",
        norm=norm,
        s=30,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.25,
    )
    x_line = np.linspace(frame["bpb_300m"].min(), frame["bpb_300m"].max(), 200)
    ax.plot(
        x_line,
        summary.regression_slope * x_line + summary.regression_intercept,
        color="#23313d",
        linestyle="--",
        linewidth=1.2,
    )
    ax.set_title(f"{POLICY_LABELS[summary.policy_class]}: BPB", fontweight="semibold")
    ax.set_xlabel("300M / 6B BPB")
    ax.set_ylabel("Delphi 3e18 BPB")
    ax.text(
        0.03,
        0.96,
        f"Pearson $r$ = {summary.pearson_r:.3f}\nSpearman $\\rho$ = {summary.spearman_rho:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "#d7d0c3", "alpha": 0.88, "boxstyle": "round,pad=0.35"},
    )
    ax.grid(alpha=0.2, linewidth=0.6)
    return scatter


def _plot_rank(ax: plt.Axes, frame: pd.DataFrame, summary: CorrelationSummary, norm: Normalize) -> None:
    ax.scatter(
        frame["rank_300m"],
        frame["rank_3e18"],
        c=frame["bpb_3e18"],
        cmap="RdYlGn_r",
        norm=norm,
        s=30,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.25,
    )
    ax.plot([1, EXPECTED_ROWS], [1, EXPECTED_ROWS], color="#23313d", linestyle="--", linewidth=1.2)
    ax.set_title(f"{POLICY_LABELS[summary.policy_class]}: rank", fontweight="semibold")
    ax.set_xlabel("300M / 6B rank")
    ax.set_ylabel("Delphi 3e18 rank")
    ax.text(
        0.03,
        0.96,
        f"Spearman $\\rho$ = {summary.spearman_rho:.3f}\nKendall $\\tau$ = {summary.kendall_tau:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "#d7d0c3", "alpha": 0.88, "boxstyle": "round,pad=0.35"},
    )
    ax.grid(alpha=0.2, linewidth=0.6)


def _plot_objective(
    objective: str,
    frames: dict[str, pd.DataFrame],
    summaries: dict[str, CorrelationSummary],
) -> Path:
    all_3e18 = np.concatenate([frames[policy]["bpb_3e18"].to_numpy() for policy in POLICY_LABELS])
    norm = Normalize(vmin=float(all_3e18.min()), vmax=float(all_3e18.max()))
    figure, axes = plt.subplots(2, 2, figsize=(13.4, 11.0), constrained_layout=True)
    scatter = None
    for column, policy_class in enumerate(POLICY_LABELS):
        scatter = _plot_bpb(axes[0, column], frames[policy_class], summaries[policy_class], norm)
        _plot_rank(axes[1, column], frames[policy_class], summaries[policy_class], norm)
    assert scatter is not None
    colorbar = figure.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.78, pad=0.02)
    colorbar.set_label(f"Delphi 3e18 {TARGETS[objective]['label']}")
    figure.suptitle(
        f"300M / 6B to Delphi 3e18 scale transfer: {TARGETS[objective]['label']}\n"
        f"{EXPECTED_ROWS} exactly matched logical policies per policy class; lower is better",
        fontsize=15,
        fontweight="semibold",
    )
    output = OUTPUT_DIR / f"300m_vs_3e18_{objective}_bpb_and_rank_correlations.png"
    figure.savefig(output, **OUTPUT_CONFIG)
    plt.close(figure)
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    two_phase_3e18 = _load_two_phase_3e18()
    one_phase_3e18 = _load_one_phase_3e18(two_phase_3e18)
    all_rows: list[pd.DataFrame] = []
    all_summaries: list[CorrelationSummary] = []
    sensitivity_summaries: list[dict[str, object]] = []
    outputs: dict[str, str] = {}
    for objective in TARGETS:
        frames = _matched_frames(objective, two_phase_3e18, one_phase_3e18)
        summaries = {policy: _summary(objective, policy, frame) for policy, frame in frames.items()}
        all_summaries.extend(summaries.values())
        for policy, frame in frames.items():
            non_deletion = frame.loc[frame["panel_source"] != "domain_deletion"]
            sensitivity_summaries.append(
                {
                    "subset": "non_deletion_core",
                    **asdict(_summary(objective, policy, non_deletion)),
                }
            )
        all_rows.extend(frames.values())
        outputs[objective] = str(_plot_objective(objective, frames, summaries))

    matched = pd.concat(all_rows, ignore_index=True)
    matched.to_csv(OUTPUT_DIR / "matched_swarm_outcomes.csv", index=False)
    summary_frame = pd.DataFrame(asdict(summary) for summary in all_summaries)
    summary_frame.to_csv(OUTPUT_DIR / "correlation_summary.csv", index=False)
    pd.DataFrame(sensitivity_summaries).to_csv(
        OUTPUT_DIR / "correlation_sensitivity_non_deletion.csv",
        index=False,
    )
    payload = {
        "matched_rows_per_objective_policy": EXPECTED_ROWS,
        "new_one_phase_rows": EXPECTED_NEW_ONE_PHASE_ROWS,
        "exact_checkpoint_aliases": EXPECTED_ALIAS_ROWS,
        "metrics": [asdict(summary) for summary in all_summaries],
        "non_deletion_sensitivity": sensitivity_summaries,
        "plots": outputs,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(summary_frame.to_string(index=False))
    for output in outputs.values():
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
