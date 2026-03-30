# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot uncheatable-eval and MMLU-SL-Verb rank distributions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

UNCHEDATABLE_CSV_PATH = Path(__file__).with_name("two_phase_many.csv")
SL_VERB_CSV_PATH = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_qsplit240_mmlu_sl_verb_rerun/collect_results-ef2602/results.csv"
)
BPB_OUTPUT_PNG = Path(__file__).with_name("uncheatable_and_mmlu_sl_verb_bpb_rank_distributions.png")
BPB_OUTPUT_CSV = Path(__file__).with_name("uncheatable_and_mmlu_sl_verb_bpb_rank_distribution_highlights.csv")
CHOICE_OUTPUT_PNG = Path(__file__).with_name("uncheatable_and_mmlu_sl_verb_choice_logprob_norm_rank_distributions.png")
CHOICE_OUTPUT_CSV = Path(__file__).with_name(
    "uncheatable_and_mmlu_sl_verb_choice_logprob_norm_rank_distribution_highlights.csv"
)
BASELINE_CSV = Path(__file__).with_name("uncheatable_and_mmlu_sl_verb_rank_distribution_wandb_baselines.csv")

UNCHEDATABLE_METRIC = "eval/uncheatable_eval/bpb"
SL_VERB_BPB_METRIC = "lm_eval/mmlu_sl_verb_5shot/bpb"
SL_VERB_CHOICE_NORM_METRIC = "lm_eval/mmlu_sl_verb_5shot/choice_logprob_norm"
WANDB_ENTITY = "marin-community"
WANDB_PROJECT = "marin"

EXCLUDED_SOURCE_RUN_NAMES = {
    UNCHEDATABLE_METRIC: {"baseline_olmix_loglinear"},
    SL_VERB_BPB_METRIC: set(),
    SL_VERB_CHOICE_NORM_METRIC: set(),
}


@dataclass(frozen=True)
class WandbBaselineSpec:
    run_name: str
    wandb_run_id: str


@dataclass(frozen=True)
class PanelConfig:
    metric: str
    title: str
    csv_path: Path | str
    lower_is_better: bool
    invert_yaxis: bool = False


WANDB_BASELINES: tuple[WandbBaselineSpec, ...] = (
    WandbBaselineSpec(
        run_name="baseline_dsre_ceq_predicted",
        wandb_run_id="baseline_dsre_ceq_predicted-540565",
    ),
    WandbBaselineSpec(
        run_name="baseline_olmix_loglinear_uncheatable_bpb",
        wandb_run_id="baseline_olmix_loglinear_uncheatable_bpb-97ffd9",
    ),
    WandbBaselineSpec(
        run_name="baseline_genericfamily_retainedtotal_tuned_uncheatable_bpb",
        wandb_run_id="baseline_genericfamily_retainedtotal_tuned_uncheatable_bpb-d97130",
    ),
)

HIGHLIGHT_STYLES = {
    "baseline_proportional": {
        "color": "#E15759",
        "name": "Proportional",
        "marker": "D",
        "params": 0,
    },
    "baseline_unimax": {
        "color": "#4E79A7",
        "name": "UniMax",
        "marker": "s",
        "params": 0,
    },
    "baseline_dsre_ceq_predicted": {
        "color": "#7B1FA2",
        "name": "DS-RE-CEQ",
        "marker": "^",
        "params": 162,
    },
    "baseline_olmix_loglinear_uncheatable_bpb": {
        "color": "#F28E2B",
        "name": "Olmix loglinear",
        "marker": "P",
        "params": 79,
    },
    "baseline_genericfamily_retainedtotal_tuned_uncheatable_bpb": {
        "color": "#2E7D32",
        "name": "GRP",
        "marker": "o",
        "params": 37,
    },
}

PLOT_CONFIGS: tuple[tuple[Path, Path, tuple[PanelConfig, PanelConfig]], ...] = (
    (
        BPB_OUTPUT_PNG,
        BPB_OUTPUT_CSV,
        (
            PanelConfig(
                metric=UNCHEDATABLE_METRIC,
                title="Uncheatable-Eval BPB",
                csv_path=UNCHEDATABLE_CSV_PATH,
                lower_is_better=True,
            ),
            PanelConfig(
                metric=SL_VERB_BPB_METRIC,
                title="MMLU-SL-Verb 5-shot BPB",
                csv_path=SL_VERB_CSV_PATH,
                lower_is_better=True,
            ),
        ),
    ),
    (
        CHOICE_OUTPUT_PNG,
        CHOICE_OUTPUT_CSV,
        (
            PanelConfig(
                metric=UNCHEDATABLE_METRIC,
                title="Uncheatable-Eval BPB",
                csv_path=UNCHEDATABLE_CSV_PATH,
                lower_is_better=True,
            ),
            PanelConfig(
                metric=SL_VERB_CHOICE_NORM_METRIC,
                title="MMLU-SL-Verb 5-shot choice\\_logprob\\_norm",
                csv_path=SL_VERB_CSV_PATH,
                lower_is_better=False,
                invert_yaxis=False,
            ),
        ),
    ),
)


def _fetch_wandb_baselines() -> pd.DataFrame:
    api = wandb.Api()
    rows: list[dict[str, str | float]] = []
    for spec in WANDB_BASELINES:
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{spec.wandb_run_id}")
        metrics = {
            UNCHEDATABLE_METRIC: run.summary.get(UNCHEDATABLE_METRIC),
            SL_VERB_BPB_METRIC: run.summary.get(SL_VERB_BPB_METRIC),
            SL_VERB_CHOICE_NORM_METRIC: run.summary.get(SL_VERB_CHOICE_NORM_METRIC),
        }
        missing = [metric for metric, value in metrics.items() if value is None]
        if missing:
            raise ValueError(f"Missing required metrics {missing} in W&B run {spec.wandb_run_id}")
        rows.append({"run_name": spec.run_name, **{metric: float(value) for metric, value in metrics.items()}})
    baseline_df = pd.DataFrame(rows).sort_values("run_name", ignore_index=True)
    baseline_df.to_csv(BASELINE_CSV, index=False)
    return baseline_df


def _load_metric_frame(panel: PanelConfig, wandb_baselines: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(panel.csv_path)
    excluded = EXCLUDED_SOURCE_RUN_NAMES[panel.metric]
    if excluded:
        df = df.loc[~df["run_name"].isin(excluded)].copy()
    augmented = pd.concat([df, wandb_baselines], ignore_index=True, sort=False)
    deduped = augmented.drop_duplicates(subset=["run_name"], keep="last")
    return deduped[["run_name", panel.metric]].dropna().copy()


def _rank_metric_frame(df: pd.DataFrame, panel: PanelConfig) -> pd.DataFrame:
    ranked = df.sort_values(panel.metric, ascending=panel.lower_is_better, ignore_index=True).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def _legend_label(style: dict[str, str | int], rank: int) -> str:
    return f"{style['name']} ({style['params']} params, rank {rank})"


def _plot_pair(
    output_png: Path,
    output_csv: Path,
    panels: tuple[PanelConfig, PanelConfig],
    wandb_baselines: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=180)
    fig.suptitle("Two-phase many-domain swarm: rank distributions", fontsize=18, y=0.98)

    cmap = plt.colormaps["RdYlGn_r"]
    highlight_rows: list[dict[str, str | int | float]] = []

    for ax, panel in zip(np.atleast_1d(axes), panels, strict=True):
        ranked = _rank_metric_frame(_load_metric_frame(panel, wandb_baselines), panel)
        ranks = ranked["rank"].to_numpy()
        values = ranked[panel.metric].to_numpy()

        point_colors = cmap(np.linspace(0.0, 1.0, len(ranked)))
        ax.plot(ranks, values, color="#4C78A8", linewidth=2.0, alpha=0.95, zorder=1)
        ax.scatter(ranks, values, c=point_colors, s=26, edgecolors="none", alpha=0.9, zorder=2)

        for run_name, style in HIGHLIGHT_STYLES.items():
            point = ranked.loc[ranked["run_name"] == run_name].iloc[0]
            rank = int(point["rank"])
            value = float(point[panel.metric])
            highlight_rows.append(
                {
                    "metric": panel.metric,
                    "run_name": run_name,
                    "name": style["name"],
                    "params": int(style["params"]),
                    "rank": rank,
                    "value": value,
                }
            )
            ax.scatter(
                [rank],
                [value],
                marker=style["marker"],
                s=70,
                color=style["color"],
                edgecolors="black",
                linewidths=0.7,
                zorder=4,
                label=_legend_label(style, rank),
            )

        if panel.invert_yaxis:
            ax.invert_yaxis()

        ax.set_title(panel.title, fontsize=14)
        ax.set_xlabel("Rank (1 = best)")
        ax.set_ylabel(panel.metric)
        ax.set_xlim(1, len(ranked))
        ax.legend(loc="lower right", fontsize=10, frameon=True)

    highlight_df = pd.DataFrame(highlight_rows).sort_values(["metric", "rank", "name"])
    highlight_df.to_csv(output_csv, index=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["text.latex.preamble"] = r"\usepackage{DejaVuSans}\renewcommand{\familydefault}{\sfdefault}"

    wandb_baselines = _fetch_wandb_baselines()
    for output_png, output_csv, panels in PLOT_CONFIGS:
        _plot_pair(output_png, output_csv, panels, wandb_baselines)
        print(f"Saved plot to {output_png}")
        print(f"Saved highlights to {output_csv}")
    print(f"Saved W&B baseline metrics to {BASELINE_CSV}")


if __name__ == "__main__":
    main()
