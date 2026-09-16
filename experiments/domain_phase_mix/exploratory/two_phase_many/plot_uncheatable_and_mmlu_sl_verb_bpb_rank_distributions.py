# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot uncheatable-eval and MMLU-SL-Verb BPB rank distributions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

UNCHEDATABLE_CSV_PATH = Path(__file__).with_name("two_phase_many.csv")
SL_VERB_CSV_PATH = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_qsplit240_mmlu_sl_verb_rerun/collect_results-ef2602/results.csv"
)
BPB_OUTPUT_PNG = Path(__file__).with_name("uncheatable_and_mmlu_sl_verb_bpb_rank_distributions.png")
BPB_OUTPUT_CSV = Path(__file__).with_name("uncheatable_and_mmlu_sl_verb_bpb_rank_distribution_highlights.csv")
BASELINE_CSV = Path(__file__).with_name("uncheatable_and_mmlu_sl_verb_rank_distribution_wandb_baselines.csv")
BASELINE_METRICS_CSV = Path(__file__).with_name("power_family_penalty_vs_baselines_ranked_metrics_common_wide.csv")

UNCHEDATABLE_METRIC = "eval/uncheatable_eval/bpb"
SL_VERB_BPB_METRIC = "lm_eval/mmlu_sl_verb_5shot/bpb"


@dataclass(frozen=True)
class BaselineSpec:
    run_name: str
    label: str
    params: int


@dataclass(frozen=True)
class PanelConfig:
    metric: str
    title: str
    csv_path: Path | str
    lower_is_better: bool
    invert_yaxis: bool = False


BASELINES: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        run_name="baseline_proportional",
        label="Proportional",
        params=0,
    ),
    BaselineSpec(
        run_name="baseline_unimax",
        label="UniMax",
        params=0,
    ),
    BaselineSpec(
        run_name="baseline_stratified",
        label="Uniform",
        params=0,
    ),
    BaselineSpec(
        run_name="baseline_olmix_loglinear_uncheatable_bpb",
        label="Olmix",
        params=79,
    ),
    BaselineSpec(
        run_name="baseline_genericfamily_power_family_penalty_raw_optimum",
        label="GRP (Power-Family Penalty)",
        params=43,
    ),
)

HIGHLIGHT_STYLES = {
    "baseline_proportional": {
        "color": "#E15759",
        "marker": "D",
    },
    "baseline_unimax": {
        "color": "#4E79A7",
        "marker": "s",
    },
    "baseline_stratified": {
        "color": "#59A14F",
        "marker": "X",
    },
    "baseline_olmix_loglinear_uncheatable_bpb": {
        "color": "#F28E2B",
        "marker": "P",
    },
    "baseline_genericfamily_power_family_penalty_raw_optimum": {
        "color": "#2E7D32",
        "marker": "o",
    },
}

PLOT_CONFIG = (
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
)


def _load_baseline_metrics() -> tuple[pd.DataFrame, dict[str, BaselineSpec]]:
    frame = pd.read_csv(BASELINE_METRICS_CSV)
    selected_rows: list[dict[str, str | float | int]] = []
    spec_by_run = {spec.run_name: spec for spec in BASELINES}
    for spec in BASELINES:
        row = frame.loc[frame["run_name"] == spec.run_name]
        if row.empty:
            raise ValueError(f"Missing baseline row for {spec.run_name!r} in {BASELINE_METRICS_CSV}")
        selected_rows.append(
            {
                "run_name": spec.run_name,
                UNCHEDATABLE_METRIC: float(row.iloc[0][UNCHEDATABLE_METRIC]),
                SL_VERB_BPB_METRIC: float(row.iloc[0][SL_VERB_BPB_METRIC]),
                "label": spec.label,
                "params": spec.params,
            }
        )
    baseline_df = pd.DataFrame(selected_rows).sort_values("run_name", ignore_index=True)
    baseline_df.to_csv(BASELINE_CSV, index=False)
    return baseline_df, spec_by_run


def _load_metric_frame(panel: PanelConfig, baseline_metrics: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(panel.csv_path)
    df = df.loc[~df["run_name"].isin(set(baseline_metrics["run_name"]))].copy()
    augmented = pd.concat([df, baseline_metrics], ignore_index=True, sort=False)
    deduped = augmented.drop_duplicates(subset=["run_name"], keep="last")
    return deduped[["run_name", panel.metric]].dropna().copy()


def _rank_metric_frame(df: pd.DataFrame, panel: PanelConfig) -> pd.DataFrame:
    ranked = df.sort_values(panel.metric, ascending=panel.lower_is_better, ignore_index=True).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def _legend_label(spec: BaselineSpec, value: float) -> str:
    return f"{spec.label} ({value:.3f})"


def _plot_pair(
    output_png: Path,
    output_csv: Path,
    panels: tuple[PanelConfig, PanelConfig],
    baseline_metrics: pd.DataFrame,
    spec_by_run: dict[str, BaselineSpec],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=180)
    fig.suptitle("Two-phase many-domain swarm: rank distributions", fontsize=18, y=0.98)

    cmap = plt.colormaps["RdYlGn_r"]
    highlight_rows: list[dict[str, str | int | float]] = []

    for ax, panel in zip(np.atleast_1d(axes), panels, strict=True):
        ranked = _rank_metric_frame(_load_metric_frame(panel, baseline_metrics), panel)
        ranks = ranked["rank"].to_numpy()
        values = ranked[panel.metric].to_numpy()

        point_colors = cmap(np.linspace(0.0, 1.0, len(ranked)))
        ax.plot(ranks, values, color="#4C78A8", linewidth=2.0, alpha=0.95, zorder=1)
        ax.scatter(ranks, values, c=point_colors, s=26, edgecolors="none", alpha=0.9, zorder=2)

        for run_name, style in HIGHLIGHT_STYLES.items():
            spec = spec_by_run[run_name]
            point = ranked.loc[ranked["run_name"] == run_name].iloc[0]
            rank = int(point["rank"])
            value = float(point[panel.metric])
            highlight_rows.append(
                {
                    "metric": panel.metric,
                    "run_name": run_name,
                    "name": spec.label,
                    "params": spec.params,
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
                label=_legend_label(spec, value),
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

    baseline_metrics, spec_by_run = _load_baseline_metrics()
    output_png, output_csv, panels = PLOT_CONFIG
    _plot_pair(output_png, output_csv, panels, baseline_metrics, spec_by_run)
    print(f"Saved plot to {output_png}")
    print(f"Saved highlights to {output_csv}")
    print(f"Saved baseline metrics to {BASELINE_CSV}")


if __name__ == "__main__":
    main()
