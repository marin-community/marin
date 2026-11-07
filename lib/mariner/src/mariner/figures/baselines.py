# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Redraws of the paper's figures that compare MARINER with its ablations and baselines (Appendix B.2): Figures 18
(``fig:a-learning-ablations``), 19 (``fig:a-learning-baselines``) and 20 (``fig:a-comparator-paths``).

- Figures 18 and 19 follow MARINER and its three ablations (18), and MARINER and the six baselines (19), through the
  swarm sizes of Figure 6, from the out-of-fold learning-curve studies of ``fits_learning_curves``: each model's
  selection regret relative to a random pick (top row) and the Spearman correlation of its pooled out-of-fold
  predictions (bottom row) on each objective, with 95% t-intervals over the ten subset and fold draws
  (``surrogate.learning_curve_summary``).
- Figure 20 follows the blends w(s) = (1 - s) w_MARINER + s w_baseline of MARINER's trained proposal with each of five
  baselines' trained proposals: MARINER's and the baseline's own predictions along the blend
  (``fits_path_predictions``), the three trainer seeds of each proposal at s = 0 and 1, and the seed-0 run of the
  trained blend at s = 0.5 (``validation_3e18``).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from mariner.data import OBJECTIVES, OUTCOME_PREFIX, read_table, require_columns
from mariner.figures.layout import (
    BAND_ALPHA,
    LOSS_LABELS,
    METHOD_COLORS,
    METHOD_LABELS,
    NEUTRAL,
    OBJECTIVE_TITLES,
    FigureData,
    Method,
    data_limits,
    figure_legend,
    mosaic,
)
from mariner.figures.surrogate import (
    RATIO_AXIS_MARGIN,
    RATIO_AXIS_TOP,
    SWARM_SIZE_LABEL,
    draw_regret_panel,
    learning_curve_handles,
    learning_curve_summary,
    proposal_runs,
    swarm_size_ticks,
)

# ---------------------------------------------------------------------------------------------------------------
# Figures 18 and 19: learning curves of the ablations and the baselines
# ---------------------------------------------------------------------------------------------------------------

ABLATION_MODELS = (Method.MARINER, Method.MARINER_CONVEX, Method.MARINER_ADDITIVE, Method.MARINER_BUCKET_THRESHOLDS)
BASELINE_MODELS = (
    Method.MARINER,
    Method.OLMIX,
    Method.REGMIX,
    Method.REGMIX_RELEASED,
    Method.QUADRATIC,
    Method.SPLINE,
    Method.KERNEL_RIDGE,
)


def draw_learning_pair(root: Path, models: Sequence[Method]) -> FigureData:
    """Selection regret relative to a random pick (top) and the pooled out-of-fold Spearman correlation (bottom)
    against swarm size, for ``models`` on each objective; a constant fit has no correlation."""
    summary = learning_curve_summary(root, models)
    layout = [[f"regret {objective}" for objective in OBJECTIVES], [f"spearman {objective}" for objective in OBJECTIVES]]
    figure, axes = mosaic(layout, columns=len(OBJECTIVES), rows=len(layout), legend_height=1.1)
    ranked = summary[~summary["constant_fit"]]
    for objective in OBJECTIVES:
        regret, spearman = axes[f"regret {objective}"], axes[f"spearman {objective}"]
        draw_regret_panel(regret, summary, objective, models)
        regret.set_ylim(summary["ratio"].min() / RATIO_AXIS_MARGIN, RATIO_AXIS_TOP)
        regret.set_title(OBJECTIVE_TITLES[objective])
        for model in models:
            rows = ranked[(ranked["objective"] == objective) & (ranked["model"] == model)].sort_values("mixtures")
            color = METHOD_COLORS[model]
            spearman.fill_between(
                rows["mixtures"], rows["spearman_low"], rows["spearman_high"], color=color, alpha=BAND_ALPHA, linewidth=0
            )
            spearman.plot(rows["mixtures"], rows["spearman"], color=color, marker="o", markerfacecolor="white")
        swarm_size_ticks(spearman, summary)
        spearman.set_ylim(*data_limits(ranked[["spearman_low", "spearman_high"]].to_numpy()))
        spearman.set_xlabel(SWARM_SIZE_LABEL)
    axes[f"regret {OBJECTIVES[0]}"].set_ylabel("Selection regret / random pick's")
    axes[f"spearman {OBJECTIVES[0]}"].set_ylabel("Out-of-fold Spearman correlation")
    figure_legend(figure, learning_curve_handles(summary, models), columns=3)
    return FigureData(figure, {"learning_curves": summary})


def draw_learning_ablations(root: Path) -> FigureData:
    """Figure 18: the learning curves of MARINER and its three ablations."""
    return draw_learning_pair(root, ABLATION_MODELS)


def draw_learning_baselines(root: Path) -> FigureData:
    """Figure 19: the learning curves of MARINER and the six baselines."""
    return draw_learning_pair(root, BASELINE_MODELS)


# ---------------------------------------------------------------------------------------------------------------
# Figure 20: predictions along blends of MARINER's proposal with each baseline's
# ---------------------------------------------------------------------------------------------------------------

PATHS_TABLE = "fits_path_predictions"
VALIDATION_TABLE = "validation_3e18"
MIXTURE_COLUMNS = ["start_mixture_id", "end_mixture_id", "midpoint_mixture_id"]
CURVE_COLUMNS = ["objective", "path", "predictor", "position", "predicted_bpb"]
PATH_BASELINES = (Method.OLMIX, Method.QUADRATIC, Method.SPLINE, Method.REGMIX, Method.KERNEL_RIDGE)  # the columns
MIDPOINT_LAUNCH = "path_midpoints"  # the validation launch that trained each path's blend at s = 0.5
MIDPOINT = 0.5
ENDPOINT_STYLE = {"marker": "o", "markersize": 4.5, "markerfacecolor": "white", "markeredgewidth": 1.0}
MIDPOINT_STYLE = {"marker": "D", "markersize": 5.5, "markeredgecolor": "white"}


def path_values(root: Path) -> dict[str, pd.DataFrame]:
    """Each path's predictions and mixtures, the measured runs of its two proposals at s = 0 and 1, and the run of its
    trained midpoint at s = 0.5.

    Raises ``ValueError`` unless every path names one start, end and midpoint mixture and holds the predictions of
    MARINER and of its own baseline, and unless its midpoint has one run in the midpoint launch.
    """
    predictions = read_table(root, PATHS_TABLE)
    require_columns(predictions, [*CURVE_COLUMNS, *MIXTURE_COLUMNS], PATHS_TABLE)
    validation = read_table(root, VALIDATION_TABLE)
    outcomes = [f"{OUTCOME_PREFIX}{objective}" for objective in OBJECTIVES]
    require_columns(validation, ["mixture_id", "launch", "trainer_seed", *outcomes], VALIDATION_TABLE)
    paths, endpoints, midpoints = [], [], []
    for (objective, path), rows in predictions.groupby(["objective", "path"], sort=False):
        mixtures = rows[MIXTURE_COLUMNS].drop_duplicates()
        if len(mixtures) != 1:
            raise ValueError(f"{PATHS_TABLE} names {len(mixtures)} sets of mixtures for the {objective} path to {path}")
        if set(rows["predictor"]) != {Method.MARINER.value, path}:
            raise ValueError(
                f"{PATHS_TABLE}'s {objective} path to {path} has predictors {sorted(set(rows['predictor']))}"
            )
        start, end, midpoint = mixtures.iloc[0]
        paths.append({"objective": objective, "path": path, **mixtures.iloc[0].to_dict()})
        measured = f"{OUTCOME_PREFIX}{objective}"
        for position, mixture in ((0.0, start), (1.0, end)):
            runs = proposal_runs(validation, mixture)
            endpoints += [
                {
                    "objective": objective,
                    "path": path,
                    "position": position,
                    "mixture_id": mixture,
                    "trainer_seed": seed,
                    "bpb": bpb,
                }
                for seed, bpb in zip(runs["trainer_seed"], runs[measured], strict=True)
            ]
        run = validation[(validation["mixture_id"] == midpoint) & (validation["launch"] == MIDPOINT_LAUNCH)]
        if len(run) != 1:
            raise ValueError(f"{MIDPOINT_LAUNCH} holds {len(run)} runs of {midpoint}, not one")
        midpoints.append(
            {
                "objective": objective,
                "path": path,
                "position": MIDPOINT,
                "mixture_id": midpoint,
                "trainer_seed": run["trainer_seed"].iloc[0],
                "bpb": run[measured].iloc[0],
            }
        )
    return {
        "curves": (
            predictions[CURVE_COLUMNS].sort_values(["objective", "path", "predictor", "position"], ignore_index=True)
        ),
        "paths": pd.DataFrame(paths),
        "endpoints": pd.DataFrame(endpoints),
        "midpoints": pd.DataFrame(midpoints),
    }


def draw_comparator_paths(root: Path) -> FigureData:
    """Figure 20: MARINER's and each baseline's own predictions along the blends of their proposals, with the measured
    loss of both proposals and of the trained midpoint; each row shares its loss axis."""
    values = path_values(root)
    curves, endpoints, midpoints = values["curves"], values["endpoints"], values["midpoints"]
    layout = [[f"{objective} {baseline.value}" for baseline in PATH_BASELINES] for objective in OBJECTIVES]
    figure, axes = mosaic(layout, columns=len(PATH_BASELINES), rows=len(OBJECTIVES), legend_height=0.5)
    for objective in OBJECTIVES:
        shown = np.concatenate(
            [
                curves.loc[curves["objective"] == objective, "predicted_bpb"],
                endpoints.loc[endpoints["objective"] == objective, "bpb"],
                midpoints.loc[midpoints["objective"] == objective, "bpb"],
            ]
        )
        for column, baseline in enumerate(PATH_BASELINES):
            axis = axes[f"{objective} {baseline.value}"]
            path = (curves["objective"] == objective) & (curves["path"] == baseline.value)
            for predictor, linestyle in ((Method.MARINER, "-"), (baseline, "--")):
                curve = curves[path & (curves["predictor"] == predictor.value)]
                axis.plot(curve["position"], curve["predicted_bpb"], color=METHOD_COLORS[predictor], linestyle=linestyle)
            ends = endpoints[(endpoints["objective"] == objective) & (endpoints["path"] == baseline.value)]
            axis.plot(ends["position"], ends["bpb"], color=NEUTRAL, linestyle="none", zorder=4, **ENDPOINT_STYLE)
            middle = midpoints[(midpoints["objective"] == objective) & (midpoints["path"] == baseline.value)]
            axis.plot(middle["position"], middle["bpb"], color=NEUTRAL, linestyle="none", zorder=4, **MIDPOINT_STYLE)
            axis.set_ylim(*data_limits(shown))
            axis.set_xticks([0.0, MIDPOINT, 1.0])
            if objective == OBJECTIVES[0]:
                axis.set_title(f"MARINER to {METHOD_LABELS[baseline]}")
            if objective == OBJECTIVES[-1]:
                axis.set_xlabel("Blend position s")
            if column == 0:
                axis.set_ylabel(LOSS_LABELS[objective])
    handles = [
        Line2D([], [], color=METHOD_COLORS[Method.MARINER], label="MARINER's prediction"),
        Line2D([], [], color=NEUTRAL, linestyle="--", label="the baseline's own prediction (its color)"),
        Line2D(
            [],
            [],
            color=NEUTRAL,
            linestyle="none",
            label="measured proposals: s = 0, MARINER's; s = 1, the baseline's (3 seeds)",
            **ENDPOINT_STYLE,
        ),
        Line2D([], [], color=NEUTRAL, linestyle="none", label="measured midpoint (seed 0)", **MIDPOINT_STYLE),
    ]
    figure_legend(figure, handles, columns=4)
    return FigureData(figure, values)
