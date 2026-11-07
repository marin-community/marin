# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Redraws of the paper's figures on held-out runs: Figures 14, 16 and 17.

- ``fig:r3`` (Figure 14): the main swarm's 280 designs, each trained at the three proxy settings, compared between
  pairs of settings by Uncheatable BPB (top row, with the least-squares line) and by rank (bottom row, with equal
  ranks). Beside them are the Uncheatable proposals of MARINER, Olmix and both RegMix variants, learned from the Qwen3
  360M/1.6B swarm and trained at every setting, and the proportional and UniMax-8 baselines among the designs. A
  proposal's BPB at Qwen3 360M/1.6B is the mean of its runs there in ``heldout``, and at each Llama setting its run in
  ``heldout``'s ``scale_transfer`` cohort. Its rank is the position it would take among the 280 designs.
- ``fig:analysis-floor`` (Figure 16): MARINER's (top) and Olmix's (bottom) predictions against the measured loss of
  the 30 held-out mixtures that measure lowest on each objective and on three components, Arithmetic, arXiv physics
  and SciQ. Each panel gives the Spearman correlation, and a band marks one proportional-repeat standard deviation
  around equality.
- ``fig:analysis-extrapolation`` (Figure 17): MARINER's residual, measured minus predicted loss, on arXiv physics and
  SciQ at each held-out mixture, against the mixture's total variation to the nearest of the main swarm's 280 fit
  runs.

Figures 16 and 17 read each objective's frontier bank (``frontier_bank``): the held-out runs that
``fits_heldout_predictions`` predicts, grouped by exposure, without the two sources of designed interventions that
the paper's analysis of the best held-out mixtures leaves out. That leaves 170 Uncheatable and 157 OlmoBaseEval Easy
exposures. MARINER's predictions are the released per-task predictions, Olmix's the released per-task laws at each
exposure's mixture, and an objective's prediction weights its tasks by their ``aggregate_weight``. Figure 16's titles
print ``fits_reliability``'s signal-to-noise ratios, and its bands are the repeat standard deviations there, both over
the main swarm's ten proportional repeats.

Figure 14's row of the dataset's index also names ``fits``, which the redraw does not read.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from mariner.data import (
    BASELINE_PREFIX,
    CALIBRATION_RUN,
    EXPOSURE_ID,
    FIT_INCLUSION,
    HELDOUT_PREDICTIONS_TABLE,
    HELDOUT_TABLE,
    MAIN_SWARM,
    OBSERVATION_ID,
    OUTCOME_PREFIX,
    RELIABILITY_SOURCES,
    RELIABILITY_TABLE,
    RUN_LABEL,
    TASK_ID,
    TRAINED_MARINER,
    TRAINED_OLMIX,
    WEIGHT_PREFIX,
    heldout_bank,
    mixture_weights,
    objective_from_tasks,
    read_table,
    release_partition,
    release_runs,
    release_tasks,
    released_olmix_laws,
    released_task_predictions,
    reliability_rows,
    require_columns,
)
from mariner.design import UNIMAX
from mariner.figures.layout import (
    BAND_ALPHA,
    METHOD_COLORS,
    NEUTRAL,
    OBJECTIVE_TITLES,
    FigureData,
    data_limits,
    figure_legend,
    mosaic,
    panels,
)
from mariner.figures.surrogate import PROPOSAL_MIXTURES
from mariner.methods import METHOD_LABELS, Method
from mariner.model import HeldoutBank, Objective

PROPOSAL_METHOD = "proposal_method"  # heldout's column naming the design or launch a run comes from
EDGE = "#222222"  # the outline of a marker drawn over others

# ---------------------------------------------------------------------------------------------------------------
# Figure 14: transfer of the swarm's designs and the proposals across proxy settings
# ---------------------------------------------------------------------------------------------------------------

TRANSFER_OBJECTIVE = "uncheatable"
SETTING_LABELS = {
    "llama_160m_1p2b": "Llama 160M/1.2B",
    "llama_200m_6b": "Llama 200M/6B",
    "qwen3_360m_1p6b": "Qwen3 360M/1.6B",
}
# The paper's columns, as (source, target) settings.
TRANSFER_PAIRS = (
    ("llama_160m_1p2b", "qwen3_360m_1p6b"),
    ("llama_160m_1p2b", "llama_200m_6b"),
    ("qwen3_360m_1p6b", "llama_200m_6b"),
)
SCALE_TRANSFER = "scale_transfer"  # heldout's cohort of the proposals trained at the Llama settings
# The Uncheatable proposals learned from the main swarm that Figure 14 trains at every setting, by mixture id.
TRANSFER_PROPOSALS = {
    Method.MARINER: TRAINED_MARINER[TRANSFER_OBJECTIVE],
    Method.OLMIX: TRAINED_OLMIX[TRANSFER_OBJECTIVE].mixture_id,
    Method.REGMIX: PROPOSAL_MIXTURES[TRANSFER_OBJECTIVE][Method.REGMIX],
    Method.REGMIX_RELEASED: "mix_0a09804c6e21af02",  # released RegMix's recipe, rgref_u_endpoint_cap64
}
# The baselines among the designs, by run label (the swarms name UniMax-8's run without its cap).
TRANSFER_BASELINES = {Method.PROPORTIONAL: CALIBRATION_RUN, Method.UNIMAX: f"{BASELINE_PREFIX}{UNIMAX}"}
TRANSFER_MARKERS = {
    Method.MARINER: ("*", 12.0),
    Method.OLMIX: ("D", 6.0),
    Method.REGMIX: ("^", 7.0),
    Method.REGMIX_RELEASED: ("v", 7.0),
    Method.PROPORTIONAL: ("s", 6.0),
    Method.UNIMAX: ("s", 6.0),
}


def setting_designs(root: Path, setting: str) -> pd.Series:
    """The Uncheatable BPB of a swarm's fit runs at one proxy setting, by run label."""
    runs = release_runs(root, setting)
    outcome = f"{OUTCOME_PREFIX}{TRANSFER_OBJECTIVE}"
    require_columns(runs, [FIT_INCLUSION, RUN_LABEL, outcome], f"swarm_{setting}")
    fit = runs.loc[runs[FIT_INCLUSION]]
    return fit.set_index(RUN_LABEL)[outcome]


def proposal_bpb(heldout: pd.DataFrame, mixture: str, setting: str) -> tuple[float, int]:
    """A proposal's mean Uncheatable BPB at one setting and its number of runs: its runs at the main swarm's setting,
    and its one ``scale_transfer`` run at each other setting."""
    runs = heldout[(heldout["mixture_id"] == mixture) & (heldout["swarm"] == setting)]
    if setting != MAIN_SWARM:
        runs = runs[runs["cohort"] == SCALE_TRANSFER]
        if len(runs) != 1:
            raise ValueError(f"{HELDOUT_TABLE} has {len(runs)} {SCALE_TRANSFER} runs of {mixture} at {setting}, not 1")
    values = runs[f"{OUTCOME_PREFIX}{TRANSFER_OBJECTIVE}"]
    if values.empty or values.isna().any():
        raise ValueError(f"{HELDOUT_TABLE} has no Uncheatable BPB of every run of {mixture} at {setting}")
    return float(values.mean()), len(values)


def transfer_values(root: Path) -> dict[str, pd.DataFrame]:
    """The designs' BPB and ranks at each setting, the proposals' BPB and ranks, and each pair's statistics.

    The designs are the main swarm's 280 fit runs, matched across the settings by run label; the Llama 160M/1.2B
    swarm's fourth baseline is not among them. Ranks count from 1 for the lowest BPB, and tied designs share the
    lowest rank.
    """
    designs = {setting: setting_designs(root, setting) for setting in SETTING_LABELS}
    labels = designs[MAIN_SWARM].index
    bpb = pd.DataFrame({setting: values.reindex(labels) for setting, values in designs.items()})
    if bpb.isna().to_numpy().any():
        missing = bpb.columns[bpb.isna().any()].tolist()
        raise ValueError(f"the swarms {', '.join(missing)} lack the Uncheatable BPB of some of {MAIN_SWARM}'s designs")
    ranks = bpb.rank(method="min")
    heldout = read_table(root, HELDOUT_TABLE)
    require_columns(heldout, ["mixture_id", "swarm", "cohort", f"{OUTCOME_PREFIX}{TRANSFER_OBJECTIVE}"], HELDOUT_TABLE)
    transferred = set(heldout.loc[heldout["cohort"] == SCALE_TRANSFER, "mixture_id"])
    if transferred != set(TRANSFER_PROPOSALS.values()):
        raise ValueError(f"the {SCALE_TRANSFER} cohort trains {sorted(transferred)}, not Figure 14's proposals")
    proposals = []
    for method, mixture in TRANSFER_PROPOSALS.items():
        for setting in SETTING_LABELS:
            value, runs = proposal_bpb(heldout, mixture, setting)
            rank = 1 + int((bpb[setting] < value).sum())
            proposals.append({"method": method.value, "setting": setting, "bpb": value, "runs": runs, "rank": rank})
    pairs = []
    for source, target in TRANSFER_PAIRS:
        x, y = bpb[source].to_numpy(), bpb[target].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        pairs.append(
            {
                "source": source,
                "target": target,
                "pearson": float(stats.pearsonr(x, y).statistic),
                "spearman": float(stats.spearmanr(x, y).statistic),
                "slope": float(slope),
                "intercept": float(intercept),
            }
        )
    long = bpb.melt(ignore_index=False, var_name="setting", value_name="bpb").rename_axis(RUN_LABEL).reset_index()
    long["rank"] = ranks.melt(ignore_index=False)["value"].to_numpy()
    return {"designs": long, "proposals": pd.DataFrame(proposals), "pairs": pd.DataFrame(pairs)}


def transfer_marker(method: Method) -> dict[str, Any]:
    marker, size = TRANSFER_MARKERS[method]
    return {
        "marker": marker,
        "markersize": size,
        "color": METHOD_COLORS[method],
        "markeredgecolor": EDGE,
        "markeredgewidth": 0.5,
        "linestyle": "none",
    }


def draw_scale_transfer(root: Path) -> FigureData:
    """Figure 14: the designs' Uncheatable BPB and ranks between pairs of settings, with the proposals and baselines."""
    values = transfer_values(root)
    designs = values["designs"].pivot(index=RUN_LABEL, columns="setting", values=["bpb", "rank"])
    proposals = values["proposals"].set_index(["method", "setting"])
    count = len(designs)
    layout = [[f"{kind} {index}" for index in range(len(TRANSFER_PAIRS))] for kind in ("bpb", "rank")]
    figure, axes = mosaic(layout, columns=len(TRANSFER_PAIRS), rows=len(layout), legend_height=0.8)
    statistics = values["pairs"].set_index(["source", "target"])
    for index, (source, target) in enumerate(TRANSFER_PAIRS):
        pair = statistics.loc[(source, target)].astype(float)
        source_label, target_label = SETTING_LABELS[source], SETTING_LABELS[target]
        for kind in ("bpb", "rank"):
            axis = axes[f"{kind} {index}"]
            axis.plot(
                designs[(kind, source)],
                designs[(kind, target)],
                "o",
                color=NEUTRAL,
                alpha=0.4,
                markersize=3.0,
                markeredgewidth=0,
                zorder=2,
            )
            for method, label in TRANSFER_BASELINES.items():
                axis.plot(
                    designs.loc[label, (kind, source)],
                    designs.loc[label, (kind, target)],
                    zorder=4,
                    **transfer_marker(method),
                )
            for method in TRANSFER_PROPOSALS:
                axis.plot(
                    proposals.loc[(method.value, source), kind],
                    proposals.loc[(method.value, target), kind],
                    zorder=5,
                    **transfer_marker(method),
                )
            if kind == "bpb":
                drawn = pd.concat([designs[(kind, source)], proposals.xs(source, level="setting")[kind]])
                line = np.array([drawn.min(), drawn.max()])
                fitted = pair["slope"] * line + pair["intercept"]
                axis.plot(line, fitted, color=NEUTRAL, linestyle="--", linewidth=0.9)
                axis.set_title(f"{source_label} → {target_label}\nPearson r = {pair['pearson']:.2f}")
                axis.set_xlabel(f"Uncheatable BPB, {source_label}")
                axis.set_ylabel(f"Uncheatable BPB, {target_label}")
            else:
                axis.plot([1, count], [1, count], color=NEUTRAL, linestyle="--", linewidth=0.9)
                axis.set_title(f"Spearman \N{GREEK SMALL LETTER RHO} = {pair['spearman']:.2f}")
                axis.set_xlabel(f"Rank at {source_label}")
                axis.set_ylabel(f"Rank at {target_label}")
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            color=NEUTRAL,
            alpha=0.4,
            markersize=3.0,
            markeredgewidth=0,
            label=f"the swarm's {count} designs",
        ),
        *(
            Line2D([], [], label=f"{METHOD_LABELS[method]} proposal", **transfer_marker(method))
            for method in TRANSFER_PROPOSALS
        ),
        *(Line2D([], [], label=METHOD_LABELS[method], **transfer_marker(method)) for method in TRANSFER_BASELINES),
        Line2D(
            [], [], color=NEUTRAL, linestyle="--", linewidth=0.9, label="least-squares line (top), equal ranks (bottom)"
        ),
    ]
    figure_legend(figure, handles, columns=4)
    return FigureData(figure, values)


# ---------------------------------------------------------------------------------------------------------------
# Figures 16 and 17: the best held-out mixtures
# ---------------------------------------------------------------------------------------------------------------

# Sources of designed interventions rather than proposed mixtures: the conditional epoch dose-response design and an
# archived launch of the baseline mixtures. The paper's analysis of the best held-out mixtures leaves them out.
EXCLUDED_SOURCES = ("designed_conditional_epoch_dose", "archive::delphi_baseline_mixtures_issue6607_20260623")


class FrontierBank(NamedTuple):
    """One objective's held-out exposures, and MARINER's released predictions at them in the bank's order."""

    bank: HeldoutBank
    mariner: pd.DataFrame  # by exposure id and task


def frontier_bank(root: Path, objective: str, buckets: Sequence[str], outcomes: Objective) -> FrontierBank:
    """The held-out runs that ``fits_heldout_predictions`` predicts for ``objective``, without ``EXCLUDED_SOURCES``,
    as a bank over ``buckets`` that averages the outcomes of ``outcomes`` (``mariner.data.heldout_bank``), with
    MARINER's released per-task predictions, which the runs of an exposure must share.
    """
    predictions = released_task_predictions(root, HELDOUT_PREDICTIONS_TABLE, objective)
    heldout = read_table(root, HELDOUT_TABLE)
    require_columns(heldout, [OBSERVATION_ID, PROPOSAL_METHOD, EXPOSURE_ID], HELDOUT_TABLE)
    chosen = heldout[OBSERVATION_ID].isin(predictions.index) & ~heldout[PROPOSAL_METHOD].isin(EXCLUDED_SOURCES)
    runs = heldout[chosen]
    bank = heldout_bank(runs, buckets, outcomes)
    by_exposure = predictions.loc[runs[OBSERVATION_ID]].groupby(runs[EXPOSURE_ID].to_numpy(), sort=False)
    if (by_exposure.nunique() > 1).to_numpy().any():
        raise ValueError(
            f"{HELDOUT_PREDICTIONS_TABLE} predicts runs at one exposure of the {objective} bank differently"
        )
    return FrontierBank(bank, by_exposure.first().loc[list(bank.exposure_ids)])


# ---------------------------------------------------------------------------------------------------------------
# Figure 16: predictions against the measured loss of the 30 best held-out mixtures
# ---------------------------------------------------------------------------------------------------------------


class FrontierColumn(NamedTuple):
    """A column of Figure 16: an objective, or one of its tasks."""

    title: str
    objective: str
    task_id: str | None  # None: the objective itself


FRONTIER_COLUMNS = (  # the objectives, then three components in ascending SNR, as the paper orders them
    FrontierColumn(OBJECTIVE_TITLES["uncheatable"], "uncheatable", None),
    FrontierColumn(OBJECTIVE_TITLES["obe_easy"], "obe_easy", None),
    FrontierColumn("Arithmetic", "obe_easy", "basic_skills_arithmetic"),
    FrontierColumn("arXiv physics", "uncheatable", "uncheatable/arxiv_physics"),
    FrontierColumn("SciQ", "obe_easy", "sciq"),
)
FRONTIER_MODELS = (Method.MARINER, Method.OLMIX)
FRONTIER_BEST = 30  # the lowest-loss held-out mixtures of each column


def reliability_row(reliability: pd.DataFrame, column: FrontierColumn) -> pd.Series:
    """``fits_reliability``'s row of the column's objective (its task-less row) or task."""
    rows = reliability_rows(reliability, RELIABILITY_SOURCES[column.objective])
    require_columns(rows, [TASK_ID, "snr", "repeat_sd"], RELIABILITY_TABLE)
    if column.task_id is None:
        rows = rows[rows[TASK_ID].isna()]
    else:
        rows = rows[rows[TASK_ID] == column.task_id]
    if len(rows) != 1:
        raise ValueError(f"{RELIABILITY_TABLE} has {len(rows)} rows for {column.title}, not 1")
    return rows.iloc[0]


def frontier_values(root: Path) -> dict[str, pd.DataFrame]:
    """Each column's 30 best held-out exposures with both models' predictions, its SNR and repeat SD from
    ``fits_reliability``, and the Spearman correlation of each panel."""
    tasks = release_tasks(root)
    buckets = release_partition(root).bucket_ids
    reliability = read_table(root, RELIABILITY_TABLE)
    frontiers, objectives, olmix_tasks = {}, {}, {}
    for name in dict.fromkeys(column.objective for column in FRONTIER_COLUMNS):
        objectives[name] = objective_from_tasks(tasks, name)
        frontiers[name] = frontier_bank(root, name, buckets, objectives[name])
        bank = frontiers[name].bank
        olmix_tasks[name] = pd.DataFrame(
            np.column_stack([law.predict(bank.weights) for law in released_olmix_laws(root, name)]),
            index=list(bank.exposure_ids),
            columns=list(objectives[name].tasks),
        )
    points, columns, correlations = [], [], []
    for column in FRONTIER_COLUMNS:
        frontier, objective, olmix = (
            frontiers[column.objective],
            objectives[column.objective],
            olmix_tasks[column.objective],
        )
        exposures = list(frontier.bank.exposure_ids)
        if column.task_id is None:
            measured = pd.Series(frontier.bank.measured_objective, index=exposures)
            predicted = {
                Method.MARINER: frontier.mariner[list(objective.tasks)] @ objective.weights,
                Method.OLMIX: olmix @ objective.weights,
            }
        else:
            measured = frontier.bank.measured[column.task_id].set_axis(exposures)
            predicted = {Method.MARINER: frontier.mariner[column.task_id], Method.OLMIX: olmix[column.task_id]}
        measured = measured.dropna()
        best = measured.iloc[np.argsort(measured.to_numpy(), kind="stable")[:FRONTIER_BEST]]
        row = reliability_row(reliability, column)
        columns.append(
            {
                "column": column.title,
                "objective": column.objective,
                "task_id": column.task_id,
                "snr": float(row["snr"]),
                "repeat_sd": float(row["repeat_sd"]),
                "exposures": len(measured),
            }
        )
        for model, values in predicted.items():
            chosen = values.loc[best.index]
            points.append(
                pd.DataFrame(
                    {
                        "column": column.title,
                        "model": model.value,
                        EXPOSURE_ID: best.index,
                        "measured": best.to_numpy(),
                        "predicted": chosen.to_numpy(),
                    }
                )
            )
            correlation = float(stats.spearmanr(best.to_numpy(), chosen.to_numpy()).statistic)
            correlations.append({"column": column.title, "model": model.value, "spearman": correlation})
    return {
        "points": pd.concat(points, ignore_index=True),
        "columns": pd.DataFrame(columns),
        "spearman": pd.DataFrame(correlations),
    }


def draw_frontier(root: Path) -> FigureData:
    """Figure 16: each model's predictions against the measured loss of each column's 30 best held-out mixtures."""
    values = frontier_values(root)
    points = values["points"]
    columns = values["columns"].set_index("column")
    correlations = values["spearman"].set_index(["column", "model"])["spearman"]
    titles = [column.title for column in FRONTIER_COLUMNS]
    layout = [[f"{model.value} {title}" for title in titles] for model in FRONTIER_MODELS]
    figure, axes = mosaic(layout, columns=len(titles), rows=len(layout), legend_height=0.5)
    for title in titles:
        column = columns.loc[title]
        drawn = points[points["column"] == title]
        span = np.array(data_limits(np.concatenate([drawn["measured"], drawn["predicted"]])))
        for row, model in enumerate(FRONTIER_MODELS):
            axis = axes[f"{model.value} {title}"]
            shown = drawn[drawn["model"] == model.value]
            sd = column["repeat_sd"]
            axis.fill_between(span, span - sd, span + sd, color=NEUTRAL, alpha=BAND_ALPHA, linewidth=0)
            axis.plot(span, span, color=NEUTRAL, linewidth=0.8)
            axis.plot(
                shown["measured"],
                shown["predicted"],
                "o",
                color=METHOD_COLORS[model],
                markeredgecolor="white",
                markeredgewidth=0.4,
                markersize=4.5,
            )
            axis.set_xlim(*span)
            axis.set_ylim(*span)
            axis.set_aspect("equal")
            # The correlation goes in the title, where no point can lie under it.
            heading = f"Spearman {correlations[(title, model.value)]:.2f}"
            if row == 0:
                heading = f"{title}\nSNR {column['snr']:.0f}, {FRONTIER_BEST} best\n{heading}"
            axis.set_title(heading)
            if row == len(FRONTIER_MODELS) - 1:
                axis.set_xlabel("Measured loss (BPB)")
            if title == titles[0]:
                axis.set_ylabel(f"{METHOD_LABELS[model]}'s prediction (BPB)")
    handles = [
        *(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                color=METHOD_COLORS[model],
                markeredgecolor="white",
                markersize=6.0,
                label=f"{METHOD_LABELS[model]}'s prediction",
            )
            for model in FRONTIER_MODELS
        ),
        Line2D([], [], color=NEUTRAL, linewidth=0.8, label="prediction equals measurement"),
        Patch(color=NEUTRAL, alpha=BAND_ALPHA, linewidth=0, label="one proportional-repeat SD around equality"),
    ]
    figure_legend(figure, handles, columns=4)
    return FigureData(figure, values)


# ---------------------------------------------------------------------------------------------------------------
# Figure 17: MARINER's residuals against the distance from the swarm
# ---------------------------------------------------------------------------------------------------------------


class ResidualComponent(NamedTuple):
    label: str
    objective: str
    task_id: str
    color: str


EXTRAPOLATION_COMPONENTS = (
    ResidualComponent("arXiv physics", "uncheatable", "uncheatable/arxiv_physics", "#8A9099"),
    ResidualComponent("SciQ", "obe_easy", "sciq", "#D55E00"),
)
BEST_MARKED = 5  # each component's five best held-out mixtures are circled, the best starred
RESIDUAL_FLOOR = -0.3  # BPB: the axis ends here, as the paper's does, and lower residuals are drawn on it


def weight_buckets(runs: pd.DataFrame) -> list[str]:
    """The buckets of a run table's weight columns, in table order."""
    return [column.removeprefix(WEIGHT_PREFIX) for column in runs.columns if column.startswith(WEIGHT_PREFIX)]


def residual_values(root: Path) -> dict[str, pd.DataFrame]:
    """Each component's held-out exposures with a measured value: MARINER's residual, measured minus predicted, the
    total variation to the nearest of the main swarm's fit runs, and the rank of the measured loss (1 the lowest)."""
    swarm, name = release_runs(root, MAIN_SWARM), f"swarm_{MAIN_SWARM}"
    require_columns(swarm, [FIT_INCLUSION], name)
    buckets = weight_buckets(swarm)
    swarm_weights = mixture_weights(swarm.loc[swarm[FIT_INCLUSION]], buckets, name)
    rows = []
    for component in EXTRAPOLATION_COMPONENTS:
        # The bank averages the component alone, an objective of one task whose recorded value is the task's own.
        outcomes = Objective(component.task_id, (component.task_id,), np.ones(1))
        frontier = frontier_bank(root, component.objective, buckets, outcomes)
        measured = frontier.bank.measured[component.task_id].to_numpy(float)
        kept = ~np.isnan(measured)
        weights, measured = frontier.bank.weights[kept], measured[kept]
        exposures = np.asarray(frontier.bank.exposure_ids)[kept]
        distance = 0.5 * np.abs(weights[:, None, :] - swarm_weights[None, :, :]).sum(axis=2).min(axis=1)
        predicted = frontier.mariner.loc[exposures, component.task_id].to_numpy(float)
        rank = np.empty(len(measured), dtype=int)
        rank[np.argsort(measured, kind="stable")] = np.arange(1, len(measured) + 1)
        rows.append(
            pd.DataFrame(
                {
                    "component": component.label,
                    TASK_ID: component.task_id,
                    EXPOSURE_ID: exposures,
                    "distance": distance,
                    "measured": measured,
                    "predicted": predicted,
                    "residual": measured - predicted,
                    "rank": rank,
                }
            )
        )
    return {"residuals": pd.concat(rows, ignore_index=True)}


def draw_extrapolation_residuals(root: Path) -> FigureData:
    """Figure 17: MARINER's residual on arXiv physics and SciQ against each mixture's distance from the swarm."""
    values = residual_values(root)
    residuals = values["residuals"]
    figure, axes = panels(1, legend_width=1.6, legend_height=0.9)
    axis = axes[0]
    axis.axhline(0.0, color=NEUTRAL, linewidth=0.8, zorder=1)
    for component in EXTRAPOLATION_COMPONENTS:
        rows = residuals[residuals["component"] == component.label]
        shown = np.maximum(rows["residual"], RESIDUAL_FLOOR)
        inside = rows["residual"] >= RESIDUAL_FLOOR
        axis.plot(
            rows.loc[inside, "distance"],
            shown[inside],
            "o",
            color=component.color,
            alpha=0.75,
            markersize=3.0,
            markeredgewidth=0,
            zorder=2,
        )
        axis.plot(
            rows.loc[~inside, "distance"],
            shown[~inside],
            "v",
            color=component.color,
            markersize=5.0,
            markeredgewidth=0,
            zorder=2,
        )
        marked = rows["rank"].between(2, BEST_MARKED)
        axis.plot(
            rows.loc[marked, "distance"],
            shown[marked],
            "o",
            markerfacecolor="none",
            markeredgecolor=EDGE,
            markersize=7.0,
            markeredgewidth=0.8,
            zorder=3,
        )
        best = rows["rank"] == 1
        axis.plot(
            rows.loc[best, "distance"],
            shown[best],
            "*",
            color=component.color,
            markeredgecolor=EDGE,
            markersize=12.0,
            markeredgewidth=0.6,
            zorder=4,
        )
    high = float(residuals["residual"].max())
    margin = 0.05 * (high - RESIDUAL_FLOOR)
    axis.set_ylim(RESIDUAL_FLOOR - margin, high + margin)
    axis.set_xlim(0.0, data_limits(residuals["distance"])[1])
    axis.text(0.02, 0.98, "MARINER optimistic", transform=axis.transAxes, ha="left", va="top", color=NEUTRAL)
    axis.text(0.02, 0.02, "MARINER pessimistic", transform=axis.transAxes, ha="left", va="bottom", color=NEUTRAL)
    axis.set_xlabel("Total variation to the nearest swarm run")
    axis.set_ylabel("Measured \N{MINUS SIGN} predicted loss (BPB)")
    floor = f"{RESIDUAL_FLOOR:g}".replace("-", "\N{MINUS SIGN}")  # as the axis prints it
    handles = [
        *(
            Line2D([], [], linestyle="none", marker="o", color=component.color, markeredgewidth=0, label=component.label)
            for component in EXTRAPOLATION_COMPONENTS
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor="none",
            markeredgecolor=EDGE,
            label=f"each component's {BEST_MARKED} best held-out mixtures",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="*",
            color=NEUTRAL,
            markeredgecolor=EDGE,
            markersize=10.0,
            label="each component's best held-out mixture",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="v",
            color=NEUTRAL,
            label=f"residual below {floor} BPB, drawn at {floor}",
        ),
    ]
    figure_legend(figure, handles, columns=2)
    return FigureData(figure, values)
