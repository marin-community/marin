# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Redraws of the paper's figures on the surrogate: Figures 4 to 7.

- ``fig:b6`` (Figure 4): MARINER's and Olmix's in-sample fits to the fixed-model StarCoder sweeps of Figure 3B (1.0
  to 7.4B tokens), MARINER's fitted minima, and Olmix's picks, the swept mixture where its law is lowest, at their
  measured loss. Both models are fitted with this package.
- ``fig:mariner-terms`` (Figure 5): ``mariner.model``'s benefit and harm at the parameter values the paper's panels
  print, and one bucket's predicted loss.
- ``fig:r6`` (Figure 6): each surrogate's selection regret relative to a random pick against swarm size, from the
  out-of-fold learning-curve study in ``fits_learning_curves``.
- ``fig:proposal-predictions`` (Figure 7): each model's trained proposal, its measured loss over three trainer seeds,
  and all four models' predictions for it, from ``fits_proposal_predictions``.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from mariner.data import (
    OBJECTIVES,
    OBSERVATION_ID,
    OUTCOME_PREFIX,
    TRAINED_MARINER,
    TRAINED_OLMIX,
    VALIDATION_TABLE,
    read_table,
    require_columns,
)
from mariner.figures.layout import (
    BAND_ALPHA,
    CONFIDENCE,
    METHOD_COLORS,
    METHOD_LABELS,
    NEUTRAL,
    OBJECTIVE_TITLES,
    FigureData,
    Method,
    figure_legend,
    panel_legend,
    panels,
    scale_colors,
    t_interval,
)
from mariner.figures.starcoder import (
    FIXED_PARAMETERS_CURVES,
    SHARE_AXIS_LIMITS,
    atlas_sweeps,
    fitted_sweep,
    tokens_label,
)
from mariner.model import Head, Shape, benefit, design_matrix, harm
from mariner.olmix import fit_log_linear

# ---------------------------------------------------------------------------------------------------------------
# Figure 4: monotone Olmix heads against MARINER on the fixed-model sweeps
# ---------------------------------------------------------------------------------------------------------------


def draw_olmix_capacity_failure(root: Path) -> FigureData:
    """Figure 4: MARINER's and Olmix's fits to the 1.0 to 7.4B-token sweeps of a 210M model, and their picks."""
    sweeps = atlas_sweeps(root, FIXED_PARAMETERS_CURVES)
    figure, axes = panels(1, legend_width=1.6, legend_height=1.0)
    axis = axes[0]
    observations, mariner_rows, olmix_rows, minima, picks = [], [], [], [], []
    handles = []
    for sweep, color in zip(sweeps.values(), scale_colors(len(sweeps)), strict=True):
        weights = np.column_stack([1.0 - sweep.shares, sweep.shares])
        curve = fitted_sweep(sweep)
        optimum, dense = curve.optimum, curve.shares
        law = fit_log_linear(weights, sweep.bpb)
        pick = int(np.argmin(law.predict(weights)))
        olmix_curve = law.predict(np.column_stack([1.0 - dense, dense]))
        axis.plot(dense, curve.predicted_bpb, color=color)
        axis.plot(dense, olmix_curve, color=METHOD_COLORS[Method.OLMIX], linestyle="--", linewidth=1.0)
        axis.scatter(sweep.shares, sweep.bpb, s=12, facecolor="white", edgecolor=color, linewidth=0.8, zorder=4)
        axis.scatter([optimum.share], [optimum.bpb], s=70, marker="*", color=color, edgecolor=NEUTRAL, zorder=5)
        axis.scatter(
            [sweep.shares[pick]],
            [sweep.bpb[pick]],
            s=28,
            marker="D",
            color=METHOD_COLORS[Method.OLMIX],
            edgecolor=NEUTRAL,
            linewidth=0.5,
            zorder=6,
            clip_on=False,
        )
        epochs = curve.optimum_epochs
        label = f"{tokens_label(sweep.train_tokens)} tokens: MARINER's minimum at {epochs:.1f} epochs"
        handles.append(Line2D([], [], color=color, marker="o", markerfacecolor="white", label=label))
        observations += [
            {"curve": sweep.name, "train_tokens": sweep.train_tokens, "share": share, "bpb": bpb}
            for share, bpb in zip(sweep.shares, sweep.bpb, strict=True)
        ]
        mariner_rows += [
            {"curve": sweep.name, "share": share, "predicted_bpb": value}
            for share, value in zip(dense, curve.predicted_bpb, strict=True)
        ]
        olmix_rows += [
            {"curve": sweep.name, "share": share, "predicted_bpb": value}
            for share, value in zip(dense, olmix_curve, strict=True)
        ]
        minima.append({"curve": sweep.name, "share": optimum.share, "epochs": epochs, "bpb": optimum.bpb})
        picks.append({"curve": sweep.name, "share": sweep.shares[pick], "measured_bpb": sweep.bpb[pick]})
    axis.set_xlim(*SHARE_AXIS_LIMITS)
    axis.set_xlabel("StarCoder share of the mixture, p")
    axis.set_ylabel("Programming-languages loss (BPB)")
    axis.set_title("Fits to the fixed-model sweeps (210M parameters)")
    figure_legend(
        figure,
        [
            *handles,
            Line2D([], [], color=NEUTRAL, label="MARINER fit"),
            Line2D([], [], color=METHOD_COLORS[Method.OLMIX], linestyle="--", label="Olmix fit"),
            Line2D([], [], color=NEUTRAL, marker="*", linestyle="none", markersize=8, label="MARINER's fitted minimum"),
            Line2D(
                [],
                [],
                color=METHOD_COLORS[Method.OLMIX],
                marker="D",
                linestyle="none",
                label="Olmix's pick, at its measured loss",
            ),
        ],
        columns=2,
    )
    values = {
        "observations": pd.DataFrame(observations),
        "mariner_fits": pd.DataFrame(mariner_rows),
        "olmix_fits": pd.DataFrame(olmix_rows),
        "mariner_minima": pd.DataFrame(minima),
        "olmix_picks": pd.DataFrame(picks),
    }
    return FigureData(figure, values)


# ---------------------------------------------------------------------------------------------------------------
# Figure 5: the benefit and harm terms and one bucket's predicted loss
# ---------------------------------------------------------------------------------------------------------------

# The parameter values that the paper's panels print. Left: three benefit shapes (rho, kappa) that the fits select.
BENEFIT_SHAPES = ((2.0, 1.0), (0.5, 1.0), (0.5, 0.5))
HARM_THRESHOLDS = (1.0, 2.0, 3.0)  # middle: tau
# Right: one bucket with the second benefit shape and the second threshold, and the panel's amplitudes alpha 1 and
# beta 0.445, floor 0.9 and intercept log 0.5.
LOSS_SHAPE = Shape(rho=0.5, kappa=1.0, tau=2.0)
LOSS_HEAD = Head(intercept=math.log(0.5), coefficients=np.array([1.0, 0.445]), phi=0.9)
LOG_EPOCHS = np.geomspace(0.045, 72.0, 1_500)
LINEAR_EPOCHS = np.linspace(1.0, 16.0, 1_500)
BENEFIT_COLOR, HARM_COLOR = "#4C78A8", "#B06056"
SHADES = (1.0, 0.68, 0.4)  # the three curves of a panel, darkest first
BENEFIT_STYLES = ("-", "-", "--")


def draw_mariner_terms(root: Path) -> FigureData:
    """Figure 5: the benefit and harm terms against a bucket's epochs, and one bucket's predicted loss.

    It reads no table: every curve is ``mariner.model``'s at the paper's parameter values.
    """
    del root
    figure, axes = panels(3, legend_height=1.0)
    benefit_rows, harm_rows, activations = [], [], []
    for (rho, kappa), shade, linestyle in zip(BENEFIT_SHAPES, SHADES, BENEFIT_STYLES, strict=True):
        curve = benefit(LOG_EPOCHS, rho, kappa)
        axes[0].plot(
            LOG_EPOCHS,
            curve,
            color=BENEFIT_COLOR,
            alpha=shade,
            linestyle=linestyle,
            label=rf"$\rho$ = {rho:g}, $\kappa$ = {kappa:g}",
        )
        benefit_rows += [
            {"rho": rho, "kappa": kappa, "epochs": e, "benefit": b} for e, b in zip(LOG_EPOCHS, curve, strict=True)
        ]
    for tau, shade in zip(HARM_THRESHOLDS, SHADES, strict=True):
        curve = harm(LOG_EPOCHS, tau)
        onset = math.exp(tau) - 1.0
        at_onset = float(harm(np.array([onset]), tau)[0])
        axes[1].plot(
            LOG_EPOCHS, curve, color=HARM_COLOR, alpha=shade, label=rf"$\tau$ = {tau:g}, activates at {onset:.2g} epochs"
        )
        axes[1].plot([onset], [at_onset], "o", color=HARM_COLOR, alpha=shade, markeredgecolor="white", zorder=5)
        harm_rows += [{"tau": tau, "epochs": e, "harm": h} for e, h in zip(LOG_EPOCHS, curve, strict=True)]
        activations.append({"tau": tau, "epochs": onset, "harm": at_onset})
    for axis, title, label in ((axes[0], "Benefit: Weibull CDF", "b(E)"), (axes[1], "Harm: squared softplus", "h(E)")):
        axis.set_xscale("log")
        axis.set_xlabel("Epochs E (log scale)")
        axis.set_ylabel(f"{label} (unitless)")
        axis.set_title(title)
        panel_legend(axis)
    axes[1].set_ylim(0.0, 6.0)
    alpha, beta = LOSS_HEAD.coefficients
    matrix = design_matrix(LINEAR_EPOCHS[:, None], LOSS_SHAPE)
    loss = LOSS_HEAD.predict(matrix)
    benefit_only = dataclasses.replace(LOSS_HEAD, coefficients=np.array([alpha, 0.0])).predict(matrix)
    lowest = int(np.argmin(loss))
    axes[2].fill_between(LINEAR_EPOCHS, benefit_only, loss, color=HARM_COLOR, alpha=0.22, linewidth=0, label="harm")
    axes[2].plot(LINEAR_EPOCHS, benefit_only, color=BENEFIT_COLOR, linestyle="--", label="benefit only")
    axes[2].plot(LINEAR_EPOCHS, loss, color=METHOD_COLORS[Method.MARINER], linewidth=1.6, label="predicted loss")
    axes[2].plot(
        LINEAR_EPOCHS[lowest],
        loss[lowest],
        "*",
        color=METHOD_COLORS[Method.MARINER],
        markersize=9,
        markeredgecolor="white",
        label=f"minimum at {LINEAR_EPOCHS[lowest]:.1f} epochs",
    )
    panel_legend(
        axes[2],
        title=(
            rf"$\rho$ {LOSS_SHAPE.rho:g}, $\kappa$ {LOSS_SHAPE.kappa:g}, $\tau$ {LOSS_SHAPE.tau:g}, $\alpha$ {alpha:g},"
            "\n"
            rf"$\beta$ {beta:g}, $\phi$ {LOSS_HEAD.phi:g}, c = log {math.exp(LOSS_HEAD.intercept):g}"
        ),
    )
    axes[2].set_xlabel("Epochs E")
    axes[2].set_ylabel("Predicted loss (BPB)")
    axes[2].set_title("One bucket's predicted loss")
    values = {
        "benefit": pd.DataFrame(benefit_rows),
        "harm": pd.DataFrame(harm_rows),
        "activations": pd.DataFrame(activations),
        "predicted_loss": pd.DataFrame({"epochs": LINEAR_EPOCHS, "bpb": loss, "benefit_only_bpb": benefit_only}),
        "minimum": pd.DataFrame({"epochs": [LINEAR_EPOCHS[lowest]], "bpb": [loss[lowest]]}),
    }
    return FigureData(figure, values)


# ---------------------------------------------------------------------------------------------------------------
# Figure 6: selection regret against swarm size
# ---------------------------------------------------------------------------------------------------------------

LEARNING_CURVES_TABLE = "fits_learning_curves"
# The studies of fits_learning_curves, which fit their models on the same subsets and folds of the main swarm. MARINER,
# which all three fit, has the same rows in each; a model's rows come from the first study here that fits it.
LEARNING_CURVE_STUDIES = ("mariner_regmix_oof", "pbt", "extra")
LEARNING_CURVE_MODELS = (Method.MARINER, Method.OLMIX, Method.REGMIX, Method.REGMIX_RELEASED, Method.SPLINE)
OUT_OF_FOLD = "oof"
# Strata: each fold, the folds' mean in a draw, and the Spearman correlation over the folds' pooled predictions.
FOLD, FOLD_MEAN, POOLED = "fold", "fold_mean", "pooled"
CURVE_KEYS = ["target", "model", "k", "draw", "stratum", "fold"]  # one row of a study
CURVE_METRICS = ["regret_at_1", "random_regret_at_1", "spearman"]
REGRET_COLUMNS = ["objective", "model", "swarm_size", "mixtures", "ratio", "low", "high", "constant_fit", "draws"]
BAND_FLOOR = 1e-3  # the log axis cannot reach a band's lower end at or below zero
RANDOM_COLOR = "#7F7F7F"
RATIO_AXIS_MARGIN = 2.5  # the regret axis starts this factor below the lowest mean ratio
RATIO_AXIS_TOP = 1.6
TICKED_SWARM_SIZES = (20, 40, 80, 160)  # the x axis labels these sizes, doubling from the smallest, and the largest
SWARM_SIZE_LABEL = "Distinct mixtures in the swarm (including the anchor)"


def learning_curve_summary(root: Path, models: Sequence[Method]) -> pd.DataFrame:
    """Each model's selection regret relative to a random pick and its out-of-fold Spearman correlation, by objective
    and swarm size.

    Both are means over the subset and fold draws with a 95% t-interval: the folds' mean regret over a random pick's,
    and the Spearman correlation of the five folds' pooled predictions. A model whose fit is constant in every fold of
    every draw at a size picks arbitrarily, so its ratio there is the random pick's, 1, and its correlation is undefined
    (NaN). Raises ``ValueError`` when two studies disagree on a model's metrics.
    """
    curves = read_table(root, LEARNING_CURVES_TABLE)
    require_columns(curves, ["study", "evaluation", *CURVE_KEYS, *CURVE_METRICS], LEARNING_CURVES_TABLE)
    selected = curves[
        curves["study"].isin(LEARNING_CURVE_STUDIES)
        & (curves["evaluation"] == OUT_OF_FOLD)
        & curves["model"].isin([model.value for model in models])
    ]
    priority = selected["study"].map({study: rank for rank, study in enumerate(LEARNING_CURVE_STUDIES)})
    selected = selected.iloc[np.argsort(priority.to_numpy(), kind="stable")]
    selected = selected.drop_duplicates([*CURVE_KEYS, *CURVE_METRICS])
    disagreeing = selected.loc[selected.duplicated(CURVE_KEYS), "model"].unique()
    if len(disagreeing):
        raise ValueError(f"the studies of {LEARNING_CURVES_TABLE} disagree on {', '.join(map(str, disagreeing))}")
    keys = ["target", "model", "k"]
    folds = selected[selected["stratum"] == FOLD]
    constant = folds["spearman"].isna().groupby([folds[key] for key in keys]).all()
    pooled = {
        cell: rows["spearman"].to_numpy(float) for cell, rows in selected[selected["stratum"] == POOLED].groupby(keys)
    }
    rows = []
    for (objective, model, size), draws in selected[selected["stratum"] == FOLD_MEAN].groupby(keys):
        ratio = (draws["regret_at_1"] / draws["random_regret_at_1"]).to_numpy(float)
        constant_fit = bool(constant.loc[(objective, model, size)])
        if constant_fit:
            ratio = np.ones_like(ratio)
        center, half = t_interval(ratio)
        if constant_fit:
            spearman, spread = math.nan, math.nan
        elif (objective, model, size) in pooled:
            spearman, spread = t_interval(pooled[(objective, model, size)])
        else:
            raise ValueError(f"{LEARNING_CURVES_TABLE} has no pooled rows of {model} on {objective} at {size} runs")
        rows.append(
            {
                "objective": objective,
                "model": model,
                "swarm_size": int(size),
                "mixtures": int(size) + 1,  # the size counts the runs besides the proportional anchor
                "ratio": center,
                "low": max(center - half, BAND_FLOOR),
                "high": center + half,
                "constant_fit": constant_fit,
                "draws": len(ratio),
                "spearman": spearman,
                "spearman_low": spearman - spread,
                "spearman_high": spearman + spread,
            }
        )
    return pd.DataFrame(rows)


def swarm_size_ticks(axis: Axes, summary: pd.DataFrame) -> None:
    """A log axis of distinct mixtures that labels ``TICKED_SWARM_SIZES`` and the largest swarm."""
    axis.set_xscale("log")
    ticks = [size + 1 for size in (*TICKED_SWARM_SIZES, int(summary["swarm_size"].max()))]
    axis.set_xticks(ticks, [str(tick) for tick in ticks])
    axis.minorticks_off()


def draw_regret_panel(axis: Axes, summary: pd.DataFrame, objective: str, models: Sequence[Method]) -> None:
    """One objective's regret ratios against swarm size: each model's mean over the draws with its interval as a band,
    and its constant fits as hollow diamonds joined to its first ranked size; a random pick's ratio is 1."""
    axis.axhline(1.0, color=RANDOM_COLOR, linestyle="--", linewidth=0.9)
    for model in models:
        rows = summary[(summary["objective"] == objective) & (summary["model"] == model)].sort_values("mixtures")
        ranked, constant = rows[~rows["constant_fit"]], rows[rows["constant_fit"]]
        color = METHOD_COLORS[model]
        axis.fill_between(ranked["mixtures"], ranked["low"], ranked["high"], color=color, alpha=BAND_ALPHA, linewidth=0)
        axis.plot(ranked["mixtures"], ranked["ratio"], color=color, marker="o", markerfacecolor="white")
        if len(constant):
            joined = pd.concat([constant, ranked.iloc[:1]])
            axis.plot(joined["mixtures"], joined["ratio"], color=color, linestyle=":")
            axis.plot(
                constant["mixtures"], constant["ratio"], "D", color=color, markerfacecolor="white", linestyle="none"
            )
    axis.set_yscale("log")
    swarm_size_ticks(axis, summary)


def learning_curve_handles(summary: pd.DataFrame, models: Sequence[Method]) -> list[Artist]:
    """The legend of the learning-curve panels: the models, the bands, a random pick and any constant fits."""
    handles: list[Artist] = [
        Line2D([], [], color=METHOD_COLORS[model], marker="o", markerfacecolor="white", label=METHOD_LABELS[model])
        for model in models
    ]
    handles += [
        Patch(color=NEUTRAL, alpha=BAND_ALPHA, linewidth=0, label=f"{CONFIDENCE:.0%} t-interval over the draws"),
        Line2D([], [], color=RANDOM_COLOR, linestyle="--", label="random pick"),
    ]
    handles += [
        Line2D(
            [],
            [],
            color=METHOD_COLORS[Method(model)],
            marker="D",
            markerfacecolor="white",
            linestyle=":",
            label="constant fit (a random pick)",
        )
        for model in summary.loc[summary["constant_fit"], "model"].unique()
    ]
    return handles


def draw_learning_curves(root: Path) -> FigureData:
    """Figure 6: selection regret relative to a random pick against swarm size, for each objective."""
    ratios = learning_curve_summary(root, LEARNING_CURVE_MODELS)[REGRET_COLUMNS]
    figure, axes = panels(len(OBJECTIVES), legend_height=0.6, sharey=True)
    for axis, objective in zip(axes, OBJECTIVES, strict=True):
        draw_regret_panel(axis, ratios, objective, LEARNING_CURVE_MODELS)
        axis.set_xlabel(SWARM_SIZE_LABEL)
        axis.set_title(OBJECTIVE_TITLES[objective])
    axes[0].set_ylabel("Selection regret / random pick's")
    # From below the lowest mean, as the paper's axis starts; a band whose interval reaches zero runs off the bottom.
    axes[0].set_ylim(ratios["ratio"].min() / RATIO_AXIS_MARGIN, RATIO_AXIS_TOP)
    figure_legend(figure, learning_curve_handles(ratios, LEARNING_CURVE_MODELS), columns=4)
    return FigureData(figure, {"regret_ratios": ratios})


# ---------------------------------------------------------------------------------------------------------------
# Figure 7: each model's proposal, its measured loss and every model's prediction
# ---------------------------------------------------------------------------------------------------------------

PROPOSAL_MODELS = (Method.MARINER, Method.OLMIX, Method.SPLINE, Method.REGMIX)
# The trained proposal of each model, by mixture id: MARINER's and Olmix's as mariner.data names them, and the
# spline's and tuned RegMix's comparator proposals.
PROPOSAL_MIXTURES = {
    "uncheatable": {
        Method.MARINER: TRAINED_MARINER["uncheatable"],
        Method.OLMIX: TRAINED_OLMIX["uncheatable"].mixture_id,
        Method.SPLINE: "mix_60ff53ad7686c66a",
        Method.REGMIX: "mix_2ffc67cf6a7d478e",
    },
    "obe_easy": {
        Method.MARINER: TRAINED_MARINER["obe_easy"],
        Method.OLMIX: TRAINED_OLMIX["obe_easy"].mixture_id,
        Method.SPLINE: "mix_5031e9bf4ca27afa",
        Method.REGMIX: "mix_40e3c45f36f0c460",
    },
}
# The validation launches that trained the proposals' three trainer seeds. The Olmix cap and KL sweep retrained
# Olmix's matched proposal once more; the figure leaves that run out.
PROPOSAL_LAUNCHES = ("frozen_procedure", "fairness_repeats", "matched_olmix", "comparator_proposals", "table5_repeats")
TRAINER_SEEDS = (0, 1, 2)
NORMAL_QUANTILE_95 = 1.96  # the figure's interval: the mean plus or minus 1.96 standard errors
BAR_HALF_WIDTH = 0.32
MARKERS = {Method.MARINER: "o", Method.OLMIX: "D", Method.SPLINE: "s", Method.REGMIX: "^"}


def proposal_runs(runs: pd.DataFrame, mixture: str) -> pd.DataFrame:
    """The ``validation_3e18`` runs that trained the proposal ``mixture`` at its three trainer seeds.

    Raises ``ValueError`` unless the proposal launches hold one run at each seed.
    """
    rows = runs[(runs["mixture_id"] == mixture) & runs["launch"].isin(PROPOSAL_LAUNCHES)]
    if sorted(rows["trainer_seed"]) != list(TRAINER_SEEDS):
        raise ValueError(f"{mixture} has trainer seeds {sorted(rows['trainer_seed'])}, not {TRAINER_SEEDS}")
    return rows


def proposal_values(root: Path) -> dict[str, pd.DataFrame]:
    """Each proposal's measured runs and their summary, and the predictions ``fits_proposal_predictions`` holds.

    The summary is the mean over the three trainer seeds and the half-width of the figure's 95% interval.
    """
    runs = read_table(root, VALIDATION_TABLE)
    outcomes = [f"{OUTCOME_PREFIX}{objective}" for objective in PROPOSAL_MIXTURES]
    require_columns(runs, [OBSERVATION_ID, "mixture_id", "launch", "trainer_seed", *outcomes], VALIDATION_TABLE)
    released = read_table(root, "fits_proposal_predictions")
    columns = [OBSERVATION_ID, "objective", "predictor", "predicted_bpb"]
    require_columns(released, columns, "fits_proposal_predictions")
    measured, predictions = [], []
    for objective, proposals in PROPOSAL_MIXTURES.items():
        for proposal, mixture in proposals.items():
            rows = proposal_runs(runs, mixture)
            measured += [
                {
                    "objective": objective,
                    "proposal": proposal.value,
                    OBSERVATION_ID: run,
                    "trainer_seed": seed,
                    "bpb": bpb,
                }
                for run, seed, bpb in zip(
                    rows[OBSERVATION_ID], rows["trainer_seed"], rows[f"{OUTCOME_PREFIX}{objective}"], strict=True
                )
            ]
            own = released[(released[OBSERVATION_ID].isin(rows[OBSERVATION_ID])) & (released["objective"] == objective)]
            for predictor, values in own.groupby("predictor"):
                if predictor not in {model.value for model in PROPOSAL_MODELS}:
                    continue
                if values["predicted_bpb"].nunique() != 1:
                    raise ValueError(f"{predictor} predicts the runs of {mixture} differently")
                predictions.append(
                    {
                        "objective": objective,
                        "proposal": proposal.value,
                        "predictor": predictor,
                        "predicted_bpb": float(values["predicted_bpb"].iloc[0]),
                    }
                )
    measured_runs = pd.DataFrame(measured)
    summary = (
        measured_runs.groupby(["objective", "proposal"], sort=False)["bpb"]
        .agg(mean_bpb="mean", sd_bpb="std", runs="size")
        .reset_index()
    )
    summary["half_width"] = NORMAL_QUANTILE_95 * summary["sd_bpb"] / np.sqrt(summary["runs"])
    return {"measured": measured_runs, "measured_summary": summary, "predictions": pd.DataFrame(predictions)}


def draw_proposal_predictions(root: Path) -> FigureData:
    """Figure 7: each proposal's measured loss with its 95% interval, and every model's prediction for it.

    Raises ``ValueError`` naming the (proposal, predictor) pairs the snapshot has no prediction for.
    """
    values = proposal_values(root)
    summary, predictions = values["measured_summary"], values["predictions"]
    held = set(zip(predictions["objective"], predictions["proposal"], predictions["predictor"], strict=True))
    missing = [
        f"{predictor.value} on {proposal.value}'s {objective} proposal"
        for objective in PROPOSAL_MIXTURES
        for proposal in PROPOSAL_MODELS
        for predictor in PROPOSAL_MODELS
        if (objective, proposal.value, predictor.value) not in held
    ]
    if missing:
        raise ValueError(f"fits_proposal_predictions has no prediction of {', '.join(missing)}")
    figure, axes = panels(len(PROPOSAL_MIXTURES), legend_height=0.4)
    for axis, objective in zip(axes, PROPOSAL_MIXTURES, strict=True):
        for column, proposal in enumerate(PROPOSAL_MODELS):
            row = summary[(summary["objective"] == objective) & (summary["proposal"] == proposal)].iloc[0]
            color = METHOD_COLORS[proposal]
            span = [column - BAR_HALF_WIDTH, column + BAR_HALF_WIDTH]
            low, high = row["mean_bpb"] - row["half_width"], row["mean_bpb"] + row["half_width"]
            axis.fill_between(span, low, high, color=color, alpha=0.2, linewidth=0)
            axis.plot(span, [row["mean_bpb"]] * 2, color=color, linewidth=1.8)
            for predictor in PROPOSAL_MODELS:
                predicted = predictions[
                    (predictions["objective"] == objective)
                    & (predictions["proposal"] == proposal)
                    & (predictions["predictor"] == predictor)
                ]["predicted_bpb"].iloc[0]
                axis.plot(
                    [column],
                    [predicted],
                    MARKERS[predictor],
                    color=METHOD_COLORS[predictor],
                    markerfacecolor=METHOD_COLORS[predictor] if predictor == Method.MARINER else "none",
                )
        axis.set_xticks(
            range(len(PROPOSAL_MODELS)), [METHOD_LABELS[model].replace(" ", "\n") for model in PROPOSAL_MODELS]
        )
        axis.set_xlabel("Proposal of")
        axis.set_title(OBJECTIVE_TITLES[objective])
    axes[0].set_ylabel("Loss at 3e18 FLOPs (BPB)")
    handles = [Line2D([], [], color=NEUTRAL, linewidth=1.8, label="measured (mean and 95% interval)")]
    handles += [
        Line2D(
            [],
            [],
            marker=MARKERS[model],
            linestyle="none",
            color=METHOD_COLORS[model],
            markerfacecolor=METHOD_COLORS[model] if model == Method.MARINER else "none",
            label=f"{METHOD_LABELS[model]}'s prediction",
        )
        for model in PROPOSAL_MODELS
    ]
    figure_legend(figure, handles, columns=3)
    return FigureData(figure, values)
