# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The two-bucket Nemotron-CC/StarCoder sweeps that Figures 1 to 4 plot, and MARINER's fits to them.

A sweep varies the StarCoder share p of a mixture with Nemotron-CC web text at one model size and token budget. Figures 3
and 4 fit MARINER to each sweep as a two-bucket swarm (``fit_sweep``), the way the paper fitted them. Figures 1 and 2
draw the epoch-matching sweep (``epoch_matching``): proxies with and without simulated epoching select a share, and a
target run measures each selection.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import minimize_scalar

from mariner.data import OBSERVATION_ID, read_table
from mariner.figures.layout import BAND_ALPHA, NEUTRAL, flops_label, panel_legend, rounded_label
from mariner.fit import fit_objective
from mariner.model import Anchor, Objective, ObjectiveFit, Swarm

FIXED_TOKENS_CURVES = ("C05", "C06", "C07", "C08")  # starcoder_curves refs: 1.00B tokens, 210M to 1.18B parameters
FIXED_PARAMETERS_CURVES = ("C05", "C10", "C13", "C14")  # 210M parameters, 1.00B to 7.41B tokens
# Under simulated epoching the sweeps' target budget is Nemotron-CC's pool, so a share of 1 is one Nemotron-CC epoch
# (the fitter description of starcoder_curves).
NEMOTRON_EPOCHS_PER_UNIT_WEIGHT = 1.0
SWEEP_BUCKETS = ("nemotron_cc", "starcoder")
SWEEP_TASK = "programming_languages_bpb"
SWEEP_OBJECTIVE = Objective(SWEEP_TASK, (SWEEP_TASK,), np.ones(1))
MINIMUM_GRID = 4_001  # shares that bracket a fit's local minima before a bounded search refines them
MINIMUM_XATOL = 1e-12
CURVE_POINTS = 400  # shares at which a fitted curve is drawn
EPOCH_SCALE_TOLERANCE = 1e-12  # relative: a diagonal cell's epochs over its shares, one value up to rounding
SHARE_AXIS_LIMITS = (0.0, 0.95)  # the fitted sweeps run from p = 0.036 to 0.9; the margin keeps their last markers


@dataclasses.dataclass(frozen=True)
class StarcoderSweep:
    """One sweep of the StarCoder share at a fixed model and token budget, sorted by share.

    ``name`` is the atlas's ``curve_ref`` or the diagonal's ``cell_id``.
    """

    name: str
    total_parameters: float
    train_tokens: float
    starcoder_epochs_at_p1: float  # StarCoder's materialized epochs at share 1
    compute_flops: float
    shares: np.ndarray
    bpb: np.ndarray


def sorted_sweep(
    name: str,
    total_parameters: float,
    train_tokens: float,
    starcoder_epochs_at_p1: float,
    compute_flops: float,
    shares: pd.Series,
    bpb: pd.Series,
) -> StarcoderSweep:
    order = np.argsort(shares.to_numpy(float))
    return StarcoderSweep(
        name,
        float(total_parameters),
        float(train_tokens),
        float(starcoder_epochs_at_p1),
        float(compute_flops),
        shares.to_numpy(float)[order],
        bpb.to_numpy(float)[order],
    )


def atlas_sweeps(root: Path, refs: Sequence[str]) -> dict[str, StarcoderSweep]:
    """The atlas's sweeps ``refs`` (``starcoder_curves``), with their runs' shares and BPB."""
    curves = read_table(root, "starcoder_curves").set_index("curve_ref")
    membership = read_table(root, "starcoder_curve_membership")
    observations = read_table(root, "starcoder_observations").set_index(OBSERVATION_ID)
    sweeps = {}
    for ref in dict.fromkeys(refs):
        curve = curves.loc[ref]
        runs = observations.loc[membership.loc[membership["curve_id"] == curve["curve_id"], OBSERVATION_ID]]
        sweeps[ref] = sorted_sweep(
            ref,
            curve["total_parameters"],
            curve["train_tokens"],
            curve["starcoder_epochs_at_p1"],
            curve["compute_flops"],
            runs["starcoder_weight"],
            runs["observed_bpb"],
        )
    return sweeps


def diagonal_sweeps(root: Path) -> list[StarcoderSweep]:
    """The fixed-TPP sweeps (``starcoder_diagonal``), one per cell, from the smallest compute to the largest."""
    diagonal = read_table(root, "starcoder_diagonal").sort_values(["rung", "starcoder_weight"])
    sweeps = []
    for cell, rows in diagonal.groupby("cell_id", sort=False):
        scales = (rows["starcoder_epochs"] / rows["starcoder_weight"]).to_numpy(float)
        if np.ptp(scales) > EPOCH_SCALE_TOLERANCE * scales.max():
            raise ValueError(f"the runs of diagonal cell {cell} disagree on StarCoder's epochs per unit share")
        first = rows.iloc[0]
        sweeps.append(
            sorted_sweep(
                str(cell),
                first["total_parameters"],
                first["materialized_tokens"],
                float(np.median(scales)),
                first["compute_flops"],
                rows["starcoder_weight"],
                rows["starcoder_bpb"],
            )
        )
    return sweeps


def fit_shares(shares: np.ndarray, bpb: np.ndarray, starcoder_epochs_at_p1: float) -> ObjectiveFit:
    """MARINER fitted to losses at StarCoder shares, sorted by share, as the paper fitted Figures 1, 3 and 4.

    The runs form a two-bucket swarm, Nemotron-CC at share 1 - p and StarCoder at share p. The smallest-share run is
    the calibration run: it stays in every training fold, and its BPB anchors the floor without a noise margin, since
    the runs have no repeats. Leave-one-out cross-validation over the other runs selects the shape, the ridge weight
    and the floor multiplier.
    """
    rows = np.arange(len(shares))
    swarm = Swarm(
        runs=tuple(str(row) for row in rows),
        buckets=SWEEP_BUCKETS,
        weights=np.column_stack([1.0 - shares, shares]),
        epochs_per_unit_weight=np.array([NEMOTRON_EPOCHS_PER_UNIT_WEIGHT, starcoder_epochs_at_p1]),
        pool_fractions=np.ones((len(rows), len(SWEEP_BUCKETS))),
        outcomes=pd.DataFrame({SWEEP_TASK: bpb}),
        calibration=rows == 0,
    )
    folds = tuple((np.delete(rows, row), rows[row : row + 1]) for row in rows[1:])
    return fit_objective(swarm, SWEEP_OBJECTIVE, {SWEEP_TASK: Anchor(float(bpb[0]), 0.0)}, folds=folds)


def fit_sweep(sweep: StarcoderSweep) -> ObjectiveFit:
    """MARINER fitted to one sweep (``fit_shares``); its smallest share is the proportional mixture."""
    return fit_shares(sweep.shares, sweep.bpb, sweep.starcoder_epochs_at_p1)


def sweep_prediction(fit: ObjectiveFit, shares: np.ndarray) -> np.ndarray:
    return fit.predict(np.column_stack([1.0 - shares, shares]))


class Optimum(NamedTuple):
    share: float
    bpb: float


def fitted_minimum(fit: ObjectiveFit, low: float, high: float) -> Optimum:
    """The share in [``low``, ``high``] at which the fit predicts the least BPB, and that BPB.

    A grid of shares brackets each local minimum, which a bounded scalar search refines; the interval's ends compete
    too.
    """
    grid = np.linspace(low, high, MINIMUM_GRID)
    loss = sweep_prediction(fit, grid)
    candidates = [Optimum(grid[0], loss[0]), Optimum(grid[-1], loss[-1])]
    for index in np.flatnonzero((loss[1:-1] < loss[:-2]) & (loss[1:-1] <= loss[2:])) + 1:
        search = minimize_scalar(
            lambda share: float(sweep_prediction(fit, np.array([share]))[0]),
            bounds=(grid[index - 1], grid[index + 1]),
            method="bounded",
            options={"xatol": MINIMUM_XATOL},
        )
        candidates.append(Optimum(float(search.x), float(search.fun)))
    best = min(candidates, key=lambda optimum: optimum.bpb)
    return Optimum(float(best.share), float(best.bpb))


@dataclasses.dataclass(frozen=True)
class FittedSweep:
    """A sweep with MARINER's fit, the fit's minimum over the sweep's shares, and the fitted curve to draw."""

    sweep: StarcoderSweep
    fit: ObjectiveFit
    optimum: Optimum
    shares: np.ndarray  # CURVE_POINTS shares from the sweep's smallest to its largest
    predicted_bpb: np.ndarray

    @property
    def optimum_epochs(self) -> float:
        """StarCoder's materialized epochs at the fitted minimum."""
        return self.optimum.share * self.sweep.starcoder_epochs_at_p1


def fitted_sweep(sweep: StarcoderSweep) -> FittedSweep:
    fit = fit_sweep(sweep)
    shares = np.linspace(sweep.shares[0], sweep.shares[-1], CURVE_POINTS)
    optimum = fitted_minimum(fit, float(sweep.shares[0]), float(sweep.shares[-1]))
    return FittedSweep(sweep, fit, optimum, shares, sweep_prediction(fit, shares))


def parameters_label(count: float) -> str:
    return f"{count / 1e9:.2f}B" if count >= 1e9 else f"{count / 1e6:.0f}M"


def tokens_label(count: float) -> str:
    return f"{count / 1e9:.2f}B"


# ---------------------------------------------------------------------------------------------------------------
# The epoch-matching sweep (Figures 1 and 2)
# ---------------------------------------------------------------------------------------------------------------

EPOCH_MATCHING_TABLE = "starcoder_epoch_matching"
EPOCH_MATCHING_BPB = "programming_languages_bpb"
RUN_KEYS = ["arm", "subset_seed", "trainer_seed", "starcoder_weight"]  # one run of the sweep each
SELECTION_TOLERANCE = 1e-12  # the target's follow-up run trains the matched selection, a mean of grid shares


class Arm(StrEnum):
    """The epoch-matching sweep's runs: the target, and the proxy without and with simulated epoching."""

    TARGET = "target"
    UNMATCHED = "unmatched"
    MATCHED = "matched"


PROXY_ARMS = (Arm.MATCHED, Arm.UNMATCHED)
ARM_COLORS = {Arm.TARGET: "#2F4B7C", Arm.UNMATCHED: "#CC3311", Arm.MATCHED: "#469C76"}
AVOIDED_COLOR = "#0072B2"
DIAMOND = {"marker": "D", "markersize": 6, "markeredgecolor": "white", "linestyle": "none", "zorder": 6}


@dataclasses.dataclass(frozen=True)
class EpochMatching:
    """The epoch-matching sweep (Figure 2) on its grid of StarCoder shares.

    ``matched`` has one row per subset of the proxy with simulated epoching, the mean over its two trainer seeds, and
    ``unmatched`` is the mean over the other proxy's two; the runs without StarCoder serve both. ``target`` holds the
    target's grid runs and its follow-up at the matched selection. A proxy selects the mean of its curves' grid minima,
    a curve being one trainer seed's runs (of one subset, with simulated epoching).
    """

    matched: pd.DataFrame  # subset seeds x grid shares
    unmatched: pd.Series  # by grid share
    target: pd.Series  # by share
    selections: dict[Arm, float]  # each proxy's selected share
    epochs: pd.DataFrame  # arms x shares: StarCoder's materialized epochs
    epochs_at_p1: dict[Arm, float]  # each arm's tokens over its StarCoder pool: its epochs at share 1
    compute_flops: dict[Arm, float]

    @property
    def shares(self) -> np.ndarray:
        return self.unmatched.index.to_numpy(float)

    def target_optimum(self) -> Optimum:
        """The grid share with the least target loss, and that loss."""
        grid = self.target.loc[self.shares]
        return Optimum(float(grid.idxmin()), float(grid.min()))

    def proxy_bpb(self, arm: Arm) -> float:
        """A proxy's loss at its selection, on its curve: for simulated epoching, the subsets' mean."""
        curve = self.matched.mean() if arm == Arm.MATCHED else self.unmatched
        return float(np.interp(self.selections[arm], self.shares, curve.to_numpy(float)))

    def target_bpb(self, arm: Arm) -> float:
        """The target's loss at a proxy's selection, from the target run that trained it."""
        trained = self.target[np.isclose(self.target.index, self.selections[arm], rtol=0, atol=SELECTION_TOLERANCE)]
        if len(trained) != 1:
            raise ValueError(f"{len(trained)} target runs trained the {arm} selection {self.selections[arm]:.6f}")
        return float(trained.iloc[0])

    def regret(self, arm: Arm) -> float:
        return self.target_bpb(arm) - self.target_optimum().bpb

    @property
    def excess_loss_avoided(self) -> float:
        """The fraction of the unmatched proxy's target regret that the matched proxy's selection avoids."""
        return 1.0 - self.regret(Arm.MATCHED) / self.regret(Arm.UNMATCHED)

    def epochs_at(self, arm: Arm, share: float) -> float:
        row = self.epochs.loc[arm]
        return float(row[np.isclose(row.index, share, rtol=0, atol=SELECTION_TOLERANCE)].iloc[0])

    def selection_epochs(self, arm: Arm) -> float:
        """StarCoder's materialized epochs at a proxy's selection. Simulated epoching reproduces the target's epochs,
        and the matched selection is the share that the target's follow-up run trained."""
        return self.epochs_at(Arm.TARGET if arm == Arm.MATCHED else arm, self.selections[arm])

    def realized_shares(self, arm: Arm) -> pd.Series:
        """The StarCoder share that an arm's runs drew at each nominal share: its materialized epochs over its epochs at
        share 1. A run without StarCoder draws none."""
        epochs = self.epochs.loc[arm].dropna()
        shares = epochs.index.to_numpy(float)
        realized = pd.Series(epochs.to_numpy(float) / self.epochs_at_p1[arm], index=shares)
        return pd.concat([pd.Series([0.0], index=[0.0]), realized[shares != 0.0]]).sort_index()


def epoch_matching(root: Path) -> EpochMatching:
    """The epoch-matching sweep (``starcoder_epoch_matching``) reduced to the curves and selections Figures 1 and 2 plot.

    A run without StarCoder is the same for every subset, so the unmatched proxy's runs at share 0 are also each
    matched subset's point at 0. Raises ``ValueError`` when two runs share an arm, subset, trainer seed and share,
    when a proxy curve lacks a share, or when an arm's runs at one share differ in materialized epochs.
    """
    runs = read_table(root, EPOCH_MATCHING_TABLE)
    repeated = runs[runs.duplicated(RUN_KEYS, keep=False)]
    if len(repeated):
        raise ValueError(f"{EPOCH_MATCHING_TABLE} repeats runs: {', '.join(repeated['observation_id'])}")
    epochs = runs.groupby(["arm", "starcoder_weight"])["starcoder_epochs"]
    differing = epochs.nunique()[lambda counts: counts > 1]
    if len(differing):
        raise ValueError(f"runs of one arm and share differ in materialized epochs: {list(differing.index)}")
    unmatched = runs[runs["arm"] == Arm.UNMATCHED]
    matched = runs[runs["arm"] == Arm.MATCHED]
    without = unmatched[unmatched["starcoder_weight"] == 0.0]
    matched = pd.concat([matched, *(without.assign(subset_seed=seed) for seed in matched["subset_seed"].unique())])
    matched_curves = matched.pivot(
        index=["subset_seed", "trainer_seed"], columns="starcoder_weight", values=EPOCH_MATCHING_BPB
    )
    unmatched_curves = unmatched.pivot(index="trainer_seed", columns="starcoder_weight", values=EPOCH_MATCHING_BPB)
    if not matched_curves.columns.equals(unmatched_curves.columns):
        raise ValueError("the proxies' curves cover different StarCoder shares")
    if matched_curves.isna().any().any() or unmatched_curves.isna().any().any():
        raise ValueError(f"a proxy curve of {EPOCH_MATCHING_TABLE} lacks a run at some StarCoder share")
    target = runs[runs["arm"] == Arm.TARGET].set_index("starcoder_weight")[EPOCH_MATCHING_BPB].sort_index()
    if not set(matched_curves.columns) <= set(target.index):
        raise ValueError(f"the target of {EPOCH_MATCHING_TABLE} lacks a run at some share of the proxies' grid")
    first = runs.groupby("arm").first()
    return EpochMatching(
        matched=matched_curves.groupby(level="subset_seed").mean(),
        unmatched=unmatched_curves.mean(),
        target=target,
        selections={
            Arm.MATCHED: float(matched_curves.idxmin(axis=1).mean()),
            Arm.UNMATCHED: float(unmatched_curves.idxmin(axis=1).mean()),
        },
        epochs=epochs.first().unstack(),
        epochs_at_p1={
            Arm(arm): float(row["train_tokens"] / row["starcoder_subset_tokens"]) for arm, row in first.iterrows()
        },
        compute_flops={Arm(arm): float(row["compute_flops"]) for arm, row in first.iterrows()},
    )


def selection_values(matching: EpochMatching) -> pd.DataFrame:
    """Each proxy's selected share, its loss there, the target's loss there and the target's regret."""
    return pd.DataFrame(
        [
            {
                "arm": arm.value,
                "share": matching.selections[arm],
                "proxy_bpb": matching.proxy_bpb(arm),
                "target_bpb": matching.target_bpb(arm),
                "regret": matching.regret(arm),
                "epochs": matching.selection_epochs(arm),
            }
            for arm in PROXY_ARMS
        ]
    )


def draw_proxies(axis: Axes, matching: EpochMatching, matched_curves: pd.DataFrame, first_share: float) -> None:
    """The proxies' loss from ``first_share``: the rows of ``matched_curves`` over the band of the subsets' range, the
    unmatched proxy's curve, and each proxy's selection on its curve."""
    shown = matching.shares >= first_share
    shares = matching.shares[shown]
    color = ARM_COLORS[Arm.MATCHED]
    low, high = matching.matched.min().to_numpy(float), matching.matched.max().to_numpy(float)
    axis.fill_between(shares, low[shown], high[shown], color=color, alpha=BAND_ALPHA, linewidth=0)
    for _, curve in matched_curves.iterrows():
        axis.plot(shares, curve.to_numpy(float)[shown], color=color, marker="o", markersize=3)
    unmatched = matching.unmatched.to_numpy(float)[shown]
    axis.plot(shares, unmatched, color=ARM_COLORS[Arm.UNMATCHED], marker="o", markerfacecolor="white")
    for arm in PROXY_ARMS:
        axis.plot([matching.selections[arm]], [matching.proxy_bpb(arm)], color=ARM_COLORS[arm], **DIAMOND)


def draw_target(axis: Axes, matching: EpochMatching, first_share: float) -> None:
    """The target's loss from ``first_share``, its optimum, each proxy selection's target loss above the optimum (its
    regret), and the excess loss that the matched selection avoids."""
    target = matching.target[matching.target.index >= first_share]
    optimum = matching.target_optimum()
    color = ARM_COLORS[Arm.TARGET]
    axis.plot(target.index, target.to_numpy(float), color=color, marker="o", markerfacecolor="white")
    axis.axhline(optimum.bpb, color=NEUTRAL, linestyle="--", linewidth=0.7)
    axis.plot([optimum.share], [optimum.bpb], "*", color=color, markersize=11, markeredgecolor="white", zorder=6)
    for arm in PROXY_ARMS:
        share, bpb = matching.selections[arm], matching.target_bpb(arm)
        axis.plot([share, share], [optimum.bpb, bpb], color=ARM_COLORS[arm], linewidth=3, solid_capstyle="butt")
        axis.plot([share], [bpb], color=ARM_COLORS[arm], **DIAMOND)
    matched, unmatched = matching.target_bpb(Arm.MATCHED), matching.target_bpb(Arm.UNMATCHED)
    share = matching.selections[Arm.MATCHED]
    axis.plot([share, matching.selections[Arm.UNMATCHED]], [unmatched, unmatched], color=AVOIDED_COLOR, linestyle=":")
    axis.plot([share, share], [matched, unmatched], color=AVOIDED_COLOR, linewidth=1.2, marker="_", markersize=8)


def draw_repetition(
    proxy: Axes,
    target: Axes,
    matching: EpochMatching,
    matched_curves: pd.DataFrame,
    matched_label: str,
    first_shares: tuple[float, float],
) -> dict[str, pd.DataFrame]:
    """Figures 1 and 2's proxy and target panels, with their titles and legends, and the values they plot.

    ``matched_curves`` are the curves drawn with simulated epoching, each subset's or their mean, which
    ``matched_label`` names; ``first_shares`` are the first shares in view in the proxy and the target panel.
    """
    proxy_first, target_first = first_shares
    draw_proxies(proxy, matching, matched_curves, proxy_first)
    draw_target(target, matching, target_first)
    selections = selection_values(matching).set_index("arm")
    optimum = matching.target_optimum()
    matched, unmatched = selections.loc[Arm.MATCHED], selections.loc[Arm.UNMATCHED]
    with_color, without_color = ARM_COLORS[Arm.MATCHED], ARM_COLORS[Arm.UNMATCHED]
    with_selection = f"p = {matched['share']:.3f}, {rounded_label(matched['epochs'])} epochs"
    without_selection = f"p = {unmatched['share']:.2f}, {rounded_label(unmatched['epochs'])} epochs"
    band = f"range of its {len(matching.matched)} subsets"
    panel_legend(
        proxy,
        [
            Line2D(
                [], [], color=with_color, marker="o", markersize=3, label=f"with simulated epoching ({matched_label})"
            ),
            Patch(color=with_color, alpha=BAND_ALPHA, linewidth=0, label=band),
            Line2D([], [], color=without_color, marker="o", markerfacecolor="white", label="without simulated epoching"),
            Line2D([], [], color=with_color, **DIAMOND, label=f"selected with: {with_selection}"),
            Line2D([], [], color=without_color, **DIAMOND, label=f"selected without: {without_selection}"),
        ],
    )
    avoided = f"{100 * matching.excess_loss_avoided:.1f}% less excess loss with simulated epoching"
    target_color = ARM_COLORS[Arm.TARGET]
    panel_legend(
        target,
        [
            Line2D([], [], color=target_color, marker="o", markerfacecolor="white", label="target run"),
            Line2D(
                [],
                [],
                color=target_color,
                marker="*",
                linestyle="none",
                markersize=11,
                label=f"optimum: p = {optimum.share:.2f}, {optimum.bpb:.3f}",
            ),
            Line2D([], [], color=with_color, **DIAMOND, label=f"at the selection with: {matched['target_bpb']:.3f}"),
            Line2D(
                [], [], color=without_color, **DIAMOND, label=f"at the selection without: {unmatched['target_bpb']:.3f}"
            ),
            Line2D([], [], color=AVOIDED_COLOR, label=avoided),
        ],
    )
    proxy.set_title(f"Proxy runs, {flops_label(matching.compute_flops[Arm.MATCHED])} FLOPs")
    target.set_title(f"Target run, {flops_label(matching.compute_flops[Arm.TARGET])} FLOPs")
    proxy.set_ylabel("Proxy loss (BPB)")
    target.set_ylabel("Target loss (BPB)")
    for axis in (proxy, target):
        axis.set_xlabel("StarCoder share of the mixture, p")
    shown = matching.shares >= proxy_first
    shares = matching.shares[shown]
    proxy_curves = [
        {"arm": Arm.MATCHED.value, "curve": str(name), "share": share, "bpb": bpb}
        for name, curve in matched_curves.iterrows()
        for share, bpb in zip(shares, curve.to_numpy(float)[shown], strict=True)
    ]
    proxy_curves += [
        {"arm": Arm.UNMATCHED.value, "curve": Arm.UNMATCHED.value, "share": share, "bpb": bpb}
        for share, bpb in zip(shares, matching.unmatched.to_numpy(float)[shown], strict=True)
    ]
    band = pd.DataFrame(
        {
            "share": shares,
            "low": matching.matched.min().to_numpy(float)[shown],
            "high": matching.matched.max().to_numpy(float)[shown],
        }
    )
    target_curve = matching.target[matching.target.index >= target_first]
    return {
        "proxy_curves": pd.DataFrame(proxy_curves),
        "proxy_band": band,
        "target_curve": target_curve.rename_axis("share").rename("bpb").reset_index(),
        "selections": selections.reset_index(),
        "target_optimum": pd.DataFrame([optimum._asdict()]),
        "summary": pd.DataFrame(
            {
                "excess_loss_avoided": [matching.excess_loss_avoided],
                "proxy_flops": [matching.compute_flops[Arm.MATCHED]],
                "target_flops": [matching.compute_flops[Arm.TARGET]],
            }
        ),
    }
