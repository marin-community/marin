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
from scipy.optimize import minimize_scalar

from mariner.data import OBSERVATION_ID, read_table
from mariner.figures.layout import NEUTRAL
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


def fit_sweep(sweep: StarcoderSweep) -> ObjectiveFit:
    """MARINER fitted to one sweep as the paper fitted Figures 3 and 4.

    The sweep is a two-bucket swarm, Nemotron-CC at share 1 - p and StarCoder at share p. Its smallest-share run, the
    proportional mixture, is the calibration run: it stays in every training fold, and its BPB anchors the floor
    without a noise margin, since the sweeps have no repeats. Leave-one-out cross-validation over the other runs
    selects the shape, the ridge weight and the floor multiplier.
    """
    rows = np.arange(len(sweep.shares))
    swarm = Swarm(
        runs=tuple(f"{sweep.name}/{row}" for row in rows),
        buckets=SWEEP_BUCKETS,
        weights=np.column_stack([1.0 - sweep.shares, sweep.shares]),
        epochs_per_unit_weight=np.array([NEMOTRON_EPOCHS_PER_UNIT_WEIGHT, sweep.starcoder_epochs_at_p1]),
        pool_fractions=np.ones((len(rows), len(SWEEP_BUCKETS))),
        outcomes=pd.DataFrame({SWEEP_TASK: sweep.bpb}),
        calibration=rows == 0,
    )
    folds = tuple((np.delete(rows, row), rows[row : row + 1]) for row in rows[1:])
    return fit_objective(swarm, SWEEP_OBJECTIVE, {SWEEP_TASK: Anchor(float(sweep.bpb[0]), 0.0)}, folds=folds)


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


def epoch_matching(root: Path) -> EpochMatching:
    """The epoch-matching sweep (``starcoder_epoch_matching``) reduced to the curves and selections Figures 1 and 2 plot.

    A run without StarCoder is the same for every subset, so the unmatched proxy's runs at share 0 are also each
    matched subset's point at 0.
    """
    runs = read_table(root, EPOCH_MATCHING_TABLE)
    unmatched = runs[runs["arm"] == Arm.UNMATCHED]
    matched = runs[runs["arm"] == Arm.MATCHED]
    without = unmatched[unmatched["starcoder_weight"] == 0.0]
    matched = pd.concat([matched, *(without.assign(subset_seed=seed) for seed in matched["subset_seed"].unique())])
    matched_curves = matched.pivot_table(
        index=["subset_seed", "trainer_seed"], columns="starcoder_weight", values=EPOCH_MATCHING_BPB
    )
    unmatched_curves = unmatched.pivot_table(index="trainer_seed", columns="starcoder_weight", values=EPOCH_MATCHING_BPB)
    if not matched_curves.columns.equals(unmatched_curves.columns):
        raise ValueError("the proxies' curves cover different StarCoder shares")
    target = runs[runs["arm"] == Arm.TARGET].set_index("starcoder_weight")[EPOCH_MATCHING_BPB].sort_index()
    epochs = runs.groupby(["arm", "starcoder_weight"])["starcoder_epochs"].mean().unstack()
    compute = runs.groupby("arm")["compute_flops"].first()
    return EpochMatching(
        matched=matched_curves.groupby(level="subset_seed").mean(),
        unmatched=unmatched_curves.mean(),
        target=target,
        selections={
            Arm.MATCHED: float(matched_curves.idxmin(axis=1).mean()),
            Arm.UNMATCHED: float(unmatched_curves.idxmin(axis=1).mean()),
        },
        epochs=epochs,
        compute_flops={Arm(arm): float(flops) for arm, flops in compute.items()},
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
    axis.fill_between(shares, low[shown], high[shown], color=color, alpha=0.16, linewidth=0)
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
