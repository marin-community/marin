# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The two-bucket Nemotron-CC/StarCoder sweeps that Figures 1 to 4 plot, and MARINER's fits to them.

A sweep varies the StarCoder share p of a mixture with Nemotron-CC web text at one model size and token budget, under
simulated epoching. Figures 3 and 4 fit MARINER to each sweep as a two-bucket swarm (``fit_sweep``), the way the paper
fitted them.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from mariner.data import OBSERVATION_ID, read_table
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
            math.nan,
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
