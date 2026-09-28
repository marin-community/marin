# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hierarchical phase replay averaged over the configurations cross-validation cannot separate.

Selecting a surrogate normally means scoring every candidate configuration by grouped
out-of-fold RMSE and keeping the single winner. On this panel that step is close to a coin
flip. Across the distinct hierarchical-replay shapes, out-of-fold RMSE rank-correlates +0.97
with absolute censored bias, so the criterion is a good global proxy for proposal quality --
but restricted to configurations within one run sigma of the best, the correlation collapses
to +0.40 on Uncheatable and inverts to -0.85 on Table-9 while absolute bias still spans 2.8
run sigma. Two configurations differing by 0.10 run sigma in out-of-fold RMSE disagree by a
median 0.0011 BPB, and up to 0.0092, on individual policies. The aggregate score averages that
disagreement away; a proposal acts on it.

So this model keeps every configuration whose out-of-fold RMSE falls inside that unresolvable
band and averages their predictions, rather than pretending to choose. Weights are nonnegative
and sum to one, fitted by least squares on the members' own out-of-fold predictions. Because
"all weight on the winner" is a feasible weight vector, the combination cannot fit worse than
the single best configuration, which is what a plain band average does lose. The simplex
constraint also keeps the combination inside the convex hull of its members, so the ensemble
cannot extrapolate further than any member does.

Candidate configurations come from the same two-stage screen the single-configuration model
uses -- a baseline shape screen, then structural configurations over the top shapes -- so the
two differ only in whether the band is collapsed to its argmin.

Two honest limits. The gains are modest: on the 300M fit panel this improves out-of-fold RMSE
by about 0.4 percent and censored-extrapolation RMSE by about 9 percent, the latter with a
paired interval excluding zero, while phase decision skill is unchanged under resampling. And
the stacked weights come out sparse, typically two or three active members dominated by the
argmin at 0.84 to 0.90, because weighting on out-of-fold error runs into the same resolution
limit that motivated the ensemble. The residual mass on the other members is where the gain
comes from.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import nnls

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_hierarchical_coverage_grp_20260715 import (  # noqa: E402
    Config,
    Model,
    fit_model,
    oof_prediction,
)
from fit_production_grp_quality_variants import Dataset  # noqa: E402

# Half-width of the unresolvable band, in run standard deviations. One sigma is the width at
# which the criterion's rank correlation with censored bias was measured to break down.
BAND_SIGMA = 1.0
# Run-to-run standard deviation for the targets where replicate runs of identical mixtures
# measured it. Other swarms fall back to a relative width.
RUN_SIGMA = {"uncheatable": 0.00096, "table9": 0.0031}
# Fallback band half-width as a fraction of the best out-of-fold RMSE, used where run sigma has
# not been measured. Chosen to sit between the measured ratios on the two dolma3 targets, which
# are 12 percent (Uncheatable) and 21 percent (Table-9).
RELATIVE_BAND = 0.15
# Cap on members so a very flat criterion cannot make a cell unaffordable. Members are taken in
# ascending out-of-fold error, so a truncated band keeps its best part.
MAX_MEMBERS = 24
EPSILON = 1e-12


def band_half_width(target_id: str, best_rmse: float) -> float:
    """Band half-width in target units, from measured run sigma where it exists."""
    sigma = RUN_SIGMA.get(target_id)
    return BAND_SIGMA * sigma if sigma is not None else RELATIVE_BAND * best_rmse


@dataclass(frozen=True)
class BandMember:
    config: Config
    oof_rmse: float
    weight: float


@dataclass(frozen=True)
class BandModel:
    """Averaged predictor over the configurations the criterion cannot separate."""

    members: tuple[BandMember, ...]
    fitted: tuple[Model, ...]
    best_oof_rmse: float
    band_half_width: float
    n_candidates: int

    @property
    def config(self) -> Config:
        """The highest-weighted member, so single-config accessors keep working."""
        return max(zip(self.members, strict=True), key=lambda pair: pair[0].weight)[0].config

    @property
    def active_members(self) -> int:
        return sum(1 for member in self.members if member.weight > 1e-6)

    def predict(self, weights: np.ndarray) -> np.ndarray:
        total = None
        for member, model in zip(self.members, self.fitted, strict=True):
            if member.weight <= 0.0:
                continue
            contribution = member.weight * np.asarray(model.predict(weights), dtype=float)
            total = contribution if total is None else total + contribution
        assert total is not None, "band has no active members"
        return total


def stack_weights(predictions: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Nonnegative weights summing to one that minimize error of the combination.

    Solved on the simplex by writing ``w = e_0 + sum_k d_k (e_k - e_0)`` with ``d >= 0`` and
    ``sum d <= 1``, so nonnegative least squares on the differenced predictions gives ``d`` and
    the residual mass returns to the best member.
    """
    if predictions.shape[1] == 1:
        return np.ones(1)
    finite = np.isfinite(observed) & np.isfinite(predictions).all(axis=1)
    design, truth = predictions[finite], observed[finite]
    differences = design[:, 1:] - design[:, :1]
    coefficients, _ = nnls(differences, truth - design[:, 0], maxiter=200 * differences.shape[1])
    total = coefficients.sum()
    if total > 1.0:
        coefficients = coefficients / total
    weights = np.concatenate([[1.0 - coefficients.sum()], coefficients])
    return np.maximum(weights, 0.0) / max(np.maximum(weights, 0.0).sum(), EPSILON)


def build_band(
    dataset: Dataset,
    configs: list[Config],
    splits: list[tuple[np.ndarray, np.ndarray]],
    target_id: str,
    indices: np.ndarray,
) -> tuple[BandModel, dict[str, Any]]:
    """Score every candidate, keep the unresolvable band, and stack it.

    Scoring and weighting use out-of-fold predictions on the fitting rows only, so the band is
    identified by the same panel every other Observatory model is selected on.
    """
    observed = np.asarray(dataset.target, dtype=float)
    # One out-of-fold pass per candidate; scoring and stacking both reuse it.
    predictions = {index: oof_prediction(dataset, config, splits) for index, config in enumerate(configs)}
    scored = [(float(np.sqrt(np.nanmean((predictions[index] - observed) ** 2))), index) for index in predictions]
    scored.sort(key=lambda item: item[0])
    best = scored[0][0]
    half_width = band_half_width(target_id, best)
    inside = [(rmse, index) for rmse, index in scored if rmse <= best + half_width][:MAX_MEMBERS]

    stacked = np.column_stack([predictions[index] for _, index in inside])
    weights = stack_weights(stacked, observed)
    members = tuple(
        BandMember(config=configs[index], oof_rmse=rmse, weight=float(weight))
        for (rmse, index), weight in zip(inside, weights, strict=True)
    )
    model = BandModel(
        members=members,
        fitted=tuple(fit_model(dataset, member.config, indices) for member in members),
        best_oof_rmse=best,
        band_half_width=half_width,
        n_candidates=len(configs),
    )
    detail = {
        "bandSize": len(members),
        "activeMembers": model.active_members,
        "candidatesScored": len(configs),
        "bestOofRmse": best,
        "bandHalfWidth": half_width,
        "bandWorstOofRmse": float(inside[-1][0]),
        "memberWeights": [
            {"oofRmse": member.oof_rmse, "weight": member.weight} for member in members if member.weight > 1e-6
        ],
    }
    return model, detail


def refit(dataset: Dataset, model: BandModel, indices: np.ndarray) -> BandModel:
    """Refit every member on a new row subset, keeping band membership and weights fixed."""
    from dataclasses import replace  # noqa: PLC0415

    return replace(model, fitted=tuple(fit_model(dataset, member.config, indices) for member in model.members))
