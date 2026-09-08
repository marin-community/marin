# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Average over the shapes the selection criterion cannot tell apart, instead of picking one.

The continuous-shape exploration established the constraint this model is built around.
Across all distinct hierarchical-replay shapes, out-of-fold RMSE rank-correlates +0.97 with
absolute censored bias, so globally the criterion is a good proxy for proposal quality. But
restricted to shapes within one run sigma of the best, that correlation collapses to +0.40 on
Uncheatable and flips to -0.85 on Table-9, while absolute bias still spans 2.8 sigma. For
hierarchical replay on Table-9 the argmin-OOF shape carries a bias of +5.38e-3 while another
shape in the same band reaches -0.01e-3 for 5.3e-3 of OOF RMSE.

Taking the argmin of a criterion that cannot resolve its own band is therefore closer to
drawing a shape at random from that band than to selecting one. Every tuning decision in this
experiment happens inside it, which is why four structurally unrelated attacks all improved
fit and all failed to improve the phase call.

If the band is unresolvable, the honest estimator averages over it rather than pretending to
choose. Two ways to do that are tested here:

``uniform``   equal weight on every configuration whose out-of-fold RMSE is within
              ``BAND_SIGMA`` run sigma of the best.
``softmin``   weight proportional to ``exp(-(oof - best) / (BAND_SIGMA * sigma))``, which
              degrades gracefully to the argmin as the temperature falls and to the uniform
              band average as it rises.
``stacked``   nonnegative weights summing to one, fitted to minimize out-of-fold error of the
              combination itself. The argmin is a feasible weight vector, so the stacked
              combination cannot have worse out-of-fold error than the single best member.
              This is the arm that repairs the one metric plain averaging loses.

Both are panel-identified: the band is defined by out-of-fold error on the fitting rows
alone, and no censored row or evaluation policy enters the weighting. Averaging predictions
rather than coefficients is deliberate, because the shapes index different nonlinear
responses and their coefficients are not commensurable.

The prediction is a plain average of member predictions, so the ensemble is not a member of
the model class. That is the point: the class was never the constraint, the criterion's
resolution was.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dual_objective_harness_20260726 import RUN_SIGMA, fit_on, out_of_fold_predictions  # noqa: E402
from swarm39_harness_20260725 import Model, Panel  # noqa: E402

# Half-width of the unresolvable band, in run sigma. 1.0 is the width at which the
# criterion's rank correlation with censored bias was measured to collapse.
BAND_SIGMA = 1.0
# Cap on members so a very flat criterion cannot make scoring unaffordable. Members are
# taken in ascending out-of-fold error, so a truncated band keeps its best part.
MAX_MEMBERS = 40


def stack_weights(
    panel: Panel,
    model: Model,
    target: str,
    members: list[tuple[float, dict, float]],
    rows: np.ndarray | None,
) -> np.ndarray:
    """Nonnegative weights summing to one that minimize out-of-fold error of the combination.

    Fitted on the members' own out-of-fold predictions, so no row is used to fit a member and
    to weight it in the same fold. The simplex constraint keeps the combination inside the
    convex hull of its members, which prevents the extrapolating behaviour an unconstrained
    least-squares stack would license on policies far from the panel.
    """
    observed = panel.targets[target]
    if rows is not None:
        observed = np.where(rows, observed, np.nan)
    columns = [out_of_fold_predictions(panel, model, target, shape, l2, rows=rows) for _, shape, l2 in members]
    stacked = np.column_stack(columns)
    finite = np.isfinite(observed) & np.isfinite(stacked).all(axis=1)
    design, truth = stacked[finite], observed[finite]

    # Solve on the simplex by centring on the first member: w = w0 + sum_k d_k (e_k - e_0)
    # with d >= 0 and sum d <= 1. Nonnegative least squares on the differenced design gives
    # the d, and the residual mass returns to member 0.
    differences = design[:, 1:] - design[:, :1]
    coefficients, _ = nnls(differences, truth - design[:, 0], maxiter=200 * max(differences.shape[1], 1))
    total = coefficients.sum()
    if total > 1.0:
        coefficients = coefficients / total
    weights = np.concatenate([[1.0 - coefficients.sum()], coefficients])
    return weights / weights.sum()


@dataclass(frozen=True)
class BandMember:
    shape: dict
    l2: float
    oof_rmse: float
    weight: float


@dataclass(frozen=True)
class BandEnsemble:
    """A set of configurations the criterion cannot separate, with prediction weights."""

    model: Model
    target: str
    members: tuple[BandMember, ...]
    best_oof_rmse: float

    @property
    def size(self) -> int:
        return len(self.members)

    def predict(self, fit_panel: Panel, rows: np.ndarray | None, target_panel: Panel) -> np.ndarray:
        """Fit every member on ``fit_panel`` rows and average their predictions elsewhere."""
        total = None
        for member in self.members:
            fitted = fit_on(fit_panel, self.model, self.target, member.shape, member.l2, rows=rows)
            contribution = member.weight * fitted.predict(target_panel)
            total = contribution if total is None else total + contribution
        assert total is not None, "empty ensemble"
        return total


def build_band(
    panel: Panel,
    model: Model,
    target: str,
    rows: np.ndarray | None = None,
    weighting: str = "uniform",
    band_sigma: float = BAND_SIGMA,
) -> BandEnsemble:
    """Collect the configurations within ``band_sigma`` run sigma of the best out-of-fold error.

    Selection uses only the supplied rows, so a censored evaluation set never informs the
    band or its weights.
    """
    observed = panel.targets[target]
    if rows is not None:
        observed = np.where(rows, observed, np.nan)

    # Deduplicate configurations that produce a byte-identical design. Two designs in this
    # family ignore one of their nominal shape keys -- hierarchical replay never reads
    # ``rate`` and compact retained state never reads ``penalty_threshold`` -- so the nominal
    # grid contains exact duplicates. Duplicates are harmless for a uniform average but make
    # the stacking design exactly collinear, which lets the weight land arbitrarily on one of
    # an identical pair and inflates the apparent band size.
    scored: list[tuple[float, dict, float]] = []
    seen: set[tuple[bytes, float]] = set()
    for shape in model.shapes():
        signature = model.build(panel, shape).matrix.tobytes()
        for l2 in model.l2_grid:
            if (signature, l2) in seen:
                continue
            seen.add((signature, l2))
            prediction = out_of_fold_predictions(panel, model, target, shape, l2, rows=rows)
            finite = np.isfinite(prediction) & np.isfinite(observed)
            score = float(np.sqrt(np.mean((prediction[finite] - observed[finite]) ** 2)))
            scored.append((score, shape, l2))

    scored.sort(key=lambda item: item[0])
    best = scored[0][0]
    cutoff = best + band_sigma * RUN_SIGMA[target]
    inside = [item for item in scored if item[0] <= cutoff][:MAX_MEMBERS]

    if weighting == "uniform":
        weights = np.ones(len(inside)) / len(inside)
    elif weighting == "softmin":
        excess = np.array([score - best for score, _, _ in inside])
        raw = np.exp(-excess / max(band_sigma * RUN_SIGMA[target], 1e-12))
        weights = raw / raw.sum()
    elif weighting == "stacked":
        weights = stack_weights(panel, model, target, inside, rows)
    else:
        raise ValueError(f"unknown weighting {weighting!r}")

    members = tuple(
        BandMember(shape=shape, l2=l2, oof_rmse=score, weight=float(weight))
        for (score, shape, l2), weight in zip(inside, weights, strict=True)
    )
    return BandEnsemble(model=model, target=target, members=members, best_oof_rmse=best)


def out_of_fold_ensemble(panel: Panel, ensemble: BandEnsemble, rows: np.ndarray | None = None) -> np.ndarray:
    """Grouped out-of-fold prediction of the ensemble, averaging members fold by fold."""
    total = None
    for member in ensemble.members:
        prediction = out_of_fold_predictions(panel, ensemble.model, ensemble.target, member.shape, member.l2, rows=rows)
        contribution = member.weight * prediction
        total = contribution if total is None else total + contribution
    assert total is not None, "empty ensemble"
    return total
