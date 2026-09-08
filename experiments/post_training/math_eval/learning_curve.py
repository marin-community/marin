# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Descriptive two-parameter sigmoid fits on the optimizer-update axis."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


def fit_sigmoid(updates, completion_rates):
    """Fit p(u)=sigmoid(A+B*u/200) after at least 200 updates and five checkpoints.

    Checkpoint proportions have equal weight. The fit is descriptive: overlapping
    questions and correlated updates do not create independent observations or
    justify coefficient confidence intervals. No extrapolated time-to-target is
    reported. Boundary-only rates and nonidentifiable fits return explicit status.
    """
    u = np.asarray(updates, dtype=float)
    y = np.asarray(completion_rates, dtype=float)
    if u.ndim != 1 or y.shape != u.shape or not len(u):
        raise ValueError("Provide aligned one-dimensional update and completion vectors")
    if (
        not np.all(np.isfinite(u))
        or not np.all(np.isfinite(y))
        or np.any(u < 0)
        or np.any(np.diff(u) <= 0)
        or np.any((y < 0) | (y > 1))
    ):
        raise ValueError("Updates must increase and completion rates must be finite probabilities")
    base = {
        "definition": "p(update)=sigmoid(A+B*update/200)",
        "axis": "optimizer_updates",
        "weighting": "equal checkpoint proportions",
        "coefficient_intervals": None,
        "checkpoint_count": len(u),
        "observed_update_range": (float(u[0]), float(u[-1])),
    }
    if u[-1] - u[0] < 200:
        return base | {"status": "requires_200_update_span"}
    if len(u) < 5:
        return base | {"status": "requires_five_checkpoints"}
    if np.all((y == 0) | (y == 1)):
        return base | {"status": "boundary_only_rates_not_identifiable"}
    x = np.column_stack((np.ones_like(u), u / 200))

    def loss(params):
        logits = x @ params
        return float(np.mean(np.logaddexp(0, logits) - y * logits))

    def gradient(params):
        return x.T @ (expit(x @ params) - y) / len(y)

    result = minimize(
        loss,
        [float(logit(np.clip(y.mean(), 1e-6, 1 - 1e-6))), 0],
        jac=gradient,
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 1000},
    )
    prediction = expit(x @ result.x)
    hessian = (x.T * (prediction * (1 - prediction))) @ x / len(y)
    if np.linalg.cond(hessian) > 1e12 or np.linalg.norm(gradient(result.x)) > 1e-7:
        return base | {"status": "nonconverged_or_nonidentifiable"}
    return base | {
        "status": "fit",
        "A": float(result.x[0]),
        "B": float(result.x[1]),
        "rmse": float(np.sqrt(np.mean((prediction - y) ** 2))),
        "predicted_at_observed_updates": prediction.tolist(),
    }
