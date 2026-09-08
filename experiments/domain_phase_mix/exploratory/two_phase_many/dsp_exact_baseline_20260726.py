# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
#   "scipy",
# ]
# ///
"""The real Observatory effective-exposure DSP as a swarm39 harness baseline.

Replaces the strawman ``build_effective_exposure_dsp`` in
``swarm39_models_20260725``, which used a single shared power exponent and 43
design columns. The Observatory model is ``standalone_code/dsp_exact.py``: it
fits a per-bucket saturation rate ``rho`` and a per-bucket overexposure
threshold ``tau`` by nonlinear optimization, plus separate per-bucket benefit
and penalty coefficients and a shared late multiplier ``gamma``. At 39 buckets
that is 39*4 + 2 = 158 parameters against the strawman's 43 columns.

The two differ structurally, not only in count. The strawman's benefit is
``exposure ** power``, which is unbounded, so an optimizer can keep gaining by
piling mass into one bucket. The real benefit is ``1 - exp(-rho * exposure)``,
bounded in [0, 1] and saturating at a fitted per-bucket rate, so each bucket's
contribution caps out. That difference bears directly on out-of-support
extrapolation and therefore on where the raw optimum lands.

Fidelity
--------
``dsp_exact`` profiles out the linear head at every nonlinear step, so given the
fitted ``(rho, tau, gamma)`` the design is exactly ``[-signal, penalty]`` and the
head is its own uncolumn-scaled NNLS. This module therefore carries the fitted
nonlinear parameters as the harness ``shape`` and reuses ``dsp_exact``'s own
linear head rather than the harness's column-scaled one. ``verify_adapter``
asserts that ``Fit.predict`` reproduces ``dsp_exact.predict`` to machine
precision; every entry point calls it before returning a fit.

Exposure units
--------------
The harness rescales the catalog multipliers so the proportional policy lands on
0.905353 epochs. The DSP packet's own 39-bucket metadata is the same catalog
scaled by 4021.44 against the harness's 4012.08, a 0.23 percent difference that
``rho`` absorbs and that leaves it far from its [1e-4, 2] bounds. The harness
panel can therefore be handed to ``dsp_exact`` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# dsp_exact is a standalone module rather than an installed package, so its
# directory joins the path before it is imported.
sys.path.insert(0, str(Path(__file__).resolve().parent / "standalone_code"))

import dsp_exact as dsp
from swarm39_harness_20260725 import Design, Fit, Model, Panel, grouped_splits

VARIANT_NAME = "effective_exposure"
MODEL_NAME = "effective_exposure_dsp"
BASIN_HOPPING_ITERS = 3
PREDICT_TOLERANCE = 1e-9


def _variant() -> dsp.DSPVariant:
    return dsp.VARIANTS[VARIANT_NAME]


def _weights(panel: Panel) -> np.ndarray:
    return np.stack([panel.phase0, panel.phase1], axis=1)


def _packet(panel: Panel, target: str) -> dsp.PacketData:
    observed = panel.targets[target]
    return dsp.PacketData(
        frame=pd.DataFrame({"run_name": panel.row_id, "objective_metric": observed}),
        name_col="run_name",
        y=observed,
        w=_weights(panel),
        m=len(panel.buckets),
        c0=panel.c0,
        c1=panel.c1,
        domain_names=list(panel.buckets),
    )


def _shape_from_params(params: dict) -> dict:
    return {
        "rho": np.asarray(params["rho"], dtype=float).tolist(),
        "tau": np.asarray(params["tau"], dtype=float).tolist(),
        "gamma": float(params["gamma"]),
    }


def _params_from_shape(shape: dict) -> dict:
    return {
        "rho": np.asarray(shape["rho"], dtype=float),
        "tau": np.asarray(shape["tau"], dtype=float),
        "gamma": float(shape["gamma"]),
    }


def build_dsp_exact(panel: Panel, shape: dict) -> Design:
    """Design for fixed nonlinear parameters, matching ``dsp_exact.fit_linear_head``."""
    signal, penalty = dsp.features(_weights(panel), panel.c0, panel.c1, _variant(), _params_from_shape(shape))
    return Design(
        matrix=np.hstack([-signal, penalty]),
        names=tuple(
            [
                *(f"dsp_benefit:{bucket}" for bucket in panel.buckets),
                *(f"dsp_penalty:{bucket}" for bucket in panel.buckets),
            ]
        ),
    )


def dsp_exact_model() -> Model:
    """Harness model whose shape is fitted per panel rather than chosen from a grid."""
    return Model(name=MODEL_NAME, build=build_dsp_exact, shapes=lambda: ({},))


def _fit_once(panel: Panel, target: str) -> tuple[dsp.FittedDSPModel, dict]:
    fitted, _ = dsp.fit_variant(
        _packet(panel, target),
        _variant(),
        maxiter=dsp.FIT_MAXITER,
        coarse_top_k=dsp.START_TOP_K,
        basin_hopping_iters=BASIN_HOPPING_ITERS,
    )
    return fitted, _shape_from_params(fitted.params)


def _as_fit(fitted: dsp.FittedDSPModel, shape: dict, panel: Panel, oof_rmse: float) -> Fit:
    return Fit(
        model=MODEL_NAME,
        shape=shape,
        l2=dsp.LINEAR_REG,
        intercept=float(fitted.intercept),
        coefficients=np.concatenate([fitted.benefit_coef, fitted.penalty_coef]),
        names=build_dsp_exact(panel, shape).names,
        oof_rmse=oof_rmse,
    )


def verify_adapter(fit: Fit, fitted: dsp.FittedDSPModel, panel: Panel) -> float:
    """Assert the harness fit reproduces ``dsp_exact.predict`` exactly."""
    mine = fit.predict(panel, dsp_exact_model())
    theirs = dsp.predict(fitted, _weights(panel))
    worst = float(np.max(np.abs(mine - theirs)))
    assert worst < PREDICT_TOLERANCE, f"adapter diverges from dsp_exact by {worst:.3e}"
    return worst


def fit_dsp_exact(panel: Panel, target: str, n_splits: int = 5, seed: int = 0) -> Fit:
    """Fit on the panel and score grouped out-of-fold RMSE.

    The nonlinear parameters are refitted inside every fold rather than carried
    over from the full-panel fit, so the out-of-fold number is not leaked into by
    the 78 per-bucket parameters.
    """
    usable = np.isfinite(panel.targets[target])
    panel = panel.subset(usable)
    errors = []
    for train, test in grouped_splits(panel, n_splits, seed):
        # grouped_splits yields boolean masks, not integer indices.
        assert train.dtype == bool and len(train) == len(panel), "expected a boolean fold mask"
        train_fit, _ = _fit_once(panel.subset(train), target)
        held = panel.subset(test)
        errors.append(dsp.predict(train_fit, _weights(held)) - held.targets[target])
    oof_rmse = float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))
    fitted, shape = _fit_once(panel, target)
    fit = _as_fit(fitted, shape, panel, oof_rmse)
    verify_adapter(fit, fitted, panel)
    return fit
