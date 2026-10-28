# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The MARINER surrogate: the procedure's inputs, the response functions and the fitted model.

The materialized epochs E_i(w) = w_i * (D_tgt / P_i) / pf_i count the passes over bucket i at weight w_i, for a run
that draws from the fraction pf_i of the bucket's pool of P_i tokens. Per task t, the surrogate is a floored
log-deficit model on the shared response functions,
f_t(w) = phi_t + exp(c_t - sum_i alpha_ti b_t(E_i) + sum_i beta_ti h_t(E_i)) with alpha, beta >= 0,
b_t(E) = 1 - exp(-(rho_t E)^kappa_t) and h_t(E) = softplus(log(1 + E) - tau_t)^2. An objective aggregates its tasks
as sum_t a_t f_t(w) with the evaluation's own weights a_t.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

LOG_CLIP = 30.0  # numerical overflow guard on the linear predictor; not a modeling constant
WEIGHT_SUM_TOLERANCE = 1e-6  # how far a mixture's weights may sum from one


# ---------------------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Swarm:
    """A proxy swarm: runs x buckets weights and pool fractions, the exposure scale per bucket, per-task outcomes."""

    runs: tuple[str, ...]  # observation ids
    buckets: tuple[str, ...]
    weights: np.ndarray  # (runs, buckets), rows sum to one
    epochs_per_unit_weight: np.ndarray  # (buckets,) D_tgt / P_i
    pool_fractions: np.ndarray  # (runs, buckets) fraction of each bucket's pool the run drew from
    outcomes: pd.DataFrame  # (runs, tasks) with task names as columns
    calibration: np.ndarray  # (runs,) bool: rows that are calibration data

    @property
    def exposures(self) -> np.ndarray:
        return self.weights * self.epochs_per_unit_weight[None, :] / self.pool_fractions

    def subset(self, rows: np.ndarray) -> Swarm:
        """The runs at the integer positions ``rows``, in that order."""
        return dataclasses.replace(
            self,
            runs=tuple(self.runs[row] for row in rows),
            weights=self.weights[rows],
            pool_fractions=self.pool_fractions[rows],
            outcomes=self.outcomes.iloc[rows].reset_index(drop=True),
            calibration=self.calibration[rows],
        )


@dataclasses.dataclass(frozen=True)
class Objective:
    name: str
    tasks: tuple[str, ...]
    weights: np.ndarray  # evaluation weights a_t (byte shares or 1/n)


@dataclasses.dataclass(frozen=True)
class Anchor:
    proportional: float  # the proportional mixture's mean over its repeats
    repeat_sd: float  # its repeat SD (0 disables the noise margin)


# ---------------------------------------------------------------------------------------------------------------
# Response functions and design
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Shape:
    """A task's response shape: benefit rate rho, Weibull power kappa and harm threshold tau."""

    rho: float
    kappa: float
    tau: float


def benefit(exposure: np.ndarray, rho: float | np.ndarray, kappa: float | np.ndarray) -> np.ndarray:
    return -np.expm1(-((np.maximum(rho * exposure, 0.0)) ** kappa))


def harm(exposure: np.ndarray, tau: float | np.ndarray) -> np.ndarray:
    return np.logaddexp(np.log1p(np.maximum(exposure, 0.0)) - tau, 0.0) ** 2


def design_matrix(exposures: np.ndarray, shape: Shape) -> np.ndarray:
    """Columns: minus the benefit of every bucket, then the harm of every bucket (ridge multiplier one each)."""
    return np.hstack([-benefit(exposures, shape.rho, shape.kappa), harm(exposures, shape.tau)])


# ---------------------------------------------------------------------------------------------------------------
# Fitted surrogate: the per-task heads and the objective's vectorized prediction
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Head:
    intercept: float
    coefficients: np.ndarray
    phi: float

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        linear = self.intercept + matrix @ self.coefficients
        return self.phi + np.exp(np.clip(linear, -LOG_CLIP, LOG_CLIP))


@dataclasses.dataclass(frozen=True)
class TaskFit:
    task: str
    shape: Shape
    gamma: float
    nu: float
    flat_profile: bool
    inner_cv_rmse: float
    head: Head

    def to_json(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "rho": self.shape.rho,
            "kappa": self.shape.kappa,
            "tau": self.shape.tau,
            "gamma": self.gamma,
            "nu": self.nu,
            "phi": self.head.phi,
            "flat_profile": self.flat_profile,
            "inner_cv_rmse": self.inner_cv_rmse,
            "intercept": self.head.intercept,
            "coefficients": self.head.coefficients.tolist(),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> TaskFit:
        shape = Shape(float(payload["rho"]), float(payload["kappa"]), float(payload["tau"]))
        head = Head(float(payload["intercept"]), np.asarray(payload["coefficients"], float), float(payload["phi"]))
        return cls(
            str(payload["task"]),
            shape,
            float(payload["gamma"]),
            float(payload["nu"]),
            bool(payload["flat_profile"]),
            float(payload["inner_cv_rmse"]),
            head,
        )


@dataclasses.dataclass(frozen=True)
class ObjectiveFit:
    objective: str
    buckets: tuple[str, ...]
    epochs_per_unit_weight: np.ndarray
    weights: np.ndarray  # evaluation weights per task
    tasks: tuple[TaskFit, ...]
    inner_fold_labels: dict[str, int]  # inner fold validating each run, by observation id; -1: never (calibration)

    def _arrays(self) -> tuple[np.ndarray, ...]:
        rho = np.asarray([t.shape.rho for t in self.tasks])
        kappa = np.asarray([t.shape.kappa for t in self.tasks])
        tau = np.asarray([t.shape.tau for t in self.tasks])
        intercept = np.asarray([t.head.intercept for t in self.tasks])
        coefficients = np.stack([t.head.coefficients for t in self.tasks])
        phi = np.asarray([t.head.phi for t in self.tasks])
        return rho, kappa, tau, intercept, coefficients, phi

    def predict_tasks(self, mixtures: np.ndarray, pool_fractions: np.ndarray | None = None) -> np.ndarray:
        """Per-task predictions for every mixture row, shape (mixtures, tasks).

        ``pool_fractions``, broadcast against ``mixtures``, divide the exposures as in ``Swarm.exposures``; omitted,
        every mixture draws from whole pools.
        """
        rho, kappa, tau, intercept, coefficients, phi = self._arrays()
        exposure = np.atleast_2d(mixtures) * self.epochs_per_unit_weight[None, :]
        if pool_fractions is not None:
            exposure = exposure / pool_fractions
        exposure = exposure[:, None, :]
        columns = np.concatenate(
            [-benefit(exposure, rho[None, :, None], kappa[None, :, None]), harm(exposure, tau[None, :, None])],
            axis=2,
        )
        linear = intercept[None, :] + np.einsum("nck,ck->nc", columns, coefficients)
        return phi[None, :] + np.exp(np.clip(linear, -LOG_CLIP, LOG_CLIP))

    def predict(self, mixtures: np.ndarray, pool_fractions: np.ndarray | None = None) -> np.ndarray:
        return self.predict_tasks(mixtures, pool_fractions) @ self.weights

    def to_json(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "buckets": list(self.buckets),
            "epochs_per_unit_weight": self.epochs_per_unit_weight.tolist(),
            "task_weights": self.weights.tolist(),
            "tasks": [t.to_json() for t in self.tasks],
            "inner_fold_labels": dict(self.inner_fold_labels),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ObjectiveFit:
        return cls(
            str(payload["objective"]),
            tuple(payload["buckets"]),
            np.asarray(payload["epochs_per_unit_weight"], float),
            np.asarray(payload["task_weights"], float),
            tuple(TaskFit.from_json(t) for t in payload["tasks"]),
            {str(run): int(label) for run, label in payload["inner_fold_labels"].items()},
        )
