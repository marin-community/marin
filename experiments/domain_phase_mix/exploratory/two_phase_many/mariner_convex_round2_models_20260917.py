# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Distinct convex response geometries and an isolated softplus-link fitter."""

from __future__ import annotations

import dataclasses
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit

from experiments.domain_phase_mix.exploratory.two_phase_many import mariner_convex_candidates_20260917 as prior
from experiments.domain_phase_mix.exploratory.two_phase_many import single_phase_observatory_models_20260902 as base
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_registry_20260902 as registry,
)

SOURCE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
RAW_SCALE = 32.0
OVERLOAD_KNOTS = (1.0, 4.0, 16.0)


@dataclasses.dataclass(frozen=True)
class CandidateSpec:
    model_id: str
    label: str
    benefit: str
    interaction: str
    link: str
    equation: str
    proof: str


def candidate_specs() -> tuple[CandidateSpec, ...]:
    proof = "Nonnegative sum of convex columns composed with a convex increasing outer link."
    return (
        CandidateSpec(
            "cv2_mariner", "MARINER control", "weibull", "mariner", "exp", "canonical MARINER", "No convexity guarantee."
        ),
        CandidateSpec(
            "cv2_linear", "Convex linear control", "weibull", "none", "exp", "exp(c-alpha*b+beta*E/32)", proof
        ),
        CandidateSpec(
            "cv2_raw_additive", "Raw additive convex", "weibull", "none", "raw", "max(c-alpha*b+beta*E/32,0)", proof
        ),
        CandidateSpec(
            "cv2_softplus",
            "Softplus outer link",
            "weibull",
            "none",
            "softplus",
            "phi+A*softplus(c-alpha*b+beta*E/32); phi>=0,A=anchor-phi",
            proof,
        ),
        CandidateSpec(
            "cv2_log_benefit",
            "Logarithmic benefit",
            "log",
            "none",
            "exp",
            "b=log(1+rho*E),h=E/32",
            proof + " log1p is concave.",
        ),
        CandidateSpec(
            "cv2_shifted_power",
            "Shifted-power benefit",
            "power",
            "none",
            "exp",
            "b=(1+rho*E)^kappa-1,h=E/32",
            proof + " Shifted power is concave for0<kappa<=1.",
        ),
        CandidateSpec(
            "cv2_inverse_power",
            "Inverse-power saturation",
            "inverse",
            "none",
            "exp",
            "b=1-(1+rho*E)^(-kappa),h=E/32",
            proof + " b''=-kappa*(kappa+1)*rho^2*(1+rho*E)^(-kappa-2)<=0.",
        ),
        CandidateSpec(
            "cv2_global_overload",
            "Global epoch overload",
            "weibull",
            "global",
            "exp",
            "g=c-alpha*b+beta*E/32+sum_k gamma_k*(mean(E)-q_k)+/32",
            proof + " A hinge of an affine functional is convex.",
        ),
        CandidateSpec(
            "cv2_family_overload",
            "Family epoch overload",
            "weibull",
            "family",
            "exp",
            "g=c-alpha*b+beta*E/32+sum_fk gamma_fk*(mean_family(E)-q_k)+/32",
            proof + " Family means are fixed affine functionals.",
        ),
        CandidateSpec(
            "cv2_family_imbalance",
            "Family epoch imbalance",
            "weibull",
            "variance",
            "exp",
            "g=c-alpha*b+beta*E/32+sum_f gamma_f*sum_i(E_i-mean_family(E))^2/32^2",
            proof + " Squared centered norms have PSD Hessians.",
        ),
    )


def benefit(exposure: np.ndarray, rate: float, power: float, kind: str) -> np.ndarray:
    if not 0.0 < power <= 1.0:
        raise ValueError("Convex candidates require power in (0,1].")
    if kind == "weibull":
        return base.weibull_response(exposure, rate, power)
    logged = np.log1p(rate * exposure)
    if kind == "log":
        return logged
    if kind == "power":
        return np.expm1(power * logged)
    if kind == "inverse":
        return -np.expm1(-power * logged)
    raise ValueError(kind)


def design(spec: CandidateSpec, features: base.Features, shape: base.Shape) -> base.Design:
    if spec.interaction == "mariner":
        return prior.mariner_control_design(features, shape, ())
    signal = benefit(features.exposures, float(shape["rate"]), float(shape.get("power", 1.0)), spec.benefit)
    columns = [-signal, features.exposures / RAW_SCALE]
    names = [f"benefit:{i}" for i in range(features.buckets)] + [f"linear:{i}" for i in range(features.buckets)]
    if spec.interaction == "global":
        mean = features.exposures.mean(axis=1, keepdims=True)
        for knot in OVERLOAD_KNOTS:
            columns.append(np.maximum(mean - knot, 0.0) / RAW_SCALE)
            names.append(f"global_overload:{knot:g}")
    if spec.interaction in ("family", "variance"):
        for family in features.families.nonsingleton:
            members = features.families.members[family]
            epochs = features.exposures[:, members]
            mean = epochs.mean(axis=1, keepdims=True)
            if spec.interaction == "family":
                for knot in OVERLOAD_KNOTS:
                    columns.append(np.maximum(mean - knot, 0.0) / RAW_SCALE)
                    names.append(f"family_overload:{family}:{knot:g}")
            else:
                columns.append(np.sum((epochs - mean) ** 2, axis=1, keepdims=True) / RAW_SCALE**2)
                names.append(f"family_imbalance:{family}")
    return prior.design_columns(columns, names)


def inverse_softplus(value: np.ndarray) -> np.ndarray:
    """Stable inverse for strictly positive softplus outputs."""
    if np.any(value <= 0.0) or not np.isfinite(value).all():
        raise ValueError("Inverse softplus requires positive finite values.")
    return value + np.log(-np.expm1(-value))


@dataclasses.dataclass(frozen=True)
class LinkedHead:
    linear: base.FittedHead
    floor: float
    scale: float
    kappa: float
    link: str

    @property
    def converged(self) -> bool:
        return self.linear.converged


def linked_predict(head: LinkedHead, matrix: np.ndarray) -> np.ndarray:
    linear = head.linear.intercept + matrix @ head.linear.coefficients
    if head.link == "raw":
        return np.maximum(linear, 0.0)
    return head.floor + head.scale * np.logaddexp(0.0, linear)


def linked_head(
    design: base.Design, response: np.ndarray, ridge: float, spec: base.HeadSpec, link: str, kappa: float
) -> LinkedHead:
    if link == "raw":
        floor, scale, target = 0.0, 1.0, response
    else:
        anchor, gap = base.floor_anchor_and_gap(response, spec)
        floor = max(base.fitted_floor(kappa, anchor, gap, spec), 0.0)
        scale = max(anchor - floor, 1e-9)
        target = inverse_softplus(np.maximum((response - floor) / scale, 1e-9))
    linear = base.fit_head(design, target, ridge, dataclasses.replace(spec, link=base.LinkKind.IDENTITY))
    return LinkedHead(linear, floor, scale, kappa, link)


def linked_cv(
    design: base.Design,
    response: np.ndarray,
    ridge: float,
    spec: base.HeadSpec,
    link: str,
    kappa: float,
    inner: base.InnerFolds,
) -> float:
    error, count = 0.0, 0
    for train, validation in inner:
        head = linked_head(
            base.Design(design.values[train], design.ridge, design.names), response[train], ridge, spec, link, kappa
        )
        predicted = linked_predict(head, design.values[validation])
        error += float(np.sum((predicted - response[validation]) ** 2))
        count += len(validation)
    return math.sqrt(error / count)


@dataclasses.dataclass(frozen=True)
class LinkedConvexModel:
    base: base.GridModel
    link: str
    floor_bounds: tuple[float, float]
    floor_evaluations: int
    flat_profile_kappa: float | None

    @property
    def model_id(self) -> str:
        return self.base.model_id

    def selected_head(
        self,
        features: base.Features,
        response: np.ndarray,
        train: np.ndarray,
        shape: base.Shape,
        ridge: float,
        kappa: float,
    ) -> LinkedHead:
        matrix = self.base.design(features, shape)
        return linked_head(
            base.Design(matrix.values[train], matrix.ridge, matrix.names),
            response[train],
            ridge,
            self.base.head_for(shape),
            self.link,
            kappa,
        )

    def fit(
        self, features: base.Features, response: np.ndarray, train: np.ndarray, inner: base.InnerFolds, seed: int
    ) -> base.Fitted:
        del seed
        shapes = self.base.candidate_shapes(features)
        table = np.full((len(shapes), len(self.base.ridge_grid)), np.inf)
        prior_kappa = self.base.head.floor_kappa_prior
        for i, shape in enumerate(shapes):
            matrix = self.base.design(features, shape)
            for j, ridge in enumerate(self.base.ridge_grid):
                table[i, j] = linked_cv(
                    matrix, response, ridge, self.base.head_for(shape), self.link, prior_kappa, inner
                )
        if not np.isfinite(table).any():
            raise ValueError(f"{self.model_id}: no finite inner-CV candidate")
        i, j = np.unravel_index(int(np.argmin(table)), table.shape)
        shape, ridge = dict(shapes[i]), self.base.ridge_grid[j]
        matrix = self.base.design(features, shape)
        selected_cv, kappa, evaluations = float(table[i, j]), prior_kappa, 0
        if self.link == "softplus":

            def objective(log_kappa: float) -> float:
                return linked_cv(
                    matrix, response, ridge, self.base.head_for(shape), self.link, math.exp(log_kappa), inner
                )

            lower, upper = self.floor_bounds
            result = minimize_scalar(
                objective,
                bounds=(math.log(lower), math.log(upper)),
                method="bounded",
                options={"maxiter": self.floor_evaluations, "xatol": 0.02},
            )
            kappa, evaluations = float(math.exp(result.x)), int(result.nfev)
            if self.flat_profile_kappa is not None and math.log(kappa) >= (1 - base.FLAT_PROFILE_FRACTION) * math.log(
                upper
            ):
                kappa = self.flat_profile_kappa
            final_cv = objective(math.log(kappa))
        else:
            final_cv = selected_cv
        head = self.selected_head(features, response, train, shape, ridge, kappa)
        return base.Fitted(
            shape,
            ridge,
            head,
            {
                "inner_cv_rmse": final_cv,
                "selection_inner_cv_rmse": selected_cv,
                "link": self.link,
                "kappa": kappa,
                "floor": head.floor,
                "scale": head.scale,
                "converged": True,
                "columns": matrix.values.shape[1],
                "fitted_dof": head.linear.active + 1 + self.nonlinear_dof(features),
                "nonlinear_dof": self.nonlinear_dof(features),
                "kappa_search_evaluations": evaluations,
            },
            table,
        )

    def predict(self, fitted: base.Fitted, features: base.Features, rows: np.ndarray) -> np.ndarray:
        return linked_predict(fitted.head, self.base.design(features, fitted.shape).values[rows])

    def nonlinear_dof(self, features: base.Features) -> int:
        return self.base.nonlinear_dof(features) + int(self.link == "softplus")


def build_model(spec: CandidateSpec, features: base.Features) -> base.FittedFloorModel | LinkedConvexModel:
    frozen = registry.ENTRY_BY_ID[registry.FROZEN_ID].build(features)
    if not isinstance(frozen, base.FittedFloorModel) or not isinstance(frozen.base, base.GridModel):
        raise TypeError("Expected canonical fitted-floor grid.")
    keys = ("rate", "power", "threshold") if spec.interaction == "mariner" else ("rate", "power")
    if spec.benefit == "log":
        keys = ("rate",)
    shapes = tuple(
        dict(values) for values in dict.fromkeys(tuple((key, float(s[key])) for key in keys) for s in frozen.base.shapes)
    )
    identity = (spec, SOURCE_HASH)

    def builder(query: base.Features, shape: base.Shape) -> base.Design:
        return design(identity[0], query, shape)

    grid = dataclasses.replace(frozen.base, model_id=spec.model_id, builder=builder, shapes=shapes, shape_dof=len(keys))
    if spec.link == "exp":
        return dataclasses.replace(frozen, base=grid)
    return LinkedConvexModel(grid, spec.link, frozen.kappa_bounds, frozen.search_evaluations, frozen.flat_profile_kappa)


def design_epoch_jacobian(spec: CandidateSpec, features: base.Features, shape: base.Shape) -> np.ndarray:
    """Analytic basis Jacobian: rows by columns by raw-epoch coordinates."""
    matrix = design(spec, features, shape)
    jacobian = np.zeros((features.rows, matrix.values.shape[1], features.buckets))
    rate, power = float(shape["rate"]), float(shape.get("power", 1.0))
    epochs = features.exposures
    unit = 1.0 + rate * epochs
    if spec.benefit == "weibull":
        negative_derivative = prior.negative_benefit_derivative(epochs, rate, power)
    elif spec.benefit == "log":
        negative_derivative = -rate / unit
    elif spec.benefit == "power":
        negative_derivative = -power * rate * unit ** (power - 1.0)
    elif spec.benefit == "inverse":
        negative_derivative = -power * rate * unit ** (-power - 1.0)
    else:
        raise ValueError(spec.benefit)
    for i in range(features.buckets):
        jacobian[:, i, i] = negative_derivative[:, i]
        if spec.interaction == "mariner":
            denominator = 1.0 + np.exp(float(shape["threshold"])) + epochs[:, i]
            jacobian[:, features.buckets + i, i] = 2 * (np.log(denominator) - float(shape["threshold"])) / denominator
        else:
            jacobian[:, features.buckets + i, i] = 1.0 / RAW_SCALE
    column = 2 * features.buckets
    if spec.interaction == "global":
        mean = epochs.mean(axis=1)
        for knot in OVERLOAD_KNOTS:
            jacobian[:, column, :] = ((mean > knot) / (features.buckets * RAW_SCALE))[:, None]
            column += 1
    if spec.interaction in ("family", "variance"):
        for family in features.families.nonsingleton:
            members = features.families.members[family]
            mean = epochs[:, members].mean(axis=1)
            if spec.interaction == "family":
                for knot in OVERLOAD_KNOTS:
                    jacobian[:, column, members] = ((mean > knot) / (len(members) * RAW_SCALE))[:, None]
                    column += 1
            else:
                jacobian[:, column, members] = 2 * (epochs[:, members] - mean[:, None]) / RAW_SCALE**2
                column += 1
    return jacobian


def linear_and_weight_gradient(
    spec: CandidateSpec, fitted: base.Fitted, features: base.Features
) -> tuple[np.ndarray, np.ndarray]:
    """Return uncapped linear predictor and gradient for E_i=inventory_i*w_i.

    Fractional Weibull benefits have an infinite right derivative at zero. Zero
    coefficients are removed before contraction, avoiding spurious zero*infinity.
    """
    head = fitted.head.linear if isinstance(fitted.head, LinkedHead) else fitted.head
    matrix = design(spec, features, fitted.shape)
    active = head.coefficients > 0.0
    jacobian = design_epoch_jacobian(spec, features, fitted.shape)
    gradient = np.einsum("rcb,c->rb", jacobian[:, active, :], head.coefficients[active])
    return head.intercept + matrix.values @ head.coefficients, gradient * features.inventory[None, :]


def prediction_and_weight_gradient(
    spec: CandidateSpec, fitted: base.Fitted, features: base.Features
) -> tuple[np.ndarray, np.ndarray]:
    """Uncapped mathematical response and analytic mixture gradient."""
    linear, gradient = linear_and_weight_gradient(spec, fitted, features)
    if spec.link == "raw":
        return np.maximum(linear, 0.0), gradient * (linear > 0.0)[:, None]
    if spec.link == "softplus":
        head = fitted.head
        return head.floor + head.scale * np.logaddexp(0.0, linear), gradient * (head.scale * expit(linear))[:, None]
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        exponential = np.exp(linear)
    return fitted.head.floor + exponential, gradient * exponential[:, None]
