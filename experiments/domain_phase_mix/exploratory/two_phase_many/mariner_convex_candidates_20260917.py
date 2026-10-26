# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convex-in-mixture response families for the archived MARINER screen.

For nonnegative coefficients, each new column is convex in raw epochs: a
negative concave Weibull benefit or a convex increasing harm. Epochs are affine
in mixture weights. The uncapped exponential of their sum and positive weighted
sums across tasks are therefore convex. This is a guarantee for mixture
optimization after fitting, not joint convexity of nonlinear parameter fitting.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.special import expit

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    mariner_harm_screen_models_20260917 as prior,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_registry_20260902 as registry,
)

RAW_SCALE = 32.0
MODULE_SOURCE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DesignFunction = Callable[[base.Features, base.Shape, tuple[float, ...]], base.Design]


@dataclasses.dataclass(frozen=True)
class CandidateSpec:
    model_id: str
    label: str
    category: str
    equation: str
    proof: str
    params: tuple[float, ...]
    shape_keys: tuple[str, ...]
    design_function: DesignFunction
    guaranteed_convex: bool = True


def design_columns(columns: list[np.ndarray], names: list[str]) -> base.Design:
    matrix = np.concatenate(columns, axis=1)
    return base.Design(matrix, np.ones(matrix.shape[1]), tuple(names))


def benefit_columns(features: base.Features, shape: base.Shape) -> tuple[list[np.ndarray], list[str]]:
    power = float(shape["power"])
    if not 0.0 < power <= 1.0:
        raise ValueError("Convex mixture objectives require Weibull power in (0, 1].")
    benefit = base.weibull_response(features.exposures, float(shape["rate"]), power)
    return [-benefit], [f"benefit:{i}" for i in range(features.buckets)]


def append_harm(features: base.Features, shape: base.Shape, harm: np.ndarray, name: str) -> base.Design:
    columns, names = benefit_columns(features, shape)
    columns.append(harm)
    names.extend(f"{name}:{i}" for i in range(features.buckets))
    return design_columns(columns, names)


def linear_design(features: base.Features, shape: base.Shape, params: tuple[float, ...]) -> base.Design:
    del params
    return append_harm(features, shape, features.exposures / RAW_SCALE, "linear")


def smooth_hinge_design(features: base.Features, shape: base.Shape, params: tuple[float, ...]) -> base.Design:
    width = params[0]
    onset = np.expm1(float(shape["threshold"]))
    harm = (
        width * (np.logaddexp(0.0, (features.exposures - onset) / width) - np.logaddexp(0.0, -onset / width)) / RAW_SCALE
    )
    return append_harm(features, shape, harm, "smooth_hinge")


def hinge_design(features: base.Features, shape: base.Shape, params: tuple[float, ...]) -> base.Design:
    onset = np.expm1(float(shape["threshold"]))
    harm = (np.maximum(features.exposures - onset, 0.0) / RAW_SCALE) ** params[0]
    return append_harm(features, shape, harm, "hinge")


def huber_hinge(exposure: np.ndarray, onset: float, width: float) -> np.ndarray:
    """One-sided Huber loss, with a continuous derivative rising from zero to one."""
    excess = np.maximum(exposure - onset, 0.0)
    shoulder = np.minimum(excess, width)
    return (shoulder**2 / (2.0 * width) + np.maximum(excess - width, 0.0)) / RAW_SCALE


def huber_design(features: base.Features, shape: base.Shape, params: tuple[float, ...]) -> base.Design:
    harm = huber_hinge(features.exposures, float(np.expm1(shape["threshold"])), params[0])
    return append_harm(features, shape, harm, "huber_hinge")


def multiknot_design(features: base.Features, shape: base.Shape, params: tuple[float, ...]) -> base.Design:
    columns, names = benefit_columns(features, shape)
    for knot in params:
        columns.append(np.maximum(features.exposures - knot, 0.0) / RAW_SCALE)
        names.extend(f"hinge_at_{knot:g}:{i}" for i in range(features.buckets))
    return design_columns(columns, names)


def convex_bank_design(features: base.Features, shape: base.Shape, params: tuple[float, ...]) -> base.Design:
    del shape
    columns = []
    names = []
    for rate in (0.1, 0.5, 2.0):
        columns.append(-base.weibull_response(features.exposures, rate, 0.7))
        names.extend(f"benefit_rate_{rate:g}:{i}" for i in range(features.buckets))
    for knot in params:
        columns.append(np.maximum(features.exposures - knot, 0.0) / RAW_SCALE)
        names.extend(f"hinge_at_{knot:g}:{i}" for i in range(features.buckets))
    return design_columns(columns, names)


def mariner_control_design(features: base.Features, shape: base.Shape, params: tuple[float, ...]) -> base.Design:
    del params
    return prior.log_harm_design(features, shape, (("harm_power", 2.0),))


def old_convex_design(features: base.Features, shape: base.Shape, params: tuple[float, ...]) -> base.Design:
    del params
    return prior.convex_linear_design(features, shape, ())


def negative_benefit_derivative(exposure: np.ndarray, rate: float, power: float) -> np.ndarray:
    """Derivative on positive epochs; right derivative is -infinity at zero for power<1."""
    if power == 1.0:
        return -rate * np.exp(-rate * exposure)
    derivative = np.full_like(exposure, -np.inf, dtype=float)
    positive = exposure > 0.0
    epochs = exposure[positive]
    derivative[positive] = -power * rate**power * epochs ** (power - 1.0) * np.exp(-((rate * epochs) ** power))
    return derivative


def design_epoch_derivative(
    spec: CandidateSpec, features: base.Features, shape: base.Shape
) -> tuple[np.ndarray, np.ndarray]:
    """Return each column's own-bucket epoch derivative and its bucket index.

    At a hinge knot the left derivative is a valid subgradient. Negative Weibull
    derivatives diverge at zero for power<1; callers must not silently replace
    them with finite numbers when claiming an optimization certificate.
    """
    epochs = features.exposures
    if spec.category == "convex_response":
        columns = [negative_benefit_derivative(epochs, rate, 0.7) for rate in (0.1, 0.5, 2.0)]
        columns.extend((epochs > knot).astype(float) / RAW_SCALE for knot in spec.params)
    else:
        columns = [negative_benefit_derivative(epochs, float(shape["rate"]), float(shape["power"]))]
        if spec.category == "linear":
            columns.append(np.full_like(epochs, 1.0 / RAW_SCALE))
        elif spec.category == "multiknot":
            columns.extend((epochs > knot).astype(float) / RAW_SCALE for knot in spec.params)
        else:
            onset = np.expm1(float(shape["threshold"]))
            if spec.model_id == "cv17_mariner":
                denominator = 1.0 + np.exp(float(shape["threshold"])) + epochs
                columns.append(2.0 * (np.log(denominator) - float(shape["threshold"])) / denominator)
            elif spec.model_id == "cv17_old_convex":
                columns.append(expit((epochs - onset) / (1.0 + onset)) / (1.0 + onset))
            elif spec.category == "smooth_hinge":
                columns.append(expit((epochs - onset) / spec.params[0]) / RAW_SCALE)
            elif spec.category == "hinge":
                columns.append((epochs > onset).astype(float) / RAW_SCALE)
            elif spec.category == "quadratic":
                columns.append(2.0 * np.maximum(epochs - onset, 0.0) / RAW_SCALE**2)
            elif spec.category == "huber_hinge":
                columns.append(np.clip((epochs - onset) / spec.params[0], 0.0, 1.0) / RAW_SCALE)
            else:
                raise ValueError(f"Unknown derivative family: {spec.model_id}")
    return np.concatenate(columns, axis=1), np.tile(np.arange(features.buckets), len(columns))


def candidate_specs() -> tuple[CandidateSpec, ...]:
    """Return the preregistered concrete variants and their convexity arguments."""
    shared = ("rate", "power", "threshold")
    benefit = ("rate", "power")
    proof = "Negative Weibull is convex for 0<kappa<=1; nonnegative convex columns + uncapped exp preserve convexity."
    specifications = [
        CandidateSpec(
            "cv17_mariner",
            "MARINER",
            "control",
            "h=s^2-s(0)^2",
            "No convexity guarantee.",
            (),
            shared,
            mariner_control_design,
            False,
        ),
        CandidateSpec(
            "cv17_old_convex",
            "Old convex softplus",
            "control",
            "h=softplus((E-q)/(1+q))-h(0)",
            proof,
            (),
            shared,
            old_convex_design,
        ),
        CandidateSpec("cv17_linear", "Linear harm", "linear", "h=E/32", proof, (), benefit, linear_design),
    ]
    for width in (1.0, 8.0):
        specifications.append(
            CandidateSpec(
                f"cv17_smooth_d{width:g}",
                f"Smooth hinge width {width:g}",
                "smooth_hinge",
                "h=[d*softplus((E-q)/d)-d*softplus(-q/d)]/32; q=expm1(tau)",
                proof + " Softplus is convex; its affine argument preserves convexity.",
                (width,),
                shared,
                smooth_hinge_design,
            )
        )
    specifications.append(
        CandidateSpec(
            "cv17_hinge",
            "Exact hinge",
            "hinge",
            "h=(E-q)+/32; q=expm1(tau)",
            proof + " Positive part is convex.",
            (1.0,),
            shared,
            hinge_design,
        )
    )
    for width in (1.0, 8.0):
        specifications.append(
            CandidateSpec(
                f"cv17_huber_d{width:g}",
                f"Huber hinge width {width:g}",
                "huber_hinge",
                "h=Huber_positive(E-q,d)/32; q=expm1(tau)",
                proof + " Huber hinge derivative rises continuously from zero to one.",
                (width,),
                shared,
                huber_design,
            )
        )
    specifications.append(
        CandidateSpec(
            "cv17_quadratic_hinge",
            "Squared hinge",
            "quadratic",
            "h=((E-q)+/32)^2; q=expm1(tau)",
            proof + " Squaring is convex increasing on the positive-part range.",
            (2.0,),
            shared,
            hinge_design,
        )
    )
    for knots in ((0.0, 4.0, 32.0), (0.0, 1.0, 4.0, 16.0, 64.0)):
        specifications.append(
            CandidateSpec(
                f"cv17_multiknot{len(knots)}",
                f"Independent {len(knots)}-knot harm",
                "multiknot",
                "h_i=sum_k beta_ik*(E_i-q_k)+/32, beta_ik>=0",
                proof + " Each hinge has an independent nonnegative coefficient.",
                knots,
                benefit,
                multiknot_design,
            )
        )
    specifications.append(
        CandidateSpec(
            "cv17_convex_bank",
            "Convex response bank",
            "convex_response",
            "g=c-sum_ij alpha_ij*b(E_i;rho_j,.7)+sum_ik beta_ik*(E_i-q_k)+/32; rho=(.1,.5,2), q=(0,4,32)",
            proof + " Every basis is convex, with separate nonnegative coefficients.",
            (0.0, 4.0, 32.0),
            (),
            convex_bank_design,
        )
    )
    return tuple(specifications)


def build_model(spec: CandidateSpec, features: base.Features) -> base.FittedFloorModel:
    """Use the canonical floor and nonnegative ridge fit with an isolated design."""
    frozen = registry.ENTRY_BY_ID[registry.FROZEN_ID].build(features)
    if not isinstance(frozen, base.FittedFloorModel) or not isinstance(frozen.base, base.GridModel):
        raise TypeError("The convex screen requires the canonical fitted-floor grid model.")
    if frozen.base.head.kind is not base.HeadKind.NNLS or np.isfinite(frozen.base.head.cap_margin):
        raise ValueError("The convexity proof requires nonnegative coefficients and an uncapped exponential.")
    identity = (spec, MODULE_SOURCE_HASH)

    def builder(query: base.Features, shape: base.Shape) -> base.Design:
        candidate = identity[0]
        return candidate.design_function(query, shape, candidate.params)

    distinct = {tuple((key, float(shape[key])) for key in spec.shape_keys): None for shape in frozen.base.shapes}
    grid = dataclasses.replace(
        frozen.base,
        model_id=spec.model_id,
        builder=builder,
        shapes=tuple(dict(shape) for shape in distinct),
        shape_dof=len(spec.shape_keys),
    )
    return dataclasses.replace(frozen, base=grid)
