# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convex majorant and finite-domain convex envelope of original MARINER harm."""

import dataclasses
import functools
import hashlib
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    mariner_convex_candidates_20260917 as convex,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    mariner_harm_screen_models_20260917 as original,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_registry_20260902 as registry,
)

MODULE_SOURCE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclasses.dataclass(frozen=True)
class CandidateSpec:
    model_id: str
    label: str
    equation: str


def candidate_specs() -> tuple[CandidateSpec, ...]:
    """Return the two predeclared convexifications, in unchanged original harm units."""
    return (
        CandidateSpec(
            "ce2_majorant",
            "Original-harm convex majorant",
            "H(E) until E*=max((e-1)exp(tau)-1,0), then its tangent continuation",
        ),
        CandidateSpec(
            "ce2_envelope",
            "Original-harm finite-domain convex envelope",
            "Greatest convex minorant of H on each fixed [0,inventory_i], then terminal tangent continuation",
        ),
    )


def original_harm(epochs: np.ndarray, threshold: float) -> np.ndarray:
    return original.centered_log_harm(epochs, threshold, 2.0)


def harm_derivative(epochs: np.ndarray, threshold: float) -> np.ndarray:
    denominator = epochs + 1.0 + np.exp(threshold)
    return 2.0 * (np.log(denominator) - threshold) / denominator


def harm_inflection(threshold: float) -> float:
    return max(float(np.expm1(1.0) * np.exp(threshold) - 1.0), 0.0)


@functools.lru_cache(maxsize=4096)
def envelope_join(threshold: float, maximum: float) -> tuple[float, float]:
    """Return envelope join and terminal slope using the physical epoch bound."""
    if maximum <= 0 or not np.isfinite(maximum):
        raise ValueError("The physical epoch bound must be finite and positive.")
    inflection = harm_inflection(threshold)
    if maximum <= inflection:
        return maximum, float(harm_derivative(np.asarray(maximum), threshold))
    value_at_maximum = float(original_harm(np.asarray(maximum), threshold))
    chord_from_zero = value_at_maximum / maximum
    initial_derivative = float(harm_derivative(np.asarray(0.0), threshold))
    if chord_from_zero <= initial_derivative:
        return 0.0, chord_from_zero

    def residual(join: float) -> float:
        chord = (value_at_maximum - float(original_harm(np.asarray(join), threshold))) / (maximum - join)
        return float(harm_derivative(np.asarray(join), threshold)) - chord

    join = float(brentq(residual, 0.0, inflection, xtol=1e-13, rtol=1e-13, full_output=False))
    slope = (value_at_maximum - float(original_harm(np.asarray(join), threshold))) / (maximum - join)
    return float(join), float(slope)


def harm_and_derivative(
    model_id: str, epochs: np.ndarray, threshold: float, physical_maxima: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate each bucket's convex harm and continuous first derivative."""
    if epochs.ndim != 2 or epochs.shape[1] != len(physical_maxima):
        raise ValueError("Epoch columns and the fixed physical inventory must agree.")
    if np.any(epochs < 0):
        raise ValueError("Epochs must be nonnegative.")
    if model_id == "ce2_majorant":
        joins = np.full(len(physical_maxima), harm_inflection(threshold))
        slopes = harm_derivative(joins, threshold)
    elif model_id == "ce2_envelope":
        joins, slopes = np.asarray([envelope_join(threshold, maximum) for maximum in physical_maxima]).T
    else:
        raise ValueError(f"Unknown convexification: {model_id}")
    prefix = np.minimum(epochs, joins)
    values = original_harm(prefix, threshold) + (epochs - prefix) * slopes
    derivatives = np.where(epochs < joins, harm_derivative(epochs, threshold), slopes)
    return values, derivatives


def design_epoch_derivative(
    spec: CandidateSpec, features: base.Features, shape: base.Shape
) -> tuple[np.ndarray, np.ndarray]:
    """Return own-bucket basis derivatives; benefit may have -infinity slope at zero."""
    negative_benefit = convex.negative_benefit_derivative(
        features.exposures, float(shape["rate"]), float(shape["power"])
    )
    _, harm = harm_and_derivative(
        spec.model_id, features.exposures, float(shape["threshold"]), tuple(map(float, features.inventory))
    )
    return np.concatenate([negative_benefit, harm], axis=1), np.tile(np.arange(features.buckets), 2)


def design(
    features: base.Features,
    shape: base.Shape,
    model_id: str,
    physical_maxima: tuple[float, ...],
    source_identity: str,
) -> base.Design:
    del source_identity
    if not 0 < float(shape["power"]) <= 1:
        raise ValueError("Convexity requires Weibull shape in (0,1].")
    harm, _ = harm_and_derivative(model_id, features.exposures, float(shape["threshold"]), physical_maxima)
    return convex.append_harm(features, shape, harm, "bucket_overexposure")


def build_model(spec: CandidateSpec, features: base.Features) -> base.FittedFloorModel:
    """Replace only harm; bind physical bounds at construction, independent of queries."""
    frozen = registry.ENTRY_BY_ID[registry.FROZEN_ID].build(features)
    if not isinstance(frozen, base.FittedFloorModel) or not isinstance(frozen.base, base.GridModel):
        raise TypeError("This extension requires the frozen fitted-floor grid model.")
    if frozen.base.head.kind is not base.HeadKind.NNLS or np.isfinite(frozen.base.head.cap_margin):
        raise ValueError("The mathematical objective requires nonnegative amplitudes and no upper cap.")
    identity = (spec.model_id, tuple(map(float, features.inventory)), MODULE_SOURCE_HASH)

    def builder(query: base.Features, shape: base.Shape) -> base.Design:
        return design(query, shape, *identity)

    grid = dataclasses.replace(frozen.base, model_id=spec.model_id, builder=builder)
    return dataclasses.replace(frozen, base=grid)
