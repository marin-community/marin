# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-basis harm hypotheses fitted through the frozen MARINER procedure.

Harm is centered after its complete transformation. This leaves the canonical
squared-softplus model unchanged up to its free intercept. Every candidate keeps
the canonical amplitude ridge; changing a basis therefore changes its implied
function-space regularization, which the screen must report as a limitation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_registry_20260902 as registry,
)

VariantParams = tuple[tuple[str, float], ...]
DesignFunction = Callable[[base.Features, base.Shape, VariantParams], base.Design]
MODULE_SOURCE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclasses.dataclass(frozen=True)
class CandidateSpec:
    model_id: str
    label: str
    category: str
    equation: str
    cost: str
    falsifier: str
    variant_params: VariantParams
    shape_keys: tuple[str, ...]
    design_function: DesignFunction
    link: base.LinkKind = base.LinkKind.KAPPA_FLOOR


def centered_log_harm(exposure: np.ndarray, threshold: float, power: float = 2.0) -> np.ndarray:
    """Power of log-epoch softplus, centered after exponentiation."""
    value = np.logaddexp(0.0, np.log1p(np.maximum(exposure, 0.0)) - threshold)
    origin = np.logaddexp(0.0, -threshold)
    return value**power - origin**power


def coupled_threshold(rate: float, power: float, ratio: float) -> float:
    """Set harm's epoch inflection to ``ratio`` times benefit half-saturation."""
    half_saturation = math.log(2.0) ** (1.0 / power) / rate
    return math.log1p(ratio * half_saturation) - math.log(math.e - 1.0)


def pair_interaction(values: np.ndarray) -> np.ndarray:
    """Mean product over distinct ordered bucket pairs, one column per row."""
    buckets = values.shape[1]
    if buckets < 2:
        raise ValueError("a pair-interaction candidate needs at least two buckets")
    # Sum distinct products directly, avoiding cancellation for a sparse row.
    pair_sum = np.zeros(values.shape[0])
    for bucket in range(buckets - 1):
        pair_sum += values[:, bucket] * values[:, bucket + 1 :].sum(axis=1)
    return (2.0 * pair_sum / (buckets * (buckets - 1)))[:, None]


def _benefit(features: base.Features, shape: base.Shape) -> np.ndarray:
    return base.weibull_response(features.exposures, float(shape["rate"]), float(shape["power"]))


def _design(benefit: np.ndarray, harm: np.ndarray) -> base.Design:
    buckets = benefit.shape[1]
    names = tuple(f"bucket_signal:{index}" for index in range(buckets)) + tuple(
        f"bucket_overexposure:{index}" for index in range(harm.shape[1])
    )
    values = np.concatenate([-benefit, harm], axis=1)
    return base.Design(values, np.ones(values.shape[1]), names)


def log_harm_design(features: base.Features, shape: base.Shape, params: VariantParams) -> base.Design:
    harm = centered_log_harm(features.exposures, float(shape["threshold"]), dict(params)["harm_power"])
    return _design(_benefit(features, shape), harm)


def rational_bounded_design(features: base.Features, shape: base.Shape, params: VariantParams) -> base.Design:
    harm = centered_log_harm(features.exposures, float(shape["threshold"]))
    cap = dict(params)["cap"]
    return _design(_benefit(features, shape), harm / (1.0 + harm / cap))


def exponential_bounded_design(features: base.Features, shape: base.Shape, params: VariantParams) -> base.Design:
    harm = centered_log_harm(features.exposures, float(shape["threshold"]))
    cap = dict(params)["cap"]
    return _design(_benefit(features, shape), -cap * np.expm1(-harm / cap))


def coupled_design(features: base.Features, shape: base.Shape, params: VariantParams) -> base.Design:
    threshold = coupled_threshold(float(shape["rate"]), float(shape["power"]), dict(params)["ratio"])
    harm = centered_log_harm(features.exposures, threshold)
    return _design(_benefit(features, shape), harm)


def tied_benefit_design(features: base.Features, shape: base.Shape, params: VariantParams) -> base.Design:
    benefit = _benefit(features, shape)
    return _design(benefit, benefit ** dict(params)["harm_power"])


def harm_interaction_design(features: base.Features, shape: base.Shape, params: VariantParams) -> base.Design:
    design = log_harm_design(features, shape, params)
    interaction = pair_interaction(design.values[:, features.buckets :])
    return base.Design(
        np.concatenate([design.values, interaction], axis=1),
        np.append(design.ridge, 1.0),
        (*design.names, "global_overload_harm"),
    )


def benefit_interaction_design(features: base.Features, shape: base.Shape, params: VariantParams) -> base.Design:
    design = log_harm_design(features, shape, params)
    interaction = pair_interaction(-design.values[:, : features.buckets])
    return base.Design(
        np.concatenate([design.values, interaction], axis=1),
        np.append(design.ridge, 1.0),
        (*design.names, "global_overload_benefit"),
    )


def no_harm_design(features: base.Features, shape: base.Shape, params: VariantParams) -> base.Design:
    del params
    return _design(_benefit(features, shape), np.empty((features.rows, 0)))


def convex_linear_design(features: base.Features, shape: base.Shape, params: VariantParams) -> base.Design:
    del params
    threshold = float(shape["threshold"])
    harm = base.softplus_raw_epoch_hinge_harm(features.exposures, threshold)
    origin = base.softplus_raw_epoch_hinge_harm(np.asarray(0.0), threshold)
    return _design(_benefit(features, shape), harm - origin)


def candidate_specs() -> tuple[CandidateSpec, ...]:
    """Return the fixed, individually reported hypotheses; no variant adds a shape grid axis."""
    three_shapes = ("rate", "power", "threshold")
    two_shapes = ("rate", "power")
    full_cost = "168 shapes x 5 ridges; canonical fitted floor"
    reduced_cost = "28 shapes x 5 ridges; canonical fitted floor"
    specs = [
        CandidateSpec(
            "hs17_mariner",
            "MARINER centered control",
            "control",
            "h=s^2-s(0)^2",
            full_cost,
            "Must reproduce canonical predictions and CV scores.",
            (("harm_power", 2.0),),
            three_shapes,
            log_harm_design,
        ),
        CandidateSpec(
            "hs17_log1",
            "Logarithmic harm",
            "tail",
            "h=s-s(0)",
            full_cost,
            "Worse high-repetition ranking or selection than squared-log harm.",
            (("harm_power", 1.0),),
            three_shapes,
            log_harm_design,
        ),
        CandidateSpec(
            "hs17_log15",
            "Log-power 1.5 harm",
            "tail",
            "h=s^1.5-s(0)^1.5",
            full_cost,
            "No ranking or selection improvement between the existing log and squared-log tails.",
            (("harm_power", 1.5),),
            three_shapes,
            log_harm_design,
        ),
    ]
    for cap in (1.0, 4.0):
        for tag, label, equation, design in (
            ("rat", "Rational bounded harm", "h=H/(1+H/C)", rational_bounded_design),
            ("exp", "Exponential bounded harm", "h=C*(1-exp(-H/C))", exponential_bounded_design),
        ):
            specs.append(
                CandidateSpec(
                    f"hs17_bounded_{tag}_c{cap:g}",
                    f"{label}, C={cap:g}",
                    "tail",
                    equation,
                    full_cost,
                    "Underprediction at high repetition after benefit has saturated.",
                    (("cap", cap),),
                    three_shapes,
                    design,
                )
            )
    for ratio, tag in ((0.5, "05"), (1.0, "1"), (2.0, "2"), (4.0, "4")):
        specs.append(
            CandidateSpec(
                f"hs17_coupled_c{tag}",
                f"Coupled inflection, c={ratio:g}",
                "scale_coupling",
                "E_inf=c*(log(2)^(1/kappa)/rho); tau=log(1+E_inf)-log(e-1)",
                reduced_cost,
                "A fixed benefit-to-harm clock loses ranking or selects implausible mixtures.",
                (("ratio", ratio),),
                two_shapes,
                coupled_design,
            )
        )
    for power in (2.0, 3.0):
        specs.append(
            CandidateSpec(
                f"hs17_tied_b{power:g}",
                f"Benefit-tied harm b^{power:g}",
                "benefit_coupling",
                "h=b^q",
                reduced_cost,
                "Fails where harm continues changing after the benefit is saturated.",
                (("harm_power", power),),
                two_shapes,
                tied_benefit_design,
            )
        )
    for tag, design in (("harm", harm_interaction_design), ("benefit", benefit_interaction_design)):
        specs.append(
            CandidateSpec(
                f"hs17_interaction_{tag}",
                f"Global {tag} interaction",
                "interaction",
                f"eta=eta_MARINER+gamma*mean(i!=j,v_i*v_j); v={tag}",
                full_cost + "; one additional nonnegative amplitude",
                "No held-out improvement from simultaneous bucket exposure beyond additive effects.",
                (("harm_power", 2.0),),
                three_shapes,
                design,
            )
        )
    specs.extend(
        (
            CandidateSpec(
                "hs17_no_harm",
                "No harm",
                "control",
                "h=0",
                reduced_cost,
                "Misses turnover or selects excessive repetition.",
                (),
                two_shapes,
                no_harm_design,
            ),
            CandidateSpec(
                "hs17_convex_linear",
                "Convex linear-tail harm",
                "control",
                "h=softplus((E-expm1(tau))/exp(tau))-h(0)",
                full_cost,
                "Persistent marginal log-deficit cost worsens ranking or selection.",
                (),
                three_shapes,
                convex_linear_design,
            ),
            CandidateSpec(
                "hs17_identity",
                "Additive loss control",
                "link",
                "f=c-sum(alpha*b)+sum(beta*H)",
                "168 shapes x 5 ridges; identity link, no fitted floor",
                "Poor calibration or implausible off-support loss exposes the link's importance.",
                (("harm_power", 2.0),),
                three_shapes,
                log_harm_design,
                base.LinkKind.IDENTITY,
            ),
        )
    )
    return tuple(specs)


def build_model(spec: CandidateSpec, features: base.Features) -> base.FittedFloorModel | base.GridModel:
    """Build one candidate without modifying the shared registry or canonical procedure."""
    frozen = registry.ENTRY_BY_ID[registry.FROZEN_ID].build(features)
    if not isinstance(frozen, base.FittedFloorModel) or not isinstance(frozen.base, base.GridModel):
        raise TypeError("the harm screen requires the frozen fitted-floor grid model")
    # The benchmark's model description serializes closure values, including this
    # immutable configuration and the source digest that implements its bases.
    identity = (spec, MODULE_SOURCE_HASH)

    def builder(query: base.Features, shape: base.Shape) -> base.Design:
        candidate = identity[0]
        return candidate.design_function(query, shape, candidate.variant_params)

    distinct = {tuple((key, float(shape[key])) for key in spec.shape_keys): None for shape in frozen.base.shapes}
    shapes = tuple(dict(shape) for shape in distinct)
    grid = dataclasses.replace(
        frozen.base,
        model_id=spec.model_id,
        builder=builder,
        shapes=shapes,
        shape_dof=len(spec.shape_keys),
        head=dataclasses.replace(frozen.base.head, link=spec.link),
    )
    if spec.link is base.LinkKind.IDENTITY:
        return grid
    if spec.link is not base.LinkKind.KAPPA_FLOOR:
        raise ValueError(f"unsupported screen link {spec.link}")
    return dataclasses.replace(frozen, base=grid)
