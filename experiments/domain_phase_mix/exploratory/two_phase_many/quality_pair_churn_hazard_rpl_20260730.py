# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RPL with conditional composition churn inside predeclared quality pairs.

Dolma 3 Common Crawl supplies high- and low-quality versions of the same
thirteen topics. Those pairs define a mechanism-specific partition independent
of RPL's broader coefficient-pooling families. All other buckets are
singletons.

For each quality pair ``f``, normalize the two phase mixtures within the pair
and compute conditional squared Hellinger displacement:

    C_f = 1 - sum_{i in f} sqrt(p0_i p1_i)

``C_f`` is a bounded churn *rate*. It is independent of the pair's total mass;
the amount of early state exposed to the hazard still scales through
``w0_i``. The retained-state transition is:

    gate_i = retention * (w1_i - w0_i) - churn_hazard * C_{f(i)}

Singleton buckets have ``C_f = 0``. At ``churn_hazard=0`` the model is exactly
RPL. The mechanism differs from an additive global Hellinger penalty: it acts
inside the retained-state transition and uses thirteen topic-matched quality
pairs rather than one global phase-distance scalar.
"""

from __future__ import annotations

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import retained_power_law_model_20260728 as rpl

Geometry = rpl.Geometry
Shape = rpl.Shape

COMMON_CRAWL_PREFIX = "dolma3_cc/"
QUALITY_SUFFIXES = ("_high", "_low")


def quality_pair_families(domain_names: tuple[str, ...]) -> np.ndarray:
    """Group topic-matched Common Crawl quality tiers and leave others singleton."""
    keys = []
    for name in domain_names:
        if name.startswith(COMMON_CRAWL_PREFIX) and name.endswith(QUALITY_SUFFIXES):
            keys.append(name.rsplit("_", 1)[0])
        else:
            keys.append(name)
    index = {key: family for family, key in enumerate(dict.fromkeys(keys))}
    families = np.asarray([index[key] for key in keys])

    for key, family in index.items():
        if not key.startswith(COMMON_CRAWL_PREFIX):
            continue
        members = np.flatnonzero(families == family)
        suffixes = {domain_names[member].rsplit("_", 1)[1] for member in members}
        if len(members) != 2 or suffixes != {"high", "low"}:
            raise ValueError(f"expected one high/low quality pair for {key}")
    return families


def conditional_family_churn(
    weights: np.ndarray,
    churn_families: np.ndarray,
) -> np.ndarray:
    """Return conditional squared Hellinger churn for every family and row."""
    phase_0, phase_1 = weights[:, 0, :], weights[:, 1, :]
    values = []
    for family in np.unique(churn_families):
        members = churn_families == family
        if members.sum() == 1:
            values.append(np.zeros(len(weights)))
            continue
        mass_0 = phase_0[:, members].sum(axis=1)
        mass_1 = phase_1[:, members].sum(axis=1)
        denominator = np.sqrt(mass_0 * mass_1)
        overlap = np.sqrt(phase_0[:, members] * phase_1[:, members]).sum(axis=1)
        affinity = np.divide(
            overlap,
            denominator,
            out=np.ones_like(overlap),
            where=denominator > 1e-15,
        )
        if np.any(affinity > 1.0 + 1e-12) or np.any(affinity < -1e-12):
            raise ValueError("conditional Hellinger affinity left [0, 1]")
        values.append(1.0 - np.clip(affinity, 0.0, 1.0))
    return np.column_stack(values)


def bucket_churn(
    weights: np.ndarray,
    churn_families: np.ndarray,
) -> np.ndarray:
    """Broadcast each quality pair's churn rate to its member buckets."""
    by_family = conditional_family_churn(weights, churn_families)
    unique = np.unique(churn_families)
    family_values = {family: by_family[:, index] for index, family in enumerate(unique)}
    return np.column_stack([family_values[family] for family in churn_families])


def retained_share(
    weights: np.ndarray,
    geometry: Geometry,
    retention: float,
    late_multiplier: float,
    churn_hazard: float,
    churn_families: np.ndarray,
) -> np.ndarray:
    """Return retained token share after recency and quality-pair churn."""
    if churn_hazard < 0.0:
        raise ValueError("churn_hazard must be nonnegative")
    phase_0, phase_1 = weights[:, 0, :], weights[:, 1, :]
    gate = retention * (phase_1 - phase_0) - churn_hazard * bucket_churn(weights, churn_families)
    survival = np.exp(rpl.GATE_CLIP * np.tanh(gate / rpl.GATE_CLIP))
    return survival * geometry.phase_0_fraction * phase_0 + late_multiplier * geometry.phase_1_fraction * phase_1


def design_matrix(
    weights: np.ndarray,
    geometry: Geometry,
    shape: Shape,
    churn_hazard: float,
    churn_families: np.ndarray,
) -> np.ndarray:
    """Return the RPL design under the quality-pair churn transition."""
    if churn_hazard < 0.0:
        raise ValueError("churn_hazard must be nonnegative")
    if churn_hazard == 0.0 or len(np.unique(churn_families)) == len(churn_families):
        return rpl.design_matrix(weights, geometry, shape)

    retained = retained_share(
        weights,
        geometry,
        shape.retention,
        shape.late_multiplier,
        churn_hazard,
        churn_families,
    )
    benefit = (retained + shape.benefit_offset) ** (-shape.benefit_exponent)
    excess = np.maximum(
        rpl.total_epochs(weights, geometry) - shape.damage_threshold,
        0.0,
    )
    blocks = [
        rpl._hierarchical_block(benefit, geometry),
        rpl._hierarchical_block(excess**shape.damage_exponent, geometry),
        rpl._signed(rpl.concentration_gap(weights, geometry)),
    ]
    if shape.ordering_channel:
        blocks.append(rpl.marginal_phase_block(weights, geometry, shape))
    return np.column_stack(blocks)
