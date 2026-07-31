# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RPL with a within-family composition-churn hazard.

RPL lets bucket ``i``'s early state survive according to its own phase contrast
``d_i = w1_i - w0_i``. This variant adds a shared forgetting hazard when the
composition *within* a predeclared semantic family changes between phases.

For family ``f``:

    H_f = 2 (sqrt(W0_f W1_f) - sum_{i in f} sqrt(w0_i w1_i))

where ``Wt_f = sum_{i in f} wt_i``. ``H_f`` is the squared Hellinger
displacement inside the family after factoring out movement of the family's
total mass. It is nonnegative, symmetric under swapping phases, zero for tied
policies, zero for singleton families, and zero when every member of a family
changes by the same multiplicative factor.

The retained-state transition is:

    gate_i = retention * d_i - churn_hazard * H_{f(i)}

The first term is RPL's bucket-specific recency signal. The second is a shared
interference hazard: replacing a semantic family's internal composition makes
early state from every member less likely to survive. At ``churn_hazard=0`` the
model is exactly RPL. Unlike an additive Hellinger output correction, this term
changes the latent state before the power-law response.
"""

from __future__ import annotations

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import retained_power_law_model_20260728 as rpl

Geometry = rpl.Geometry
Shape = rpl.Shape


def family_churn(weights: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Return within-family squared Hellinger displacement for every row."""
    phase_0, phase_1 = weights[:, 0, :], weights[:, 1, :]
    values = []
    for family in np.unique(geometry.families):
        members = geometry.families == family
        family_overlap = np.sqrt(phase_0[:, members].sum(axis=1) * phase_1[:, members].sum(axis=1))
        bucket_overlap = np.sqrt(phase_0[:, members] * phase_1[:, members]).sum(axis=1)
        values.append(np.maximum(2.0 * (family_overlap - bucket_overlap), 0.0))
    return np.column_stack(values)


def bucket_churn(weights: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Broadcast each family's churn hazard to its member buckets."""
    by_family = family_churn(weights, geometry)
    families = geometry.families
    family_values = {family: by_family[:, index] for index, family in enumerate(np.unique(families))}
    return np.column_stack([family_values[family] for family in families])


def retained_share(
    weights: np.ndarray,
    geometry: Geometry,
    retention: float,
    late_multiplier: float,
    churn_hazard: float,
) -> np.ndarray:
    """Return retained token share after bucket recency and family churn."""
    if churn_hazard < 0.0:
        raise ValueError("churn_hazard must be nonnegative")
    phase_0, phase_1 = weights[:, 0, :], weights[:, 1, :]
    gate = retention * (phase_1 - phase_0)
    if churn_hazard > 0.0:
        gate = gate - churn_hazard * bucket_churn(weights, geometry)
    survival = np.exp(rpl.GATE_CLIP * np.tanh(gate / rpl.GATE_CLIP))
    return survival * geometry.phase_0_fraction * phase_0 + late_multiplier * geometry.phase_1_fraction * phase_1


def design_matrix(
    weights: np.ndarray,
    geometry: Geometry,
    shape: Shape,
    churn_hazard: float,
) -> np.ndarray:
    """Return the RPL design under the family-churn state transition."""
    if churn_hazard == 0.0 or len(np.unique(geometry.families)) == len(geometry.families):
        return rpl.design_matrix(weights, geometry, shape)

    retained = retained_share(
        weights,
        geometry,
        shape.retention,
        shape.late_multiplier,
        churn_hazard,
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
