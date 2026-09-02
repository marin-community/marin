# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RPL with a shared semantic-family component in the retained-state transition.

Let ``a = beta0 * w0 + beta1 * w1`` be the token aggregate and
``d = w1 - w0`` the phase contrast. RPL lets the survival of bucket ``i``'s
early state depend only on ``d_i``. This variant decomposes that displacement
into bucket-specific and family-shared components:

    D_f = sum_{j in f} d_j
    pi_i(a) = a_i / sum_{j in f} a_j
    u_i(q) = (1 - q) d_i + q pi_i(a) D_f

The survival gate uses ``u_i`` while the raw early and late doses remain the
observed policy. ``q`` is the fraction of retained state shared within a
predeclared semantic family. At ``q=0`` the model is exactly RPL. Tied policies
are unchanged for every ``q``. Singleton families are also exactly RPL.

This is not a family-specific retention rate: for ``q>0``, the state transition
for bucket ``i`` depends on the contrasts of the other buckets in its family.
The family-total contrast is conserved, so the mechanism changes only how
family displacement is allocated among bucket-specific retained states.
"""

from __future__ import annotations

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import retained_power_law_model_20260728 as rpl

Geometry = rpl.Geometry
Shape = rpl.Shape


def aggregate_and_contrast(
    weights: np.ndarray,
    geometry: Geometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Return token aggregate and late-minus-early phase contrast."""
    phase_0, phase_1 = weights[:, 0, :], weights[:, 1, :]
    aggregate = geometry.phase_0_fraction * phase_0 + geometry.phase_1_fraction * phase_1
    return aggregate, phase_1 - phase_0


def mediated_contrast(
    weights: np.ndarray,
    geometry: Geometry,
    family_mediation: float,
) -> np.ndarray:
    """Blend bucket-specific contrast with aggregate-proportional family contrast."""
    if not 0.0 <= family_mediation <= 1.0:
        raise ValueError("family_mediation must lie in [0, 1]")
    aggregate, contrast = aggregate_and_contrast(weights, geometry)
    families = geometry.families
    if len(np.unique(families)) == len(families) or family_mediation == 0.0:
        return contrast.copy()

    shared = np.zeros_like(contrast)
    for family in np.unique(families):
        members = families == family
        family_mass = aggregate[:, members].sum(axis=1)
        family_contrast = contrast[:, members].sum(axis=1)
        shares = np.divide(
            aggregate[:, members],
            family_mass[:, None],
            out=np.zeros_like(aggregate[:, members]),
            where=family_mass[:, None] > 0.0,
        )
        shared[:, members] = shares * family_contrast[:, None]
    return (1.0 - family_mediation) * contrast + family_mediation * shared


def retained_share(
    weights: np.ndarray,
    geometry: Geometry,
    retention: float,
    late_multiplier: float,
    family_mediation: float,
) -> np.ndarray:
    """Return RPL retained share under the family-mediated survival state."""
    phase_0, phase_1 = weights[:, 0, :], weights[:, 1, :]
    contrast = mediated_contrast(weights, geometry, family_mediation)
    survival = np.exp(rpl.GATE_CLIP * np.tanh(retention * contrast / rpl.GATE_CLIP))
    return survival * geometry.phase_0_fraction * phase_0 + late_multiplier * geometry.phase_1_fraction * phase_1


def design_matrix(
    weights: np.ndarray,
    geometry: Geometry,
    shape: Shape,
    family_mediation: float,
) -> np.ndarray:
    """RPL design with only the retained-state transition changed."""
    if family_mediation == 0.0 or len(np.unique(geometry.families)) == len(geometry.families):
        return rpl.design_matrix(weights, geometry, shape)

    retained = retained_share(
        weights,
        geometry,
        shape.retention,
        shape.late_multiplier,
        family_mediation,
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
