# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compact retained state with an explicit odd/even phase law.

Design follows what the swarm39 work established rather than adding capacity.

*Small.* Every measurement this experiment has made says capacity hurts transfer
from 300M to 3e18: on the 238-pair transfer test the 40-column
``compact_retained_state`` leads Uncheatable at 0.710 phase skill and the
158-parameter DSP is last at 0.334, and the real DSP is the worst of four models
on 3e18 heldouts while fitting the panel twice as tightly in sample. So the
aggregate response here is exactly the compact retained-state core, unchanged.

*Explicit phase law.* On exact aggregate-matched pairs the response splits as
``Delta = O + C`` with ``O`` odd and ``C`` even in the contrast ``d = p1 - p0``,
and two-phase beats tied only where ``|O| > C``. The 39-bucket panels give

    |O| ~ kappa * rho^p   with p in 1.66 to 1.85,
    C   ~ c * rho^q       with q in 2.10 to 2.68,

where ``rho`` is phase total variation. Existing models approximate this
implicitly: ``compact_retained_state`` only through retained-state gating, and
``crs_plus_phase`` through unscaled family recency columns. Neither carries the
power law. This model parameterizes it directly, with ``p`` and ``q`` fitted from
the shape grid:

* one even column ``rho^q``, whose nonnegative coefficient makes phase separation
  a cost that grows superlinearly;
* family-resolved odd columns scaled by ``rho^(p-1)``, so a family's signed
  late-minus-early exposure contributes an ordering term of total order
  ``rho^p``. Late and early enter as separate nonnegative columns because the
  head is nonnegative least squares and either direction may help.

Three families rather than 39 buckets keeps the odd channel low dimensional,
which is what separates this from hierarchical phase replay: HPR's richer phase
block took the top within-scale phase skill (0.768) and lost most of it on
transfer (0.479), the signature of fitting phase structure that does not survive
the scale jump.

Shapes are meant to be selected cross-scale and then held fixed. Reselecting per
panel makes the proposal 2.5 to 5 times rougher across panel sizes without
converging any faster.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import numpy as np
from swarm39_harness_20260725 import Design, Model, Panel
from swarm39_models_20260725 import _weibull, build_crs_plus_phase, crs_plus_shapes

# Measured on the 39-bucket panels: the odd channel is sublinear-to-quadratic in
# contrast radius and the even cost is steeper, which is why the attainable gain
# is capped rather than growing with contrast.
ODD_EXPONENTS = (1.0, 1.5, 2.0)
EVEN_EXPONENTS = (2.0, 2.5, 3.0)
CONTRAST_EPSILON = 1e-12


def phase_radius(panel: Panel) -> np.ndarray:
    """Phase total variation, the contrast radius rho, as a column."""
    return (0.5 * np.abs(panel.phase1 - panel.phase0).sum(axis=1)).reshape(-1, 1)


def partition_of(panel: Panel, shape: dict) -> tuple[np.ndarray, tuple[str, ...]]:
    """Bucket grouping for the odd block, overridable through the shape.

    The family sum is where design information is lost. The handcrafted three
    families retain only 5.4 percent of the panel's contrast variance and exactly
    0 percent of the quality direction; splitting them by quality reaches 11.7 and
    100 percent, and a label-free quintile split on corpus size reaches 11.1 and
    25 percent while transferring to any bucket set.
    """
    supplied = shape.get("family_index")
    if supplied is None:
        return np.asarray(panel.family_index), tuple(panel.family_names)
    index = np.asarray(supplied, dtype=int)
    return index, tuple(f"g{k}" for k in range(int(index.max()) + 1))


def consolidated_state(panel: Panel, shape: dict) -> np.ndarray:
    """Retained state whose forgetting gate weakens with early exposure.

    The baseline gate ``exp(-f (1 - p1))`` depends only on the late weight, so a
    bucket given 10 epochs in phase 0 and one given 0.1 epoch decay by the same
    factor when dropped. More early exposure should mean better consolidation and
    less forgetting. ``consolidation`` = 0 recovers the baseline exactly, so the
    enrichment costs one shared parameter and nests what it extends.
    """
    early = panel.phase0 * panel.c0
    late = panel.phase1 * panel.c1
    theta = float(shape.get("consolidation", 0.0))
    # Gate sharpness: the baseline decays linearly in the late-absence fraction
    # (1 - p1). A power lets forgetting be abrupt (k>1, only near-total dropout
    # hurts) or gradual (k<1, any reduction hurts). k=1 recovers the baseline.
    absence = np.maximum(1.0 - panel.phase1, 0.0) ** float(shape.get("gate_power", 1.0))
    gate = float(shape["forgetting_rate"]) * absence
    if theta > 0.0:
        gate = gate / (1.0 + early) ** theta
    # Two-timescale retention: a fast-decaying component and a slow consolidated
    # one, the standard form in the memory literature and structurally different
    # from reweighting a single gate. slow_share = 0 recovers the single-gate
    # baseline exactly; slow_ratio scales the second component's forgetting rate.
    share = float(shape.get("slow_share", 0.0))
    if share > 0.0:
        slow = np.exp(-gate * float(shape.get("slow_ratio", 0.25)))
        retained = ((1.0 - share) * np.exp(-gate) + share * slow) * early
    else:
        retained = np.exp(-gate) * early
    # Recency saturates: the late multiplier is otherwise constant, so a bucket
    # already seen 50 epochs late earns the same marginal recency credit as one
    # seen 0.1 epoch. Large saturation recovers the linear baseline exactly.
    saturation = float(shape.get("recency_saturation", np.inf))
    late_term = late if not np.isfinite(saturation) else saturation * (-np.expm1(-late / saturation))
    return np.maximum(retained + float(shape["late_multiplier"]) * late_term, 0.0)


def jensen_even_cost(panel: Panel) -> np.ndarray:
    """Per-bucket Jensen gap of a convex repetition harm, summed.

    With harm convex in within-phase sampling intensity and time-weighted by phase
    duration, Jensen gives alpha psi(x0) + (1-alpha) psi(x1) >= psi(xbar) with
    equality only at the tied policy. So this is non-negative by construction,
    exactly zero at d = 0, quadratic in the contrast near the origin, and carries
    no fitted exponent. It also gives per-bucket resolution where rho^q gives one
    global scalar, and it reuses the max(x-1,0)^2 harm already in the replay term.
    """
    alpha = panel.alpha
    intensity0 = panel.phase0 * panel.c0 / alpha
    intensity1 = panel.phase1 * panel.c1 / (1.0 - alpha)
    mean = alpha * intensity0 + (1.0 - alpha) * intensity1
    psi = lambda x: np.maximum(x - 1.0, 0.0) ** 2  # noqa: E731
    gap = alpha * psi(intensity0) + (1.0 - alpha) * psi(intensity1) - psi(mean)
    return np.maximum(gap, 0.0).sum(axis=1, keepdims=True)


def build_phase_law(panel: Panel, shape: dict) -> Design:
    """Compact retained-state benefit plus a power-law odd/even phase block."""
    state = consolidated_state(panel, shape)
    total = panel.phase0 * panel.c0 + panel.phase1 * panel.c1
    # Per-bucket saturation rate as a FUNCTION of corpus size rather than 39 free
    # parameters. Physically a small curated set and a large web dump should not
    # saturate alike, but HSB showed 39 shrunk per-bucket coefficients win in panel
    # and lose 12 of 12 on heldout. This costs one shared parameter: rate_exponent
    # = 0 recovers the flat rate exactly.
    rate = float(shape["rate"])
    exponent = float(shape.get("rate_exponent", 0.0))
    if exponent != 0.0:
        rate = rate * (panel.c1 / np.median(panel.c1)) ** (-exponent)
    benefit = _weibull(state, rate, float(shape["power"]))
    replay = np.sum(np.maximum(total - 1.0, 0.0) ** 2, axis=1, keepdims=True)

    radius = np.maximum(phase_radius(panel), CONTRAST_EPSILON)
    if shape.get("even_mode", "power") == "jensen":
        even_cost = jensen_even_cost(panel)
    else:
        even_cost = radius ** float(shape["even_exponent"])
    # Scaling the signed family contrast by rho^(p-1) makes the odd contribution
    # scale as rho^p overall, matching the measured ordering law.
    odd_scale = radius ** (float(shape["odd_exponent"]) - 1.0)
    # The odd channel is a RECENCY effect: its size scales with how many training
    # tokens of a bucket move late, which is d_i, not with epochs of the source
    # corpus, d_i * c_1i. Epoch weighting is right for the even (repetition)
    # channel and wrong here: c_1 spans 0.96 to 344.8, and applying it collapses
    # the 300M panel's contrast design from effective rank 31.46 to 1.80, leaving
    # two single-bucket directions (wikipedia, stem_heavy_crawl) at 95 percent of
    # variance. "epoch" is retained only to reproduce the earlier variant.
    weights = panel.phase1 - panel.phase0
    if shape.get("odd_weighting", "token") == "epoch":
        weights = weights * panel.c1
    group, group_names = partition_of(panel, shape)
    family_contrast = np.column_stack([weights[:, group == k].sum(axis=1) for k in range(len(group_names))])
    late = odd_scale * np.maximum(family_contrast, 0.0)
    early = odd_scale * np.maximum(-family_contrast, 0.0)

    return Design(
        matrix=np.hstack([-benefit, replay, even_cost, -late, -early]),
        names=tuple(
            [
                *(f"retained_benefit:{bucket}" for bucket in panel.buckets),
                "shared_literal_replay",
                "even_phase_cost",
                *(f"odd_late_benefit:{family}" for family in group_names),
                *(f"odd_early_benefit:{family}" for family in group_names),
            ]
        ),
    )


def phase_law_shapes() -> Iterator[dict]:
    """Retained-state shapes crossed with the two phase-law exponents.

    ``rate`` and ``power`` span the compact retained-state grid so the aggregate
    core can reproduce its baseline exactly; the phase block is what is new.
    """
    for rate in (0.25, 1.0):
        for power in (0.4, 0.7, 1.0):
            for late_multiplier in (0.5, 1.0, 2.0, 4.0):
                for forgetting_rate in (0.0, 0.25, 1.0):
                    for odd_exponent in ODD_EXPONENTS:
                        for even_exponent in EVEN_EXPONENTS:
                            yield {
                                "rate": rate,
                                "power": power,
                                "late_multiplier": late_multiplier,
                                "forgetting_rate": forgetting_rate,
                                "odd_exponent": odd_exponent,
                                "even_exponent": even_exponent,
                            }


def phase_law_model() -> Model:
    return Model(name="phase_law_crs", build=build_phase_law, shapes=phase_law_shapes)


def build_phase_law_hybrid(panel: Panel, shape: dict) -> Design:
    """``crs_plus_phase`` plus the even-cost column, which is what actually won.

    ``phase_law_crs`` lost the phase-skill gate it was designed for and won
    proposal quality instead, which points at the ``rho^q`` even-cost column
    rather than the power-law odd block: that column penalizes high-contrast
    policies, exactly the ones a surrogate over-favours in the frontier region.
    ``crs_plus_phase`` carries the best transferable Table-9 phase skill of the
    incumbents. This grafts the one column onto that model and changes nothing
    else, so any gain is attributable to the even cost alone.
    """
    base = build_crs_plus_phase(panel, shape)
    radius = np.maximum(phase_radius(panel), CONTRAST_EPSILON)
    if shape.get("even_mode", "power") == "jensen":
        even_cost = jensen_even_cost(panel)
    else:
        even_cost = radius ** float(shape["even_exponent"])
    return Design(matrix=np.hstack([base.matrix, even_cost]), names=(*base.names, "even_phase_cost"))


def phase_law_hybrid_shapes() -> Iterator[dict]:
    """The crs_plus grid crossed with the even exponent only."""
    for shape in crs_plus_shapes():
        for even_exponent in EVEN_EXPONENTS:
            yield {**shape, "even_exponent": even_exponent}


def phase_law_hybrid_model() -> Model:
    return Model(name="phase_law_hybrid", build=build_phase_law_hybrid, shapes=phase_law_hybrid_shapes)


# Every model measured at its own constrained optimum predicts 0.073 to 0.181 BPB
# below the best policy actually observed nearby, which is 7 to 18 times the
# target gain and is almost pure bias rather than variance. An additive response
# can predict below any entropy floor; a multiplicative one cannot. These floors
# admit a 1 to 10 percent improvement on the smallest observed target while
# blocking the 8 to 17 percent optimism the additive models exhibit.
# The floor is FIXED, not cross-validated. Out-of-fold RMSE is flat to the fourth
# decimal across floors 0.5 to 0.95 (0.01701 to 0.01720 on Table-9) while excess
# local optimism at the proposed optimum varies 3.6-fold (-0.01017 at no floor,
# -0.00279 at 0.95), so cross-validation is structurally blind to the quantity the
# floor exists to control. Left to CV it selects 0.90, which is 38 percent worse on
# calibration than 0.95. Selecting it on frontier calibration costs nothing in
# out-of-fold terms. The archive ranking is invariant to the floor: top-5 percentile
# is identical at every value tested, so this choice moves only the calibration and
# the aggressiveness of the continuous optimum.
DEFICIT_FLOOR_FRACTION = 0.95


def phase_law_bounded_shapes(
    odd_weighting: str = "token", even_mode: str = "power", consolidations: tuple[float, ...] = (0.0,)
) -> Iterator[dict]:
    """Phase-law shapes crossed with the structural floor, odd exponent pinned.

    The odd exponent is fixed at its mid value because the floor, not the odd
    power, is what this variant is testing, and the grid would otherwise be four
    times larger than the additive model's.
    """
    # late_multiplier extends past 4.0 and even_exponent below 2.0 because the
    # original grid selected both at an edge.
    for consolidation in consolidations:
        for rate in (0.25, 1.0):
            for power in (0.4, 0.7, 1.0):
                for late_multiplier in (0.5, 1.0, 2.0, 4.0, 8.0):
                    for forgetting_rate in (0.0, 0.25, 1.0):
                        for even_exponent in (1.5, *EVEN_EXPONENTS) if even_mode == "power" else (0.0,):
                            yield {
                                "rate": rate,
                                "power": power,
                                "late_multiplier": late_multiplier,
                                "forgetting_rate": forgetting_rate,
                                "odd_exponent": 1.5,
                                "even_exponent": even_exponent,
                                "deficit_floor_fraction": DEFICIT_FLOOR_FRACTION,
                                "odd_weighting": odd_weighting,
                                "even_mode": even_mode,
                                "consolidation": consolidation,
                            }


def phase_law_bounded_model(odd_weighting: str = "token") -> Model:
    """The phase law fitted on log reducible loss, so predictions cannot undercut the floor."""
    return Model(
        name=f"phase_law_bounded_{odd_weighting}odd",
        build=build_phase_law,
        shapes=lambda: phase_law_bounded_shapes(odd_weighting),
        link="log_deficit",
    )


def quality_label(bucket: str) -> str:
    match = re.match(r".*_(high|low)$", bucket)
    return match.group(1) if match else "none"


def partition_index(panel: Panel, scheme: str) -> np.ndarray:
    """Bucket grouping under a named scheme. ``size5`` uses no labels at all."""
    if scheme == "current3":
        return np.asarray(panel.family_index)
    if scheme == "current3x_quality":
        keys = [f"{panel.family_index[i]}|{quality_label(b)}" for i, b in enumerate(panel.buckets)]
    elif scheme == "size5":
        edges = np.quantile(panel.c1, [0.2, 0.4, 0.6, 0.8])
        keys = [str(int(np.searchsorted(edges, c))) for c in panel.c1]
    else:
        raise ValueError(f"unknown partition scheme {scheme!r}")
    order = sorted(set(keys))
    return np.asarray([order.index(k) for k in keys], dtype=int)


CONSOLIDATION_GRID = (0.0, 0.25, 0.5, 1.0)


def phase_law_partition_model(
    panel: Panel,
    scheme: str,
    odd_weighting: str = "epoch",
    even_mode: str = "power",
    consolidation: bool = False,
) -> Model:
    """Phase-law variant with an explicit bucket partition for the odd block."""
    index = partition_index(panel, scheme).tolist()
    grid = CONSOLIDATION_GRID if consolidation else (0.0,)

    def shapes() -> Iterator[dict]:
        for shape in phase_law_bounded_shapes(odd_weighting, even_mode, grid):
            yield {**shape, "family_index": index}

    tag = f"{scheme}_{odd_weighting}_{even_mode}{'_consol' if consolidation else ''}"
    return Model(name=f"plaw_{tag}", build=build_phase_law, shapes=shapes, link="log_deficit")


def build_enriched_crs(panel: Panel, shape: dict) -> Design:
    """Compact retained state with a consolidation-gated retention, and NOTHING else.

    This is the direct test of the observation that motivated the enrichment:
    compact retained state carries no dedicated phase columns, expresses ordering
    through two shared nonlinear parameters, and still matches or beats every
    explicit phase block built in this experiment. If that is because the implicit
    mechanism is the right place to spend capacity, then enriching it should beat
    both plain compact retained state and the phase-block models WITHOUT adding a
    single phase column. 41 columns against the phase law's 51.
    """
    state = consolidated_state(panel, shape)
    total = panel.phase0 * panel.c0 + panel.phase1 * panel.c1
    # Per-bucket saturation rate as a FUNCTION of corpus size rather than 39 free
    # parameters. Physically a small curated set and a large web dump should not
    # saturate alike, but HSB showed 39 shrunk per-bucket coefficients win in panel
    # and lose 12 of 12 on heldout. This costs one shared parameter: rate_exponent
    # = 0 recovers the flat rate exactly.
    rate = float(shape["rate"])
    exponent = float(shape.get("rate_exponent", 0.0))
    if exponent != 0.0:
        rate = rate * (panel.c1 / np.median(panel.c1)) ** (-exponent)
    benefit = _weibull(state, rate, float(shape["power"]))
    replay = np.sum(np.maximum(total - 1.0, 0.0) ** 2, axis=1, keepdims=True)
    return Design(
        matrix=np.hstack([-benefit, replay]),
        names=tuple([*(f"retained_benefit:{b}" for b in panel.buckets), "shared_literal_replay"]),
    )


RECENCY_SATURATIONS = (np.inf, 30.0, 10.0, 3.0)


def enriched_crs_shapes(
    consolidations: tuple[float, ...] = CONSOLIDATION_GRID,
    saturations: tuple[float, ...] = (np.inf,),
) -> Iterator[dict]:
    for recency_saturation in saturations:
        for consolidation in consolidations:
            for rate in (0.25, 1.0):
                for power in (0.4, 0.7, 1.0):
                    for late_multiplier in (0.5, 1.0, 2.0, 4.0, 8.0):
                        for forgetting_rate in (0.0, 0.25, 1.0):
                            yield {
                                "rate": rate,
                                "power": power,
                                "late_multiplier": late_multiplier,
                                "forgetting_rate": forgetting_rate,
                                "consolidation": consolidation,
                                "recency_saturation": recency_saturation,
                                "deficit_floor_fraction": DEFICIT_FLOOR_FRACTION,
                            }


def enriched_crs_model(bounded: bool = True, consolidation: bool = True) -> Model:
    """Compact retained state, optionally consolidation-gated and floor-bounded."""
    grid = CONSOLIDATION_GRID if consolidation else (0.0,)
    return Model(
        name=f"crs{'_consol' if consolidation else ''}{'_bounded' if bounded else ''}",
        build=build_enriched_crs,
        shapes=lambda: enriched_crs_shapes(grid),
        link="log_deficit" if bounded else "identity",
    )
