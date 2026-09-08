# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "scipy",
# ]
# ///
"""Observatory baselines and unique-coverage candidates for the 39-bucket swarm.

Baselines reproduce the maintained common-API implementations in the collaborator
packet (``standalone_code/reference_models.py``) so that a candidate is compared
against the same response laws the Observatory reports. They are cross-checked
against the packet numerically in ``benchmark_swarm39_models_20260725.py``.

The candidate family is motivated by a measured property of this swarm rather
than by a new functional form. Proportional sampling gives every bucket 0.905
epochs, and the fit panels oversample aggressively: the median oversampling ratio
is close to one but the 99th percentile is about 117x and the maximum about 283x,
so roughly 44 percent of (policy, bucket) cells exceed one epoch and
``dolma3_wikipedia`` sits near 91 epochs in the median fit policy. A response that
keeps rewarding exposure past one pass is therefore crediting re-reads of the same
tokens, which is the mechanism that lets an optimizer pile mass onto small
high-value corpora and produce an unsupported optimum.

Marin's simulated epoching materializes a fixed subset per bucket and recycles it,
so the distinct fraction seen is exactly ``min(E, 1)``. The candidates split
exposure into that saturating unique-coverage term, a repeat dose with
logarithmic returns, and an overload penalty.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from swarm39_harness_20260725 import PROPORTIONAL_POLICY_EPOCHS, Design, Model, Panel

EPSILON = 1e-12


def _softplus(value: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, value)


# ---------------------------------------------------------------------------
# Observatory baselines
# ---------------------------------------------------------------------------


def _retained_state(panel: Panel, shape: dict) -> np.ndarray:
    """Packet ``retained_state``: early exposure decayed by late absence, plus late."""
    early = panel.phase0 * panel.c0
    late = panel.phase1 * panel.c1
    retained = np.exp(-float(shape["forgetting_rate"]) * (1.0 - panel.phase1)) * early
    return np.maximum(retained + float(shape["late_multiplier"]) * late, 0.0)


def _weibull(exposure: np.ndarray, rate: float, power: float) -> np.ndarray:
    return -np.expm1(-((np.maximum(rate * exposure, 0.0)) ** power))


def _power(exposure: np.ndarray, power: float) -> np.ndarray:
    return np.maximum(exposure, EPSILON) ** power


def _overexposure(exposure: np.ndarray, threshold: float) -> np.ndarray:
    return _softplus(np.log1p(np.maximum(exposure, 0.0)) - threshold) ** 2


def _state_shapes(include_threshold: bool) -> Iterator[dict]:
    thresholds = (0.0, 1.0, 2.0) if include_threshold else (0.0,)
    for rate in (0.25, 1.0):
        for power in (0.4, 0.7, 1.0):
            for late_multiplier in (0.5, 1.0, 2.0, 4.0):
                for forgetting_rate in (0.0, 0.25, 1.0):
                    for threshold in thresholds:
                        yield {
                            "rate": rate,
                            "power": power,
                            "late_multiplier": late_multiplier,
                            "forgetting_rate": forgetting_rate,
                            "penalty_threshold": threshold,
                        }


def build_compact_retained_state(panel: Panel, shape: dict) -> Design:
    state = _retained_state(panel, shape)
    total = panel.phase0 * panel.c0 + panel.phase1 * panel.c1
    signal = _weibull(state, float(shape["rate"]), float(shape["power"]))
    replay = np.sum(np.maximum(total - 1.0, 0.0) ** 2, axis=1, keepdims=True)
    return Design(
        matrix=np.hstack([-signal, replay]),
        names=tuple([*(f"retained_benefit:{b}" for b in panel.buckets), "shared_literal_replay"]),
    )


def build_bucket_family_grp(panel: Panel, shape: dict) -> Design:
    state = _retained_state(panel, shape)
    power = float(shape["power"])
    threshold = float(shape["penalty_threshold"])
    family_total = panel.family_pool(state)
    return Design(
        matrix=np.hstack([-_power(state, power), -_power(family_total, power), _overexposure(family_total, threshold)]),
        names=tuple(
            [
                *(f"bucket_benefit:{b}" for b in panel.buckets),
                *(f"family_benefit:{f}" for f in panel.family_names),
                *(f"family_replay:{f}" for f in panel.family_names),
            ]
        ),
    )


def build_hierarchical_phase_replay(panel: Panel, shape: dict) -> Design:
    state = _retained_state(panel, shape)
    power = float(shape["power"])
    threshold = float(shape["penalty_threshold"])
    bucket_signal = _power(state, power)
    family_total = panel.family_pool(state)
    family_signal = _power(family_total, power)
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for index, family in enumerate(panel.family_names):
        members = np.flatnonzero(panel.family_index == index)
        if len(members) == 1:
            blocks.append(-bucket_signal[:, members])
            names.append(f"singleton_benefit:{panel.buckets[int(members[0])]}")
            continue
        blocks.append(-bucket_signal[:, members].sum(axis=1, keepdims=True))
        names.append(f"pooled_family_benefit:{family}")
        blocks.append(-bucket_signal[:, members])
        names.extend(f"bucket_residual:{panel.buckets[int(m)]}" for m in members)
    blocks.extend([-family_signal, _overexposure(family_total, threshold)])
    names.extend(f"family_coverage:{f}" for f in panel.family_names)
    names.extend(f"family_overexposure:{f}" for f in panel.family_names)
    bucket_replay = _overexposure(state, threshold)
    blocks.append(panel.family_pool(bucket_replay) / np.bincount(panel.family_index, minlength=len(panel.family_names)))
    names.extend(f"family_member_replay:{f}" for f in panel.family_names)
    blocks.append(panel.phase_tv[:, None])
    names.append("phase_shift_tv")
    return Design(matrix=np.hstack(blocks), names=tuple(names))


def build_separate_heads(panel: Panel, shape: dict) -> Design:
    early = panel.phase0 * panel.c0
    late = panel.phase1 * panel.c1
    mu0 = np.asarray(shape["mu0"], dtype=float)
    mu1 = np.asarray(shape["mu1"], dtype=float)
    d0 = np.log1p(early) - mu0
    d1 = np.log1p(late) - mu1
    return Design(
        matrix=np.hstack(
            [np.minimum(d0, 0.0) ** 2, np.maximum(d0, 0.0) ** 2, np.minimum(d1, 0.0) ** 2, np.maximum(d1, 0.0) ** 2]
        ),
        names=tuple(
            f"phase{phase}:{side}:{b}" for phase in range(2) for side in ("under", "over") for b in panel.buckets
        ),
    )


def _head_centre(exposure: np.ndarray, shift: float) -> np.ndarray:
    positive = np.where(exposure > 1e-8, exposure, np.nan)
    with np.errstate(invalid="ignore"):
        median = np.nanmedian(np.log1p(positive), axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    return np.clip(median + shift, -2.0, 8.0)


def separate_head_shapes(panel: Panel) -> Iterator[dict]:
    early = panel.phase0 * panel.c0
    late = panel.phase1 * panel.c1
    for shift0 in (-1.0, 0.0, 1.0):
        for shift1 in (-1.0, 0.0, 1.0):
            yield {"mu0": _head_centre(early, shift0).tolist(), "mu1": _head_centre(late, shift1).tolist()}


def build_effective_exposure_dsp(panel: Panel, shape: dict) -> Design:
    """Phase-weighted effective exposure with a saturating power response.

    This is the DSP form: a single late multiplier reweights phase-1 exposure and
    the response is a fitted power of the resulting effective dose.
    """
    effective = panel.phase0 * panel.c0 + float(shape["late_multiplier"]) * panel.phase1 * panel.c1
    family_total = panel.family_pool(effective)
    return Design(
        matrix=np.hstack(
            [
                -_power(effective, float(shape["power"])),
                -_power(family_total, float(shape["power"])),
                np.sum(np.maximum(effective - 1.0, 0.0) ** 2, axis=1, keepdims=True),
            ]
        ),
        names=tuple(
            [
                *(f"effexp_benefit:{b}" for b in panel.buckets),
                *(f"effexp_family:{f}" for f in panel.family_names),
                "shared_literal_replay",
            ]
        ),
    )


def dsp_shapes() -> Iterator[dict]:
    for power in (0.4, 0.7, 1.0):
        for late_multiplier in (0.5, 1.0, 2.0, 4.0):
            yield {"power": power, "late_multiplier": late_multiplier}


# ---------------------------------------------------------------------------
# Unique-coverage candidates
# ---------------------------------------------------------------------------


def coverage_parts(panel: Panel, shape: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split exposure into unique coverage, repeat dose, and overload load.

    ``unique = min(E, 1)`` is exact for a materialized subset that is recycled:
    once a bucket has been traversed there are no further distinct tokens. The
    repeat dose uses logarithmic returns, the standard empirical shape for
    multi-epoch training, and overload is a squared excess above a threshold in
    epochs.
    """
    epochs = panel.epochs
    unique = np.minimum(epochs, 1.0)
    repeat = np.log1p(np.maximum(epochs - 1.0, 0.0))
    overload = np.maximum(epochs - float(shape["overload_threshold"]), 0.0) ** 2
    return unique, repeat, overload


def phase_intensity(panel: Panel) -> np.ndarray:
    """Within-phase repetition intensity, the phase-sensitive part of exposure.

    Total epochs depend only on the aggregate, but their concentration in time
    does not. A bucket packed into a short phase is re-read back to back at
    higher intensity than the same total spread across training. Intensity is
    epochs divided by the fraction of training the phase occupies.
    """
    early = panel.c0 * panel.phase0 / panel.alpha
    late = panel.c1 * panel.phase1 / (1.0 - panel.alpha)
    baseline = panel.epochs
    return np.maximum(np.maximum(early, late) - baseline, 0.0)


def build_unique_coverage(panel: Panel, shape: dict) -> Design:
    """UCR: unique coverage, logarithmic repeat returns, and overload harm.

    Aggregate-only by construction, so this is the tied backbone. The benefit
    gradient vanishes above one epoch, which is what bounds the optimum.
    """
    unique, repeat, overload = coverage_parts(panel, shape)
    return Design(
        matrix=np.hstack(
            [
                -unique,
                -panel.family_pool(repeat),
                panel.family_pool(overload),
                -panel.family_pool(unique),
            ]
        ),
        names=tuple(
            [
                *(f"unique_benefit:{b}" for b in panel.buckets),
                *(f"repeat_benefit:{f}" for f in panel.family_names),
                *(f"overload_harm:{f}" for f in panel.family_names),
                *(f"family_coverage:{f}" for f in panel.family_names),
            ]
        ),
    )


def build_unique_coverage_phase(panel: Panel, shape: dict) -> Design:
    """UCR-P: the coverage backbone plus a phase block.

    The phase block has two parts. A within-phase intensity penalty is even in
    the contrast and physically grounded in back-to-back re-reading. A
    family-pooled recency term is the odd part, kept to three parameters because
    a higher-dimensional ordering field is not identifiable from this design.
    """
    base = build_unique_coverage(panel, shape)
    intensity = panel.family_pool(phase_intensity(panel))
    recency = panel.family_pool(panel.contrast)
    return Design(
        matrix=np.hstack([base.matrix, intensity, -recency, recency]),
        names=tuple(
            [
                *base.names,
                *(f"phase_intensity_harm:{f}" for f in panel.family_names),
                *(f"late_benefit:{f}" for f in panel.family_names),
                *(f"early_benefit:{f}" for f in panel.family_names),
            ]
        ),
    )


def coverage_shapes() -> Iterator[dict]:
    for overload_threshold in (1.0, 2.0, 4.0, 8.0):
        yield {"overload_threshold": overload_threshold}


def discounted_dose(panel: Panel, shape: dict) -> np.ndarray:
    """Effective distinct-token dose with a discounted repeat channel.

    The hard cap ``min(E, 1)`` removes the optimism that comes from crediting
    re-reads, but it also discards all design variation above one epoch, which is
    where 44 to 67 percent of cells sit. Keeping a logarithmic repeat channel with
    a single global discount ``delta`` restores per-bucket resolution while
    holding the marginal value of repetition far below that of fresh tokens.
    ``delta = 0`` recovers the hard cap.
    """
    epochs = panel.epochs
    return np.minimum(epochs, 1.0) + float(shape["repeat_discount"]) * np.log1p(np.maximum(epochs - 1.0, 0.0))


def build_discounted_coverage(panel: Panel, shape: dict) -> Design:
    """DRC: discounted-repeat coverage with family complementarity and overload harm."""
    dose = discounted_dose(panel, shape)
    power = float(shape["power"])
    overload = np.maximum(panel.epochs - float(shape["overload_threshold"]), 0.0) ** 2
    return Design(
        matrix=np.hstack(
            [
                -_power(dose, power),
                -_power(panel.family_pool(dose), power),
                panel.family_pool(overload),
            ]
        ),
        names=tuple(
            [
                *(f"dose_benefit:{b}" for b in panel.buckets),
                *(f"family_dose:{f}" for f in panel.family_names),
                *(f"overload_harm:{f}" for f in panel.family_names),
            ]
        ),
    )


def build_discounted_coverage_phase(panel: Panel, shape: dict) -> Design:
    """DRC-P: the discounted-coverage backbone plus the phase block."""
    base = build_discounted_coverage(panel, shape)
    intensity = panel.family_pool(phase_intensity(panel))
    recency = panel.family_pool(panel.contrast)
    return Design(
        matrix=np.hstack([base.matrix, intensity, -recency, recency]),
        names=tuple(
            [
                *base.names,
                *(f"phase_intensity_harm:{f}" for f in panel.family_names),
                *(f"late_benefit:{f}" for f in panel.family_names),
                *(f"early_benefit:{f}" for f in panel.family_names),
            ]
        ),
    )


def discounted_shapes() -> Iterator[dict]:
    for repeat_discount in (0.0, 0.05, 0.1, 0.2, 0.4):
        for power in (0.7, 1.0):
            for overload_threshold in (1.0, 2.0, 4.0):
                yield {
                    "repeat_discount": repeat_discount,
                    "power": power,
                    "overload_threshold": overload_threshold,
                }


def build_bounded_saturation(panel: Panel, shape: dict) -> Design:
    """Compact retained state with the saturation scale disciplined to epochs.

    Structurally this is the packet's compact retained state, but the Weibull
    scale is restricted so that the response saturates within a few epochs
    instead of absorbing the ninety-epoch tail. It isolates whether the
    baseline's out-of-support optimism comes from an unbounded-in-practice
    saturation scale rather than from its state or its penalty.
    """
    state = _retained_state(panel, shape)
    signal = _weibull(state, 1.0 / float(shape["saturation_epochs"]), float(shape["power"]))
    overload = np.maximum(panel.epochs - float(shape["overload_threshold"]), 0.0) ** 2
    return Design(
        matrix=np.hstack([-signal, panel.family_pool(overload)]),
        names=tuple(
            [
                *(f"bounded_benefit:{b}" for b in panel.buckets),
                *(f"overload_harm:{f}" for f in panel.family_names),
            ]
        ),
    )


def bounded_saturation_shapes() -> Iterator[dict]:
    """Shape grid for bounded saturation.

    The saturation range extends to 64 epochs so the selection is interior rather
    than clamped. On the Delphi panel the widened grid selects 8 epochs for both
    targets, which is the physically interesting answer: benefit keeps accruing
    for several passes and then flattens, well short of the ninety-epoch tail the
    unconstrained baselines credit.
    """
    for saturation_epochs in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0):
        for power in (0.7, 1.0):
            for late_multiplier in (0.5, 1.0, 2.0):
                for forgetting_rate in (0.0, 0.25):
                    for overload_threshold in (1.0, 2.0, 4.0):
                        yield {
                            "saturation_epochs": saturation_epochs,
                            "power": power,
                            "late_multiplier": late_multiplier,
                            "forgetting_rate": forgetting_rate,
                            "overload_threshold": overload_threshold,
                        }


def observatory_baselines(panel: Panel) -> list[Model]:
    """Baseline models with the packet's shape grids."""
    return [
        Model("compact_retained_state", build_compact_retained_state, lambda: _state_shapes(False)),
        Model("bucket_family_grp", build_bucket_family_grp, lambda: _state_shapes(True)),
        Model("hierarchical_phase_replay", build_hierarchical_phase_replay, lambda: _state_shapes(True)),
        Model("separate_heads", build_separate_heads, lambda: separate_head_shapes(panel)),
        Model("effective_exposure_dsp", build_effective_exposure_dsp, dsp_shapes),
    ]


def candidates() -> list[Model]:
    return [
        Model("unique_coverage", build_unique_coverage, coverage_shapes),
        Model("unique_coverage_phase", build_unique_coverage_phase, coverage_shapes),
        Model("discounted_coverage", build_discounted_coverage, discounted_shapes),
        Model("discounted_coverage_phase", build_discounted_coverage_phase, discounted_shapes),
        Model("bounded_saturation", build_bounded_saturation, bounded_saturation_shapes),
    ]


def build_bounded_hierarchical(panel: Panel, shape: dict) -> Design:
    """Bounded saturation with the family-benefit term the baselines carry.

    ``bounded_saturation`` has per-bucket benefit and a family overload penalty but
    no family-level benefit, while GRP and hierarchical phase replay both pool
    benefit at the family level. Family benefit encodes complementarity: a broad
    evaluation needs several capabilities, so surplus in one family cannot fully
    substitute for a starved one. Adding it back keeps the disciplined saturation
    scale while restoring the resolution the baselines get from hierarchy.
    """
    state = _retained_state(panel, shape)
    scale = 1.0 / float(shape["saturation_epochs"])
    power = float(shape["power"])
    family_state = panel.family_pool(state)
    overload = np.maximum(panel.epochs - float(shape["overload_threshold"]), 0.0) ** 2
    return Design(
        matrix=np.hstack(
            [
                -_weibull(state, scale, power),
                -_weibull(family_state, scale / len(panel.family_names), power),
                panel.family_pool(overload),
            ]
        ),
        names=tuple(
            [
                *(f"bounded_benefit:{b}" for b in panel.buckets),
                *(f"family_benefit:{f}" for f in panel.family_names),
                *(f"overload_harm:{f}" for f in panel.family_names),
            ]
        ),
    )


def build_bounded_hierarchical_phase(panel: Panel, shape: dict) -> Design:
    """Bounded hierarchical backbone plus the within-phase intensity and recency block."""
    base = build_bounded_hierarchical(panel, shape)
    intensity = panel.family_pool(phase_intensity(panel))
    recency = panel.family_pool(panel.contrast)
    return Design(
        matrix=np.hstack([base.matrix, intensity, -recency, recency]),
        names=tuple(
            [
                *base.names,
                *(f"phase_intensity_harm:{f}" for f in panel.family_names),
                *(f"late_benefit:{f}" for f in panel.family_names),
                *(f"early_benefit:{f}" for f in panel.family_names),
            ]
        ),
    )


def hierarchical_candidates() -> list[Model]:
    return [
        Model("bounded_hierarchical", build_bounded_hierarchical, bounded_saturation_shapes),
        Model("bounded_hierarchical_phase", build_bounded_hierarchical_phase, bounded_saturation_shapes),
    ]


def build_crs_plus(panel: Panel, shape: dict) -> Design:
    """Compact retained state, strictly nested, plus two structural penalties.

    Under cross-scale selection, compact retained state chooses
    ``rate = 0.25, power = 0.7, forgetting_rate = 1.0, late_multiplier = 2.0``.
    A saturation scale of 4 epochs is exactly ``rate = 0.25``, so the earlier
    bounded-saturation candidate differed from the baseline only in its penalty and
    in a forgetting grid that excluded 1.0, which locked it out of the baseline's
    own best configuration.

    This design fixes that. It contains compact retained state exactly: setting the
    family-benefit and family-overload coefficients to zero recovers the baseline's
    per-bucket benefit plus shared literal replay, and nonnegative least squares can
    select zero. So it cannot be worse in sample, and any gain has to come from the
    two added blocks earning their place:

    * family benefit, the complementarity term GRP and hierarchical phase replay
      carry and the earlier candidate lacked;
    * family-pooled overload above a threshold in epochs, which prices repetition
      per family rather than through one global scalar.
    """
    state = _retained_state(panel, shape)
    scale = 1.0 / float(shape["saturation_epochs"])
    power = float(shape["power"])
    total = panel.epochs
    family_state = panel.family_pool(state)
    overload = np.maximum(total - float(shape["overload_threshold"]), 0.0) ** 2
    literal_replay = np.sum(np.maximum(total - 1.0, 0.0) ** 2, axis=1, keepdims=True)
    return Design(
        matrix=np.hstack(
            [
                -_weibull(state, scale, power),
                -_weibull(family_state, scale / len(panel.family_names), power),
                literal_replay,
                panel.family_pool(overload),
            ]
        ),
        names=tuple(
            [
                *(f"retained_benefit:{b}" for b in panel.buckets),
                *(f"family_benefit:{f}" for f in panel.family_names),
                "shared_literal_replay",
                *(f"family_overload:{f}" for f in panel.family_names),
            ]
        ),
    )


def build_crs_plus_phase(panel: Panel, shape: dict) -> Design:
    """CRS-plus with the within-phase intensity and family recency block."""
    base = build_crs_plus(panel, shape)
    intensity = panel.family_pool(phase_intensity(panel))
    recency = panel.family_pool(panel.contrast)
    return Design(
        matrix=np.hstack([base.matrix, intensity, -recency, recency]),
        names=tuple(
            [
                *base.names,
                *(f"phase_intensity_harm:{f}" for f in panel.family_names),
                *(f"late_benefit:{f}" for f in panel.family_names),
                *(f"early_benefit:{f}" for f in panel.family_names),
            ]
        ),
    )


def crs_plus_shapes() -> Iterator[dict]:
    """Grid that spans the baseline's own selected configuration.

    ``saturation_epochs = 4`` reproduces the baseline's ``rate = 0.25`` and
    ``forgetting_rate`` now includes 1.0, so the nesting is reachable by search and
    not only in principle.
    """
    for saturation_epochs in (1.0, 2.0, 4.0, 8.0, 16.0, 64.0):
        for power in (0.4, 0.7, 1.0):
            for late_multiplier in (0.5, 1.0, 2.0, 4.0):
                for forgetting_rate in (0.0, 0.25, 1.0):
                    for overload_threshold in (1.0, 2.0, 4.0):
                        yield {
                            "saturation_epochs": saturation_epochs,
                            "power": power,
                            "late_multiplier": late_multiplier,
                            "forgetting_rate": forgetting_rate,
                            "overload_threshold": overload_threshold,
                        }


def nested_candidates() -> list[Model]:
    return [
        Model("crs_plus", build_crs_plus, crs_plus_shapes),
        Model("crs_plus_phase", build_crs_plus_phase, crs_plus_shapes),
    ]


def bucket_features(panel: Panel) -> tuple[np.ndarray, tuple[str, ...]]:
    """Nonnegative observable properties of each bucket, used to structure benefit.

    Every column is measurable before any training run: corpus size through the
    token-proportional weight, the declared family, the quality split encoded in
    the bucket name, and the source collection. Nonnegativity keeps a nonnegative
    coefficient vector interpretable as marginal value per unit of acquired state.
    """
    proportional = panel.proportional
    log_size = np.log(proportional)
    span = log_size.max() - log_size.min()
    scaled = (log_size - log_size.min()) / span
    columns: list[np.ndarray] = [np.ones(len(panel.buckets))]
    names: list[str] = ["intercept"]
    for index, family in enumerate(panel.family_names):
        columns.append((panel.family_index == index).astype(float))
        names.append(f"family:{family}")
    for label, test in (
        ("quality_high", lambda b: b.endswith("_high")),
        ("quality_low", lambda b: b.endswith("_low")),
        ("dolmino", lambda b: b.startswith("dolmino_")),
        ("synthetic", lambda b: "synth" in b),
        ("common_crawl", lambda b: "_cc/" in b),
    ):
        columns.append(np.asarray([float(test(b)) for b in panel.buckets]))
        names.append(label)
    # Both orientations of corpus size, so a nonnegative coefficient vector can
    # express value rising or falling with the size of the available pool.
    columns.append(scaled)
    names.append("large_corpus")
    columns.append(1.0 - scaled)
    names.append("small_corpus")
    return np.column_stack(columns), tuple(names)


def build_structured_benefit(panel: Panel, shape: dict) -> Design:
    """SBF: per-bucket marginal value is a function of observables, not 39 free numbers.

    Compact retained state spends 39 free coefficients on per-bucket benefit. That
    is the capacity that overfits a 280-row panel and that lets an optimizer pile
    mass onto whichever bucket drew the largest coefficient. Since adding capacity
    to this swarm was shown to damage extrapolation even when the larger model
    strictly nested the smaller, the move here is the opposite one: constrain the
    benefit field.

    Writing the benefit coefficients as ``beta = U gamma`` for a fixed matrix of
    bucket observables ``U`` makes the response

        sum_i beta_i g(state_i) = gamma . (U^T g(state)),

    so the design has one column per observable rather than one per bucket, and the
    model drops from 40 columns to 12. A bucket can only earn a large marginal value
    if its measurable properties justify it, which is what rules out the
    single-bucket optima the high-capacity baselines produce.
    """
    features, feature_names = bucket_features(panel)
    state = _retained_state(panel, shape)
    signal = _weibull(state, 1.0 / float(shape["saturation_epochs"]), float(shape["power"]))
    total = panel.epochs
    literal_replay = np.sum(np.maximum(total - 1.0, 0.0) ** 2, axis=1, keepdims=True)
    return Design(
        matrix=np.hstack([-(signal @ features), literal_replay]),
        names=tuple([*(f"benefit:{n}" for n in feature_names), "shared_literal_replay"]),
    )


def build_structured_benefit_phase(panel: Panel, shape: dict) -> Design:
    """SBF with a family-pooled recency block and within-phase intensity penalty."""
    base = build_structured_benefit(panel, shape)
    intensity = panel.family_pool(phase_intensity(panel))
    recency = panel.family_pool(panel.contrast)
    return Design(
        matrix=np.hstack([base.matrix, intensity, -recency, recency]),
        names=tuple(
            [
                *base.names,
                *(f"phase_intensity_harm:{f}" for f in panel.family_names),
                *(f"late_benefit:{f}" for f in panel.family_names),
                *(f"early_benefit:{f}" for f in panel.family_names),
            ]
        ),
    )


def structured_shapes() -> Iterator[dict]:
    for saturation_epochs in (1.0, 2.0, 4.0, 8.0, 16.0, 64.0):
        for power in (0.4, 0.7, 1.0):
            for late_multiplier in (0.5, 1.0, 2.0, 4.0):
                for forgetting_rate in (0.0, 0.25, 1.0):
                    yield {
                        "saturation_epochs": saturation_epochs,
                        "power": power,
                        "late_multiplier": late_multiplier,
                        "forgetting_rate": forgetting_rate,
                    }


def structured_candidates() -> list[Model]:
    return [
        Model("structured_benefit", build_structured_benefit, structured_shapes),
        Model("structured_benefit_phase", build_structured_benefit_phase, structured_shapes),
    ]


def build_multiplicative_deficit(panel: Panel, shape: dict) -> Design:
    """MDL: reducible loss decays multiplicatively, so BPB is bounded below by a floor.

    Every Observatory baseline and every earlier candidate in this round is additive
    in BPB, ``L = b - sum_i beta_i g(state_i)``. Nothing in that form stops a
    prediction falling below the entropy floor of the evaluation, which is the
    mechanism behind severe out-of-support optimism: 1358 heldout rows of the 3e18
    Table-9 panel exceeded 0.05 BPB of optimism for compact retained state under
    single-panel selection.

    Here the response is

        L = floor + exp(a - sum_i beta_i g(state_i)),

    fitted as a linear model on ``log(L - floor)``. Reducible loss is multiplicative
    in acquired state, which is the standard shape for a learning curve, and the
    floor is a hard structural lower bound rather than a fitted offset. No mixture,
    however extreme, can be predicted below it.

    The design block is the same per-bucket saturating state the baseline uses, so
    this isolates the link from the state and from the benefit parameterization.
    """
    state = _retained_state(panel, shape)
    signal = _weibull(state, 1.0 / float(shape["saturation_epochs"]), float(shape["power"]))
    total = panel.epochs
    literal_replay = np.sum(np.maximum(total - 1.0, 0.0) ** 2, axis=1, keepdims=True)
    return Design(
        matrix=np.hstack([signal, -literal_replay]),
        names=tuple([*(f"log_deficit_benefit:{b}" for b in panel.buckets), "log_deficit_replay"]),
    )


def build_multiplicative_deficit_phase(panel: Panel, shape: dict) -> Design:
    """MDL with a family-pooled recency block and within-phase intensity penalty."""
    base = build_multiplicative_deficit(panel, shape)
    intensity = panel.family_pool(phase_intensity(panel))
    recency = panel.family_pool(panel.contrast)
    return Design(
        matrix=np.hstack([base.matrix, -intensity, recency, -recency]),
        names=tuple(
            [
                *base.names,
                *(f"log_intensity_harm:{f}" for f in panel.family_names),
                *(f"log_late_benefit:{f}" for f in panel.family_names),
                *(f"log_early_benefit:{f}" for f in panel.family_names),
            ]
        ),
    )


def multiplicative_shapes() -> Iterator[dict]:
    """Shapes for the multiplicative law, including the structural floor.

    ``deficit_floor_fraction`` is the floor as a fraction of the smallest target
    observed on the fitting panel. Values below one keep the floor strictly under
    every fitted observation while still bounding predictions away from zero.
    """
    for deficit_floor_fraction in (0.0, 0.5, 0.8, 0.95):
        for saturation_epochs in (1.0, 2.0, 4.0, 8.0, 16.0):
            for power in (0.4, 0.7, 1.0):
                for late_multiplier in (0.5, 1.0, 2.0, 4.0):
                    for forgetting_rate in (0.0, 0.25, 1.0):
                        yield {
                            "deficit_floor_fraction": deficit_floor_fraction,
                            "saturation_epochs": saturation_epochs,
                            "power": power,
                            "late_multiplier": late_multiplier,
                            "forgetting_rate": forgetting_rate,
                        }


def multiplicative_candidates() -> list[Model]:
    return [
        Model("multiplicative_deficit", build_multiplicative_deficit, multiplicative_shapes, link="log_deficit"),
        Model(
            "multiplicative_deficit_phase",
            build_multiplicative_deficit_phase,
            multiplicative_shapes,
            link="log_deficit",
        ),
    ]


def build_hierarchical_shrinkage(panel: Panel, shape: dict) -> Design:
    """HSB: benefit is a pooled family value plus a shrunk per-bucket deviation.

    The capacity probe closed both extremes. A flat field of 39 free per-bucket
    coefficients overfits 280 rows and lets an optimizer concentrate on one small
    corpus; collapsing to 11 observable-driven coefficients destroys accuracy
    because per-bucket value is genuinely idiosyncratic. The untested regime is
    between them, and it is reached by a prior rather than by a different design.

    Benefit is reparameterized as a pooled family term plus a per-bucket deviation,
    so the same span is available as the flat field but the deviations can be
    penalized separately. With the deviation ridge multiplier at one this is the
    flat model; as it grows, buckets shrink toward their family mean and the
    effective parameter count falls continuously from 39 toward 3. The data chooses
    the shrinkage level through the same heldout-free cross-scale criterion.

    This is also the parameterization hierarchical phase replay uses, which is the
    baseline that wins at 300M, so it tests whether that win comes from the
    shrinkage rather than from the extra penalty blocks.
    """
    state = _retained_state(panel, shape)
    signal = _weibull(state, 1.0 / float(shape["saturation_epochs"]), float(shape["power"]))
    family_signal = panel.family_pool(signal)
    overload = np.maximum(panel.epochs - float(shape["overload_threshold"]), 0.0) ** 2
    return Design(
        matrix=np.hstack([-family_signal, -signal, panel.family_pool(overload)]),
        names=tuple(
            [
                *(f"family_pooled:{f}" for f in panel.family_names),
                *(f"bucket_deviation:{b}" for b in panel.buckets),
                *(f"overload_harm:{f}" for f in panel.family_names),
            ]
        ),
    )


def build_hierarchical_shrinkage_phase(panel: Panel, shape: dict) -> Design:
    """HSB with the within-phase intensity penalty and family recency block."""
    base = build_hierarchical_shrinkage(panel, shape)
    intensity = panel.family_pool(phase_intensity(panel))
    recency = panel.family_pool(panel.contrast)
    return Design(
        matrix=np.hstack([base.matrix, intensity, -recency, recency]),
        names=tuple(
            [
                *base.names,
                *(f"phase_intensity_harm:{f}" for f in panel.family_names),
                *(f"late_benefit:{f}" for f in panel.family_names),
                *(f"early_benefit:{f}" for f in panel.family_names),
            ]
        ),
    )


def shrinkage_penalty(panel: Panel, shape: dict) -> np.ndarray:
    """Ridge multipliers implementing the hierarchical prior.

    Pooled family terms, penalties and any phase block keep a multiplier of one.
    Per-bucket deviations are multiplied by ``deviation_penalty``, so raising it
    shrinks buckets toward their family mean.
    """
    n_families = len(panel.family_names)
    n_buckets = len(panel.buckets)
    multipliers = [np.ones(n_families), np.full(n_buckets, float(shape["deviation_penalty"])), np.ones(n_families)]
    total = n_families + n_buckets + n_families
    # Phase variants append three blocks of family width; they are unshrunk.
    if shape.get("with_phase_block"):
        multipliers.append(np.ones(3 * n_families))
        total += 3 * n_families
    stacked = np.concatenate(multipliers)
    assert len(stacked) == total
    return stacked


def _shrinkage_shape_grid(with_phase_block: bool) -> Iterator[dict]:
    for deviation_penalty in (1.0, 3.0, 10.0, 30.0, 100.0, 1000.0):
        for saturation_epochs in (2.0, 4.0, 8.0, 16.0):
            for power in (0.4, 0.7, 1.0):
                for late_multiplier in (1.0, 2.0, 4.0):
                    for forgetting_rate in (0.25, 1.0):
                        yield {
                            "deviation_penalty": deviation_penalty,
                            "saturation_epochs": saturation_epochs,
                            "power": power,
                            "late_multiplier": late_multiplier,
                            "forgetting_rate": forgetting_rate,
                            "overload_threshold": 1.0,
                            "with_phase_block": with_phase_block,
                        }


def shrinkage_candidates() -> list[Model]:
    return [
        Model(
            "hierarchical_shrinkage",
            build_hierarchical_shrinkage,
            lambda: _shrinkage_shape_grid(False),
            l2_grid=(0.01, 0.1, 1.0),
            penalty_scale=shrinkage_penalty,
        ),
        Model(
            "hierarchical_shrinkage_phase",
            build_hierarchical_shrinkage_phase,
            lambda: _shrinkage_shape_grid(True),
            l2_grid=(0.01, 0.1, 1.0),
            penalty_scale=shrinkage_penalty,
        ),
    ]


def centered_log_ratio(panel: Panel, pseudo_count: float) -> np.ndarray:
    """Centered log-ratio of the aggregate against the token-proportional reference.

    All fifteen earlier candidates are linear in absolute exposure. The simplex has a
    multiplicative geometry, so the natural linear model on it is linear in log
    ratios. Working relative to the proportional policy makes the coordinate zero at
    proportional sampling, negative where a bucket is starved and positive where it is
    oversampled, and it is dimensionless.

    A pseudo-count is required because the fit panel contains 39 domain-deletion
    policies with exactly zero weight, where the log ratio diverges. The pseudo-count
    is a shape parameter rather than a fitted coefficient, so deleted domains sit at a
    large but finite distance instead of being silently clipped.
    """
    reference = panel.proportional
    smoothed = (panel.aggregate + pseudo_count) / (1.0 + pseudo_count * len(panel.buckets))
    ratio = np.log(smoothed / reference)
    return ratio - ratio.mean(axis=1, keepdims=True)


def build_log_ratio_deficit(panel: Panel, shape: dict) -> Design:
    """LRD: loss is driven by log starvation relative to proportional sampling.

    The response splits the centered log ratio into the two directions that mean
    different things physically. Below proportional a bucket is starved and the model
    pays a per-bucket penalty, which is where the identifiable per-bucket structure
    lives. Above proportional a bucket is being repeated, and the return is pooled to
    families because the earlier rounds showed per-bucket resolution on the
    oversampling side is what overfits.

    Domain deletions are the most extreme starvation available and they are 39 of the
    280 fit rows, so this parameterization puts the panel's strongest signal in the
    per-bucket block rather than in the tail of a saturating exposure curve.
    """
    ratio = centered_log_ratio(panel, float(shape["pseudo_count"]))
    starvation = np.maximum(-ratio, 0.0) ** float(shape["starvation_power"])
    surplus = np.maximum(ratio, 0.0)
    overload = np.maximum(panel.epochs - float(shape["overload_threshold"]), 0.0) ** 2
    return Design(
        matrix=np.hstack([starvation, panel.family_pool(surplus), panel.family_pool(overload)]),
        names=tuple(
            [
                *(f"starvation_harm:{b}" for b in panel.buckets),
                *(f"surplus_return:{f}" for f in panel.family_names),
                *(f"overload_harm:{f}" for f in panel.family_names),
            ]
        ),
    )


def build_log_ratio_deficit_phase(panel: Panel, shape: dict) -> Design:
    """LRD with the within-phase intensity penalty and family recency block."""
    base = build_log_ratio_deficit(panel, shape)
    intensity = panel.family_pool(phase_intensity(panel))
    recency = panel.family_pool(panel.contrast)
    return Design(
        matrix=np.hstack([base.matrix, intensity, -recency, recency]),
        names=tuple(
            [
                *base.names,
                *(f"phase_intensity_harm:{f}" for f in panel.family_names),
                *(f"late_benefit:{f}" for f in panel.family_names),
                *(f"early_benefit:{f}" for f in panel.family_names),
            ]
        ),
    )


def log_ratio_shapes() -> Iterator[dict]:
    for pseudo_count in (1e-4, 1e-3, 1e-2):
        for starvation_power in (0.5, 1.0, 2.0):
            for overload_threshold in (1.0, 2.0, 4.0):
                yield {
                    "pseudo_count": pseudo_count,
                    "starvation_power": starvation_power,
                    "overload_threshold": overload_threshold,
                }


def log_ratio_candidates() -> list[Model]:
    return [
        Model("log_ratio_deficit", build_log_ratio_deficit, log_ratio_shapes),
        Model("log_ratio_deficit_phase", build_log_ratio_deficit_phase, log_ratio_shapes),
    ]


def build_crs_plus_breadth(panel: Panel, shape: dict) -> Design:
    """CRS-plus with a single global concentration penalty.

    crs_plus prices per-bucket benefit, family benefit, literal replay and
    family-pooled overload, but nothing in it prices *concentration*. A mixture can
    put a quarter of its mass on one bucket without penalty as long as that bucket's
    corpus is large enough to keep its epoch count low. Its Table-9 optimum does
    exactly that, landing at 6.3 effective buckets.

    The physical claim is that a broad evaluation requires several capabilities at
    once, so a mixture concentrated on a few buckets underperforms even when each of
    those buckets is individually valuable. The Herfindahl index of the aggregate is
    the minimal expression of that: one nonnegative column, no per-bucket capacity,
    and nonnegative least squares can zero it if breadth carries no signal.

    This is a one-parameter addition and it is checked against the accuracy wins as
    well as the optimum, so it cannot be adopted purely because it moves the optimum.
    """
    base = build_crs_plus(panel, shape)
    concentration = (panel.aggregate**2).sum(axis=1, keepdims=True)
    return Design(
        matrix=np.hstack([base.matrix, concentration]),
        names=tuple([*base.names, "concentration_penalty"]),
    )


def build_crs_plus_breadth_phase(panel: Panel, shape: dict) -> Design:
    """CRS-plus-breadth with the within-phase intensity and family recency block."""
    base = build_crs_plus_breadth(panel, shape)
    intensity = panel.family_pool(phase_intensity(panel))
    recency = panel.family_pool(panel.contrast)
    return Design(
        matrix=np.hstack([base.matrix, intensity, -recency, recency]),
        names=tuple(
            [
                *base.names,
                *(f"phase_intensity_harm:{f}" for f in panel.family_names),
                *(f"late_benefit:{f}" for f in panel.family_names),
                *(f"early_benefit:{f}" for f in panel.family_names),
            ]
        ),
    )


def breadth_candidates() -> list[Model]:
    return [
        Model("crs_plus_breadth", build_crs_plus_breadth, crs_plus_shapes),
        Model("crs_plus_breadth_phase", build_crs_plus_breadth_phase, crs_plus_shapes),
    ]


# ---------------------------------------------------------------------------
# Extensions to crs_plus: separate phase heads, and aggregate geometry measured
# against real reference mixtures rather than uniform.
# ---------------------------------------------------------------------------


def unimax_weights(panel: Panel, epoch_cap: float) -> np.ndarray:
    """Unimax mixture: as uniform as possible subject to an epoch cap per bucket.

    Buckets whose uniform share would exceed ``epoch_cap`` epochs are clamped to
    the cap and the freed mass is redistributed uniformly over the rest, iterating
    until no constraint binds. Uniform is a poor concentration reference in this
    swarm because bucket corpora differ by orders of magnitude; unimax and the
    token-proportional policy are the priors an operator would actually consider.
    """
    proportional = panel.proportional
    ceiling = proportional * (epoch_cap / PROPORTIONAL_POLICY_EPOCHS)
    n = len(proportional)
    weights = np.full(n, 1.0 / n)
    free = np.ones(n, dtype=bool)
    for _ in range(n):
        over = free & (weights > ceiling)
        if not over.any():
            break
        weights[over] = ceiling[over]
        free &= ~over
        remaining = 1.0 - weights[~free].sum()
        if not free.any() or remaining <= 0:
            break
        weights[free] = remaining / free.sum()
    return weights / weights.sum()


def _kl_to(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    safe = np.clip(values, EPSILON, None)
    return (safe * np.log(safe / np.clip(reference, EPSILON, None)[None, :])).sum(axis=1, keepdims=True)


def reference_mixtures(panel: Panel) -> dict[str, np.ndarray]:
    """Concentration references, ordered from most to least concentrated."""
    return {
        "proportional": panel.proportional,
        "unimax8": unimax_weights(panel, 8.0),
        "unimax2": unimax_weights(panel, 2.0),
        "uniform": np.full(len(panel.buckets), 1.0 / len(panel.buckets)),
    }


def build_crs_plus_geometry(panel: Panel, shape: dict) -> Design:
    """crs_plus plus aggregate concentration and phase divergence.

    Concentration is measured as KL from the aggregate to each reference mixture,
    so the model can price "how far from proportional" and "how far from unimax"
    separately instead of only "how far from uniform".
    """
    base = build_crs_plus(panel, shape)
    references = reference_mixtures(panel)
    concentration = np.hstack([_kl_to(panel.aggregate, r) for r in references.values()])
    phase0, phase1 = panel.phase0, panel.phase1
    symmetric_kl = 0.5 * (_kl_pair(phase1, phase0) + _kl_pair(phase0, phase1))
    divergence = np.hstack([symmetric_kl, panel.phase_tv[:, None]])
    return Design(
        matrix=np.hstack([base.matrix, concentration, divergence]),
        names=tuple(
            [
                *base.names,
                *(f"concentration_kl:{name}" for name in references),
                "phase_symmetric_kl",
                "phase_tv",
            ]
        ),
    )


def _kl_pair(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    safe = np.clip(left, EPSILON, None)
    return (safe * np.log(safe / np.clip(right, EPSILON, None))).sum(axis=1, keepdims=True)


def build_crs_plus_heads(panel: Panel, shape: dict) -> Design:
    """crs_plus plus family-pooled independent early and late benefit heads.

    Separate-heads style, but pooled to three families per phase rather than free
    per bucket, because the capacity probes showed 39 free per-bucket amplitudes
    are already at the ceiling this panel supports.
    """
    base = build_crs_plus(panel, shape)
    scale = 1.0 / float(shape["saturation_epochs"])
    power = float(shape["power"])
    early = panel.family_pool(panel.c0 * panel.phase0)
    late = panel.family_pool(panel.c1 * panel.phase1)
    return Design(
        matrix=np.hstack([base.matrix, -_weibull(early, scale, power), -_weibull(late, scale, power)]),
        names=tuple(
            [
                *base.names,
                *(f"early_head:{f}" for f in panel.family_names),
                *(f"late_head:{f}" for f in panel.family_names),
            ]
        ),
    )


def build_crs_plus_bucket_heads(panel: Panel, shape: dict) -> Design:
    """crs_plus plus free per-bucket early and late heads, the heavy variant."""
    base = build_crs_plus(panel, shape)
    scale = 1.0 / float(shape["saturation_epochs"])
    power = float(shape["power"])
    return Design(
        matrix=np.hstack(
            [
                base.matrix,
                -_weibull(panel.c0 * panel.phase0, scale, power),
                -_weibull(panel.c1 * panel.phase1, scale, power),
            ]
        ),
        names=tuple(
            [
                *base.names,
                *(f"early_head:{b}" for b in panel.buckets),
                *(f"late_head:{b}" for b in panel.buckets),
            ]
        ),
    )


def crs_plus_extensions() -> list[Model]:
    return [
        Model("crs_plus_geometry", build_crs_plus_geometry, crs_plus_shapes),
        Model("crs_plus_heads", build_crs_plus_heads, crs_plus_shapes),
        Model("crs_plus_bucket_heads", build_crs_plus_bucket_heads, crs_plus_shapes),
    ]
