# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bucket benefit pooled over observable covariates instead of 39 free amplitudes.

The extrapolation defect this targets was localized in
``audit_shrinkage_extrapolation_bias_20260726.py``: ridge shrinkage of the
improvement coefficients is what makes every existing surrogate under-rate a policy
better than its training data. Forcing the ridge to zero removes the bias but costs
fit, because 39 free per-bucket amplitudes on 280 rows genuinely need the
regularization. The way out is to stop needing it.

A bucket's amplitude is written as a nonnegative combination of nonnegative
covariate bases rather than a free parameter:

    a_i = sum_k theta_k z_ik,  theta_k >= 0,  z_ik >= 0.

Because the benefit contribution is ``sum_i a_i S(x_i)``, substituting gives
``sum_k theta_k (sum_i z_ik S(x_i))``, which is still linear in the fitted
coefficients. So this changes the column space without touching the nonnegative
least-squares machinery, and nonnegativity of ``theta`` guarantees every implied
amplitude stays nonnegative.

The bases are the bucket properties actually observable before training: corpus size,
quality tier, source collection, and semantic family. Size matters because the panel
oversamples small corpora heavily, spanning 359x in tokens per unit weight. Quality
tier matters because 26 of the 39 buckets carry an explicit high or low label that the
existing models see only through family membership.

Pure pooling cannot express a bucket that is worse than its covariate class, since
both ``theta`` and ``z`` are nonnegative. The ``residual`` variant restores that
freedom with 39 per-bucket columns, and relies on the caller to penalize only those
columns: pooled coefficients stay unshrunk so extrapolation is unbiased, while the
residuals carry the ridge. That is the same hierarchical idea the hierarchical replay
baseline applies at family level, with a richer basis than three families.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from swarm39_harness_20260725 import Design, Model, Panel  # noqa: E402

QUALITY_PATTERN = re.compile(r"_(high|low)$")

# Shape grid. Mirrors the compact retained-state grid so a difference in score is a
# difference in column space rather than in the nonlinear response.
RATES = (0.25, 1.0)
POWERS = (0.4, 0.7, 1.0)
LATE_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0, 8.0)
FORGETTING_RATES = (0.0, 0.25, 1.0)


def quality_tier(bucket: str) -> str:
    match = QUALITY_PATTERN.search(bucket)
    return match.group(1) if match is not None else "untiered"


def source_collection(bucket: str) -> str:
    if bucket.startswith("dolma3_cc/"):
        return "dolma3_cc"
    if bucket.startswith("dolmino_"):
        return "dolmino"
    return "dolma3_other"


def covariate_basis(panel: Panel) -> tuple[np.ndarray, tuple[str, ...]]:
    """Nonnegative bucket-by-basis matrix and its column names.

    Every column is an indicator or a ramp in [0, 1], so a nonnegative coefficient
    reads directly as the benefit amplitude that covariate class contributes.
    """
    buckets = panel.buckets
    columns: list[np.ndarray] = [np.ones(len(buckets))]
    names: list[str] = ["pooled:all"]

    for label in ("broad_text", "tech_code", "reasoning"):
        index = panel.family_names.index(label)
        columns.append((panel.family_index == index).astype(float))
        names.append(f"pooled:family_{label}")

    for tier in ("high", "low", "untiered"):
        columns.append(np.array([quality_tier(b) == tier for b in buckets], dtype=float))
        names.append(f"pooled:tier_{tier}")

    for collection in ("dolma3_cc", "dolma3_other", "dolmino"):
        columns.append(np.array([source_collection(b) == collection for b in buckets], dtype=float))
        names.append(f"pooled:source_{collection}")

    # Corpus size enters as two complementary ramps so a nonnegative combination can
    # tilt the amplitude either toward small or toward large corpora.
    scale = np.log(panel.c1)
    ramp = (scale - scale.min()) / (scale.max() - scale.min())
    columns.append(ramp)
    names.append("pooled:size_large")
    columns.append(1.0 - ramp)
    names.append("pooled:size_small")

    return np.column_stack(columns), tuple(names)


def _weibull(exposure: np.ndarray, rate: float, power: float) -> np.ndarray:
    return -np.expm1(-((np.maximum(rate * exposure, 0.0)) ** power))


def retained_state(panel: Panel, shape: dict) -> np.ndarray:
    """Revisit-gated retained state, identical to compact retained state's."""
    early = panel.phase0 * panel.c0
    late = panel.phase1 * panel.c1
    revisit = np.clip(panel.phase1, 0.0, 1.0)
    retained = np.exp(-float(shape["forgetting_rate"]) * (1.0 - revisit)) * early
    return np.maximum(retained + float(shape["late_multiplier"]) * late, 0.0)


def build_covariate_pooled(panel: Panel, shape: dict) -> Design:
    """Pooled benefit, optional per-bucket residuals, and the usual harm channels."""
    state = retained_state(panel, shape)
    total = panel.phase0 * panel.c0 + panel.phase1 * panel.c1
    benefit = _weibull(state, float(shape["rate"]), float(shape["power"]))

    basis, basis_names = covariate_basis(panel)
    blocks = [-(benefit @ basis)]
    names = list(basis_names)

    if shape.get("bucket_residuals", False):
        blocks.append(-benefit)
        names.extend(f"residual:{bucket}" for bucket in panel.buckets)

    blocks.append(np.sum(np.maximum(total - 1.0, 0.0) ** 2, axis=1, keepdims=True))
    names.append("harm:shared_literal_replay")

    overexposure = np.maximum(total - float(shape.get("overexposure_threshold", 1.0)), 0.0) ** 2
    blocks.append(panel.family_pool(overexposure))
    names.extend(f"harm:family_overexposure_{family}" for family in panel.family_names)

    return Design(matrix=np.hstack(blocks), names=tuple(names))


def pooled_shapes(bucket_residuals: bool) -> Iterator[dict]:
    for rate in RATES:
        for power in POWERS:
            for late_multiplier in LATE_MULTIPLIERS:
                for forgetting_rate in FORGETTING_RATES:
                    yield {
                        "rate": rate,
                        "power": power,
                        "late_multiplier": late_multiplier,
                        "forgetting_rate": forgetting_rate,
                        "overexposure_threshold": 1.0,
                        "bucket_residuals": bucket_residuals,
                    }


def residual_penalty(panel: Panel, shape: dict) -> np.ndarray:
    """Ridge multipliers that shrink only the per-bucket residual block.

    Pooled amplitudes stay unshrunk, so the reachable improvement is not truncated and
    extrapolation stays unbiased; the residuals carry all of the regularization.
    """
    names = build_covariate_pooled(panel, shape).names
    return np.array([1.0 if name.startswith("residual:") else 0.0 for name in names])


def covariate_pooled_model(bucket_residuals: bool, l2_grid: tuple[float, ...]) -> Model:
    suffix = "_residual" if bucket_residuals else ""
    return Model(
        name=f"covariate_pooled{suffix}",
        build=build_covariate_pooled,
        shapes=lambda: pooled_shapes(bucket_residuals),
        l2_grid=l2_grid,
        penalty_scale=residual_penalty if bucket_residuals else None,
    )
