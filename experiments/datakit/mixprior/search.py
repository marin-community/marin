# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Search the mixture simplex by acquisition score, then refine the best starts."""

from typing import Protocol

import numpy as np
from jax.typing import ArrayLike

MIXTURE_DENOMINATOR = 49_152
PREDICTION_BATCH_SIZE = 256
DIRICHLET_CONCENTRATION_RANGE = (1.0, 10_000.0)
LOCAL_STARTS = 32
LOCAL_DRAWS_PER_START = 256
LOCAL_NOISE_RANGE = (0.02, 0.5)


class Acquisition(Protocol):
    def acquisition(self, weights: ArrayLike) -> ArrayLike: ...


def quantize(weights: np.ndarray) -> np.ndarray:
    """Round phases to the training lattice while preserving their unit sums."""
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.allclose(weights.sum(axis=-1), 1):
        raise ValueError("Mixture phases must be finite simplexes")
    shape = weights.shape
    scaled = weights.reshape(-1, shape[-1]) * MIXTURE_DENOMINATOR
    counts = np.floor(scaled).astype(np.int64)
    remaining = MIXTURE_DENOMINATOR - counts.sum(axis=-1)

    # Give the remaining counts to the largest fractional parts.
    fractions = scaled - counts
    order = np.argsort(-fractions, axis=-1, kind="stable")
    for offset in range(shape[-1]):
        rows = np.flatnonzero(remaining > offset)
        counts[rows, order[rows, offset]] += 1
    return counts.reshape(shape) / MIXTURE_DENOMINATOR


def feasible_pool(
    weights: np.ndarray,
    excluded: np.ndarray,
    exposure: np.ndarray,
    max_epochs: float,
) -> np.ndarray:
    """Keep unseen mixtures within the epoch cap.

    Exposure has shape (phases, components): each entry is the phase's token
    budget divided by the component's available tokens, converting weights to epochs.
    """
    weights = quantize(weights)
    excluded = quantize(excluded)
    flat = weights.reshape(len(weights), -1)
    _, indices = np.unique(flat, axis=0, return_index=True)
    weights = weights[np.sort(indices)]

    observed_rows = excluded.reshape(len(excluded), flat.shape[1])
    candidate_rows = weights.reshape(len(weights), -1)
    seen = {row.tobytes() for row in np.round(observed_rows, 12)}
    unseen = [row.tobytes() not in seen for row in np.round(candidate_rows, 12)]

    # A component's epoch budget is shared across the two phases.
    component_epochs = (weights * exposure).sum(axis=1)
    within_limit = component_epochs.max(axis=-1) <= max_epochs
    return weights[np.asarray(unseen) & within_limit]


def acquisition_scores(model: Acquisition, weights: np.ndarray) -> np.ndarray:
    batch_count = (len(weights) + PREDICTION_BATCH_SIZE - 1) // PREDICTION_BATCH_SIZE
    scores = np.concatenate([np.asarray(model.acquisition(batch)) for batch in np.array_split(weights, batch_count)])
    if not np.isfinite(scores).all():
        raise ValueError("The model returned non-finite acquisition scores")
    return scores


def draw_mixtures(proportional: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Draw both broad and near-proportional mixtures, independently per phase."""
    minimum, maximum = DIRICHLET_CONCENTRATION_RANGE
    log_concentration = rng.uniform(np.log(minimum), np.log(maximum), size=(count, 2, 1))
    concentration = np.exp(log_concentration)

    # Normalized Gamma draws form a Dirichlet sample with this mean and concentration.
    weights = rng.gamma(concentration * proportional)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights


def refine_mixtures(centers: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Perturb promising mixtures in log space using shared noise across phases."""
    # A small pseudocount lets zero-weight components participate in refinement.
    smoothed = centers + 1 / MIXTURE_DENOMINATOR
    smoothed /= smoothed.sum(axis=-1, keepdims=True)

    start_count, _, component_count = centers.shape
    minimum, maximum = LOCAL_NOISE_RANGE
    noise_shape = (start_count, LOCAL_DRAWS_PER_START, 1, 1)
    log_scales = rng.uniform(np.log(minimum), np.log(maximum), size=noise_shape)
    scales = np.exp(log_scales)
    noise = rng.normal(size=(start_count, LOCAL_DRAWS_PER_START, 1, component_count))

    logits = np.log(smoothed[:, None]) + scales * noise
    logits -= logits.max(axis=-1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights.reshape(-1, 2, component_count)


def search(
    model: Acquisition,
    available_tokens: ArrayLike,
    phase_budgets: ArrayLike,
    observed: np.ndarray,
    *,
    pool_size: int = 65_536,
    batch_size: int = 1,
    seed: int = 111,
    max_epochs: float = 16.0,
) -> np.ndarray:
    """Return distinct unobserved mixtures satisfying the cumulative epoch cap."""
    if pool_size < batch_size or batch_size < 1 or not np.isfinite(max_epochs) or max_epochs <= 0:
        raise ValueError("Search needs a positive feasible batch size and epoch limit")
    # Proposals and duplicate filtering stay on the host; scoring runs on the GP's device.
    available_tokens = np.asarray(available_tokens)
    phase_budgets = np.asarray(phase_budgets)
    proportional = available_tokens / available_tokens.sum()
    exposure = phase_budgets[:, None] / available_tokens
    rng = np.random.default_rng(seed)

    # Search broadly, including the proportional mixture as a known starting point.
    draws = draw_mixtures(proportional, pool_size, rng)
    proportional_mixture = np.broadcast_to(proportional, (1, 2, len(available_tokens)))
    initial_candidates = np.concatenate([proportional_mixture, draws])
    pool = feasible_pool(initial_candidates, observed, exposure, max_epochs)
    if len(pool) < batch_size:
        raise ValueError("Too few feasible unobserved mixtures; increase the pool or relax the epoch cap")
    scores = acquisition_scores(model, pool)

    # Refine the best starts, retaining the starts themselves if no draw improves them.
    start_count = max(LOCAL_STARTS, batch_size)
    best_starts = np.argsort(scores)[-start_count:]
    centers = pool[best_starts]
    local_draws = refine_mixtures(centers, rng)
    refined_candidates = np.concatenate([centers, local_draws])
    pool = feasible_pool(refined_candidates, observed, exposure, max_epochs)
    scores = acquisition_scores(model, pool)
    best_candidates = np.argsort(scores)[-batch_size:][::-1]
    return pool[best_candidates]
