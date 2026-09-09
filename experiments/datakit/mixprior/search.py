# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Search the mixture simplex by posterior mean, then refine the best starts."""

from typing import Protocol

import numpy as np

MIXTURE_DENOMINATOR = 49_152


class Predictor(Protocol):
    def predict(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


def quantize(weights: np.ndarray) -> np.ndarray:
    """Round phases to the training lattice while preserving their unit sums."""
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.allclose(weights.sum(axis=-1), 1):
        raise ValueError("Mixture phases must be finite simplexes")
    shape = weights.shape
    scaled = weights.reshape(-1, shape[-1]) * MIXTURE_DENOMINATOR
    counts = np.floor(scaled).astype(np.int64)
    remaining = MIXTURE_DENOMINATOR - counts.sum(axis=-1)
    order = np.argsort(-(scaled - counts), axis=-1, kind="stable")
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
    weights = quantize(weights)
    excluded = quantize(excluded)
    flat = weights.reshape(len(weights), -1)
    _, indices = np.unique(flat, axis=0, return_index=True)
    weights = weights[np.sort(indices)]
    seen = {row.tobytes() for row in np.round(excluded.reshape(len(excluded), flat.shape[1]), 12)}
    keep = [row.tobytes() not in seen for row in np.round(weights.reshape(len(weights), -1), 12)]
    within_limit = (weights * exposure).sum(axis=1).max(axis=-1) <= max_epochs
    return weights[np.asarray(keep) & within_limit]


def posterior_mean(model: Predictor, weights: np.ndarray) -> np.ndarray:
    scores = np.concatenate([model.predict(chunk)[0] for chunk in np.array_split(weights, (len(weights) + 255) // 256)])
    if not np.isfinite(scores).all():
        raise ValueError("The model returned non-finite predictions")
    return scores


def search(
    model: Predictor,
    available_tokens: np.ndarray,
    phase_budgets: np.ndarray,
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
    proportional = available_tokens / available_tokens.sum()
    exposure = phase_budgets[:, None] / available_tokens
    rng = np.random.default_rng(seed)
    concentration = np.exp(rng.uniform(np.log(1), np.log(10_000), size=(pool_size, 2, 1)))
    draws = rng.gamma(concentration * proportional)
    draws /= draws.sum(axis=-1, keepdims=True)
    proportional = np.broadcast_to(proportional, (1, 2, len(available_tokens)))
    pool = feasible_pool(np.concatenate([proportional, draws]), observed, exposure, max_epochs)
    if len(pool) < batch_size:
        raise ValueError("Too few feasible unobserved mixtures; increase the pool or relax the epoch cap")
    scores = posterior_mean(model, pool)
    centers = pool[np.argsort(scores)[-max(32, batch_size) :]]
    smoothed = centers + 1 / MIXTURE_DENOMINATOR
    smoothed /= smoothed.sum(axis=-1, keepdims=True)
    scales = np.exp(rng.uniform(np.log(0.02), np.log(0.5), size=(len(centers), 256, 1, 1)))
    noise = rng.normal(size=(len(centers), 256, 1, centers.shape[-1]))
    logits = np.log(smoothed[:, None]) + scales * noise
    logits -= logits.max(axis=-1, keepdims=True)
    local = np.exp(logits)
    local /= local.sum(axis=-1, keepdims=True)
    pool = feasible_pool(
        np.concatenate([centers, local.reshape(-1, 2, centers.shape[-1])]), observed, exposure, max_epochs
    )
    scores = posterior_mean(model, pool)
    return pool[np.argsort(scores)[-batch_size:][::-1]]
