# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact float64 mismatch statistics and prompt-cluster uncertainty."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

LOG_TWO = math.log(2.0)
LOG_FLOAT64_MAX = math.log(np.finfo(np.float64).max)
PERCENTILES = (0, 50, 75, 90, 99, 99.9, 100)


@dataclass(frozen=True)
class BootstrapResult:
    point: dict[str, float | int]
    intervals: dict[str, tuple[float, float]]
    draws: list[dict[str, float | int]]


def _logsumexp(values: np.ndarray) -> float:
    highest = float(np.max(values))
    if highest == -math.inf:
        return -math.inf
    return highest + math.log(float(np.exp(values - highest).sum()))


def _ess_fraction(log_weights: np.ndarray) -> float:
    if log_weights.size == 0:
        return math.nan
    return float(math.exp(2 * _logsumexp(log_weights) - _logsumexp(2 * log_weights) - math.log(len(log_weights))))


def _exp_or_inf(value: float) -> float:
    return math.exp(value) if value <= LOG_FLOAT64_MAX else math.inf


def comparison_metrics(
    target: Sequence[Sequence[float]],
    reference: Sequence[Sequence[float]],
    masks: Sequence[Sequence[bool]],
    *,
    advantages: Sequence[float | None] | None = None,
    tis_cap: float | None = None,
    eps_clip_low: float = 0.2,
    eps_clip_high: float = 0.2,
) -> dict[str, float | int]:
    """Compute metrics on precisely aligned valid response tokens.

    ESS is a normalized weight-concentration statistic, not a literal count
    of independent tokens. Sequence ESS uses the product of token ratios within
    each answer; it remains descriptive when tokens came from another policy.
    """
    if not (len(target) == len(reference) == len(masks)):
        raise ValueError("comparison rows must align one-for-one")
    pieces = []
    per_sequence = []
    sign_values = []
    for index, (left, right, mask) in enumerate(zip(target, reference, masks, strict=True)):
        if not (len(left) == len(right) == len(mask)):
            raise ValueError(f"comparison token lengths differ in sample {index}")
        selected = np.asarray(mask, dtype=bool)
        delta = np.asarray(left, dtype=np.float64)[selected] - np.asarray(right, dtype=np.float64)[selected]
        if not np.isfinite(delta).all():
            raise ValueError(f"comparison has nonfinite logprobability in sample {index}")
        if delta.size:
            pieces.append(delta)
            per_sequence.append(float(delta.sum()))
            sign = None if advantages is None else advantages[index]
            sign_values.extend([0.0 if sign is None else float(sign)] * len(delta))
    if not pieces:
        raise ValueError("comparison has no unmasked tokens")
    delta = np.concatenate(pieces)
    absolute = np.abs(delta)
    quantiles = np.percentile(absolute, PERCENTILES)
    logs = np.asarray(per_sequence, dtype=np.float64)
    mean_exp_delta = _exp_or_inf(_logsumexp(delta) - math.log(len(delta)))
    mean_exp_2delta = _exp_or_inf(_logsumexp(2 * delta) - math.log(len(delta)))
    with np.errstate(over="ignore", invalid="ignore"):
        k3_terms = np.expm1(delta) - delta
    result: dict[str, float | int] = {
        "tokens": len(delta),
        "sequences": len(logs),
        "delta_min": float(delta.min()),
        "delta_max": float(delta.max()),
        "delta_mean": float(delta.mean()),
        "abs_min": float(quantiles[0]),
        "abs_p50": float(quantiles[1]),
        "abs_p75": float(quantiles[2]),
        "abs_p90": float(quantiles[3]),
        "abs_p99": float(quantiles[4]),
        "abs_p999": float(quantiles[5]),
        "abs_max": float(quantiles[6]),
        "abs_mean": float(absolute.mean()),
        "k1": float(-delta.mean()),
        "k3": float(k3_terms.mean()),
        "chi2_sample_moment": mean_exp_2delta - 1.0,
        "mean_ratio": mean_exp_delta,
        "token_ess_fraction_raw": _ess_fraction(delta),
        "sequence_ess_fraction_raw": _ess_fraction(logs),
        "share_beyond_2x": float(np.mean(absolute > LOG_TWO)),
    }
    if tis_cap is not None:
        if not math.isfinite(tis_cap) or tis_cap <= 0:
            raise ValueError("tis_cap must be finite and positive")
        capped = np.minimum(delta, math.log(tis_cap))
        capped_sequence = []
        cursor = 0
        for piece in pieces:
            capped_sequence.append(float(capped[cursor : cursor + len(piece)].sum()))
            cursor += len(piece)
        result["token_ess_fraction_capped"] = _ess_fraction(capped)
        result["sequence_ess_fraction_capped"] = _ess_fraction(np.asarray(capped_sequence))
        result["tis_cap_occupancy"] = float(np.mean(delta > math.log(tis_cap)))
    if advantages is not None:
        signs = np.asarray(sign_values)
        positive = signs > 0
        negative = signs < 0
        result["positive_advantage_clip_occupancy"] = (
            float(np.mean(delta[positive] > math.log1p(eps_clip_high))) if positive.any() else math.nan
        )
        result["negative_advantage_clip_occupancy"] = (
            float(np.mean(delta[negative] < math.log1p(-eps_clip_low))) if negative.any() else math.nan
        )
    return result


def prompt_cluster_bootstrap(
    prompt_ids: Sequence[str],
    calculate: Callable[[list[int]], dict[str, float | int]],
    *,
    seed: int,
    draws: int = 1000,
) -> BootstrapResult:
    """Return named point estimates, 95% intervals and seeded prompt-cluster draws,
    omitting counts and nonfinite draws from intervals.
    """
    if not prompt_ids or draws < 1:
        raise ValueError("bootstrap requires prompts and at least one draw")
    groups: dict[str, list[int]] = {}
    for index, prompt_id in enumerate(prompt_ids):
        groups.setdefault(prompt_id, []).append(index)
    group_rows = list(groups.values())
    point = calculate(list(range(len(prompt_ids))))
    rng = np.random.default_rng(seed)
    sampled = []
    for _ in range(draws):
        group_indices = rng.integers(0, len(group_rows), size=len(group_rows))
        row_indices = [row for group_index in group_indices for row in group_rows[group_index]]
        sampled.append(calculate(row_indices))
    intervals = {}
    for name in point:
        if name in {"tokens", "sequences"}:
            continue
        values = np.asarray([draw[name] for draw in sampled], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            lo, hi = np.percentile(values, [2.5, 97.5])
            intervals[name] = (float(lo), float(hi))
    return BootstrapResult(point=point, intervals=intervals, draws=sampled)
