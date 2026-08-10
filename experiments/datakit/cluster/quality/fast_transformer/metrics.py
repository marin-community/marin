# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rank-based holdout metrics for the quality-scorer training loop."""

import math


def _avg_ranks(xs: list[float]) -> list[float]:
    """Return tie-aware 1-indexed average ranks."""
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and xs[order[j]] == xs[order[i]]:
            j += 1
        avg = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation, average-rank handling for ties."""
    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} vs {len(y)}")
    if len(x) < 2:
        return float("nan")
    rx = _avg_ranks(x)
    ry = _avg_ranks(y)
    n = len(rx)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0.0 or dy == 0.0:
        return float("nan")
    return num / (dx * dy)


def auc(y_true: list[int], y_score: list[float]) -> float:
    """ROC AUC via the rank-based formula (O(n log n))."""
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = sum(1 for y in y_true if y == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _avg_ranks(y_score)
    sum_pos_ranks = sum(r for r, y in zip(ranks, y_true, strict=True) if y == 1)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
