"""Unbiased pass@k estimation shared by code benchmarks and trial reducers.

The Chen et al. (2021) estimator is pass@k = 1 - C(n - c, k) / C(n, k),
computed in a numerically stable product form. Code benchmark evaluators and
``eval.execution.grading`` share the estimator; functional correctness
aggregation uses ``aggregate_pass_at_k`` for per-problem means.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Union

import numpy as np


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """Estimate pass@k for each problem and return them in an array.

    Args:
        num_samples: either a single int n (same for every problem) or a
            per-problem sequence of sample counts n_i.
        num_correct: per-problem correct counts c_i.
        k: the k in pass@k.

    Returns:
        np.ndarray of per-problem pass@k estimates.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """Calculate 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def aggregate_pass_at_k(
    num_samples: Union[int, Sequence[int]],
    num_correct: Sequence[int],
    ks: Sequence[int],
) -> Dict[str, float]:
    """Aggregate per-problem (n, c) into a {``pass@k``: mean} table.

    For each requested ``k`` within the available sample count, compute the
    unbiased per-problem estimate and average over problems.

    Args:
        num_samples: n (single int, applied to every problem) or per-problem n_i.
        num_correct: per-problem correct counts c_i.
        ks: the list of k values to report.

    Returns:
        ordered dict {"pass@{k}": float} for every reportable k.
    """
    if isinstance(num_samples, int):
        max_n = num_samples
    else:
        max_n = min(int(n) for n in num_samples) if len(num_samples) else 0

    out: Dict[str, float] = {}
    for k in ks:
        if k > max_n:
            continue
        out[f"pass@{k}"] = float(np.mean(estimate_pass_at_k(num_samples, num_correct, k)))
    return out


def parse_pass_at_k(spec: Union[str, Sequence[int], None]) -> List[int]:
    """Parse a ``--pass_at_k`` spec ("1,8,32,128" or a list) into a sorted int list.

    Returns the default [1, 8, 32, 128] when ``spec`` is falsy.
    """
    if not spec:
        return [1, 8, 32, 128]
    if isinstance(spec, str):
        ks = [int(x) for x in spec.replace(" ", "").split(",") if x != ""]
    else:
        ks = [int(x) for x in spec]
    return sorted(set(ks))
