"""Shared, benchmark-agnostic pass@k utilities for evalchemy.

This module is the single home of the unbiased pass@k estimator that was
previously duplicated in
``eval/chat_benchmarks/HumanEval/human_eval/evaluation.py`` and
``eval/chat_benchmarks/MBPPPlus/mbpp_plus/evaluation.py`` (identical copies),
and re-implemented a third time in the now-retired standalone
``passatk_driver.py``. Both HumanEval and MBPPPlus now import
``estimate_pass_at_k`` from here, and ``eval.task.BaseBenchmark`` uses it for
the generic native pass@k mode (Stage 2b).

The estimator is the standard Chen et al. (2021) unbiased estimator:

    pass@k = 1 - C(n - c, k) / C(n, k)

computed in the numerically-stable product form. See
``estimate_pass_at_k`` (kept byte-for-byte equivalent to the promoted copies)
and ``aggregate_pass_at_k`` (the per-problem -> mean aggregator the base uses).
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

    Promoted verbatim from the (identical) HumanEval / MBPPPlus copies so the
    code 3a/3b/3c and the retired driver all referenced now lives once.

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

    Generic helper used by the ``BaseBenchmark`` native pass@k mode. For each
    requested ``k`` (skipping any ``k`` larger than the available samples) it
    computes the unbiased per-problem estimate and averages over problems.

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
