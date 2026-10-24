"""CPU unit tests for the shared pass@k calculations.

Covers:
 - estimate_pass_at_k matches exhaustive sampling from a finite population,
 - aggregate_pass_at_k is monotone non-decreasing in k and bounded [0,1],
 - parse_pass_at_k handles strings / lists / empty.
"""

from itertools import combinations
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.passk import aggregate_pass_at_k, estimate_pass_at_k, parse_pass_at_k


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_estimator_matches_exhaustive_sampling(k):
    counts = [0, 1, 3, 4]
    samples = list(combinations(range(4), k))
    expected = [
        sum(any(candidate < correct for candidate in sample) for sample in samples) / len(samples)
        for correct in counts
    ]
    np.testing.assert_array_equal(estimate_pass_at_k(4, counts, k), expected)


def test_estimator_edge_values():
    # c == 0 -> 0.0 ; c == n -> 1.0 ; n-c < k -> 1.0
    assert estimate_pass_at_k(128, [0], 1)[0] == 0.0
    assert estimate_pass_at_k(128, [128], 128)[0] == 1.0
    assert estimate_pass_at_k(10, [9], 8)[0] == 1.0  # n-c=1 < k=8


def test_aggregate_monotone_and_bounded():
    # per-problem correct counts out of n=128 samples
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 129, size=200).tolist()
    table = aggregate_pass_at_k(128, counts, [1, 8, 32, 128])
    vals = [table[f"pass@{k}"] for k in (1, 8, 32, 128)]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert vals == sorted(vals), f"pass@k not monotone: {vals}"


def test_aggregate_skips_k_above_n():
    table = aggregate_pass_at_k(8, [4, 0, 8], [1, 8, 32, 128])
    assert set(table) == {"pass@1", "pass@8"}  # 32, 128 > n=8 skipped


def test_parse_pass_at_k():
    assert parse_pass_at_k("1,8,32,128") == [1, 8, 32, 128]
    assert parse_pass_at_k("8, 1 ,8") == [1, 8]  # dedup + sort + whitespace
    assert parse_pass_at_k([128, 1, 8]) == [1, 8, 128]
    assert parse_pass_at_k(None) == [1, 8, 32, 128]
    assert parse_pass_at_k("") == [1, 8, 32, 128]
