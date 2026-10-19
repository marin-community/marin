"""CPU unit tests for the shared pass@k utilities (Stage 2b).

Covers:
 - estimate_pass_at_k matches the (now-promoted) reference estimator bit-for-bit,
 - aggregate_pass_at_k is monotone non-decreasing in k and bounded [0,1],
 - parse_pass_at_k handles strings / lists / empty,
 - the estimator is defined exactly once (no third copy), and HumanEval/MBPPPlus
   re-export the shared symbol (same object).
"""

import itertools
import pathlib
import subprocess
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.passk import aggregate_pass_at_k, estimate_pass_at_k, parse_pass_at_k


def _reference_estimate(num_samples, num_correct, k):
    """Verbatim copy of the pre-Stage-2b duplicated estimator, for parity."""

    def estimator(n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        it = iter(num_samples)
    return np.array([estimator(int(n), int(c), kk) for n, c, kk in zip(it, num_correct, itertools.repeat(k))])


@pytest.mark.parametrize("n", [1, 8, 32, 128])
@pytest.mark.parametrize("k", [1, 8, 32, 128])
def test_estimator_matches_reference(n, k):
    counts = list(range(0, n + 1))
    got = estimate_pass_at_k(n, counts, k)
    ref = _reference_estimate(n, counts, k)
    assert np.allclose(got, ref, atol=0.0, rtol=0.0)


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


def test_shared_estimator_is_canonical_and_humaneval_mbppplus_reexport():
    """Stage 2b dedup contract: the estimator the plan promotes (HumanEval +
    MBPPPlus, plus the retired driver) now lives once in eval/passk.py, and
    those two benchmark modules re-export the SAME object rather than defining
    their own.

    Note: several other vendored sub-packages (MBPP/human_eval, BigCodeBench,
    CruxEval, HumanEvalPlus, MultiPLE, LiveBench) ship their own independent
    `estimate_pass_at_k`; consolidating those is out of Stage 2b's named scope
    and is left for a later cleanup. This test pins the Stage-2b contract.
    """
    # Static contract (no heavy deps): neither file defines its own copy
    # anymore; both import the shared symbol.
    he_src = (REPO / "eval/chat_benchmarks/HumanEval/human_eval/evaluation.py").read_text()
    mp_src = (REPO / "eval/chat_benchmarks/MBPPPlus/mbpp_plus/evaluation.py").read_text()
    assert "def estimate_pass_at_k" not in he_src
    assert "def estimate_pass_at_k" not in mp_src
    assert "from eval.passk import estimate_pass_at_k" in he_src
    assert "from eval.passk import estimate_pass_at_k" in mp_src

    # Live identity check when the (heavy) benchmark deps are importable.
    try:
        from eval.chat_benchmarks.HumanEval.human_eval import evaluation as he
        from eval.chat_benchmarks.MBPPPlus.mbpp_plus import evaluation as mp
        from eval import passk
    except Exception as e:  # missing fire/etc. in a lean CPU env
        pytest.skip(f"benchmark deps unavailable for live identity check: {e}")
    assert he.estimate_pass_at_k is passk.estimate_pass_at_k
    assert mp.estimate_pass_at_k is passk.estimate_pass_at_k
