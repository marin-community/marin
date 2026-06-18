# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the document intruder test.

Everything here runs without API access: the panel sits behind the
``Panelist`` protocol, so we drive the driver with deterministic fakes and
the OpenAI-backed panelist with a fake OpenAI-shaped client. The statistical
core (Robbins confidence sequence, decision rule) is exercised against its
own guarantee -- anytime-valid coverage -- rather than reimplemented.
"""

from __future__ import annotations

import hashlib
import logging
from itertools import pairwise

import numpy as np
import pytest

from experiments.datakit.intruder import (
    BucketPool,
    ConfidenceSequence,
    Decision,
    IntruderTrial,
    LlmPanelist,
    Side,
    _decide,
    _robbins_radius,
    run_intruder_test,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _stable_unit(text: str) -> float:
    """Deterministic value in [0, 1) from a string (process-independent)."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") / 2.0**64


class SideAwarePanelist:
    """Fake judge that detects the intruder at a per-side rate.

    Detection is a deterministic function of the trial's content, so a run is
    reproducible and thread-safe under the driver's ThreadPoolExecutor.
    """

    def __init__(self, name: str, rate_by_side: dict[Side, float]):
        self.name = name
        self._rate_by_side = rate_by_side

    def vote(self, trial: IntruderTrial, *, max_doc_chars: int) -> int:
        rate = self._rate_by_side[trial.side]
        key = trial.side + "|" + "|".join(trial.documents)
        if _stable_unit(key) < rate:
            return trial.intruder_index
        return (trial.intruder_index + 1) % 5  # a deterministic wrong answer


class DeadPanelist:
    """Fake judge whose every call fails (bad slug / gateway down)."""

    name = "dead"

    def vote(self, trial: IntruderTrial, *, max_doc_chars: int) -> int:
        raise RuntimeError("gateway unreachable")


def _labeled_buckets(prefix: str, n_buckets: int = 4, docs_per_bucket: int = 6) -> dict[str, list[str]]:
    """Buckets whose every doc text encodes its bucket, for label checks."""
    return {f"{prefix}{b}": [f"{prefix}{b}-doc{d}" for d in range(docs_per_bucket)] for b in range(n_buckets)}


# ---------------------------------------------------------------------------
# Fake OpenAI client for LlmPanelist (the I/O boundary)
# ---------------------------------------------------------------------------


class _FakeCompletions:
    def __init__(self, content: str):
        self._content = content

    def create(self, **_kwargs):
        message = type("Msg", (), {"content": self._content})()
        choice = type("Choice", (), {"message": message})()
        return type("Completion", (), {"choices": [choice]})()


def _fake_client(content: str):
    completions = _FakeCompletions(content)
    chat = type("Chat", (), {"completions": completions})()
    return type("Client", (), {"chat": chat})()


def _trial(intruder_index: int = 2) -> IntruderTrial:
    return IntruderTrial(
        side=Side.LHS,
        in_group_bucket="A",
        intruder_bucket="B",
        documents=tuple(f"doc{i}" for i in range(5)),
        intruder_index=intruder_index,
    )


# ---------------------------------------------------------------------------
# BucketPool sampling
# ---------------------------------------------------------------------------


def test_sample_trial_labels_the_intruder_by_origin_bucket():
    """The doc at ``intruder_index`` comes from a different bucket than the other four."""
    pool = BucketPool(Side.LHS, _labeled_buckets("X"))
    rng = np.random.default_rng(1)
    for _ in range(200):
        trial = pool.sample_trial(rng)
        assert trial.intruder_bucket != trial.in_group_bucket
        assert len(trial.documents) == 5
        intruder_doc = trial.documents[trial.intruder_index]
        in_group_docs = [d for i, d in enumerate(trial.documents) if i != trial.intruder_index]
        assert intruder_doc.startswith(trial.intruder_bucket + "-")
        assert all(d.startswith(trial.in_group_bucket + "-") for d in in_group_docs)


def test_bucketpool_reads_only_the_head_not_the_whole_bucket():
    """Construction must not stream past ``head_size`` -- buckets may be huge or lazy.

    Each bucket is a generator that raises if read past the head, so a full scan
    (or a reservoir pass) would blow up here. This is the contract that makes the
    pre-shuffled-prefix sampling cheap.
    """
    head_size = 8

    def head_only(prefix: str):
        for i in range(1_000_000):
            assert i < head_size, "BucketPool streamed past the head"
            yield f"{prefix}-doc{i}"

    pool = BucketPool(Side.LHS, {f"b{b}": head_only(f"b{b}") for b in range(3)}, head_size=head_size)
    trial = pool.sample_trial(np.random.default_rng(0))
    assert len(trial.documents) == 5  # still produces valid trials from the head


def test_bucketpool_rejects_bucket_below_in_group():
    """Every bucket must hold >= 4 docs; a smaller one is rejected, not silently dropped."""
    with pytest.raises(ValueError, match="every bucket needs"):
        BucketPool(Side.LHS, {"a": ["1", "2", "3", "4", "5"], "b": ["1", "2"]})


def test_bucketpool_rejects_fewer_than_two_buckets():
    """An intruder needs a second bucket to come from."""
    with pytest.raises(ValueError, match=">= 2 buckets"):
        BucketPool(Side.LHS, {"a": ["1", "2", "3", "4", "5"]})


def test_bucketpool_rejects_head_smaller_than_in_group():
    """A head too small to hold an in-group is a misconfiguration, not a silent empty run."""
    with pytest.raises(ValueError, match="too small to form an in-group"):
        BucketPool(Side.LHS, _labeled_buckets("X"), head_size=2)


def test_bucketpool_warns_when_bucket_shorter_than_head(caplog):
    """A bucket smaller than head_size is sampled in full and surfaced as a warning, not silently."""
    with caplog.at_level(logging.WARNING, logger="experiments.datakit.intruder"):
        BucketPool(Side.LHS, {"a": ["1", "2", "3", "4", "5"], "b": ["6", "7", "8", "9"]}, head_size=128)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert 128 in warnings[0].args  # the warning carries the configured head_size, not just prose


# ---------------------------------------------------------------------------
# Robbins confidence sequence
# ---------------------------------------------------------------------------


def test_robbins_radius_infinite_before_any_data():
    assert _robbins_radius(0, alpha=0.05, rho=0.1) == np.inf


def test_robbins_radius_shrinks_with_more_data():
    radii = [_robbins_radius(n, alpha=0.05, rho=1 / np.sqrt(250)) for n in (1, 10, 100, 1000)]
    assert all(a > b for a, b in pairwise(radii))


def test_confidence_sequence_is_anytime_valid():
    """The interval covers the true mean at *every* sample size with prob >= 1 - alpha.

    This is the whole point of using a confidence sequence over a fixed-n CI:
    a coverage failure at any peeked-at n counts as a miss, and the empirical
    miss rate across many sequences must stay at or below alpha. A broken
    radius formula (dropped log term, wrong constant) fails this.
    """
    alpha = 0.05
    true_p = 0.3
    n_sequences, horizon = 600, 200
    rng = np.random.default_rng(123)
    draws = rng.random((n_sequences, horizon)) < true_p

    misses = 0
    for seq in draws:
        cs = ConfidenceSequence(alpha=alpha, rho=1 / np.sqrt(horizon))
        for hit in seq:
            cs.update(float(hit))
            lo, hi = cs.interval()
            if not (lo <= true_p <= hi):
                misses += 1
                break  # one time-uniform miss is enough to fail this sequence
    assert misses / n_sequences <= alpha


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------


def _cs_with(observations: list[float], *, alpha: float = 0.025, target: int = 250) -> ConfidenceSequence:
    cs = ConfidenceSequence(alpha=alpha, rho=1 / np.sqrt(target))
    for v in observations:
        cs.update(v)
    return cs


def test_decide_calls_winner_when_intervals_separate():
    lhs = _cs_with([1.0] * 300)
    rhs = _cs_with([0.0] * 300)
    assert _decide(lhs, rhs, rope=0.05) == Decision.LHS_MORE_COHERENT
    assert _decide(rhs, lhs, rope=0.05) == Decision.RHS_MORE_COHERENT


def test_decide_calls_tie_when_difference_is_negligible():
    """Equal means with tight enough intervals land inside the ROPE."""
    lhs = _cs_with([0.5] * 4000)
    rhs = _cs_with([0.5] * 4000)
    assert _decide(lhs, rhs, rope=0.2) == Decision.PRACTICAL_TIE


def test_decide_withholds_verdict_while_intervals_are_wide():
    """Too little data => neither a winner nor a tie; keep sampling."""
    lhs = _cs_with([1.0, 0.0, 1.0, 0.0])
    rhs = _cs_with([1.0, 0.0, 1.0, 0.0])
    assert _decide(lhs, rhs, rope=0.05) is None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def test_run_intruder_test_picks_the_more_detectable_side():
    """A panel that detects better on one side makes that side the winner."""
    panel = [SideAwarePanelist("judge", {Side.LHS: 0.95, Side.RHS: 0.25})]
    result = run_intruder_test(
        _labeled_buckets("A"),
        _labeled_buckets("B"),
        panel=panel,
        lhs_name="coherent",
        rhs_name="incoherent",
        min_trials=16,
        max_trials=400,
        batch_size=8,
        seed=1,
    )
    assert result.decision == Decision.LHS_MORE_COHERENT
    assert result.lhs_accuracy > result.rhs_accuracy
    assert result.difference_interval[0] > 0
    assert result.per_model_accuracy["judge"]["coherent"] > result.per_model_accuracy["judge"]["incoherent"]
    assert result.n_abstained == 0


def test_run_intruder_test_terminates_at_cap_when_panel_always_abstains():
    """A panel that fails every call must hit the attempt cap, not loop forever.

    Regression for the bound-on-attempts fix: all-abstention trials never
    advance the confidence sequence, so a completed-trial guard would issue
    paid calls indefinitely. The run must instead stop and report no progress.
    """
    result = run_intruder_test(
        _labeled_buckets("A"),
        _labeled_buckets("B"),
        panel=[DeadPanelist()],
        min_trials=16,
        max_trials=24,
        batch_size=8,
        seed=2,
    )
    assert result.decision == Decision.INCONCLUSIVE
    assert result.n_trials_per_side == 0
    assert result.n_abstained > 0


# ---------------------------------------------------------------------------
# LlmPanelist parsing (mocked at the OpenAI boundary)
# ---------------------------------------------------------------------------


def test_llm_panelist_maps_one_based_vote_to_zero_based_index():
    panelist = LlmPanelist(model="m", client=_fake_client('{"intruder": 3, "reasoning": "x"}'))
    assert panelist.vote(_trial(), max_doc_chars=2000) == 2


def test_llm_panelist_parses_through_a_code_fence():
    panelist = LlmPanelist(model="m", client=_fake_client('```json\n{"intruder": 1, "reasoning": "x"}\n```'))
    assert panelist.vote(_trial(), max_doc_chars=2000) == 0


def test_llm_panelist_rejects_out_of_range_vote():
    panelist = LlmPanelist(model="m", client=_fake_client('{"intruder": 9, "reasoning": "x"}'))
    with pytest.raises(ValueError, match="out-of-range"):
        panelist.vote(_trial(), max_doc_chars=2000)
