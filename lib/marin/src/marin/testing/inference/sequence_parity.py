# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare teacher-forced sequences without aligning away missing or shifted tokens."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True)
class ScoredToken:
    token_id: int
    logprob: float | None
    top1_token_id: int | None


@dataclass(frozen=True)
class ScoredSequence:
    case_id: str
    input_sha256: str
    tokens: tuple[ScoredToken, ...]


@dataclass(frozen=True)
class PositionBand:
    start: int
    end: int
    count: int
    mean_absolute_delta: float
    max_absolute_delta: float
    top1_disagreements: int

    @property
    def disagreement_fraction(self) -> float:
        return self.top1_disagreements / self.count


@dataclass(frozen=True)
class SequenceThresholds:
    pooled_mean: float
    band_mean: float
    top1_fraction: float


@dataclass(frozen=True)
class SequenceComparison:
    pooled: PositionBand
    bands: tuple[PositionBand, ...]
    failures: tuple[str, ...]


def compare_sequences(
    candidate: Sequence[ScoredSequence],
    reference: Sequence[ScoredSequence],
    *,
    boundaries: Sequence[int],
    thresholds: SequenceThresholds,
) -> SequenceComparison:
    """Require identical inputs and full token coverage before evaluating numerical error.

    Boundaries are absolute positions, including the initial unscored token. Every
    requested band must contain scores: short or partial responses cannot pass a
    long-context check. Missing initial logprobs are allowed only on both sides.
    """
    left = {item.case_id: item for item in candidate}
    right = {item.case_id: item for item in reference}
    if len(left) != len(candidate) or len(right) != len(reference) or not left or left.keys() != right.keys():
        raise ValueError("Different, empty, or duplicate case IDs")
    if len(boundaries) < 2 or boundaries[0] != 0 or any(a >= b for a, b in pairwise(boundaries)):
        raise ValueError("Position boundaries must increase from zero")
    values: list[tuple[int, float, bool]] = []
    for case_id, item in left.items():
        other = right[case_id]
        if item.input_sha256 != other.input_sha256 or len(item.tokens) != len(other.tokens):
            raise ValueError(f"Input or token coverage differs: {case_id}")
        if len(item.tokens) > boundaries[-1]:
            raise ValueError(f"Position bands omit tokens: {case_id}")
        for position, (a, b) in enumerate(zip(item.tokens, other.tokens, strict=True)):
            if a.token_id != b.token_id:
                raise ValueError(f"Token alignment differs: {case_id}:{position}")
            if position == 0 and a.logprob is None and b.logprob is None:
                continue
            if a.logprob is None or b.logprob is None or a.top1_token_id is None or b.top1_token_id is None:
                raise ValueError(f"Missing score: {case_id}:{position}")
            if not math.isfinite(a.logprob) or not math.isfinite(b.logprob):
                raise ValueError(f"Nonfinite score: {case_id}:{position}")
            values.append((position, abs(a.logprob - b.logprob), a.top1_token_id != b.top1_token_id))

    def summarize(start: int, end: int) -> PositionBand:
        selected = [(delta, mismatch) for position, delta, mismatch in values if start <= position < end]
        if not selected:
            raise ValueError(f"No scores in required band [{start}, {end})")
        return PositionBand(
            start,
            end,
            len(selected),
            sum(x[0] for x in selected) / len(selected),
            max(x[0] for x in selected),
            sum(x[1] for x in selected),
        )

    pooled = summarize(0, boundaries[-1])
    bands = tuple(summarize(a, b) for a, b in pairwise(boundaries))
    failures = []
    if pooled.mean_absolute_delta > thresholds.pooled_mean:
        failures.append("pooled_mean")
    if pooled.disagreement_fraction > thresholds.top1_fraction:
        failures.append("top1_fraction")
    failures.extend(
        f"band_mean:{band.start}:{band.end}" for band in bands if band.mean_absolute_delta > thresholds.band_mean
    )
    return SequenceComparison(pooled, bands, tuple(failures))


@dataclass(frozen=True)
class AccuracyObservation:
    case_id: str
    correct: bool
    finish_reason: str


@dataclass(frozen=True)
class AccuracyComparison:
    count: int
    candidate_accuracy: float
    reference_accuracy: float
    z: float
    gained: int
    lost: int
    changed_length_finishes: tuple[str, ...]
    failures: tuple[str, ...]


def compare_accuracy(
    candidate: Sequence[AccuracyObservation], reference: Sequence[AccuracyObservation], *, max_z: float
) -> AccuracyComparison:
    """Compare paired item coverage, truncation and an independent-binomial SEM sentinel."""
    left = {item.case_id: item for item in candidate}
    right = {item.case_id: item for item in reference}
    if len(left) != len(candidate) or len(right) != len(reference) or not left or left.keys() != right.keys():
        raise ValueError("Different, empty, or duplicate accuracy case IDs")
    n = len(left)
    p = sum(item.correct for item in left.values()) / n
    q = sum(item.correct for item in right.values()) / n
    sem = math.sqrt((p * (1 - p) + q * (1 - q)) / n)
    z = (p - q) / sem if sem else (0.0 if p == q else math.copysign(math.inf, p - q))
    truncations = tuple(
        sorted(key for key in left if (left[key].finish_reason == "length") != (right[key].finish_reason == "length"))
    )
    failures = []
    if abs(z) > max_z:
        failures.append("accuracy_z")
    if truncations:
        failures.append("length_finishes")
    return AccuracyComparison(
        n,
        p,
        q,
        z,
        sum(left[k].correct and not right[k].correct for k in left),
        sum(not left[k].correct and right[k].correct for k in left),
        truncations,
        tuple(failures),
    )
