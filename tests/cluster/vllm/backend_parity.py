# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Framework-independent next-token parity against frozen scores."""

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict
from rigging.filesystem import StoragePath

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenScore:
    logprob: float
    token_id: int


@dataclass(frozen=True)
class NextTokenObservation:
    """One backend's exact next-token response, independent of its serving API."""

    case_id: str
    backend_rank: int
    emitted_token_id: int
    returned_top_logprobs: tuple[TokenScore, ...]

    @classmethod
    def from_logprob_map(
        cls,
        case_id: str,
        emitted_token_id: int,
        actual_logprobs: dict[int, float],
        *,
        backend_rank: int,
    ) -> "NextTokenObservation":
        """Normalize a serving response so exact repeats compare as plain values."""
        _assert_valid_emitted_token(case_id, emitted_token_id, actual_logprobs, backend_rank=backend_rank)
        return cls(
            case_id=case_id,
            backend_rank=backend_rank,
            emitted_token_id=emitted_token_id,
            returned_top_logprobs=tuple(
                TokenScore(token_id=token_id, logprob=logprob) for token_id, logprob in sorted(actual_logprobs.items())
            ),
        )

    def parity_against(self, expected_top_logprobs: tuple[TokenScore, ...]) -> "NextTokenParity":
        return _parity_from_token_scores(
            self.case_id,
            expected_top_logprobs,
            self.emitted_token_id,
            {score.token_id: score.logprob for score in self.returned_top_logprobs},
            backend_rank=self.backend_rank,
        )


class ParityCaseFailure(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    backend_rank: int
    error: str


class ParityReport(BaseModel):
    """Durable successes and failures from one backend run."""

    model_config = ConfigDict(frozen=True)

    backend: str
    observations: tuple[NextTokenObservation, ...]
    case_failures: tuple[ParityCaseFailure, ...] = ()


@dataclass(frozen=True)
class CrossRankDiagnostic:
    """A non-gating summary of variation between backend ranks."""

    case_id: str
    emitted_token_ids_by_rank: tuple[tuple[int, int], ...]
    shared_top_token_count: int
    max_probability_spread: float


@dataclass(frozen=True)
class NextTokenParity:
    """One backend observation against one frozen next-token distribution."""

    case_id: str
    backend_rank: int
    greedy_token_id: int
    golden_top_token_ids: tuple[int, ...]
    golden_probability_gap_to_greedy: float
    max_probability_error: float
    top_probability_l1_error: float

    def failure_messages(self, *, max_probability_error: float) -> list[str]:
        """Describe every violated distribution contract without short-circuiting."""
        failures = []
        if self.greedy_token_id not in self.golden_top_token_ids:
            failures.append(f"{self.case_id}: emitted token {self.greedy_token_id} is absent from the golden top-k")
        if self.golden_probability_gap_to_greedy > 2 * self.max_probability_error:
            failures.append(
                f"{self.case_id}: golden probability gap {self.golden_probability_gap_to_greedy:.6f} "
                f"exceeds twice the observed error {self.max_probability_error:.6f}"
            )
        if self.max_probability_error > max_probability_error:
            failures.append(
                f"{self.case_id}: max probability error {self.max_probability_error:.6f} "
                f"exceeds {max_probability_error:.6f}"
            )
        return failures

    def assert_matches(self, *, max_probability_error: float) -> None:
        """Require golden-token coverage and a probability-supported winner."""
        failures = self.failure_messages(max_probability_error=max_probability_error)
        assert not failures, "\n".join(failures)


def _assert_valid_emitted_token(
    case_id: str,
    emitted_token_id: int,
    actual_logprobs: dict[int, float],
    *,
    backend_rank: int,
) -> None:
    assert actual_logprobs, f"{case_id} rank {backend_rank}: backend returned no logprobs"
    assert all(
        math.isfinite(value) for value in actual_logprobs.values()
    ), f"{case_id} rank {backend_rank}: backend returned a non-finite logprob"
    assert emitted_token_id in actual_logprobs, f"{case_id} rank {backend_rank}: emitted token missing from logprobs"
    maximum_actual_logprob = max(actual_logprobs.values())
    assert (
        actual_logprobs[emitted_token_id] == maximum_actual_logprob
    ), f"{case_id} rank {backend_rank}: emitted token does not have maximum returned logprob"


def assert_same_rank_repeatability(
    first_wave: Sequence[NextTokenObservation],
    second_wave: Sequence[NextTokenObservation],
) -> None:
    """Require exact emitted tokens and returned logprobs per rank across waves."""
    first_by_rank = _observations_by_rank(first_wave)
    second_by_rank = _observations_by_rank(second_wave)
    assert (
        first_by_rank.keys() == second_by_rank.keys()
    ), f"repeat waves covered different ranks: {sorted(first_by_rank)} != {sorted(second_by_rank)}"
    for rank, first_observation in first_by_rank.items():
        second_observation = second_by_rank[rank]
        assert first_observation == second_observation, (
            f"{first_observation.case_id} rank {rank} was not exactly repeatable: "
            f"first_wave={first_observation!r}, second_wave={second_observation!r}"
        )


def cross_rank_diagnostic(observations: Sequence[NextTokenObservation]) -> CrossRankDiagnostic:
    """Summarize cross-rank spread without imposing a correctness gate."""
    observations_by_rank = _observations_by_rank(observations)
    case_ids = {observation.case_id for observation in observations_by_rank.values()}
    assert len(case_ids) == 1, f"cross-rank diagnostic requires one case, got {sorted(case_ids)}"
    logprobs_by_rank = [
        {score.token_id: score.logprob for score in observation.returned_top_logprobs}
        for observation in observations_by_rank.values()
    ]
    shared_tokens = set.intersection(*(set(scores) for scores in logprobs_by_rank))
    maximum_spread = max(
        (
            max(math.exp(scores[token_id]) for scores in logprobs_by_rank)
            - min(math.exp(scores[token_id]) for scores in logprobs_by_rank)
            for token_id in shared_tokens
        ),
        default=float("nan"),
    )
    return CrossRankDiagnostic(
        case_id=case_ids.pop(),
        emitted_token_ids_by_rank=tuple(
            (rank, observation.emitted_token_id) for rank, observation in sorted(observations_by_rank.items())
        ),
        shared_top_token_count=len(shared_tokens),
        max_probability_spread=maximum_spread,
    )


def parity_from_logprob_row(
    case_id: str,
    expected_top_logprobs: tuple[TokenScore, ...],
    logprobs_row: np.ndarray,
    *,
    backend_rank: int,
) -> NextTokenParity:
    """Score a full Levanter ``[vocab]`` log-softmax row."""
    assert np.isfinite(logprobs_row).all(), f"{case_id} rank {backend_rank}: backend returned a non-finite logprob"
    greedy_token_id = int(logprobs_row.argmax())
    actual_logprobs = {score.token_id: float(logprobs_row[score.token_id]) for score in expected_top_logprobs}
    return _parity_from_token_scores(
        case_id,
        expected_top_logprobs,
        greedy_token_id,
        actual_logprobs,
        backend_rank=backend_rank,
    )


def _observations_by_rank(
    observations: Sequence[NextTokenObservation],
) -> dict[int, NextTokenObservation]:
    assert observations, "at least one backend observation is required"
    observations_by_rank = {observation.backend_rank: observation for observation in observations}
    assert len(observations_by_rank) == len(observations), "backend ranks must be unique within one wave"
    return observations_by_rank


def _parity_from_token_scores(
    case_id: str,
    expected_top_logprobs: tuple[TokenScore, ...],
    greedy_token_id: int,
    actual_logprobs: dict[int, float],
    *,
    backend_rank: int,
) -> NextTokenParity:
    golden_logprobs = {entry.token_id: entry.logprob for entry in expected_top_logprobs}
    assert golden_logprobs, f"{case_id}: golden has no top logprobs"
    assert all(math.isfinite(value) for value in golden_logprobs.values()), f"{case_id}: golden has a non-finite logprob"
    assert all(
        math.isfinite(value) for value in actual_logprobs.values()
    ), f"{case_id} rank {backend_rank}: backend returned a non-finite logprob"
    missing = golden_logprobs.keys() - actual_logprobs.keys()
    assert not missing, f"{case_id} rank {backend_rank}: golden tokens missing from backend logprobs: {sorted(missing)}"
    probability_errors = tuple(
        abs(math.exp(actual_logprobs[token_id]) - math.exp(golden_logprob))
        for token_id, golden_logprob in golden_logprobs.items()
    )
    maximum_golden_logprob = max(golden_logprobs.values())
    selected_golden_logprob = golden_logprobs.get(greedy_token_id, -math.inf)
    return NextTokenParity(
        case_id=case_id,
        backend_rank=backend_rank,
        greedy_token_id=greedy_token_id,
        golden_top_token_ids=tuple(golden_logprobs),
        golden_probability_gap_to_greedy=math.exp(maximum_golden_logprob) - math.exp(selected_golden_logprob),
        max_probability_error=max(probability_errors),
        top_probability_l1_error=sum(probability_errors),
    )


def _persist_and_index_report(
    report: ParityReport,
    report_uri: str,
    expected_case_ids: set[str],
) -> tuple[dict[str, NextTokenObservation], list[str]]:
    """Persist first, then collect report-wide failures without short-circuiting."""
    StoragePath(report_uri).write_text(report.model_dump_json(indent=2) + "\n")

    failures = []
    failed_case_ids = [failure.case_id for failure in report.case_failures]
    failures.extend(
        f"{failure.case_id} rank {failure.backend_rank}: request failed: {failure.error}"
        for failure in report.case_failures
    )
    if len(set(failed_case_ids)) != len(failed_case_ids):
        failures.append("request failures contain duplicate case IDs")

    observations_by_id: dict[str, NextTokenObservation] = {}
    for observation in report.observations:
        if observation.case_id in observations_by_id:
            failures.append(f"{observation.case_id}: duplicate observation")
            continue
        observations_by_id[observation.case_id] = observation
    covered_case_ids = observations_by_id.keys() | set(failed_case_ids)
    overlap = observations_by_id.keys() & set(failed_case_ids)
    missing = expected_case_ids - covered_case_ids
    extra = covered_case_ids - expected_case_ids
    if overlap:
        failures.append(f"cases reported as both success and failure: {sorted(overlap)}")
    if missing:
        failures.append(f"missing observations: {sorted(missing)}")
    if extra:
        failures.append(f"unexpected observations: {sorted(extra)}")
    return observations_by_id, failures


def persist_and_validate_exact_report(
    report: ParityReport,
    report_uri: str,
    expected_top_logprobs_by_case: dict[str, tuple[TokenScore, ...]],
) -> None:
    """Persist a native-backend report, then require exact frozen scores."""
    observations, failures = _persist_and_index_report(
        report,
        report_uri,
        set(expected_top_logprobs_by_case),
    )
    for case_id, expected in expected_top_logprobs_by_case.items():
        observation = observations.get(case_id)
        if observation is None:
            continue
        if observation.returned_top_logprobs != expected:
            failures.append(f"{case_id}: top logprobs differ from the frozen golden")
        if not expected:
            failures.append(f"{case_id}: frozen golden has no token scores")
        elif observation.emitted_token_id != expected[0].token_id:
            failures.append(
                f"{case_id}: emitted token {observation.emitted_token_id} differs from "
                f"frozen greedy token {expected[0].token_id}"
            )
    if failures:
        raise AssertionError("Parity report failed:\n" + "\n".join(f"- {failure}" for failure in failures))


def persist_and_validate_bounded_report(
    report: ParityReport,
    report_uri: str,
    expected_top_logprobs_by_case: dict[str, tuple[TokenScore, ...]],
    bucket_and_bound_by_case: dict[str, tuple[int, float]],
) -> None:
    """Persist a serving report, then evaluate every reachable prompt bucket."""
    if bucket_and_bound_by_case.keys() != expected_top_logprobs_by_case.keys():
        raise ValueError("Every expected case must have exactly one bucket bound")
    observations, failures = _persist_and_index_report(
        report,
        report_uri,
        set(expected_top_logprobs_by_case),
    )
    bucket_maxima: dict[int, float] = {}
    for case_id, expected in expected_top_logprobs_by_case.items():
        observation = observations.get(case_id)
        if observation is None:
            continue
        bucket, bound = bucket_and_bound_by_case[case_id]
        try:
            parity = observation.parity_against(expected)
            bucket_maxima[bucket] = max(bucket_maxima.get(bucket, 0.0), parity.max_probability_error)
            failures.extend(parity.failure_messages(max_probability_error=bound))
        except AssertionError as error:
            failures.append(f"{case_id}: {error}")
    for bucket in sorted(set(bucket for bucket, _ in bucket_and_bound_by_case.values())):
        if bucket in bucket_maxima:
            # The persisted per-case observations are the source of truth; this line
            # keeps the aggregate visible in the job log without adding report fields.
            logger.info(
                "TPU parity bucket <= %d: max_probability_error=%.6f",
                bucket,
                bucket_maxima[bucket],
            )
    if failures:
        raise AssertionError("Parity report failed:\n" + "\n".join(f"- {failure}" for failure in failures))
