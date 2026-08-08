# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pytest
from rigging.filesystem import StoragePath

from tests.cluster.vllm.backend_parity import (
    NextTokenObservation,
    NextTokenParity,
    ParityCaseFailure,
    ParityReport,
    TokenScore,
    assert_same_rank_repeatability,
    cross_rank_diagnostic,
    parity_from_logprob_row,
    persist_and_validate_bounded_report,
    persist_and_validate_exact_report,
)
from tests.cluster.vllm.snowball import read_prompt_fixture, read_representative_goldens, read_tpu_representative_goldens


def _parity(*, greedy_token_id: int, gap: float, error: float) -> NextTokenParity:
    return NextTokenParity(
        case_id="case",
        backend_rank=0,
        greedy_token_id=greedy_token_id,
        golden_top_token_ids=(2, 3),
        golden_probability_gap_to_greedy=gap,
        max_probability_error=error,
        top_probability_l1_error=error,
    )


def _observation(rank: int, *, greedy_token_id: int = 2, logprob_offset: float = 0.0) -> NextTokenObservation:
    return NextTokenObservation(
        case_id="case",
        backend_rank=rank,
        emitted_token_id=greedy_token_id,
        returned_top_logprobs=(
            TokenScore(token_id=2, logprob=-0.1 + logprob_offset),
            TokenScore(token_id=3, logprob=-2.0),
        ),
    )


@pytest.mark.parametrize(
    "parity",
    [
        _parity(greedy_token_id=2, gap=0.0, error=0.0),
        _parity(greedy_token_id=3, gap=0.01, error=0.005),
    ],
)
def test_backend_distribution_contract_accepts_exact_and_error_explained_winners(parity: NextTokenParity) -> None:
    parity.assert_matches(max_probability_error=0.075)


@pytest.mark.parametrize(
    "parity",
    [
        _parity(greedy_token_id=3, gap=0.011, error=0.005),
        _parity(greedy_token_id=9, gap=0.01, error=0.01),
        _parity(greedy_token_id=2, gap=0.0, error=0.076),
    ],
)
def test_backend_distribution_contract_rejects_unexplained_outside_or_over_bound_winners(
    parity: NextTokenParity,
) -> None:
    with pytest.raises(AssertionError):
        parity.assert_matches(max_probability_error=0.075)


def test_same_rank_repeatability_compares_every_rank_without_cross_rank_equality() -> None:
    first_wave = (_observation(0, greedy_token_id=2), _observation(1, greedy_token_id=3))
    second_wave = tuple(reversed(first_wave))

    assert_same_rank_repeatability(first_wave, second_wave)

    changed_rank_one = (_observation(0, greedy_token_id=2), _observation(1, greedy_token_id=2))
    with pytest.raises(AssertionError, match="rank 1 was not exactly repeatable"):
        assert_same_rank_repeatability(first_wave, changed_rank_one)


def test_cross_rank_spread_is_reported_without_becoming_a_gate() -> None:
    observations = (_observation(0), _observation(1, greedy_token_id=3, logprob_offset=-0.25))

    diagnostic = cross_rank_diagnostic(observations)

    assert diagnostic.case_id == "case"
    assert diagnostic.emitted_token_ids_by_rank == ((0, 2), (1, 3))
    assert diagnostic.shared_top_token_count == 2
    assert diagnostic.max_probability_spread == pytest.approx(math.exp(-0.1) - math.exp(-0.35))


def test_report_persists_successes_and_request_failures_before_aggregating(tmp_path) -> None:
    expected = {
        "numerical": (TokenScore(token_id=2, logprob=-0.1), TokenScore(token_id=3, logprob=-2.0)),
        "request": (TokenScore(token_id=2, logprob=-0.1),),
        "winner": (TokenScore(token_id=2, logprob=-0.1),),
    }
    report = ParityReport(
        backend="vllm-tpu",
        observations=(
            NextTokenObservation(
                case_id="numerical",
                backend_rank=0,
                emitted_token_id=2,
                returned_top_logprobs=(
                    TokenScore(token_id=2, logprob=-1.0),
                    TokenScore(token_id=3, logprob=-2.0),
                ),
            ),
            NextTokenObservation(
                case_id="winner",
                backend_rank=0,
                emitted_token_id=9,
                returned_top_logprobs=(
                    TokenScore(token_id=2, logprob=-0.2),
                    TokenScore(token_id=9, logprob=-0.1),
                ),
            ),
        ),
        case_failures=(ParityCaseFailure(case_id="request", backend_rank=0, error="HTTP 500"),),
    )
    report_uri = str(tmp_path / "parity.json")

    with pytest.raises(AssertionError) as raised:
        persist_and_validate_bounded_report(
            report,
            report_uri,
            expected,
            {case_id: (256, 0.01) for case_id in expected},
        )

    persisted = ParityReport.model_validate_json(StoragePath(report_uri).read_text())
    assert persisted == report
    message = str(raised.value)
    assert "request failed: HTTP 500" in message
    assert "max probability error" in message
    assert "emitted token 9 is absent" in message

    exact_report_uri = str(tmp_path / "exact-parity.json")
    with pytest.raises(AssertionError, match="top logprobs differ from the frozen golden") as exact_failure:
        persist_and_validate_exact_report(report, exact_report_uri, expected)
    assert ParityReport.model_validate_json(StoragePath(exact_report_uri).read_text()) == report
    assert "emitted token 9 differs from frozen greedy token 2" in str(exact_failure.value)


def test_comparators_reject_non_finite_backend_logprobs() -> None:
    with pytest.raises(AssertionError, match="non-finite"):
        NextTokenObservation.from_logprob_map("case", 2, {2: float("nan")}, backend_rank=0)
    with pytest.raises(AssertionError, match="non-finite"):
        parity_from_logprob_row(
            "case",
            (TokenScore(token_id=2, logprob=-0.1),),
            np.array([-1.0, np.inf, -0.1]),
            backend_rank=0,
        )


def test_prompt_fixture_digest_is_verified(tmp_path) -> None:
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text("{}")

    with pytest.raises(ValueError, match="prompt fixture digest changed"):
        read_prompt_fixture((), fixture_url=str(fixture_path))


def test_tpu_golden_covers_the_gpu_prompt_set() -> None:
    gpu_cases = read_representative_goldens()
    tpu_cases = read_tpu_representative_goldens()

    assert len(gpu_cases) == len(tpu_cases) == 64
    assert {case.id for case in gpu_cases} == {case.id for case in tpu_cases}
    assert {len(case.top_logprobs) for case in tpu_cases} == {25}
