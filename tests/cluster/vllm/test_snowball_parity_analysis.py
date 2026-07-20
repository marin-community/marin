# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import math

import pytest

from tests.cluster.vllm.backend_parity import (
    GoldenTokenObservation,
    NextTokenObservation,
    ObservationReport,
    RunProvenance,
    TokenScore,
)
from tests.cluster.vllm.snowball import RepresentativeGolden
from tests.cluster.vllm.snowball_parity_analysis import analyze_reports, round_up_one_significant_digit


def _report(*logprobs: float) -> ObservationReport:
    provenance = RunProvenance(
        backend="levanter-native",
        platform="tpu",
        process_id="process",
        code_digest="code",
        parameter_digest="parameters",
        model_config_digest="config",
        prompt_fixture_digest="prompt",
        requested_attention="tpu_splash",
        effective_attention="tpu_splash",
        requested_moe="ring",
        effective_moe="scatter",
        mesh_shape=(("data", 8),),
        device_kind="TPU v6e",
    )
    observations = tuple(
        NextTokenObservation(
            case_id="case",
            bucket_max_tokens=256,
            repeat_index=index,
            backend_index=0,
            greedy_token_id=7,
            top_logprobs=(TokenScore(token_id=7, logprob=logprob),),
            golden_tokens=(GoldenTokenObservation(token_id=7, logprob=logprob, rank=0),),
            capacity_overflow=(0.0,),
            has_nonfinite=False,
        )
        for index, logprob in enumerate(logprobs)
    )
    return ObservationReport(provenance=provenance, observations=observations)


def test_analyze_reports_separates_canonical_error_from_repeatability() -> None:
    goldens = (RepresentativeGolden(id="case", top_logprobs=(TokenScore(token_id=7, logprob=0.0),)),)

    summary = analyze_reports(
        goldens,
        (_report(math.log(0.9), math.log(0.8)),),
        require_complete=False,
    )

    platform = summary["backends"]["levanter-native"]["platforms"]["tpu"]
    assert platform["max_probability_error"] == pytest.approx(0.2)
    assert platform["repeatability_probability_error"] == pytest.approx(0.1)
    assert platform["candidate_probability_error"] == pytest.approx(0.4)
    assert platform["candidate_headroom"] == pytest.approx(0.2)
    assert platform["contract_candidate_by_bucket"] == {
        "256": {
            "max_probability_error": pytest.approx(0.2),
            "repeatability_probability_error": pytest.approx(0.1),
            "candidate_probability_error": pytest.approx(0.4),
            "candidate_headroom": pytest.approx(0.2),
        }
    }
    assert not platform["bitwise_repeatable"]
    assert platform["worst_cases_by_probability_error"] == ["case"]
    assert platform["cases"]["case"] == {
        "bucket_max_tokens": [256],
        "canonical_top_token_ranks": [0],
        "greedy_agreement": 1.0,
        "greedy_token_ids": [7],
        "max_probability_error": pytest.approx(0.2),
        "observation_count": 2,
        "probability_error": {
            "max": pytest.approx(0.2),
            "p50": pytest.approx(0.2),
            "p95": pytest.approx(0.2),
            "p99": pytest.approx(0.2),
        },
        "repeatability_probability_error": pytest.approx(0.1),
        "top25_recall": {"max": 1.0, "p50": 1.0, "p95": 1.0, "p99": 1.0},
    }


def test_round_up_one_significant_digit_is_conservative() -> None:
    assert round_up_one_significant_digit(0.0) == 0.0
    assert round_up_one_significant_digit(0.00577) == pytest.approx(0.006)
    assert round_up_one_significant_digit(0.31) == pytest.approx(0.4)


def test_exact_canonical_uses_dedicated_canonical_token_scores() -> None:
    goldens = (RepresentativeGolden(id="case", top_logprobs=(TokenScore(token_id=7, logprob=0.0),)),)
    report = _report(0.0)
    report = dataclasses.replace(
        report,
        observations=(
            dataclasses.replace(
                report.observations[0],
                top_logprobs=(TokenScore(token_id=7, logprob=-0.25),),
            ),
        ),
    )

    summary = analyze_reports(goldens, (report,), require_complete=False)

    assert summary["backends"]["levanter-native"]["platforms"]["tpu"]["exact_canonical_top25"]


def test_analyze_reports_keeps_backend_cells_separate() -> None:
    goldens = (RepresentativeGolden(id="case", top_logprobs=(TokenScore(token_id=7, logprob=0.0),)),)
    native = _report(0.0)
    exported = dataclasses.replace(
        _report(math.log(0.9)),
        provenance=dataclasses.replace(_report(0.0).provenance, backend="levanter-exported"),
    )

    summary = analyze_reports(goldens, (native, exported), require_complete=False)

    assert summary["backends"]["levanter-native"]["platforms"]["tpu"]["max_probability_error"] == 0.0
    assert summary["backends"]["levanter-exported"]["platforms"]["tpu"]["max_probability_error"] == pytest.approx(0.1)
