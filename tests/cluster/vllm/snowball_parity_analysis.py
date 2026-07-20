# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze Snowball observation reports without accelerator access."""

import argparse
import json
import math
from collections import defaultdict

import numpy as np
from rigging.filesystem import StoragePath

from tests.cluster.vllm.backend_parity import (
    NextTokenObservation,
    ObservationReport,
    TokenScore,
    observations_bitwise_equal,
)
from tests.cluster.vllm.snowball import RepresentativeGolden, read_representative_goldens


def round_up_one_significant_digit(value: float) -> float:
    if value < 0 or not math.isfinite(value):
        raise ValueError(f"Expected a finite nonnegative value, got {value}")
    if value == 0:
        return 0.0
    magnitude = 10.0 ** math.floor(math.log10(value))
    return math.ceil(value / magnitude - 1e-12) * magnitude


def _float32_bits(value: float) -> int:
    return int(np.asarray(value, dtype=np.float32).view(np.uint32))


def _token_scores_bitwise_equal(left: tuple[TokenScore, ...], right: tuple[TokenScore, ...]) -> bool:
    return len(left) == len(right) and all(
        left_score.token_id == right_score.token_id
        and _float32_bits(left_score.logprob) == _float32_bits(right_score.logprob)
        for left_score, right_score in zip(left, right, strict=True)
    )


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.quantile(array, 0.50, method="higher")),
        "p95": float(np.quantile(array, 0.95, method="higher")),
        "p99": float(np.quantile(array, 0.99, method="higher")),
        "max": float(np.max(array)),
    }


def _validate_complete_report(report: ObservationReport, expected_ids: set[str]) -> None:
    by_repeat: dict[int, list[str]] = defaultdict(list)
    for observation in report.observations:
        by_repeat[observation.repeat_index].append(observation.case_id)
    if not by_repeat:
        raise ValueError(f"Report {report.provenance.process_id} contains no observations")
    for repeat_index, case_ids in by_repeat.items():
        if len(case_ids) != len(set(case_ids)):
            raise ValueError(f"Report {report.provenance.process_id} repeat {repeat_index} has duplicate cases")
        if set(case_ids) != expected_ids:
            missing = sorted(expected_ids - set(case_ids))
            extra = sorted(set(case_ids) - expected_ids)
            raise ValueError(
                f"Report {report.provenance.process_id} repeat {repeat_index} case mismatch: "
                f"missing={missing} extra={extra}"
            )


def _platform_summary(
    expected_by_id: dict[str, RepresentativeGolden],
    reports: tuple[ObservationReport, ...],
) -> dict:
    observations = tuple(observation for report in reports for observation in report.observations)
    probability_errors: list[float] = []
    logprob_errors: list[float] = []
    top_l1_errors: list[float] = []
    top25_recalls: list[float] = []
    greedy_matches = 0
    exact_canonical = True
    missing_golden_tokens = 0
    bucket_probability_errors: dict[int, list[float]] = defaultdict(list)
    case_probability_errors: dict[str, list[float]] = defaultdict(list)
    case_top25_recalls: dict[str, list[float]] = defaultdict(list)
    case_greedy_matches: dict[str, int] = defaultdict(int)

    grouped: dict[str, list[NextTokenObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.case_id].append(observation)
        expected = expected_by_id[observation.case_id]
        actual_by_id = {token.token_id: token for token in observation.golden_tokens}
        case_errors = []
        for golden in expected.top_logprobs:
            actual = actual_by_id.get(golden.token_id)
            if actual is None or actual.logprob is None:
                missing_golden_tokens += 1
                exact_canonical = False
                continue
            probability_error = abs(math.exp(actual.logprob) - math.exp(golden.logprob))
            logprob_error = abs(actual.logprob - golden.logprob)
            probability_errors.append(probability_error)
            bucket_probability_errors[observation.bucket_max_tokens].append(probability_error)
            case_probability_errors[observation.case_id].append(probability_error)
            logprob_errors.append(logprob_error)
            case_errors.append(probability_error)
        top_l1_errors.append(sum(case_errors))
        expected_ids = {score.token_id for score in expected.top_logprobs}
        actual_top25_ids = {score.token_id for score in observation.top_logprobs[: len(expected.top_logprobs)]}
        top25_recall = len(expected_ids & actual_top25_ids) / len(expected_ids)
        top25_recalls.append(top25_recall)
        case_top25_recalls[observation.case_id].append(top25_recall)
        greedy_match = observation.greedy_token_id == expected.top_logprobs[0].token_id
        greedy_matches += greedy_match
        case_greedy_matches[observation.case_id] += greedy_match
        observed_canonical = tuple(
            TokenScore(token_id=token.token_id, logprob=token.logprob)
            for token in observation.golden_tokens
            if token.logprob is not None
        )
        exact_canonical &= _token_scores_bitwise_equal(observed_canonical, expected.top_logprobs)

    bitwise_repeatable = True
    repeatability_probability_error = 0.0
    case_repeatability_probability_error: dict[str, float] = {}
    for case_id, case_observations in grouped.items():
        first = case_observations[0]
        bitwise_repeatable &= all(observations_bitwise_equal(first, other) for other in case_observations[1:])
        expected = expected_by_id[case_id]
        case_repeatability = 0.0
        for golden in expected.top_logprobs:
            probabilities = [
                math.exp(token.logprob)
                for observation in case_observations
                for token in observation.golden_tokens
                if token.token_id == golden.token_id and token.logprob is not None
            ]
            if probabilities:
                token_repeatability = max(probabilities) - min(probabilities)
                repeatability_probability_error = max(repeatability_probability_error, token_repeatability)
                case_repeatability = max(case_repeatability, token_repeatability)
        case_repeatability_probability_error[case_id] = case_repeatability

    max_probability_error = max(probability_errors, default=0.0)
    raw_candidate = max_probability_error + max(
        2 * repeatability_probability_error,
        0.1 * max_probability_error,
    )
    candidate_probability_error = round_up_one_significant_digit(raw_candidate)
    cases = {
        case_id: {
            "observation_count": len(case_observations),
            "bucket_max_tokens": sorted({observation.bucket_max_tokens for observation in case_observations}),
            "greedy_agreement": case_greedy_matches[case_id] / len(case_observations),
            "greedy_token_ids": sorted({observation.greedy_token_id for observation in case_observations}),
            "canonical_top_token_ranks": sorted(
                {
                    token.rank
                    for observation in case_observations
                    for token in observation.golden_tokens
                    if token.token_id == expected_by_id[case_id].top_logprobs[0].token_id and token.rank is not None
                }
            ),
            "top25_recall": _quantiles(case_top25_recalls[case_id]),
            "probability_error": _quantiles(case_probability_errors[case_id]),
            "max_probability_error": max(case_probability_errors[case_id], default=0.0),
            "repeatability_probability_error": case_repeatability_probability_error[case_id],
        }
        for case_id, case_observations in sorted(grouped.items())
    }
    bucket_contract_candidates = {}
    for bucket, errors in sorted(bucket_probability_errors.items()):
        bucket_repeatability = max(
            (
                case_repeatability_probability_error[case_id]
                for case_id, case_observations in grouped.items()
                if case_observations[0].bucket_max_tokens == bucket
            ),
            default=0.0,
        )
        bucket_max_error = max(errors, default=0.0)
        bucket_raw_candidate = bucket_max_error + max(
            2 * bucket_repeatability,
            0.1 * bucket_max_error,
        )
        bucket_candidate = round_up_one_significant_digit(bucket_raw_candidate)
        bucket_contract_candidates[str(bucket)] = {
            "max_probability_error": bucket_max_error,
            "repeatability_probability_error": bucket_repeatability,
            "candidate_probability_error": bucket_candidate,
            "candidate_headroom": bucket_candidate - bucket_max_error,
        }
    return {
        "report_count": len(reports),
        "observation_count": len(observations),
        "process_ids": sorted(report.provenance.process_id for report in reports),
        "parameter_digests": sorted({report.provenance.parameter_digest for report in reports}),
        "model_config_digests": sorted({report.provenance.model_config_digest for report in reports}),
        "runtime_fingerprint_digests": sorted(
            {
                report.provenance.runtime_fingerprint.digest()
                for report in reports
                if report.provenance.runtime_fingerprint is not None
            }
        ),
        "code_digests": sorted({report.provenance.code_digest for report in reports}),
        "bitwise_repeatable": bool(bitwise_repeatable),
        "exact_canonical_top25": bool(exact_canonical),
        "greedy_agreement": greedy_matches / len(observations),
        "top25_recall": _quantiles(top25_recalls),
        "probability_error": _quantiles(probability_errors),
        "max_probability_error": max_probability_error,
        "repeatability_probability_error": repeatability_probability_error,
        "candidate_probability_error": candidate_probability_error,
        "candidate_headroom": candidate_probability_error - max_probability_error,
        "top25_l1_error": _quantiles(top_l1_errors),
        "aligned_logprob_error": _quantiles(logprob_errors),
        "probability_error_by_bucket": {
            str(bucket): _quantiles(values) for bucket, values in sorted(bucket_probability_errors.items())
        },
        "contract_candidate_by_bucket": bucket_contract_candidates,
        "cases": cases,
        "worst_cases_by_probability_error": sorted(
            cases,
            key=lambda case_id: (-cases[case_id]["max_probability_error"], case_id),
        ),
        "missing_golden_tokens": missing_golden_tokens,
        "nonfinite_observations": sum(observation.has_nonfinite for observation in observations),
        "overflow_observations": sum(
            any(value != 0.0 for value in observation.capacity_overflow) for observation in observations
        ),
    }


def analyze_reports(
    goldens: tuple[RepresentativeGolden, ...],
    reports: tuple[ObservationReport, ...],
    *,
    require_complete: bool = True,
) -> dict:
    if not reports:
        raise ValueError("At least one report is required")
    expected_by_id = {golden.id: golden for golden in goldens}
    if len(expected_by_id) != len(goldens):
        raise ValueError("Golden case IDs must be unique")
    if require_complete:
        for report in reports:
            _validate_complete_report(report, set(expected_by_id))

    by_backend: dict[str, dict[str, list[ObservationReport]]] = defaultdict(lambda: defaultdict(list))
    for report in reports:
        by_backend[report.provenance.backend][report.provenance.platform].append(report)
        unknown = {observation.case_id for observation in report.observations} - expected_by_id.keys()
        if unknown:
            raise ValueError(f"Unknown case IDs in {report.provenance.process_id}: {sorted(unknown)}")

    backends = {}
    for backend, by_platform in sorted(by_backend.items()):
        platforms = {
            platform: _platform_summary(expected_by_id, tuple(platform_reports))
            for platform, platform_reports in sorted(by_platform.items())
        }
        parameter_digests = {digest for summary in platforms.values() for digest in summary["parameter_digests"]}
        config_digests = {digest for summary in platforms.values() for digest in summary["model_config_digests"]}
        backends[backend] = {
            "platforms": platforms,
            "cross_platform_parameters_match": len(parameter_digests) == 1,
            "cross_platform_configs_match": len(config_digests) == 1,
        }
    return {
        "golden_case_count": len(goldens),
        "backends": backends,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    reports = tuple(ObservationReport.from_json_bytes(StoragePath(path).read_bytes()) for path in args.reports)
    summary = analyze_reports(read_representative_goldens(), reports, require_complete=not args.allow_partial)
    payload = (json.dumps(summary, sort_keys=True, indent=2) + "\n").encode()
    if args.output:
        StoragePath(args.output).write_bytes(payload)
    print(payload.decode(), end="")


if __name__ == "__main__":
    main()
