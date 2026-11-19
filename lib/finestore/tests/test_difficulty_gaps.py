# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Difficulty gaps are typed: workers emit enum members and typed evidence only."""

import pytest
from finestore.rl.difficulty import (
    DifficultyAttempt,
    DifficultyGap,
    DifficultyMeasurement,
    DifficultyObservation,
    MeasurementStatus,
    difficulty_gap_evidence,
    difficulty_gap_reason,
    difficulty_gaps,
    join_difficulty_gap_reasons,
    require_difficulty_gaps,
)

# Every not-measured reason spelling observed in served measurements and
# campaign catalogs (2026-11-16, atlas-preview-20261116-r20-v1 and the
# 2026.11.14.47 qualified snapshot), normalized to one spelling per form.
LEGACY_REASONS = {
    "Awaiting scored coverage": {DifficultyGap.AWAITING_SCORED_COVERAGE},
    "Selected tasks still lack scored solver attempts.": {DifficultyGap.AWAITING_SCORED_COVERAGE},
    "A candidate prompt exceeded the protocol input budget before dispatch. No solver score was assigned.": {
        DifficultyGap.PROTOCOL_INPUT_BUDGET_EXCEEDED
    },
    "Awaiting shared model provider capacity; provider waits are excluded from task attempts.": {
        DifficultyGap.PROVIDER_CAPACITY_WAIT
    },
    "daytona_snapshot_preflight_pending": {DifficultyGap.DAYTONA_SNAPSHOT_PREFLIGHT_PENDING},
    "faithful_daytona_task_spec_factory_pending": {DifficultyGap.DAYTONA_TASK_SPEC_FACTORY_PENDING},
    "Faithful Daytona TaskSpec actor factory pending": {DifficultyGap.DAYTONA_TASK_SPEC_FACTORY_PENDING},
    "Daytona snapshot and original runtime qualification pending": {DifficultyGap.DAYTONA_RUNTIME_QUALIFICATION_PENDING},
    "Awaiting qualified production Harbor trials with the original source runtime packet": {
        DifficultyGap.PRODUCTION_HARBOR_TRIALS_PENDING
    },
    "No qualified Harbor measurement has been recorded for this source/model in this snapshot.": {
        DifficultyGap.NO_QUALIFIED_MEASUREMENT_RECORDED
    },
    "math_verifier_deadline": {DifficultyGap.MATH_VERIFIER_DEADLINE},
    "native_client_effective_context_unqualified": {DifficultyGap.HOSTED_NATIVE_CONTEXT_UNQUALIFIED},
    "prior_hosted_timeout_grades_unqualified": {DifficultyGap.HOSTED_TIMEOUT_GRADES_UNQUALIFIED},
    "prior_hosted_timeout_grade_unqualified": {DifficultyGap.HOSTED_TIMEOUT_GRADES_UNQUALIFIED},
    "interrupted_hosted_dispatch_unsealed": {DifficultyGap.HOSTED_DISPATCH_INTERRUPTED},
    "interrupted_hosted_judge_canary_unsealed": {DifficultyGap.HOSTED_JUDGE_CANARY_INTERRUPTED},
    "prior_hosted_unserved_requests_with_raw_zero_grades": {DifficultyGap.HOSTED_UNSERVED_RAW_ZERO_GRADES},
    "unsupported_compose_document": {DifficultyGap.UNSUPPORTED_COMPOSE_DOCUMENT},
    "network_policy_violation": {DifficultyGap.NETWORK_POLICY_VIOLATION},
    "missing_environment_artifact:prepared_image": {DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING},
    "no_executable_task_spec_grader": {DifficultyGap.NO_EXECUTABLE_GRADER},
    "task_spec_state_capture_required": {DifficultyGap.STATE_CAPTURE_REQUIRED},
    "task_spec_tool_provider_required": {DifficultyGap.TOOL_PROVIDER_REQUIRED},
    "task_spec_shared_executor_required": {DifficultyGap.SHARED_EXECUTOR_REQUIRED},
    "task_spec_provider_contract_invalid": {DifficultyGap.PROVIDER_CONTRACT_INVALID},
    "task_spec_unresolved_build": {DifficultyGap.UNRESOLVED_DOCKER_BUILD},
    "invalid_expected_target_in_frozen_sample": {DifficultyGap.INVALID_REFERENCE},
}


@pytest.mark.parametrize("reason", sorted(LEGACY_REASONS))
def test_every_historical_reason_maps_to_typed_gaps(reason):
    gaps = {gap for gap, _ in difficulty_gaps(reason)}
    assert gaps == LEGACY_REASONS[reason]


@pytest.mark.parametrize(
    "reason,expected",
    [
        (
            "awaiting_executability_filtered_admission_population:task_spec_tool_provider_required",
            [(DifficultyGap.EXECUTABILITY_ADMISSION_PENDING, "task_spec_tool_provider_required")],
        ),
        (
            "Awaiting canonical executable admitted population after sampled filtration findings:"
            " task_spec_shared_executor_required",
            [(DifficultyGap.FILTRATION_FINDINGS_ADMISSION_PENDING, "task_spec_shared_executor_required")],
        ),
        (
            "production_judge_route_not_qualified:MiniMaxAI/MiniMax-M3-MXFP8",
            [(DifficultyGap.JUDGE_ROUTE_UNQUALIFIED, "MiniMaxAI/MiniMax-M3-MXFP8")],
        ),
        (
            "production_judge_api_not_qualified:responses",
            [(DifficultyGap.JUDGE_API_UNQUALIFIED, "responses")],
        ),
        (
            "missing_environment_artifact: s3://marin-us-east-02a/marin/images/env-4a5dde6cb8723181"
            "/2026.10.08/requirements.lock",
            [
                (
                    DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING,
                    "s3://marin-us-east-02a/marin/images" "/env-4a5dde6cb8723181/2026.10.08/requirements.lock",
                )
            ],
        ),
        (
            "Production judge route or temperature qualification pending: glm-5.3 at temperature 0.0",
            [(DifficultyGap.JUDGE_QUALIFICATION_PENDING, "glm-5.3 at temperature 0.0")],
        ),
    ],
)
def test_parameterized_legacy_reasons_keep_their_typed_detail(reason, expected):
    assert difficulty_gaps(reason) == expected


def test_legacy_missing_separator_is_still_typed():
    reason = (
        "missing_environment_artifact: s3://marin-us-east-02a/marin/images/env-4a5dde6cb8723181"
        "/2026.10.08/requirements.lock Production judge route or temperature qualification pending:"
        "  at temperature 0.0"
    )
    assert difficulty_gaps(reason) == [
        (
            DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING,
            "s3://marin-us-east-02a/marin/images" "/env-4a5dde6cb8723181/2026.10.08/requirements.lock",
        ),
        (DifficultyGap.JUDGE_QUALIFICATION_PENDING, "at temperature 0.0"),
    ]


def test_compound_reason_carries_every_gap_and_the_evidence_uri():
    reason = (
        "Faithful Daytona TaskSpec actor factory pending; missing_environment_artifact:"
        " s3://marin-us-east-02a/marin/images/env-3c8d2364e64506b0/2026.10.08/requirements.lock"
        "; qualification receipt: s3://marin-rno2a/marin/rl-data/atlas/difficulty/receipts.json"
    )
    gaps = {gap for gap, _ in difficulty_gaps(reason)}
    assert gaps == {
        DifficultyGap.DAYTONA_TASK_SPEC_FACTORY_PENDING,
        DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING,
    }
    assert difficulty_gap_evidence(reason) == "s3://marin-rno2a/marin/rl-data/atlas/difficulty/receipts.json"


def test_canonical_reasons_round_trip_through_the_parser():
    for member in DifficultyGap:
        reason = difficulty_gap_reason(member)
        assert difficulty_gaps(reason) == [(member, None)]
        require_difficulty_gaps(reason)


@pytest.mark.parametrize(
    "reason",
    [
        "the Daytona snapshot is missing",
        "qualification receipt: s3://marin-rno2a/marin/receipt.json",
    ],
)
def test_unmapped_reasons_are_rejected(reason):
    with pytest.raises(ValueError):
        require_difficulty_gaps(reason)


def _measurement(reason: str | None, status: MeasurementStatus) -> DifficultyMeasurement:
    payload = {
        "measurement_id": "fixture-measurement",
        "protocol_id": "atlas-difficulty-v3",
        "created_at": "2026-11-16T00:00:00+00:00",
        "source_id": "Fixture:source",
        "release": {"repo": "fixture/repo", "revision": "r1", "url": "https://example.invalid/r1"},
        "converter_version": "1",
        "solver": "small",
        "status": status,
        "reason": reason,
        "sample": None,
        "sample_receipt": None,
        "model": None,
        "settings": None,
        "n": 0,
        "solved": 0,
        "wilson_95": None,
        "infrastructure_attempts": 0,
        "pending_tasks": 0,
        "blocked_tasks": 0,
        "attempts_receipt": None,
        "gpu_hours": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    return DifficultyMeasurement.model_validate(payload)


def test_not_measured_records_carry_only_typed_gaps():
    reason = difficulty_gap_reason(DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING, "s3://marin/images/env-1/lock")
    measurement = _measurement(reason, MeasurementStatus.NOT_MEASURED)
    assert {gap for gap, _ in difficulty_gaps(measurement.reason)} == {DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING}
    # A historical record keeps its original spelling and stays typed.
    legacy = _measurement(
        "missing_environment_artifact: s3://marin-us-east-02a/marin/images/env-1/2026.10.08/requirements.lock",
        MeasurementStatus.NOT_MEASURED,
    )
    assert {gap for gap, _ in difficulty_gaps(legacy.reason)} == {DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING}


def test_composed_reasons_never_drift_from_the_parser():
    reason = join_difficulty_gap_reasons(
        [
            difficulty_gap_reason(DifficultyGap.DAYTONA_TASK_SPEC_FACTORY_PENDING),
            difficulty_gap_reason(DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING, "s3://lock"),
        ]
    )
    require_difficulty_gaps(reason)
    assert difficulty_gaps(reason) == [
        (DifficultyGap.DAYTONA_TASK_SPEC_FACTORY_PENDING, None),
        (DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING, "s3://lock"),
    ]


def test_free_text_after_a_gap_fails_validation():
    with pytest.raises(ValueError, match="Unmapped difficulty gap reason"):
        require_difficulty_gaps("invalid_reference:the snapshot is broken; whatever")


def test_not_measured_observation_rejects_untyped_reasons():
    release = {
        "repo": "open-athena/marin-eval-data",
        "revision": "a" * 40,
        "url": "https://huggingface.co/datasets/open-athena/marin-eval-data/tree/" + "a" * 40,
    }
    base = {
        "measurement_id": "m-1",
        "protocol_id": "atlas-difficulty-v3",
        "created_at": "2026-11-19T12:00:00+00:00",
        "source_id": "nemotron_sft_v3/math_v2/medium",
        "release": release,
        "converter_version": "v1",
        "solver": "small",
        "status": "not_measured",
        "reason": None,
        "sample": None,
        "sample_receipt": None,
        "model": None,
        "settings": None,
        "wilson_95": None,
        "attempts_receipt": None,
        "n": 0,
        "solved": 0,
        "infrastructure_attempts": 0,
        "pending_tasks": 0,
        "blocked_tasks": 0,
        "gpu_hours": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    # The observation constructor must apply the same typed-gap contract as the
    # parser: arbitrary prose and evidence-only reasons are rejected on build.
    prose = {**base, "reason": "the snapshot is broken beyond repair"}
    with pytest.raises(ValueError, match="Unmapped difficulty gap reason"):
        DifficultyObservation.model_validate(prose)
    evidence_only = {**base, "reason": "qualification receipt: jobs/oracle/result.json"}
    with pytest.raises(ValueError, match="Unmapped difficulty gap reason"):
        DifficultyObservation.model_validate(evidence_only)
    typed = {**base, "reason": DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING.value}
    assert DifficultyObservation.model_validate(typed).reason == DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING.value


def test_blocked_attempt_rejects_untyped_reasons():
    with pytest.raises(ValueError, match="Unmapped difficulty gap reason"):
        DifficultyAttempt.model_validate(
            {"task_id": "task-1", "status": "blocked", "reward": None, "reason": "gave up for unclear reasons"}
        )
    typed = DifficultyAttempt.model_validate(
        {
            "task_id": "task-1",
            "status": "blocked",
            "reward": None,
            "reason": difficulty_gap_reason(DifficultyGap.INVALID_REFERENCE, "no expected target"),
        }
    )
    assert typed.reason == "invalid_reference:no expected target"
