# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Difficulty gaps are typed: workers emit enum members and typed evidence only."""

import pytest
from finestore.rl.difficulty import (
    DIFFICULTY_GAPS,
    DifficultyGap,
    DifficultyGapOwner,
    DifficultyMeasurement,
    MeasurementStatus,
    difficulty_gap_evidence,
    difficulty_gap_reason,
    difficulty_gaps,
    join_difficulty_gap_reasons,
    require_difficulty_gaps,
)
from finestore.rl.status_taxonomy import DIFFICULTY_GAP_EXCLUSION_REASONS

# Historical reason spellings that resolve through parsing rather than the
# exact-match table: the measured artifact path rides the environment-artifact
# prefix. The complete observed-spelling record lives in the module's
# LEGACY_DIFFICULTY_GAP_REASONS table.
LEGACY_REASONS = {
    "missing_environment_artifact:prepared_image": {DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING},
}


def test_every_gap_member_declares_its_owner_meaning_and_clear_condition():
    assert {member.value for member in DifficultyGap} == set(DIFFICULTY_GAPS)
    # PIPELINE-owned gaps never exclude a source; every mapped gap is SOURCE-owned.
    for member_name in DIFFICULTY_GAP_EXCLUSION_REASONS:
        gap = getattr(DifficultyGap, member_name)
        assert DIFFICULTY_GAPS[gap].owner == DifficultyGapOwner.SOURCE


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
