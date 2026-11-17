# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Difficulty reason parsing, including known legacy spellings."""

import pytest
from finestore.rl.difficulty import (
    DifficultyGap,
    difficulty_gap_evidence,
    difficulty_gap_reason,
    difficulty_gaps,
    join_difficulty_gap_reasons,
    require_difficulty_gaps,
)


@pytest.mark.parametrize(
    "reason,expected",
    [
        (
            "missing_environment_artifact:prepared_image",
            [(DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING, "prepared_image")],
        ),
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
    # The canonical join sorts and deduplicates its segments; the parser reads
    # them back in that order, and the detail can no longer swallow the segment
    # that follows it.
    reversed_reason = join_difficulty_gap_reasons(
        [
            difficulty_gap_reason(DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING, "s3://lock"),
            difficulty_gap_reason(DifficultyGap.DAYTONA_TASK_SPEC_FACTORY_PENDING),
        ]
    )
    require_difficulty_gaps(reversed_reason)
    assert difficulty_gaps(reversed_reason) == [
        (DifficultyGap.DAYTONA_TASK_SPEC_FACTORY_PENDING, None),
        (DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING, "s3://lock"),
    ]


def test_free_text_after_a_gap_fails_validation():
    with pytest.raises(ValueError, match="Unmapped difficulty gap reason"):
        require_difficulty_gaps("invalid_reference:the snapshot is broken; whatever")
