# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Strict import of historical gap spellings into the reconciler vocabulary."""

from finestore.rl.difficulty import (
    LEGACY_DIFFICULTY_GAP_REASONS,
    QUALIFICATION_RECEIPT_PREFIX,
    DifficultyGap,
    difficulty_gaps,
)
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.glm import GLM_MODEL
from experiments.post_training.task_curation.difficulty.state import ArtifactKind, Gap, GapReason

# Capability details preserve the precise legacy cause under the admission gap.
LEGACY_GAPS = {
    DifficultyGap.AWAITING_SCORED_COVERAGE: GapReason.COVERAGE_SHORTFALL,
    DifficultyGap.PROTOCOL_INPUT_BUDGET_EXCEEDED: GapReason.PROTOCOL_INPUT_BUDGET_EXCEEDED,
    DifficultyGap.PROVIDER_CAPACITY_WAIT: GapReason.PROVIDER_CAPACITY_WAIT,
    DifficultyGap.DAYTONA_SNAPSHOT_PREFLIGHT_PENDING: GapReason.DAYTONA_SNAPSHOT_PREFLIGHT_PENDING,
    DifficultyGap.DAYTONA_TASK_SPEC_FACTORY_PENDING: GapReason.TASK_SPEC_FACTORY_PENDING,
    DifficultyGap.DAYTONA_RUNTIME_QUALIFICATION_PENDING: GapReason.EXECUTABILITY_ADMISSION_PENDING,
    DifficultyGap.PRODUCTION_HARBOR_TRIALS_PENDING: GapReason.COVERAGE_SHORTFALL,
    DifficultyGap.EXECUTABILITY_ADMISSION_PENDING: GapReason.EXECUTABILITY_ADMISSION_PENDING,
    DifficultyGap.FILTRATION_FINDINGS_ADMISSION_PENDING: GapReason.EXECUTABILITY_ADMISSION_PENDING,
    DifficultyGap.JUDGE_QUALIFICATION_PENDING: GapReason.JUDGE_ROUTE_UNQUALIFIED,
    DifficultyGap.NO_QUALIFIED_MEASUREMENT_RECORDED: GapReason.NO_MEASUREMENT_ROW,
    DifficultyGap.MATH_VERIFIER_DEADLINE: GapReason.EXECUTABILITY_ADMISSION_PENDING,
    DifficultyGap.HOSTED_NATIVE_CONTEXT_UNQUALIFIED: GapReason.EXECUTABILITY_ADMISSION_PENDING,
    DifficultyGap.HOSTED_TIMEOUT_GRADES_UNQUALIFIED: GapReason.COVERAGE_SHORTFALL,
    DifficultyGap.HOSTED_DISPATCH_INTERRUPTED: GapReason.COVERAGE_SHORTFALL,
    DifficultyGap.HOSTED_JUDGE_CANARY_INTERRUPTED: GapReason.JUDGE_ROUTE_UNQUALIFIED,
    DifficultyGap.HOSTED_UNSERVED_RAW_ZERO_GRADES: GapReason.COVERAGE_SHORTFALL,
    DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING: GapReason.ENVIRONMENT_ARTIFACT_MISSING,
    DifficultyGap.UNSUPPORTED_COMPOSE_DOCUMENT: GapReason.UNSUPPORTED_COMPOSE_DOCUMENT,
    DifficultyGap.NETWORK_POLICY_VIOLATION: GapReason.NETWORK_POLICY_VIOLATION,
    DifficultyGap.NO_EXECUTABLE_GRADER: GapReason.EXECUTABILITY_ADMISSION_PENDING,
    DifficultyGap.STATE_CAPTURE_REQUIRED: GapReason.EXECUTABILITY_ADMISSION_PENDING,
    DifficultyGap.TOOL_PROVIDER_REQUIRED: GapReason.EXECUTABILITY_ADMISSION_PENDING,
    DifficultyGap.SHARED_EXECUTOR_REQUIRED: GapReason.EXECUTABILITY_ADMISSION_PENDING,
    DifficultyGap.PROVIDER_CONTRACT_INVALID: GapReason.EXECUTABILITY_ADMISSION_PENDING,
    DifficultyGap.UNRESOLVED_DOCKER_BUILD: GapReason.ENVIRONMENT_ARTIFACT_MISSING,
    DifficultyGap.JUDGE_ROUTE_UNQUALIFIED: GapReason.JUDGE_ROUTE_UNQUALIFIED,
    DifficultyGap.JUDGE_API_UNQUALIFIED: GapReason.JUDGE_ROUTE_UNQUALIFIED,
    DifficultyGap.INVALID_REFERENCE: GapReason.EXECUTABILITY_ADMISSION_PENDING,
}


class LegacyReasonError(ValueError):
    """The complete audit found a reason outside the closed vocabulary."""


RAW_GAPS = {
    "Awaiting shared model provider capacity": GapReason.PROVIDER_CAPACITY_WAIT,
    "provider_wait": GapReason.PROVIDER_CAPACITY_WAIT,
    "input_budget_unserved": GapReason.PROTOCOL_INPUT_BUDGET_EXCEEDED,
}


def legacy_gaps(reason: str | None) -> tuple[Gap, ...]:
    """Map every segment, refusing unknown strings even in mixed legacy reasons."""
    if reason is None:
        return ()
    for spelling, legacy in LEGACY_DIFFICULTY_GAP_REASONS.items():
        if "; " in spelling:
            reason = reason.replace(spelling, legacy.value)
    segments = reason.split("; ")
    if len(segments) > 1:
        return ordered_gaps(
            [
                gap
                for segment in segments
                if not segment.startswith(QUALIFICATION_RECEIPT_PREFIX)
                for gap in legacy_gaps(segment)
            ]
        )
    if reason in RAW_GAPS:
        return (Gap(RAW_GAPS[reason]),)
    prefix = "production_judge_route_or_temperature_not_qualified:"
    if reason.startswith(prefix):
        model, separator, temperature = reason[len(prefix) :].rpartition(":temperature=")
        if not separator:
            raise LegacyReasonError(f"Unmapped judge qualification reason: {reason!r}")
        return (
            Gap(GapReason.JUDGE_ROUTE_UNQUALIFIED, judge_model=model or GLM_MODEL, judge_temperature=float(temperature)),
        )
    try:
        parsed = difficulty_gaps(reason)
    except ValueError as error:
        raise LegacyReasonError(str(error)) from error
    gaps = []
    for legacy, detail in parsed:
        canonical = LEGACY_GAPS[legacy]
        artifact_uri = detail if canonical == GapReason.ENVIRONMENT_ARTIFACT_MISSING else None
        kind = None
        if artifact_uri == "prepared_image":
            artifact_uri, kind = None, ArtifactKind.PREPARED_IMAGE
        elif artifact_uri and artifact_uri.endswith("requirements.lock"):
            kind = ArtifactKind.REQUIREMENTS_LOCK
            if artifact_uri.startswith("/"):
                suffix = "env-278169f50750da36/2026.10.08/requirements.lock"
                if not artifact_uri.endswith(suffix):
                    raise ValueError("Local environment artifact has no regional binding")
                artifact_uri = str(StoragePath("s3://marin-us-east-02a/marin/images") / suffix)
        gaps.append(
            Gap(
                canonical,
                artifact_uri=artifact_uri,
                artifact_kind=kind,
                judge_model=detail if canonical == GapReason.JUDGE_ROUTE_UNQUALIFIED else None,
                capability=(
                    f"{legacy.value}:{detail or ''}" if canonical == GapReason.EXECUTABILITY_ADMISSION_PENDING else None
                ),
            )
        )
    if not gaps:
        raise ValueError(f"Legacy reason has no typed gap: {reason!r}")
    return ordered_gaps(gaps)


def ordered_gaps(gaps: list[Gap]) -> tuple[Gap, ...]:
    priority = {
        GapReason.NETWORK_POLICY_VIOLATION: 0,
        GapReason.PROTOCOL_INPUT_BUDGET_EXCEEDED: 1,
        GapReason.PROVIDER_CAPACITY_WAIT: 3,
        GapReason.COVERAGE_SHORTFALL: 4,
    }
    return tuple(sorted(set(gaps), key=lambda gap: (priority.get(gap.reason, 2), gap.reason, gap.artifact_uri or "")))
