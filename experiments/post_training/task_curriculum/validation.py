# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-artifact validation for one curriculum generation run."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from experiments.post_training.task_curriculum.models import (
    BlindFitReview,
    BlindFitStatus,
    BlindTaskSet,
    Curriculum,
    EvidenceAccounting,
    EvidenceStatus,
    GapDispositionStatus,
    HolisticReview,
    HolisticReviewStatus,
    SystematicGapDisposition,
)

MIN_BLIND_TASK_COUNT = 24
MIN_BLIND_TASKS_PER_GUIDEPOST = 2


@dataclass(frozen=True)
class SubjectRunArtifacts:
    """Role outputs and operator dispositions retained for one subject version."""

    curriculum: Curriculum
    blind_tasks: BlindTaskSet
    fit_review: BlindFitReview
    holistic_review: HolisticReview
    gap_dispositions: Sequence[SystematicGapDisposition]


@dataclass(frozen=True)
class SubjectEvidenceIds:
    """Expected inventory and evidence identifiers for one subject run."""

    guideposts: frozenset[str]
    discovery_items: frozenset[str]
    evaluation_items: frozenset[str]


def _accounting_ids(rows: list[EvidenceAccounting], label: str) -> set[str]:
    item_ids = [row.item_id for row in rows]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError(f"{label} accounting item IDs must be unique")
    return set(item_ids)


def _validate_identities(artifacts: SubjectRunArtifacts) -> None:
    curriculum = artifacts.curriculum
    subject_ids = {
        curriculum.subject_id,
        artifacts.blind_tasks.subject_id,
        artifacts.fit_review.subject_id,
        artifacts.holistic_review.subject_id,
    }
    if len(subject_ids) != 1:
        raise ValueError(f"subject IDs do not match: {sorted(subject_ids)}")
    if artifacts.holistic_review.curriculum_version != curriculum.version:
        raise ValueError("holistic review curriculum version does not match")
    if artifacts.fit_review.curriculum_version != curriculum.version:
        raise ValueError("fit review curriculum version does not match")
    prefix = f"{curriculum.subject_id.lower()}."
    invalid_section_ids = [section.id for section in curriculum.sections if not section.id.startswith(prefix)]
    if invalid_section_ids:
        raise ValueError(f"section IDs do not start with {prefix}: {invalid_section_ids}")


def _validate_blind_sample(blind_tasks: BlindTaskSet, guidepost_ids: set[str]) -> None:
    expected_task_count = max(MIN_BLIND_TASK_COUNT, MIN_BLIND_TASKS_PER_GUIDEPOST * len(guidepost_ids))
    if len(blind_tasks.tasks) != expected_task_count:
        raise ValueError(f"expected {expected_task_count} blind tasks, found {len(blind_tasks.tasks)}")
    guidepost_counts = Counter(guidepost for task in blind_tasks.tasks for guidepost in set(task.guidepost_basis))
    unknown_guideposts = set(guidepost_counts) - guidepost_ids
    if unknown_guideposts:
        raise ValueError(f"blind tasks name unknown guideposts: {sorted(unknown_guideposts)}")
    underrepresented = sorted(
        guidepost for guidepost in guidepost_ids if guidepost_counts[guidepost] < MIN_BLIND_TASKS_PER_GUIDEPOST
    )
    if underrepresented:
        raise ValueError(f"guideposts need at least two blind tasks: {underrepresented}")


def _validate_fit_references(artifacts: SubjectRunArtifacts) -> None:
    disposition_gaps = [disposition.gap for disposition in artifacts.gap_dispositions]
    if len(disposition_gaps) != len(set(disposition_gaps)):
        raise ValueError("systematic gap dispositions must be unique")
    if set(disposition_gaps) != set(artifacts.fit_review.systematic_gaps):
        raise ValueError("systematic gap dispositions do not match the fit review")

    task_ids = {task.id for task in artifacts.blind_tasks.tasks}
    fit_task_ids = {judgment.task_id for judgment in artifacts.fit_review.judgments}
    if fit_task_ids != task_ids:
        raise ValueError("fit judgment task IDs do not match the frozen blind task set")

    capability_ids = {section.id for section in artifacts.curriculum.capability_sections()}
    fit_capability_ids = {
        section_id for judgment in artifacts.fit_review.judgments for section_id in judgment.acceptable_capability_ids
    }
    unknown_fit_capabilities = fit_capability_ids - capability_ids
    if unknown_fit_capabilities:
        raise ValueError(f"fit judgments name unknown or group sections: {sorted(unknown_fit_capabilities)}")


def _validate_guidepost_accounting(
    holistic_review: HolisticReview,
    guidepost_ids: set[str],
    capability_ids: set[str],
) -> None:
    guidepost_accounting_ids = [row.guidepost_id for row in holistic_review.guidepost_accounting]
    if len(guidepost_accounting_ids) != len(set(guidepost_accounting_ids)):
        raise ValueError("guidepost accounting IDs must be unique")
    if set(guidepost_accounting_ids) != guidepost_ids:
        raise ValueError("guidepost accounting does not match the subject inventory")
    accounting_capability_ids = {
        section_id for row in holistic_review.guidepost_accounting for section_id in row.section_ids
    }
    unknown_accounting_capabilities = accounting_capability_ids - capability_ids
    if unknown_accounting_capabilities:
        raise ValueError(
            f"guidepost accounting names unknown or group sections: {sorted(unknown_accounting_capabilities)}"
        )


def _validate_evidence_accounting(
    holistic_review: HolisticReview,
    discovery_item_ids: set[str],
    evaluation_item_ids: set[str],
    capability_ids: set[str],
) -> None:
    for label, rows, expected_ids in (
        ("discovery", holistic_review.discovery_accounting, discovery_item_ids),
        ("evaluation", holistic_review.evaluation_accounting, evaluation_item_ids),
    ):
        if _accounting_ids(rows, label) != expected_ids:
            raise ValueError(f"{label} accounting does not match the evidence manifest")
        for row in rows:
            unknown_sections = set(row.section_ids) - capability_ids
            if unknown_sections:
                raise ValueError(f"{label} accounting names unknown or group sections: {sorted(unknown_sections)}")
            if row.status == EvidenceStatus.SUPPORT and not row.section_ids:
                raise ValueError(f"supported {label} item {row.item_id} must name a capability")
            if row.status != EvidenceStatus.SUPPORT and row.section_ids:
                raise ValueError(f"non-supporting {label} item {row.item_id} cannot name capabilities")


def _validate_risk_references(curriculum: Curriculum, holistic_review: HolisticReview) -> None:
    known_section_ids = {section.id for section in curriculum.sections}
    unknown_risks = set(holistic_review.highest_risk_sections) - known_section_ids
    if unknown_risks:
        raise ValueError(f"highest-risk list names unknown sections: {sorted(unknown_risks)}")


def _validate_review_accounting(
    curriculum: Curriculum,
    holistic_review: HolisticReview,
    guidepost_ids: set[str],
    discovery_item_ids: set[str],
    evaluation_item_ids: set[str],
) -> None:
    capability_ids = {section.id for section in curriculum.capability_sections()}
    _validate_guidepost_accounting(holistic_review, guidepost_ids, capability_ids)
    _validate_evidence_accounting(
        holistic_review,
        discovery_item_ids,
        evaluation_item_ids,
        capability_ids,
    )
    _validate_risk_references(curriculum, holistic_review)


def validate_subject_run(
    artifacts: SubjectRunArtifacts,
    evidence_ids: SubjectEvidenceIds,
) -> None:
    """Validate references and accounting across one subject's artifacts."""
    _validate_identities(artifacts)
    _validate_blind_sample(artifacts.blind_tasks, set(evidence_ids.guideposts))
    _validate_fit_references(artifacts)
    _validate_review_accounting(
        artifacts.curriculum,
        artifacts.holistic_review,
        set(evidence_ids.guideposts),
        set(evidence_ids.discovery_items),
        set(evidence_ids.evaluation_items),
    )


def validate_subject_promotion(artifacts: SubjectRunArtifacts) -> None:
    """Validate score and blind-gap gates before promoting a subject."""
    if artifacts.holistic_review.status != HolisticReviewStatus.PILOT_READY:
        raise ValueError("holistic review does not pass the promotion gate")

    blocking_gaps = [
        disposition.gap
        for disposition in artifacts.gap_dispositions
        if disposition.status == GapDispositionStatus.BLOCKING
    ]
    if blocking_gaps:
        raise ValueError(f"blocking systematic gaps prevent promotion: {blocking_gaps}")

    gap_task_ids = {
        judgment.task_id for judgment in artifacts.fit_review.judgments if judgment.status == BlindFitStatus.GAP
    }
    uncovered_guideposts = {
        row.guidepost_id for row in artifacts.holistic_review.guidepost_accounting if not row.section_ids
    }
    uncovered_guidepost_gaps = [
        task.id
        for task in artifacts.blind_tasks.tasks
        if task.id in gap_task_ids and uncovered_guideposts.intersection(task.guidepost_basis)
    ]
    if uncovered_guidepost_gaps:
        raise ValueError(f"blind-task gaps expose uncovered guideposts: {uncovered_guidepost_gaps}")
