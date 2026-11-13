# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Atlas projections of opaque catalog IDs and immutable survey identities."""

from collections.abc import Callable
from copy import deepcopy

from finestore.rl.curation import Assessment, AssessmentAttempt, SnapshotSource, SourceDeclaration, source_identifier
from taskcompendium.backend_requirements import BackendSurvey
from taskcompendium.source_fingerprints import SourceIdentitySurvey

from experiments.post_training.task_curation.source_identity import (
    canonical_source_id,
    canonical_source_map,
    original_source_id,
)


def canonical_declaration(declaration: SourceDeclaration) -> SourceDeclaration:
    identifier = canonical_source_id(declaration.id)
    if identifier != source_identifier(
        declaration.origin, declaration.upstream_dataset, declaration.subset, declaration.variant
    ):
        raise ValueError("Source ID differs from generated identity")
    return declaration.model_copy(update={"id": identifier})


def canonical_snapshot_source(source: SnapshotSource) -> SnapshotSource:
    return source.model_copy(
        update={"id": canonical_source_id(source.id), "declaration": canonical_declaration(source.declaration)}
    )


def canonical_assessment(assessment: Assessment) -> Assessment:
    return assessment.model_copy(update={"source_id": canonical_source_id(assessment.source_id)})


def canonical_attempt(attempt: AssessmentAttempt) -> AssessmentAttempt:
    if attempt.source_id is None:
        return attempt
    return attempt.model_copy(update={"source_id": canonical_source_id(attempt.source_id)})


def _identity_source_ids(document: dict, source_id: Callable[[str], str]) -> dict:
    document["fingerprints"] = {source_id(key): value for key, value in document["fingerprints"].items()}
    for review in document["reviews"]:
        similarity = review["similarity"]
        for side in ("left", "right"):
            similarity[side] = source_id(similarity[side])
    for group in document["groups"]:
        group["canonical_source_id"] = source_id(group["canonical_source_id"])
        for member in group["members"]:
            member["source_id"] = source_id(member["source_id"])
    return document


def canonical_identity_survey(document: dict) -> SourceIdentitySurvey:
    document = deepcopy(document)
    document["fingerprints"] = canonical_source_map(document["fingerprints"])
    return SourceIdentitySurvey.model_validate(_identity_source_ids(document, canonical_source_id))


def identity_survey_payload(survey: SourceIdentitySurvey) -> dict:
    """Keep persisted survey and review-cache identities tied to original source IDs."""
    return _identity_source_ids(survey.model_dump(mode="json"), original_source_id)


def canonical_backend_survey(document: dict) -> BackendSurvey:
    return BackendSurvey.model_validate(
        {
            **document,
            "sources": [{**source, "source_id": canonical_source_id(source["source_id"])} for source in document["sources"]],
        }
    )


def backend_survey_payload(survey: BackendSurvey) -> dict:
    """Preserve the published backend-survey identity through Atlas projection."""
    document = survey.model_dump(mode="json")
    for source in document["sources"]:
        source["source_id"] = original_source_id(source["source_id"])
    return document
