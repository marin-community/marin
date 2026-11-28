# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical projections of immutable survey identities across historical IDs."""

from collections.abc import Callable
from copy import deepcopy

from finestore.rl.source_identity import canonical_source_id, canonical_source_map, original_source_id
from taskcompendium.source_fingerprints import SourceIdentitySurvey


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
    document = survey.model_dump(mode="json")
    for review in document["reviews"]:
        if review["measurement"] is None:
            del review["measurement"]
    return _identity_source_ids(document, original_source_id)
