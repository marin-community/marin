# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical projections of immutable survey identities across historical IDs."""

from finestore.rl.source_identity import canonical_source_id, original_source_id
from taskcompendium.source_fingerprints import SourceIdentitySurvey


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


