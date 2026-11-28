# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The command's admission boundary accepts only an existing pinned exclusion."""

from pathlib import Path

import pytest
from finestore.rl.status_taxonomy import ExclusionReason, SourceStatus

from experiments.post_training.task_curation.ingest_stages.eligibility import eligible_source, require_starting_state
from experiments.post_training.task_curation.ingest_stages.ingest_config import IngestMode, load_ingest_config

CONFIGS = Path(__file__).parents[1] / "ingest_stages/configs"


@pytest.fixture
def excluded():
    policy = load_ingest_config(CONFIGS / "litmus-train.json")
    declaration = {
        "id": policy.source_id,
        "status": SourceStatus.EXCLUDED.value,
        "dataset_id": policy.upstream.repository,
        "dataset_revision": policy.upstream.revision,
        "exclusion_claims": [
            {
                "reason": ExclusionReason.GRADER_DEFECT.value,
                "evidence": {"kind": "note", "message": "Pinned grader needs repair."},
            }
        ],
    }
    return policy, declaration


def test_existing_exclusion_preserves_the_starting_pin_and_reasons(excluded):
    policy, declaration = excluded
    result = eligible_source(policy, {policy.source_id: declaration})
    assert result.dataset_revision == declaration["dataset_revision"]
    assert [claim.model_dump(mode="json", exclude_none=True) for claim in result.exclusion_claims] == declaration[
        "exclusion_claims"
    ]
    changed = {**declaration, "notes": "An independent declaration change"}
    assert eligible_source(policy, {policy.source_id: changed}).declaration_sha256 != result.declaration_sha256


@pytest.mark.parametrize("status", [SourceStatus.AVAILABLE, SourceStatus.PENDING, SourceStatus.DEPRECATED])
def test_nonexcluded_sources_cannot_enter_even_with_supplied_reasons(excluded, status):
    policy, declaration = excluded
    with pytest.raises(ValueError, match="requires an Excluded"):
        eligible_source(policy, {policy.source_id: {**declaration, "status": status}})


@pytest.mark.parametrize(
    "change",
    [
        {"dataset_revision": None},
        {"dataset_revision": "main"},
        {"dataset_revision": "b" * 40},
        {"dataset_id": "other/dataset"},
        {"exclusion_claims": []},
    ],
)
def test_an_exclusion_without_a_matching_immutable_pin_and_reasons_cannot_enter(excluded, change):
    policy, declaration = excluded
    with pytest.raises(ValueError):
        eligible_source(policy, {policy.source_id: {**declaration, **change}})


def test_a_config_cannot_register_its_own_source(excluded):
    policy, _ = excluded
    with pytest.raises(ValueError, match="already declared"):
        eligible_source(policy, {})


def test_ingest_cannot_be_used_to_bypass_repair_eligibility(excluded):
    policy, declaration = excluded
    assert require_starting_state(policy, {}) is None
    with pytest.raises(ValueError, match="requires an unknown source"):
        require_starting_state(policy, {policy.source_id: declaration})
    repair = policy.model_copy(update={"mode": IngestMode.REPAIR})
    assert require_starting_state(repair, {policy.source_id: declaration}).source_id == policy.source_id
    with pytest.raises(ValueError, match="already declared"):
        require_starting_state(repair, {})
