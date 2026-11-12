# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source identity ignores row order and keeps overlap flags advisory."""

import json
from pathlib import Path

from finestore.rl.source_identity import canonical_source_id
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.source_fingerprints import DuplicateLabel, SourceFingerprint, SourceIdentitySurvey

from experiments.post_training.task_curation.duplicate_review import DUPLICATE_MODEL, review_pairs
from experiments.post_training.task_curation.judges import HostedJudgeEndpoint
from experiments.post_training.task_curation.source_duplicate_ci import changed_source_declarations
from experiments.post_training.task_curation.source_fingerprints import (
    candidate_pairs,
    duplicate_groups,
    fingerprint_source,
)
from experiments.post_training.task_curation.tests.review_server import JudgeReply


def task_document(identifier: str, instruction: str) -> dict:
    task = TaskSpec(
        id=identifier,
        source=Source(dataset="fixture", revision="pin", row=identifier, importer_revision="v1"),
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="Fixture has no executable reference", contract={}),
    )
    return task.model_dump(mode="json")


def test_source_fingerprint_census_is_order_independent_and_counts_duplicate_contracts():
    rows = [task_document(str(index), f"Compute the square of {index}.") for index in range(30)]
    rows.append(task_document("copied-row", "Compute the square of 2."))
    first = fingerprint_source(rows, {})
    second = fingerprint_source(reversed(rows), {})
    assert first == second
    assert first.task_count == 31 and first.within_source_duplicates == 1
    assert len(first.sample) == 20


def test_top_candidate_review_is_cached_and_acknowledgement_is_bound_to_request(review_server):
    arithmetic = [task_document(str(index), f"Compute the square of {index}.") for index in range(25)]
    chemistry = [
        task_document(f"chem-{index}", f"Name the chemical element with atomic number {index}.")
        for index in range(1, 25)
    ]
    old_ids = ("Campaign Release:tasktrove-codeforces-rebuilt-v1", "MarinSkyRL:svamp")
    original, rebuilt = map(canonical_source_id, old_ids)
    fingerprints = {
        original: fingerprint_source(arithmetic, {}),
        rebuilt: fingerprint_source(list(reversed(arithmetic)), {}),
        "distinct": fingerprint_source(chemistry, {}),
    }
    pairs = candidate_pairs(fingerprints, {old_ids[0]}, 1)
    assert [(pair.left, pair.right) for pair in pairs] == [(original, rebuilt)]
    review_server.model_ids = [DUPLICATE_MODEL]
    review_server.judge_replies = [
        JudgeReply(
            json.dumps({"label": "duplicate", "overlap_fraction": 1.0, "reason": "The same arithmetic task population."})
        )
    ]
    endpoint = HostedJudgeEndpoint(DUPLICATE_MODEL, f"http://127.0.0.1:{review_server.server_port}/v1", "test-key")
    reviewed = list(review_pairs(pairs, fingerprints, endpoint, [], {}))
    assert reviewed[0].verdict.label == DuplicateLabel.DUPLICATE
    request_key = reviewed[0].prompt_sha256
    legacy_fingerprints = {
        old_ids[0]: fingerprints[original],
        old_ids[1]: fingerprints[rebuilt],
        "distinct": fingerprints["distinct"],
    }
    acknowledged = list(
        review_pairs(pairs, legacy_fingerprints, endpoint, reviewed, {request_key: "Retain both pinned histories"})
    )
    assert len(review_server.requests) == 1
    assert acknowledged[0].acknowledgement == "Retain both pinned histories"
    assert review_server.requests[0]["model"] == DUPLICATE_MODEL
    assert review_server.requests[0]["temperature"] == 0.7
    groups = duplicate_groups(acknowledged)
    assert groups[0].canonical_source_id == original
    assert {member.source_id for member in groups[0].members} == {original, rebuilt}
    assert fingerprints[original].task_count == fingerprints[rebuilt].task_count == 25


def test_source_change_detection_includes_added_and_modified_declarations():
    before = {
        "same": {"id": "same", "pin": "a"},
        "changed": {"id": "changed", "pin": "old"},
        "removed": {"id": "removed", "pin": "a"},
    }
    after = {
        "same": {"id": "same", "pin": "a"},
        "changed": {"id": "changed", "pin": "new"},
        "added": {"id": "added", "pin": "b"},
    }
    assert changed_source_declarations(before, after) == {"changed", "added"}


def test_known_codeforces_rebuild_ranks_before_distinct_nl2bash():
    fixture = Path(__file__).parent / "fixtures/source-overlap-fingerprints.json"
    fingerprints = {
        key: SourceFingerprint.model_validate(value) for key, value in json.loads(fixture.read_text()).items()
    }
    pairs = candidate_pairs(fingerprints, {"TaskTrove:tasktrove-codeforces-rebuilt-v1"}, 1)
    assert [(pair.left, pair.right) for pair in pairs] == [
        ("TaskTrove:tasktrove-codeforces-rebuilt-v1", "TaskTrove:laion__codeforces-v3")
    ]


def test_alias_only_declaration_change_reuses_published_fingerprints():
    old = "Campaign Release:tasktrove-codeforces-rebuilt-v1"
    current = canonical_source_id(old)
    before = {old: {"id": old, "pipeline": {"version": "frozen"}}}
    after = {current: {"id": current, "pipeline": {"version": "frozen"}}}
    assert changed_source_declarations(before, after) == set()
    after[current]["pipeline"]["version"] = "changed"
    assert changed_source_declarations(before, after) == {current}


def test_frozen_survey_digest_survives_canonical_source_projection():
    raw = {
        "version": 1,
        "catalog_root": "fixture",
        "snapshot_id": "frozen",
        "generated_at": "2026-11-12T00:00:00Z",
        "fingerprints": {"MarinSkyRL:aime24": None, "MarinSkyRL:svamp": None},
        "reviews": [],
        "groups": [],
        "judge": None,
    }
    survey = SourceIdentitySurvey.model_validate(raw)
    assert set(survey.fingerprints) == {"HuggingFaceH4:aime24", "ChilleD:svamp"}
    assert canonical_sha256(survey.identity_payload()) == canonical_sha256(raw)
