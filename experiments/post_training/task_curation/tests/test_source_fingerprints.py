# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source identity ignores row order and keeps overlap flags advisory."""

import json
from pathlib import Path

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
from taskcompendium.source_fingerprints import DuplicateLabel, SourceFingerprint

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
    fingerprints = {
        "original": fingerprint_source(arithmetic, {}),
        "rebuilt": fingerprint_source(list(reversed(arithmetic)), {}),
        "distinct": fingerprint_source(chemistry, {}),
    }
    pairs = candidate_pairs(fingerprints, {"original"}, 1)
    assert [(pair.left, pair.right) for pair in pairs] == [("original", "rebuilt")]
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
    acknowledged = list(
        review_pairs(pairs, fingerprints, endpoint, reviewed, {request_key: "Retain both pinned histories"})
    )
    assert len(review_server.requests) == 1
    assert acknowledged[0].acknowledgement == "Retain both pinned histories"
    assert review_server.requests[0]["model"] == DUPLICATE_MODEL
    assert review_server.requests[0]["temperature"] == 0.7
    groups = duplicate_groups(acknowledged)
    assert groups[0].canonical_source_id == "original"
    assert {member.source_id for member in groups[0].members} == {"original", "rebuilt"}
    assert fingerprints["original"].task_count == fingerprints["rebuilt"].task_count == 25


def test_source_change_detection_includes_added_and_modified_declarations():
    before = {"same": {"pin": "a"}, "changed": {"pin": "old"}, "removed": {"pin": "a"}}
    after = {"same": {"pin": "a"}, "changed": {"pin": "new"}, "added": {"pin": "b"}}
    assert changed_source_declarations(before, after) == {"changed", "added"}


def test_known_codeforces_rebuild_ranks_before_distinct_nl2bash():
    fixture = Path(__file__).parent / "fixtures/source-overlap-fingerprints.json"
    fingerprints = {
        key: SourceFingerprint.model_validate(value) for key, value in json.loads(fixture.read_text()).items()
    }
    pairs = candidate_pairs(fingerprints, {"Campaign Release:tasktrove-codeforces-rebuilt-v1"}, 1)
    assert [(pair.left, pair.right) for pair in pairs] == [
        ("Campaign Release:tasktrove-codeforces-rebuilt-v1", "Task Trove:laion__codeforces-v3")
    ]
