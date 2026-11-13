# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source identity ignores row order and keeps overlap flags advisory."""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from iris.rpc import job_pb2
from marin.evaluation.eval_env import EVALUATION_REQUEST_TIMEOUT_SECONDS
from marin.evaluation.model_config import load_model_config
from marin.inference.types import OpenAIEndpoint, RunningModel
from rigging.filesystem.storage_path import StoragePath
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
from taskcompendium.source_fingerprints import DuplicateLabel, DuplicateReview, SourceFingerprint, SourceIdentitySurvey

from experiments.post_training.task_curation.catalog_identity import canonical_identity_survey, identity_survey_payload
from experiments.post_training.task_curation.duplicate_review import (
    DUPLICATE_MAX_TOKENS,
    DUPLICATE_MODEL,
    DUPLICATE_MODEL_CONFIG,
    LocalDuplicateEndpoint,
    UnresolvedDuplicatePair,
    review_pairs,
)
from experiments.post_training.task_curation.judges import HostedJudgeEndpoint
from experiments.post_training.task_curation.source_duplicate_ci import changed_source_declarations
from experiments.post_training.task_curation.source_duplicates import scan
from experiments.post_training.task_curation.source_fingerprints import (
    candidate_pairs,
    duplicate_groups,
    fingerprint_source,
)
from experiments.post_training.task_curation.source_identity import canonical_source_id
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


def source_input_declaration(identifier: str, revision: str = "pin") -> dict:
    return {
        "id": identifier,
        "dataset_id": "fixture/tasks",
        "dataset_revision": revision,
        "pipeline": {"name": "fixture", "version": "1", "source": "fixture/tasks", "revision": revision, "files": []},
        "actor_census_input": {
            "dataset_id": "fixture/tasks",
            "dataset_revision": revision,
            "converter_version": "1",
            "converter_revision": "converter-v1",
            "measured_tasks": 10,
            "task_specs_sha256": "a" * 64,
            "evidence_url": "s3://fixture/actor-census.json",
        },
    }


def test_source_fingerprint_census_is_order_independent_and_counts_duplicate_contracts():
    rows = [task_document(str(index), f"Compute the square of {index}.") for index in range(30)]
    rows.append(task_document("copied-row", "Compute the square of 2."))
    first = fingerprint_source(rows, {})
    second = fingerprint_source(reversed(rows), {})
    assert first == second
    assert first.task_count == 31 and first.within_source_duplicates == 1
    assert len(first.sample) == 20


def test_top_candidate_review_is_cached_and_acknowledgement_is_bound_to_request(review_server, tmp_path):
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
    responses = StoragePath(str(tmp_path / "responses"))
    reviewed = list(review_pairs(pairs, fingerprints, endpoint, [], {}, responses))
    first = reviewed[0]
    assert isinstance(first, DuplicateReview) and first.verdict.label == DuplicateLabel.DUPLICATE
    request_key = first.prompt_sha256
    legacy_fingerprints = {
        old_ids[0]: fingerprints[original],
        old_ids[1]: fingerprints[rebuilt],
        "distinct": fingerprints["distinct"],
    }
    acknowledged = list(
        review_pairs(
            pairs, legacy_fingerprints, endpoint, [first], {request_key: "Retain both pinned histories"}, responses
        )
    )
    assert len(review_server.requests) == 1
    acknowledged_review = acknowledged[0]
    assert isinstance(acknowledged_review, DuplicateReview)
    assert acknowledged_review.acknowledgement == "Retain both pinned histories"
    assert review_server.requests[0]["model"] == DUPLICATE_MODEL
    assert review_server.requests[0]["temperature"] == 0.7
    groups = duplicate_groups([acknowledged_review])
    assert groups[0].canonical_source_id == original
    assert {member.source_id for member in groups[0].members} == {original, rebuilt}
    assert fingerprints[original].task_count == fingerprints[rebuilt].task_count == 25


def test_source_change_detection_includes_added_and_modified_declarations():
    before = {
        "same": source_input_declaration("same", "a"),
        "changed": source_input_declaration("changed", "old"),
        "removed": source_input_declaration("removed", "a"),
    }
    after = {
        "same": source_input_declaration("same", "a"),
        "changed": source_input_declaration("changed", "new"),
        "added": source_input_declaration("added", "b"),
    }
    assert changed_source_declarations(before, after) == {"changed", "added"}


def test_uncached_local_review_covers_batch_gpu_recovery(review_server, monkeypatch, tmp_path):
    model_config = tmp_path / "config.json"
    model_config.write_text(json.dumps({"max_position_embeddings": 262144}))
    monkeypatch.setattr("marin.evaluation.serving_config.hf_hub_download", lambda *args, **kwargs: str(model_config))
    leases = []
    request_timeouts = []
    send = httpx.Client.send

    def record_request(client, request, *args, **kwargs):
        if request.method == "POST":
            request_timeouts.append(request.extensions["timeout"])
        return send(client, request, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", record_request)

    @contextmanager
    def serving_lease(config):
        leases.append(config)
        yield SimpleNamespace(
            jobs=(),
            model=RunningModel(
                endpoint=OpenAIEndpoint(
                    base_url=f"http://127.0.0.1:{review_server.server_port}/v1", model=DUPLICATE_MODEL
                )
            ),
            wait_until_ready=lambda: None,
        )

    monkeypatch.setattr("experiments.post_training.task_curation.duplicate_review.remote_inference", serving_lease)
    review_server.model_ids = [DUPLICATE_MODEL]
    review_server.judge_replies = [
        JudgeReply(json.dumps({"label": "duplicate", "overlap_fraction": 1.0, "reason": "Identical square tasks."}))
    ]
    rows = [task_document("square", "Compute the square of 2.")]
    fingerprints = {"original": fingerprint_source(rows, {}), "copy": fingerprint_source(rows, {})}
    pairs = candidate_pairs(fingerprints, {"copy"}, 1)
    reviews = list(
        review_pairs(
            pairs, fingerprints, LocalDuplicateEndpoint(DUPLICATE_MODEL_CONFIG), [], {}, StoragePath(str(tmp_path))
        )
    )

    assert isinstance(reviews[0], DuplicateReview) and reviews[0].verdict.label is DuplicateLabel.DUPLICATE
    assert leases[0].iris.priority == job_pb2.PRIORITY_BAND_BATCH
    assert request_timeouts[0]["read"] >= (
        leases[0].iris.endpoint_ready_timeout_seconds + EVALUATION_REQUEST_TIMEOUT_SECONDS
    )
    assert review_server.requests[0]["temperature"] == 0.7


def test_invalid_duplicate_opinions_are_checkpointed_retried_and_isolated_per_pair(review_server, monkeypatch, tmp_path):
    rows = [task_document("square", "Compute the square of 2.")]
    fingerprints = {name: fingerprint_source(rows, {}) for name in ("first", "second", "third")}
    pairs = candidate_pairs(fingerprints, set(fingerprints), 3)
    valid = json.dumps({"label": "duplicate", "overlap_fraction": 1, "reason": "Identical square tasks."})
    inconsistent = json.dumps({"label": "duplicate", "overlap_fraction": 0.95, "reason": "Estimated shared population."})
    contents = [inconsistent, valid, "not-json", inconsistent, inconsistent, valid]
    review_server.model_ids = [DUPLICATE_MODEL]
    review_server.judge_replies = [JudgeReply(content) for content in contents]
    endpoint = HostedJudgeEndpoint(DUPLICATE_MODEL, f"http://127.0.0.1:{review_server.server_port}/v1", "test-key")
    send = httpx.Client.send

    def checkpoint_before_next_request(client, request, *args, **kwargs):
        if request.method == "POST":
            assert len(list(tmp_path.rglob("*.json"))) == len(review_server.requests)
        return send(client, request, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", checkpoint_before_next_request)
    survey = SourceIdentitySurvey(
        catalog_root="fixture",
        snapshot_id="frozen",
        generated_at="2026-11-13T00:00:00Z",
        fingerprints=fingerprints,
        reviews=[],
        groups=[],
        judge=None,
    )
    receipts = tmp_path / "scan.receipts.jsonl"
    with pytest.raises(ValueError, match="1 source pairs remain unresolved"):
        scan(survey, set(fingerprints), 3, endpoint, {}, [], receipts)
    first, last = [DuplicateReview.model_validate_json(line) for line in receipts.read_text().splitlines()]
    unresolved = UnresolvedDuplicatePair.model_validate_json(receipts.with_suffix(".unresolved.jsonl").read_text())
    assert first.similarity == pairs[0] and last.similarity == pairs[2]
    assert unresolved.similarity == pairs[1]
    assert len(unresolved.response_receipts) == len(unresolved.validation_errors) == 3
    saved = [json.loads(path.read_text())["choices"][0]["message"]["content"] for path in tmp_path.rglob("*.json")]
    assert sorted(saved) == sorted(contents)
    assert all(Path(path).exists() for path in unresolved.response_receipts)
    assert len(review_server.requests) == 6
    assert review_server.requests[0] == review_server.requests[1]
    assert review_server.requests[2] == review_server.requests[3] == review_server.requests[4]


def test_longest_retained_pair_prompt_fits_local_judge_context():
    # All 745 frozen pairs were tokenized with Qwen revision 37561816e66b5b9153b1b64c10968c18c1e6f4a0.
    # Receipt SHA-256: b4d6e4d62acf61757eea387acd9144ca9d98e3c9300de77cd299305f63297433
    receipt = Path(__file__).parent / "fixtures/source-overlap-context-budget.json"
    budget = json.loads(receipt.read_text())
    model = load_model_config(DUPLICATE_MODEL_CONFIG)
    assert model.serve.max_model_len is not None
    assert budget["max_prompt_tokens"] + DUPLICATE_MAX_TOKENS <= model.serve.max_model_len


@pytest.mark.parametrize("field", ["converter_revision", "task_specs_sha256"])
def test_source_content_changes_require_fingerprints_while_measurement_metadata_reuses_them(field):
    before = {"fixture": source_input_declaration("fixture")}
    after = {"fixture": source_input_declaration("fixture")}
    after["fixture"].update(notes="New display notes", task_count=8, lexical_contamination={"measured_tasks": 10})
    after["fixture"]["actor_census_input"]["evidence_url"] = "s3://fixture/copied-census.json"
    assert changed_source_declarations(before, after) == set()
    after["fixture"]["actor_census_input"][field] = "changed"
    assert changed_source_declarations(before, after) == {"fixture"}


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
    before = {old: source_input_declaration(old)}
    after = {current: source_input_declaration(current)}
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
    survey = canonical_identity_survey(raw)
    assert set(survey.fingerprints) == {"HuggingFaceH4:aime24", "ChilleD:svamp"}
    assert canonical_sha256(identity_survey_payload(survey)) == canonical_sha256(raw)
