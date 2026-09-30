# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The public dataset projection excludes private grading material."""

import json
from hashlib import sha256

import pytest

from taskcompendium.grading import exact_answer, structured_exact
from taskcompendium.lowering import HarborEnvironmentConfig, ToolBinding
from taskcompendium.mixed_release import (
    AcceptedPublicRecord,
    CohortInput,
    HarborSample,
    ReleaseReview,
    SourceAsset,
    SourceProof,
    SourceRights,
    assemble_mixed_candidate,
    finalize_mixed_candidate,
)
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ProviderRequirement,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.public_projection import public_task
from taskcompendium.submission import PlainText, ProviderState


def test_public_task_retains_agent_input_and_ordered_tags_without_expected_state():
    specification = TaskSpec(
        id="example",
        context=ConversationInput(events=(TextMessage(role="user", content="What is the answer?"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=exact_answer("secret-gold-action"),
        source=Source(dataset="source", revision="pinned", row="1", importer_revision="1"),
        tags=("first", "second"),
    )
    record = public_task(
        specification,
        PlainText(id="plain"),
        HarborEnvironmentConfig(),
    )
    payload = json.loads(record.model_dump_json())
    assert payload["context"]["events"][0]["content"] == "What is the answer?"
    assert payload["tags"] == ["first", "second"]
    assert payload["submission_instruction"] == "Give your answer as plain text."
    assert "verifier" not in payload
    assert "secret-gold-action" not in record.model_dump_json()


def test_public_state_task_keeps_requirement_without_runtime_binding_or_expected_state():
    specification = TaskSpec(
        id="workplace-example",
        context=ConversationInput(events=(TextMessage(role="user", content="Update the calendar."),)),
        environment_requirements=EnvironmentRequirements(),
        tool_providers={"workplace": ProviderRequirement(action_interface="workplace.v1", seed_sha256="a" * 64)},
        answer_type=AnswerType.STATE,
        verifier=structured_exact({"private_marker": "secret-expected-state"}),
        source=Source(dataset="workplace", revision="pinned", row="train:0", importer_revision="1"),
    )
    environment = HarborEnvironmentConfig(
        tool_providers={
            "workplace": ToolBinding(
                action_interface="workplace.v1",
                seed_sha256="a" * 64,
                provider="python:internal.secret:NemoWorkplaceProvider",
                provider_revision="1",
                tools=("calendar_update_event",),
                tools_sha256="b" * 64,
            )
        }
    )
    record = public_task(
        specification,
        ProviderState(id="state", provider="workplace"),
        environment,
        source_category="workplace_assistant_calendar",
    )
    payload = json.loads(record.model_dump_json())
    assert payload["record_version"] == 1
    assert payload["tool_providers"]["workplace"]["action_interface"] == "workplace.v1"
    assert payload["source_category"] == "workplace_assistant_calendar"
    assert payload["submission_instruction"].startswith("Use the available tools")
    assert "secret-expected-state" not in record.model_dump_json()
    assert "internal.secret" not in record.model_dump_json()
    mismatched = environment.model_copy(
        update={
            "tool_providers": {
                "workplace": environment.tool_providers["workplace"].model_copy(update={"seed_sha256": "c" * 64})
            }
        }
    )
    with pytest.raises(ValueError, match="binding interface or seed"):
        public_task(specification, ProviderState(id="state", provider="workplace"), mismatched)


def test_mixed_candidate_keeps_configs_source_proof_and_rights_separate(tmp_path):
    source_digest = "a" * 64
    regional_pin = (
        "2026.09.18.3#manifest-sha256=" + "b" * 64 + "#parquet-size=100#parquet-etag=etag#parquet-version-id=id"
    )
    workplace = public_task(
        TaskSpec(
            id="workplace-0",
            context=ConversationInput(events=(TextMessage(role="user", content="Update the calendar."),)),
            environment_requirements=EnvironmentRequirements(),
            answer_type=AnswerType.TEXT,
            verifier=exact_answer("private-workplace-answer"),
            source=Source(
                dataset="nvidia/workplace",
                revision="source-pin",
                row=f"train:0:{source_digest}",
                importer_revision="workplace-v1",
            ),
        ),
        PlainText(id="plain"),
        HarborEnvironmentConfig(),
        source_category="calendar",
    )
    tasktrove = public_task(
        TaskSpec(
            id="tasktrove-0",
            context=ConversationInput(events=(TextMessage(role="user", content="Choose one option."),)),
            environment_requirements=EnvironmentRequirements(),
            answer_type=AnswerType.TEXT,
            verifier=exact_answer("private-tasktrove-answer"),
            source=Source(
                dataset="open-athena/task-trove",
                revision=regional_pin,
                row="knowledge:knowledge/mcqa.jsonl",
                importer_revision="mcq-v1",
            ),
            tags=("knowledge", "mcqa"),
        ),
        PlainText(id="plain"),
        HarborEnvironmentConfig(),
        source_category="knowledge",
    )
    archive_digest = "b" * 64
    accepted = AcceptedPublicRecord(
        task=tasktrove,
        source_proof=SourceProof(
            source_row="knowledge:knowledge/mcqa.jsonl",
            input_file="source.parquet",
            input_object_pin=regional_pin,
            archive_path="knowledge/mcqa.jsonl",
            archive_sha256=archive_digest,
        ),
    )
    workplace_path = tmp_path / "workplace.jsonl"
    workplace_path.write_text(workplace.model_dump_json() + "\n")
    tasktrove_path = tmp_path / "tasktrove.jsonl"
    regional_candidate = accepted.model_dump(mode="json")
    del regional_candidate["task"]["record_version"]
    tasktrove_path.write_text(json.dumps(regional_candidate) + "\n")
    rights = SourceRights(
        license="cc-by-4.0",
        attribution="NVIDIA Corporation",
        source_card_url="https://example.org/card",
        source_card_revision="1" * 40,
        change_notice="Converted source rows to agent-visible tasks.",
    )
    sample = HarborSample(
        evidence_url="https://example.org/trials",
        taskcompendium_revision="d" * 40,
        harbor_revision="e" * 40,
        trials=1,
        coverage="one accepted row",
    )
    cohorts = (
        CohortInput(
            config="workplace",
            cohort="workplace_train",
            split="train",
            record_format="public_task",
            input_path=workplace_path,
            input_sha256=sha256(workplace_path.read_bytes()).hexdigest(),
            accepted_rows=1,
            source_records=2,
            parsed_rows=1,
            source_dataset="nvidia/workplace",
            source_revision="source-pin",
            source_assets=(SourceAsset(path="train.jsonl", pin="sha256:" + "f" * 64),),
            task_spec_schema="0.11",
            importer_revision="workplace-v1",
            projection_builder_revision="1" * 40,
            rights=rights,
            harbor_samples=(sample,),
        ),
        CohortInput(
            config="tasktrove_clean",
            cohort="knowledge_mcqa",
            split="train",
            record_format="accepted_public_record",
            input_path=tasktrove_path,
            input_sha256=sha256(tasktrove_path.read_bytes()).hexdigest(),
            accepted_rows=1,
            source_records=23860,
            parsed_rows=1,
            source_dataset="open-athena/task-trove",
            source_subset="knowledge",
            source_category="knowledge",
            projection_manifest_sha256="c" * 64,
            source_assets=(SourceAsset(path="source.parquet", pin=regional_pin),),
            task_spec_schema="0.13",
            importer_revision="mcq-v1",
            projection_builder_revision="2" * 40,
            rights=rights,
            harbor_samples=(sample,),
        ),
    )
    destination = assemble_mixed_candidate(cohorts, tmp_path / "candidate", builder_revision="3" * 40)
    reversed_candidate = assemble_mixed_candidate(
        tuple(reversed(cohorts)), tmp_path / "candidate-reversed", builder_revision="3" * 40
    )
    candidate_files = sorted(path.relative_to(destination) for path in destination.rglob("*") if path.is_file())
    assert candidate_files == sorted(
        path.relative_to(reversed_candidate) for path in reversed_candidate.rglob("*") if path.is_file()
    )
    assert all((destination / path).read_bytes() == (reversed_candidate / path).read_bytes() for path in candidate_files)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert [entry["config"] for entry in manifest["data_files"]] == ["workplace", "tasktrove_clean"]
    assert [entry["source_records"] for entry in manifest["data_files"]] == [2, 23860]
    assert [entry["parsed_rows"] for entry in manifest["data_files"]] == [1, 1]
    assert [entry["accepted_rows"] for entry in manifest["data_files"]] == [1, 1]
    assert [entry["exported_rows"] for entry in manifest["data_files"]] == [1, 1]
    assert manifest["publication_ready"] is False
    assert manifest["data_files"][1]["source"]["projection_manifest_sha256"] == "c" * 64
    tasktrove_output = (destination / "data/tasktrove_clean/knowledge_mcqa.jsonl").read_text()
    assert json.loads(tasktrove_output)["task"]["record_version"] == 1
    assert json.loads(tasktrove_output)["task"]["tags"] == ["knowledge", "mcqa"]
    assert json.loads(tasktrove_output)["source_proof"]["archive_sha256"] == archive_digest
    assert "private-workplace-answer" not in (destination / "data/workplace/workplace_train.jsonl").read_text()
    assert "private-tasktrove-answer" not in tasktrove_output
    assert "config_name: workplace" in (destination / "README.md").read_text()
    assert "config_name: tasktrove_clean" in (destination / "README.md").read_text()

    candidate_manifest_sha256 = sha256((destination / "manifest.json").read_bytes()).hexdigest()
    (destination / "private.json").write_text('{"private": true}')
    with pytest.raises(ValueError, match="unlisted or unsupported file"):
        finalize_mixed_candidate(
            destination,
            tmp_path / "ready-with-private-file",
            ReleaseReview(
                candidate_manifest_sha256=candidate_manifest_sha256,
                rights_review_url="https://example.org/rights-review",
                harbor_evidence_urls=("https://example.org/trials",),
            ),
        )
    assert not (tmp_path / "ready-with-private-file").exists()
    (destination / "private.json").unlink()
    ready = finalize_mixed_candidate(
        destination,
        tmp_path / "ready",
        ReleaseReview(
            candidate_manifest_sha256=candidate_manifest_sha256,
            rights_review_url="https://example.org/rights-review",
            harbor_evidence_urls=("https://example.org/trials",),
        ),
    )
    ready_manifest = json.loads((ready / "manifest.json").read_text())
    assert ready_manifest["publication_ready"] is True
    assert ready_manifest["publication_review"]["candidate_manifest_sha256"] == candidate_manifest_sha256
    assert (
        sha256((ready / "data/tasktrove_clean/knowledge_mcqa.jsonl").read_bytes()).hexdigest()
        == sha256((destination / "data/tasktrove_clean/knowledge_mcqa.jsonl").read_bytes()).hexdigest()
    )
    assert json.loads((destination / "manifest.json").read_text())["publication_ready"] is False
    assert "TaskCompendium Alpha 1 Candidate" not in (ready / "README.md").read_text()
    tasktrove_output_path = destination / "data/tasktrove_clean/knowledge_mcqa.jsonl"
    tasktrove_output_path.write_text(tasktrove_output + " ")
    with pytest.raises(ValueError, match="does not match its manifest"):
        finalize_mixed_candidate(
            destination,
            tmp_path / "tampered-ready",
            ReleaseReview(
                candidate_manifest_sha256=candidate_manifest_sha256,
                rights_review_url="https://example.org/rights-review",
                harbor_evidence_urls=("https://example.org/trials",),
            ),
        )
    assert not (tmp_path / "tampered-ready").exists()


def test_mixed_candidate_rejects_private_or_unproved_tasktrove_record(tmp_path):
    regional_pin = (
        "2026.09.18.3#manifest-sha256=" + "a" * 64 + "#parquet-size=100#parquet-etag=etag#parquet-version-id=id"
    )
    task = public_task(
        TaskSpec(
            id="tasktrove-0",
            context=ConversationInput(events=(TextMessage(role="user", content="Choose one option."),)),
            environment_requirements=EnvironmentRequirements(),
            answer_type=AnswerType.TEXT,
            verifier=exact_answer("private-answer"),
            source=Source(
                dataset="open-athena/task-trove",
                revision=regional_pin,
                row="knowledge:knowledge/mcqa.jsonl",
                importer_revision="mcq-v1",
            ),
        ),
        PlainText(id="plain"),
        HarborEnvironmentConfig(),
        source_category="knowledge",
    )
    path = tmp_path / "accepted.jsonl"
    path.write_text(
        json.dumps(
            {
                "task": task.model_dump(mode="json"),
                "source_proof": {
                    "source_row": "knowledge:knowledge/mcqa.jsonl",
                    "input_file": "source.parquet",
                    "input_object_pin": regional_pin,
                    "archive_path": "knowledge/mcqa.jsonl",
                    "archive_sha256": "b" * 64,
                },
            }
        )
        + "\n"
    )
    cohort = CohortInput(
        config="tasktrove_clean",
        cohort="knowledge_mcqa",
        split="train",
        record_format="accepted_public_record",
        input_path=path,
        input_sha256=sha256(path.read_bytes()).hexdigest(),
        accepted_rows=1,
        source_records=1,
        source_dataset="open-athena/task-trove",
        source_subset="knowledge",
        source_category="knowledge",
        projection_manifest_sha256="f" * 64,
        source_assets=(SourceAsset(path="source.parquet", pin=regional_pin),),
        task_spec_schema="0.13",
        importer_revision="mcq-v1",
        projection_builder_revision="1" * 40,
        rights=SourceRights(
            license="cc-by-4.0",
            attribution="NVIDIA Corporation",
            source_card_url="https://example.org/card",
            source_card_revision="1" * 40,
            change_notice="Converted source row.",
        ),
        harbor_samples=(
            HarborSample(
                evidence_url="https://example.org/trials",
                taskcompendium_revision="c" * 40,
                harbor_revision="d" * 40,
                trials=1,
                coverage="one accepted row",
            ),
        ),
    )
    with pytest.raises(ValueError, match="original ordered source tags"):
        assemble_mixed_candidate((cohort,), tmp_path / "candidate", builder_revision="e" * 40)
    assert not (tmp_path / "candidate").exists()
    payload = json.loads(path.read_text())
    payload["task"]["verifier"] = {"expected": "private-answer"}
    path.write_text(json.dumps(payload) + "\n")
    cohort = cohort.model_copy(update={"input_sha256": sha256(path.read_bytes()).hexdigest()})
    with pytest.raises(ValueError):
        assemble_mixed_candidate((cohort,), tmp_path / "candidate", builder_revision="e" * 40)
    assert not (tmp_path / "candidate").exists()


def test_release_finalizer_rejects_review_for_another_candidate(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "manifest.json").write_text('{"publication_ready": false}\n')
    with pytest.raises(ValueError, match="does not match"):
        finalize_mixed_candidate(
            candidate,
            tmp_path / "ready",
            ReleaseReview(
                candidate_manifest_sha256="a" * 64,
                rights_review_url="https://example.org/rights-review",
                harbor_evidence_urls=("https://example.org/trials",),
            ),
        )
    assert not (tmp_path / "ready").exists()


def test_regional_proof_requires_object_identity():
    unpinned = "2026.09.18.3#manifest-sha256=" + "a" * 64 + "#parquet-size=100#parquet-etag=#parquet-version-id="
    with pytest.raises(ValueError, match="immutable pin"):
        SourceProof(source_row="knowledge:row", input_file="source.parquet", input_object_pin=unpinned)
