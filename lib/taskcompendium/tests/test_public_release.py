# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The public dataset projection excludes private grading material."""

import json
from hashlib import sha256

import pytest

from taskcompendium.grading import exact_answer, structured_exact
from taskcompendium.lowering import HarborEnvironmentConfig, ToolBinding
from taskcompendium.mixed_release import (
    AcceptedTaskRecord,
    CatalogJoinEvidence,
    CohortInput,
    HarborSample,
    PublishedRow,
    ReleaseReview,
    SourceAsset,
    SourceProof,
    SourceRights,
    assemble_mixed_candidate,
    finalize_mixed_candidate,
    published_task,
)
from taskcompendium.models import (
    SCHEMA_VERSION,
    AnswerType,
    ConversationInput,
    ConversationTrace,
    EnvironmentRequirements,
    FunctionDefinition,
    ProviderRequirement,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.public_projection import public_task
from taskcompendium.submission import FinalAction, GradingAttempt, PlainText, ProviderState
from taskcompendium.verifier_registry import grade_answer
from taskcompendium.verifiers.multiple_choice import multiple_choice_answer


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


def _cohort(tmp_path, config="workplace"):
    digest = "a" * 64
    regional_pin = (
        "2026.09.18.3#manifest-sha256=" + "b" * 64 + "#parquet-size=100#parquet-etag=etag#parquet-version-id=id"
    )
    tasktrove = config == "tasktrove_clean"
    dataset = "s3://example-bucket/tasktrove" if tasktrove else "nvidia/workplace"
    source_row = "knowledge:knowledge/mcqa.jsonl" if tasktrove else f"train:0:{digest}"
    source_pin = regional_pin if tasktrove else "source-pin"
    input_file = f"{dataset}/source.parquet" if tasktrove else "train.jsonl"
    object_pin = regional_pin if tasktrove else "sha256:" + digest
    specification = TaskSpec(
        id=config,
        context=ConversationInput(events=(TextMessage(role="user", content="Choose an option."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=multiple_choice_answer("B", 3) if tasktrove else exact_answer("private-workplace-answer"),
        source=Source(dataset=dataset, revision=source_pin, row=source_row, importer_revision="importer-v1"),
        tags=("knowledge", "mcqa") if tasktrove else (),
    )
    record = AcceptedTaskRecord(
        task=specification,
        source_category="knowledge" if tasktrove else "calendar",
        source_proof=SourceProof(
            source_row=source_row,
            source_row_sha256=None if tasktrove else digest,
            input_file=input_file,
            input_object_pin=object_pin,
            archive_path="knowledge/mcqa.jsonl" if tasktrove else None,
            archive_sha256="c" * 64 if tasktrove else None,
        ),
    )
    path = tmp_path / f"{config}.jsonl"
    path.write_text(record.model_dump_json() + "\n")
    sample = HarborSample(
        evidence_url="https://example.org/trials",
        taskcompendium_revision="d" * 40,
        harbor_revision="e" * 40,
        trials=1,
        coverage="one reconstructed task",
    )
    cohort = CohortInput(
        config=config,
        cohort="mcqa" if tasktrove else "workplace_train",
        split="train",
        input_path=path,
        input_sha256=sha256(path.read_bytes()).hexdigest(),
        accepted_rows=1,
        source_records=2,
        parsed_rows=1,
        source_dataset=dataset,
        source_revision=None if tasktrove else source_pin,
        source_subset="knowledge" if tasktrove else None,
        source_category="knowledge" if tasktrove else None,
        projection_manifest_sha256="f" * 64 if tasktrove else None,
        source_assets=(SourceAsset(path=input_file, pin=object_pin),),
        task_spec_schema=SCHEMA_VERSION,
        importer_revision="importer-v1",
        projection_builder_revision="1" * 40,
        rights=SourceRights(
            license="cc-by-4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            attribution="NVIDIA Corporation",
            source_card_url="https://example.org/card",
            source_card_revision="2" * 40,
            change_notice="Converted accepted source rows to complete tasks.",
        ),
        harbor_samples=(sample,),
        catalog_join=(
            CatalogJoinEvidence(
                catalog_sha256="3" * 64,
                ledger_sha256="4" * 64,
                output_schema_version=SCHEMA_VERSION,
                joined_rows=1,
                verifier_matches=1,
                projected_field_matches=1,
            )
            if tasktrove
            else None
        ),
    )
    return cohort, record


async def test_demo_export_reconstructs_gradeable_tasks_and_retains_provenance(tmp_path):
    workplace, original_workplace = _cohort(tmp_path)
    tasktrove, original_tasktrove = _cohort(tmp_path, "tasktrove_clean")
    destination = assemble_mixed_candidate((workplace, tasktrove), tmp_path / "candidate", builder_revision="5" * 40)
    reversed_candidate = assemble_mixed_candidate(
        (tasktrove, workplace), tmp_path / "reversed", builder_revision="5" * 40
    )
    for path in destination.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (reversed_candidate / path.relative_to(destination)).read_bytes()
    mcqa = PublishedRow.model_validate_json((destination / "data/tasktrove_clean/mcqa.jsonl").read_text())
    reconstructed = published_task(mcqa)
    assert reconstructed.verifier == original_tasktrove.task.verifier
    assert reconstructed.answer_type is AnswerType.TEXT
    assert mcqa.record_version == 3 and mcqa.schema_version == SCHEMA_VERSION
    assert mcqa.tags == ("knowledge", "mcqa")
    assert mcqa.source.dataset == "tasktrove_clean"
    assert mcqa.provenance.input_file == "source.parquet"
    rewards = []
    for answer in ("B", "A"):
        result = await grade_answer(
            reconstructed,
            PlainText(id="plain"),
            GradingAttempt(
                conversation=ConversationTrace(
                    events=(*reconstructed.context.events, TextMessage(role="assistant", content=answer))
                ),
                tool_providers={},
                workspace=None,
            ),
        )
        rewards.append(result.reward)
    assert rewards == [1.0, 0.0]
    state = PublishedRow.model_validate_json((destination / "data/workplace/workplace_train.jsonl").read_text())
    assert published_task(state).verifier == original_workplace.task.verifier
    assert "private-workplace-answer" in state.model_dump_json()
    assert "verifier" not in public_task(reconstructed, PlainText(id="plain"), HarborEnvironmentConfig()).model_dump()
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["public_record_version"] == 3
    assert manifest["data_files"][1]["source"]["catalog_join"]["input_schema_version"] == "0.13"
    assert manifest["publication_ready"] is False
    for path in destination.rglob("*"):
        if path.is_file():
            assert "example-bucket" not in path.read_text()
    payload = mcqa.model_dump()
    assert not {"task", "source_proof", "submission_instruction", "gold"} & payload.keys()


def test_demo_export_rejects_missing_grader_and_wrong_source_proof(tmp_path):
    cohort, record = _cohort(tmp_path, "tasktrove_clean")
    payload = record.model_dump(mode="json")
    del payload["task"]["verifier"]
    cohort.input_path.write_text(json.dumps(payload) + "\n")
    cohort = cohort.model_copy(update={"input_sha256": sha256(cohort.input_path.read_bytes()).hexdigest()})
    with pytest.raises(ValueError):
        assemble_mixed_candidate((cohort,), tmp_path / "candidate", builder_revision="5" * 40)
    assert not (tmp_path / "candidate").exists()
    payload = record.model_dump(mode="json")
    payload["source_proof"]["archive_path"] = "wrong.jsonl"
    cohort.input_path.write_text(json.dumps(payload) + "\n")
    cohort = cohort.model_copy(update={"input_sha256": sha256(cohort.input_path.read_bytes()).hexdigest()})
    with pytest.raises(ValueError, match="archive proof"):
        assemble_mixed_candidate((cohort,), tmp_path / "candidate", builder_revision="5" * 40)


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


def _workplace_candidate(tmp_path):
    cohort, _ = _cohort(tmp_path)
    candidate = assemble_mixed_candidate((cohort,), tmp_path / "candidate", builder_revision="1" * 40)
    review = ReleaseReview(
        candidate_manifest_sha256=sha256((candidate / "manifest.json").read_bytes()).hexdigest(),
        rights_review_url="https://example.org/rights-review",
        harbor_evidence_urls=("https://example.org/trials",),
    )
    return candidate, review


def test_release_finalizer_rejects_unlisted_private_file(tmp_path):
    candidate, review = _workplace_candidate(tmp_path)
    (candidate / "private.json").write_text('{"private": true}')

    with pytest.raises(ValueError, match="unlisted or unsupported file"):
        finalize_mixed_candidate(candidate, tmp_path / "ready", review)

    assert not (tmp_path / "ready").exists()


def test_release_finalizer_writes_separate_ready_artifact(tmp_path):
    candidate, review = _workplace_candidate(tmp_path)
    ready = finalize_mixed_candidate(candidate, tmp_path / "ready", review)
    candidate_data = candidate / "data/workplace/workplace_train.jsonl"
    ready_data = ready / "data/workplace/workplace_train.jsonl"

    ready_manifest = json.loads((ready / "manifest.json").read_text())
    assert ready_manifest["publication_ready"] is True
    assert ready_manifest["publication_review"]["candidate_manifest_sha256"] == review.candidate_manifest_sha256
    assert sha256(ready_data.read_bytes()).hexdigest() == sha256(candidate_data.read_bytes()).hexdigest()
    assert json.loads((candidate / "manifest.json").read_text())["publication_ready"] is False
    assert "TaskCompendium Alpha 1 Candidate" not in (ready / "README.md").read_text()
    assert "[cc-by-4.0](https://creativecommons.org/licenses/by/4.0/)" in (ready / "README.md").read_text()


def test_release_finalizer_rejects_data_changed_after_review(tmp_path):
    candidate, review = _workplace_candidate(tmp_path)
    data_path = candidate / "data/workplace/workplace_train.jsonl"
    data_path.write_text(data_path.read_text() + " ")

    with pytest.raises(ValueError, match="does not match its manifest"):
        finalize_mixed_candidate(candidate, tmp_path / "ready", review)

    assert not (tmp_path / "ready").exists()


def test_regional_proof_requires_object_identity():
    unpinned = "2026.09.18.3#manifest-sha256=" + "a" * 64 + "#parquet-size=100#parquet-etag=#parquet-version-id="
    with pytest.raises(ValueError, match="immutable pin"):
        SourceProof(source_row="knowledge:row", input_file="source.parquet", input_object_pin=unpinned)


def test_public_action_task_serializes_terminal_functions_without_verifier():
    specification = TaskSpec(
        id="terminal-action",
        context=ConversationInput(events=(TextMessage(role="user", content="Choose a calendar."),)),
        environment_requirements=EnvironmentRequirements(),
        final_tools=(
            FunctionDefinition(
                name="choose_calendar",
                description="Select a calendar.",
                parameters={"type": "object", "properties": {"name": {"type": "string"}}},
                strict=True,
            ),
        ),
        answer_type=AnswerType.NATIVE_ACTION,
        verifier=exact_answer("secret-expected-calendar"),
        source=Source(dataset="source", revision="pinned", row="1", importer_revision="1"),
    )
    record = public_task(
        specification,
        FinalAction(id="action", require_call=True, max_calls=1),
        HarborEnvironmentConfig(),
    )
    payload = json.loads(record.model_dump_json())
    assert payload["final_tools"] == [
        {
            "name": "choose_calendar",
            "description": "Select a calendar.",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}}},
            "strict": True,
        }
    ]
    assert "verifier" not in payload
    assert "secret-expected-calendar" not in record.model_dump_json()
    assert "tool_choice" not in payload and "parallel_tool_calls" not in payload
