# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from taskcompendium.models import Source, judge_spec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, NormalizedTask, RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets import s1_deepresearch

NATIVE = json.loads((Path(__file__).parent / "fixtures/s1_research.json").read_text())


def converted(native):
    row = RawRow(
        id="original-s1-row",
        source=Source(
            dataset=s1_deepresearch.REPO, revision=s1_deepresearch.REVISION, row="fixture", importer_revision="test"
        ),
        data=native,
    )
    result = s1_deepresearch.convert_research(row, ConversionContext({}, None))
    assert isinstance(result, NormalizedTask)
    return result


def rejected(native):
    row = RawRow(
        id="original-s1-row",
        source=Source(
            dataset=s1_deepresearch.REPO, revision=s1_deepresearch.REVISION, row="fixture", importer_revision="test"
        ),
        data=native,
    )
    result = s1_deepresearch.convert_research(row, ConversionContext({}, None))
    assert isinstance(result, ImportRejection)
    return result


def test_context_carries_only_the_source_task_definition_and_records_the_exclusion():
    result = converted(NATIVE)
    task = result.task
    system, user = task.context.events
    assert system.role == "system" and user.role == "user"
    assert system.content == NATIVE["messages"][0]["content"]
    assert user.content == NATIVE["messages"][1]["content"] == NATIVE["meta"]["question"]
    assert "30" not in system.content + user.content
    assert len(result.changes) == 1
    change = result.changes[0]
    assert "private provenance" in change.replacement
    assert s1_deepresearch.ANSWER_PROVENANCE_RECEIPT in change.reason
    assert s1_deepresearch.CARD_URL in change.reason
    assert task.labels["family"] == "closed-ended-multi-hop" and task.labels["language"] == "en"
    assert not task.resources.worker and not task.resources.oracle


def test_closed_ended_rows_bind_the_dataset_reference_and_no_trajectory_text():
    task = converted(NATIVE).task
    judge = judge_spec(task.grader)
    assert judge is not None and judge.references == ("30",)
    assert judge.question.startswith("Determine whether the candidate's final answer")
    assert judge.temperature == 0.0
    # The generated trajectory text must not leak into the grader inputs.
    grader_texts = " ".join(resource_bytes(resource).decode() for resource in task.resources.verifier)
    assert "private provenance" not in grader_texts
    assert "Fixture generated reasoning" not in grader_texts


def test_open_ended_rows_share_one_prompt_only_checklist_contract():
    native = {
        **NATIVE,
        "meta": {**NATIVE["meta"], "type": "Open-ended Exploration", "answer": None},
    }
    task = converted(native).task
    judge = judge_spec(task.grader)
    assert judge is not None and not judge.references and not judge.exact_gate
    assert judge.rubric == "checklist" and len(judge.criteria) == 5
    assert judge.model == "MiniMaxAI/MiniMax-M3-MXFP8"
    assert task.labels["family"] == "open-ended-exploration"
    resources = {resource.path: resource_bytes(resource) for resource in task.resources.verifier}
    conversation = json.loads(resources["conversation.json"])
    assert [event["role"] for event in conversation] == ["system", "user"]
    assert json.loads(resources["source-row.json"]) == native


def test_open_ended_answers_stay_in_the_rubric_family():
    native = {
        **NATIVE,
        "meta": {**NATIVE["meta"], "type": "Open-ended Exploration", "answer": "A reference the rubric family ignores."},
    }
    task = converted(native).task
    judge = judge_spec(task.grader)
    assert judge is not None and not judge.references and judge.rubric == "checklist"


def test_contract_defects_fail_closed():
    broken = {**NATIVE, "meta": {**NATIVE["meta"], "type": "Closed-ended Multi-hop Resolution", "answer": None}}
    assert rejected(broken).reason.startswith("missing_closed_ended_reference")
    assert rejected({**NATIVE, "messages": NATIVE["messages"][1:]}).reason.startswith("invalid_s1_native_opening")
    assert rejected({**NATIVE, "meta": {**NATIVE["meta"], "type": "Unspecified"}}).reason.startswith(
        "invalid_s1_task_type"
    )
    assert rejected({**NATIVE, "meta": {**NATIVE["meta"], "language": "fr"}}).reason.startswith(
        "invalid_s1_task_metadata"
    )
    assert rejected({"messages": NATIVE["messages"]}).reason.startswith("incomplete_s1_native_contract")


def test_source_module_binds_the_sealed_population():
    source = s1_deepresearch.source(("data.jsonl",))
    assert source.info.id == s1_deepresearch.SOURCE_ID
    assert source.info.count == 15000
    assert source.info.origin.value == "ScienceOne"
    assert source.config.source.revision == s1_deepresearch.REVISION
    assert "reference judge" in source.info.notes and "private provenance" in source.info.notes
