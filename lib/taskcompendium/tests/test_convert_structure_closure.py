# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native row conversion follows curation identity and preserves closure semantics."""

import json
from dataclasses import replace
from functools import partial

import pytest

from taskcompendium.convert.structure_closure import structure_closure_task
from taskcompendium.grading import grade_answer
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    ConversationTrace,
    EnvironmentRequirements,
    GradingAttempt,
    NoGrader,
    PlainText,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
    TokenizedInput,
)
from taskcompendium.pipeline.conversion import convert_row
from taskcompendium.pipeline.fingerprints import semantic_digest
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import NormalizedTask, RawRow
from taskcompendium.runtime.resources import inline_resource

TOKENIZER = "org/replay-tokenizer"
TOKENIZER_REVISION = "a" * 40
TOKENS = (128000, 42, 128005)
CONVERT = partial(
    structure_closure_task,
    tokenizer_model=TOKENIZER,
    tokenizer_revision=TOKENIZER_REVISION,
    terminal_markers=("<|eot_id|>", "<|end_of_text|>"),
)


@pytest.fixture
def native_row():
    def row(*, identifier="native-id", reference='{"a":[1,2]}'):
        extras = {
            "prompt_token_ids": list(TOKENS),
            "prefix": '{"a":[1',
            "expected": reference,
            "format": "json",
            "family": "json_array",
            "control": "continue_required",
        }
        config = inline_resource("config.json", json.dumps({"extras": extras, "config": {}}).encode())
        native = TaskSpec(
            id=identifier,
            source=Source(dataset="structure-closure", revision="original", row=identifier, importer_revision="old"),
            context=ConversationInput(events=(TextMessage(role="user", content="Continue the supplied prefix."),)),
            environment_requirements=EnvironmentRequirements(),
            answer_type=AnswerType.TEXT,
            answer_format=PlainText(),
            grader=NoGrader(reason="Replaced by the native session grader in the source row"),
            resources=ResourceGroups(verifier=(config,)),
        ).model_dump(mode="json")
        native["grader"] = {"kind": "session"}
        payload = json.dumps(
            {
                "task": native,
                "runtime": {"task_machine": None, "verifier_machine": None},
                "session": {"task_session": "structure_closure", "max_turns": 1},
            },
            indent=2,
        )
        return RawRow(
            "canonical-id",
            Source(dataset="published/release", revision="b" * 40, row="train.parquet:7", importer_revision="1"),
            {"lowered_task_spec": payload, "teacher_route": "structural"},
        )

    return row


def test_native_converter_preserves_pinned_input_private_resources_and_pipeline_audit(native_row):
    row = native_row()
    # Exercise the standard conversion boundary, which enforces supplied row identity.
    converted = convert_row(row, lambda raw, _context: CONVERT(raw), ConversionContext({}, None))
    assert isinstance(converted, NormalizedTask)
    task = converted.task
    assert task.id == row.id and task.source == row.source
    assert isinstance(task.context, TokenizedInput)
    assert (task.context.token_ids, task.context.tokenizer_model, task.context.tokenizer_revision) == (
        TOKENS,
        TOKENIZER,
        TOKENIZER_REVISION,
    )
    native = json.loads(row.data["lowered_task_spec"])["task"]
    assert task.resources.model_dump(mode="json") == native["resources"]
    audit = {change.field: change.original for change in converted.changes}
    assert audit["id"] == native["id"]
    assert json.loads(audit["source"]) == native["source"]
    correct = GradingAttempt(
        ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=",2]}")))
    )
    premature = GradingAttempt(
        ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content="]}")))
    )
    assert grade_answer(task, correct).reward == 1.0
    assert grade_answer(task, premature).reward == 0.0


def test_native_aliases_deduplicate_but_private_references_remain_distinct(native_row):
    rows = [native_row(), native_row(identifier="another-native-id"), native_row(reference='{"a":[1,3]}')]
    tasks = []
    for index, row in enumerate(rows):
        converted = CONVERT(
            replace(row, id=f"canonical-{index}", source=row.source.model_copy(update={"row": f"train:{index}"}))
        )
        assert isinstance(converted, NormalizedTask)
        tasks.append(converted.task)
    digests = [semantic_digest(task, include_reference=True) for task in tasks]
    assert digests[0] == digests[1]
    assert digests[0] != digests[2]
    assert semantic_digest(tasks[0], include_reference=False) == semantic_digest(tasks[2], include_reference=False)
