# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task-format detection classifies one row's payload from its structure, never guessing."""

import base64
from functools import partial
from pathlib import Path

import pytest
from finestore.rl.curation import TaskFormat
from fray.local_backend import LocalClient
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    ExecutionRequirements,
    InlineFile,
    NoGrader,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    Source,
    TaskResource,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.conversion import ConversionContext, RawRow, convert_raw_row
from taskcompendium.pipeline.models import NormalizedTask
from taskcompendium.pipeline.source_processing import write_conversion
from taskcompendium.pipeline.task_format import (
    AmbiguousTaskFormatError,
    FormatRow,
    MixedTaskFormatSourceError,
    TaskFormatMismatchError,
    UndetectableTaskFormatError,
    detect_task_format,
)


def inline(path: str, content: str) -> TaskResource:
    return TaskResource(path=path, source=InlineFile(content_base64=base64.b64encode(content.encode()).decode()))


def base_task(**overrides) -> TaskSpec:
    fields = dict(
        id="fixture-task",
        context=ConversationInput(events=(TextMessage(role="user", content="Do the task."),)),
        environment_requirements=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=f"fixture@sha256:{'0' * 64}"
        ),
        execution_requirements=ExecutionRequirements(
            cpus=1, memory_mb=2048, storage_mb=2048, agent_timeout_sec=300, verifier_timeout_sec=60
        ),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        source=Source(dataset="fixture", revision="pinned", row="row:1", importer_revision="1"),
        resources=ResourceGroups(),
    )
    fields.update(overrides)
    return TaskSpec(**fields)


def harbor_task() -> TaskSpec:
    return base_task(
        grader=NoGrader(reason="The original package verifies itself"),
        tags=("harbor",),
        resources=ResourceGroups(
            verifier=(
                inline("task.toml", "[environment]\ndocker_image = 'fixture-image'\n"),
                inline("instruction.md", "Solve the task.\n"),
                inline("tests/test.sh", "exit 0\n"),
            )
        ),
    )


def taskcompendium_task() -> TaskSpec:
    """A TaskSpec row carrying a task.toml verifier document but no Harbor package."""
    return base_task(
        grader=ScriptGrader(
            argv=("true",),
            environment=EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=f"fixture@sha256:{'0' * 64}"
            ),
        ),
        resources=ResourceGroups(verifier=(inline("task.toml", "[verifier]\nmode = 'separate'\n"),)),
    )


def native_task() -> TaskSpec:
    """A TaskSpec row with no verifier resources at all."""
    return base_task(
        grader=ScriptGrader(
            argv=("true",),
            environment=EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=f"fixture@sha256:{'0' * 64}"
            ),
        ),
    )


def blend_record() -> dict:
    return {
        "dataset": "fixture-blend",
        "responses_create_params": {"input": [{"role": "user", "content": "Do the task."}]},
    }


def agent_blend_record() -> dict:
    return {
        "agent_ref": {"name": "fixture-agent"},
        "responses_create_params": {"input": [{"role": "user", "content": "Do the task."}]},
    }


def carrier_row(task: TaskSpec) -> dict:
    serialized = task.model_dump_json()
    return {"task_spec": serialized, "task_sha256": "0" * 64}


@pytest.mark.parametrize(
    "row,expected",
    [
        (FormatRow(raw=carrier_row(harbor_task()), task=harbor_task()), TaskFormat.HARBOR),
        (FormatRow(raw=carrier_row(taskcompendium_task()), task=taskcompendium_task()), TaskFormat.TASKSPEC),
        (FormatRow(raw=blend_record(), task=None), TaskFormat.MARINSKYRL),
        (FormatRow(raw=blend_record(), task=native_task()), TaskFormat.MARINSKYRL),
        (FormatRow(raw=agent_blend_record(), task=None), TaskFormat.MARINSKYRL),
        (FormatRow(raw={"record": "answer-row"}, task=native_task()), TaskFormat.TASKSPEC),
        # The carrier is transport: a frozen cohort row whose task has no task directory
        # is a native TaskSpec row, whatever encoding stores it.
        (
            FormatRow(raw={"lowered_task_spec": {"task": native_task().model_dump_json(), "runtime": {}}}),
            TaskFormat.TASKSPEC,
        ),
    ],
)
def test_one_fixture_row_per_format_detects_exactly_that_format(row: FormatRow, expected: TaskFormat):
    assert detect_task_format(row) is expected


def test_historical_labels_parse_onto_their_shape():
    assert TaskFormat("taskcompendium") is TaskFormat.TASKSPEC
    assert TaskFormat("native_taskspec") is TaskFormat.TASKSPEC
    assert [format_.value for format_ in TaskFormat] == ["harbor", "taskspec", "marinskyrl"]


def test_ambiguous_row_raises_instead_of_guessing():
    with pytest.raises(AmbiguousTaskFormatError) as raised:
        detect_task_format(FormatRow(raw=blend_record(), task=harbor_task()))
    assert "marinskyrl" in str(raised.value) and "harbor" in str(raised.value)


def test_undetectable_row_raises():
    with pytest.raises(UndetectableTaskFormatError):
        detect_task_format(FormatRow(raw={"unrelated": "record"}, task=None))


def test_corrupt_carrier_row_is_undetectable_not_a_guess():
    with pytest.raises(UndetectableTaskFormatError):
        detect_task_format(FormatRow(raw={"task_spec": "{not json"}, task=None))


def _conversion_rows(*tasks: TaskSpec) -> Dataset:
    templates = {f"{task.id}-{index}": task for index, task in enumerate(tasks)}
    rows = [
        RawRow(
            task_id,
            Source(dataset="fixture", revision="pinned", row=f"row:{index}", importer_revision="1"),
            {"answer": "red"},
        )
        for index, task_id in enumerate(templates)
    ]

    def convert(row: RawRow, _context: ConversionContext) -> NormalizedTask:
        template = templates[row.id]
        return NormalizedTask(template.model_copy(update={"id": row.id, "source": row.source}), ())

    return Dataset.from_list(rows).map(
        partial(convert_raw_row, convert=convert, context=ConversionContext(inputs={}, grader_environment=None))
    )


def _run_conversion(rows: Dataset, declared: TaskFormat, tmp_path: Path):
    output = tmp_path / "output"
    with ZephyrContext(client=LocalClient(), max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        return write_conversion(rows, context, str(output), declared_task_format=declared)


def test_conversion_rejects_a_source_whose_rows_span_task_formats(tmp_path: Path):
    with pytest.raises(MixedTaskFormatSourceError) as raised:
        _run_conversion(_conversion_rows(native_task(), harbor_task()), TaskFormat.TASKSPEC, tmp_path)
    message = str(raised.value)
    assert "taskspec=1" in message and "harbor=1" in message
    assert "split the source into one source per task format" in message


def test_conversion_rejects_a_declared_format_that_differs_from_the_detected_payload(tmp_path: Path):
    with pytest.raises(TaskFormatMismatchError) as raised:
        _run_conversion(_conversion_rows(native_task()), TaskFormat.HARBOR, tmp_path)
    assert "harbor" in str(raised.value) and "taskspec" in str(raised.value)


def test_conversion_accepts_rows_whose_detected_format_matches_the_declaration(tmp_path: Path):
    variant = native_task().model_copy(
        update={
            "id": "fixture-task-variant",
            "context": ConversationInput(events=(TextMessage(role="user", content="A different task."),)),
        }
    )
    result = _run_conversion(_conversion_rows(native_task(), variant), TaskFormat.TASKSPEC, tmp_path)
    assert result.converted_rows == 2
