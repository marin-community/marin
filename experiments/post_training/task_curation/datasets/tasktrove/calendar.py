# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove calendar sources: schedule a conversation's events and return the final calendar as JSON.

The archive's verifier (``tests/verifier.py``, run by ``tests/test.sh``) accepts any schedule that
meets the final event constraints. It needs only the Python standard library (the source ran it in
``python:3.11-slim``), so it runs as archived with the grader packages (``GRADER_PACKAGES``). The
archive asks for the calendar in ``/app/answer.txt``; the task asks for it in the reply, which the
runtime writes to that file for the verifier. A source witness (``solution/answer.json``) is the
golden control; an archive whose witness is not a nonempty JSON list of events is a source defect.
"""

import json
from typing import Any

from taskcompendium.convert.answers import source_defect
from taskcompendium.convert.delivery import rewritten_task
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.controls import answer_reply
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import (
    Controls,
    ImportRejection,
    IntendedUse,
    NormalizedTask,
    RawRow,
    Reply,
)
from taskcompendium.runtime.resources import resource_bytes
from verifyit.spec import DEFAULT_OUTPUT

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.tasktrove.archives import TaskTroveConverter, tasktrove_source
from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import (
    archive_file,
    archive_resources,
    archive_script_grader,
)
from experiments.post_training.task_curation.pipeline import CurationRecipe, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

WITNESS_PATH = "solution/answer.json"
GRADER_FILES = ("tests/verifier.py", "tests/verifier_data.json")
DELIVERY = (
    "write your final calendar as a JSON list to `/app/answer.txt`",
    "return your final calendar as a JSON list in the assistant response",
)
REWRITE_REASON = "Adapt final calendar file delivery to the assistant response"

CALENDAR_RUBRIC = """
Read the complete source conversation, including rescheduling, removals, and unrelated messages. The task requests the
final schedule as JSON; it does not supply interactive calendar tools.

Compare every hidden expected event with the user requests: IDs, names, durations, permanent constraints, working
hours, removed events, and whether a schedule remains feasible.

Check that all events fit their allowed windows without overlap. A source oracle may be wrong; the existence of one
valid schedule does not establish agreement with the conversation.

Any schedule satisfying the actual final-state contract is acceptable. Different valid start times are not reference
conflicts. Before means ending at or before, and after means starting at or after, the named time.

Flag contradictory or absent inputs rather than inventing exceptions, durations, dates, attendees, or a tool
environment. Distinguish the source's final-schedule task from a full agent episode.
"""

IF_CALENDAR_RUBRIC = """
Read the complete conversation and derive the final state after additions, updates, removals, and refusals.

Compare event IDs, names, durations, windows, permanent constraints, and working hours with the hidden data.

Check that every required event fits its allowed window without overlaps; do not invent exceptions.

The contract accepts any feasible final schedule. Different valid times are not reference conflicts.

Before constrains event end and after constrains event start; compare these meanings with the wording.

A source witness proves only grader compatibility; missing witness controls imply verification uncertainty.

This requests a final JSON calendar, not an interactive tool episode; flag contradictory delivery promises.
"""


def calendar_defect(expected_events: Any) -> str | None:
    """Why the expected events cannot describe a calendar, or ``None`` when they can."""
    if not isinstance(expected_events, dict) or not expected_events:
        return "Expected nonempty event constraints"
    for key, event in expected_events.items():
        try:
            int(key)
        except (ValueError, TypeError) as error:
            return str(error)
        duration = event.get("duration") if isinstance(event, dict) else None
        if not isinstance(duration, int) or isinstance(duration, bool) or duration <= 0:
            return f"Malformed calendar event {key}"
    return None


def witness_defect(witness: bytes | None) -> str | None:
    """Why a source witness cannot be a final calendar, or ``None`` when it can or is absent."""
    if witness is None:
        return None
    try:
        events = json.loads(witness)
    except ValueError as error:
        return f"Witness is not JSON: {error}"
    if not isinstance(events, list) or not events or not all(isinstance(event, dict) for event in events):
        return "Witness must be a nonempty JSON list of events"
    return None


def convert_calendar(row: RawRow, context: ConversionContext) -> TaskSpec | NormalizedTask | ImportRejection:
    instruction, data = row.data["instruction"], row.data.get("verifier_data")
    if not instruction.strip() or not isinstance(data, dict):
        return source_defect("missing_input", "Instruction and calendar verifier data are required")
    defect = calendar_defect(data.get("expected_events"))
    if defect is not None:
        return source_defect("invalid_calendar", defect)
    defect = witness_defect(archive_file(row.data, WITNESS_PATH))
    if defect is not None:
        return source_defect("invalid_witness", defect)
    grader = archive_script_grader(
        row.data, required=GRADER_FILES, environment=required_grader_environment(context), answer_path=DEFAULT_OUTPUT
    )
    if isinstance(grader, ImportRejection):
        return grader
    task = TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction.replace(*DELIVERY)),)),
        environment_requirements=EnvironmentRequirements(),
        resources=archive_resources(row.data),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=grader,
        tags=("tool-use", "calendar", "scheduling", "state-tracking", "nemotron"),
    )
    return rewritten_task(task, original=instruction, reason=REWRITE_REASON)


def witness_text(task: TaskSpec) -> str | None:
    witness = next((resource for resource in task.resources.oracle if resource.path == WITNESS_PATH), None)
    return None if witness is None else resource_bytes(witness).decode(errors="replace")


def calendar_golden(task: TaskSpec) -> Reply | None:
    """The source witness schedule, when the archive ships one."""
    witness = witness_text(task)
    return None if witness is None else answer_reply(task, witness)


def sources() -> list[RlDataSource[CurationRecipe]]:
    sources = (
        (
            "tasktrove-calendar",
            "laion__nemotron-gym-agent-calendar-v2",
            CALENDAR_RUBRIC,
            SourceInfo(
                id="Task Trove:laion__nemotron-gym-agent-calendar-v2",
                title="laion/nemotron-gym-agent-calendar-v2",
                origin="Task Trove",
                family="tool-use",
                tags=("agentic", "multi-turn"),
                count=2699,
                notes="Deterministic schedule check. Templated; dedupe against calendar-v3.",
            ),
        ),
        (
            "tasktrove-if_calendar",
            "laion__nemotron-gym-instruction-following-calendar-v3",
            IF_CALENDAR_RUBRIC,
            SourceInfo(
                id="Task Trove:laion__nemotron-gym-instruction-following-calendar-v3",
                title="laion/nemotron-gym-instruction-following-calendar-v3",
                origin="Task Trove",
                family="instruction-following",
                tags=("agentic", "multi-turn"),
                count=5673,
                notes=(
                    "Deterministic overlap and constraint check. Synthetic and templated; dedupe against "
                    "agent-calendar-v2."
                ),
            ),
        ),
    )
    return [
        RlDataSource(
            info=info,
            pipeline=process_rows,
            config=CurationRecipe(
                name=name,
                source=tasktrove_source(config),
                convert=TaskTroveConverter(config, convert_calendar),
                version="1",
                intended_use=IntendedUse.TRAIN,
                rubric=rubric,
                controls=Controls(golden=calendar_golden),
                grader=GRADER_PACKAGES,
            ),
        )
        for name, config, rubric, info in sources
    ]
