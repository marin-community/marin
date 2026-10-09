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
from dataclasses import replace
from typing import Any

from taskcompendium.convert.answers import source_defect
from taskcompendium.convert.delivery import rewritten_task
from taskcompendium.convert.tasktrove import ANSWER_PATH, archive_file, archive_resources, archive_script_grader
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

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.tasktrove.archives import tasktrove_source
from experiments.post_training.task_curation.pipeline import RlDataPipeline, ShellSim
from experiments.post_training.task_curation.source import DataSourceMetadata, RlDataSource

TASKTROVE_METADATA = DataSourceMetadata(
    id="",
    name="",
    origin="Task Trove",
    url="https://huggingface.co/datasets/open-athena/task-trove",
    dataset_id="open-athena/task-trove",
    revision="ec049a4fb541ffbe5bbccb803e826563f5718dbf",
    revised_at="2026-10-08T09:34:47.000Z",
    dataset_revision="9065fa568394f286dab0081e43dc76fc87c48984",
    verifier_revision=None,
    environment="Harbor",
    type="Agentic",
    turns="Multi-turn",
    count_basis="Released Harbor tasks: manifest by_source.converted",
    count_precision="exact",
    count_url=(
        "https://huggingface.co/datasets/open-athena/task-trove/blob/ec049a4fb541ffbe5bbccb803"
        "e826563f5718dbf/manifest.json"
    ),
    benchmark_basis="Release manifest does not designate benchmarks",
    family_basis="Task Trove release manifest source_verdicts.family",
    family_url=(
        "https://huggingface.co/datasets/open-athena/task-trove/blob/ec049a4fb541ffbe5bbccb803"
        "e826563f5718dbf/manifest.json"
    ),
    classification_basis="Task Trove tasks run as Agentic interactions in Harbor",
    canonical_url="https://huggingface.co/datasets/open-athena/task-trove",
    provenance_url=(
        "https://huggingface.co/datasets/open-athena/task-trove/blob/ec049a4fb541ffbe5bbccb8"
        "03e826563f5718dbf/manifest.json"
    ),
    verification="script",
    snapshot_safe=True,
    snapshot_safety_basis="Accepted as snapshot-safe in the Atlas inventory",
    upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
    modes=("script",),
    recorded_at="2026-10-08",
)

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
        row.data, required=GRADER_FILES, environment=required_grader_environment(context), answer_path=ANSWER_PATH
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
    )
    return rewritten_task(task, original=instruction, reason=REWRITE_REASON)


def witness_text(task: TaskSpec) -> str | None:
    witness = next((resource for resource in task.resources.oracle if resource.path == WITNESS_PATH), None)
    return None if witness is None else resource_bytes(witness).decode(errors="replace")


def calendar_golden(task: TaskSpec) -> Reply | None:
    """The source witness schedule, when the archive ships one."""
    witness = witness_text(task)
    return None if witness is None else answer_reply(task, witness)


def sources() -> list[RlDataSource]:
    sources = (
        (
            "tasktrove-calendar",
            "laion__nemotron-gym-agent-calendar-v2",
            CALENDAR_RUBRIC,
            replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__nemotron-gym-agent-calendar-v2",
                name="laion__nemotron-gym-agent-calendar-v2",
                display_name="laion/nemotron-gym-agent-calendar-v2",
                family="tool-use",
                task_count=2699,
                notes="Deterministic schedule check. Templated; dedupe against calendar-v3.",
                canonical_source="laion/nemotron-gym-agent-calendar-v2",
                upstream_repository="laion/nemotron-gym-agent-calendar-v2",
                upstream_url="https://huggingface.co/datasets/laion/nemotron-gym-agent-calendar-v2",
                input_count=2699,
            ),
        ),
        (
            "tasktrove-if_calendar",
            "laion__nemotron-gym-instruction-following-calendar-v3",
            IF_CALENDAR_RUBRIC,
            replace(
                TASKTROVE_METADATA,
                id="Task Trove:laion__nemotron-gym-instruction-following-calendar-v3",
                name="laion__nemotron-gym-instruction-following-calendar-v3",
                display_name="laion/nemotron-gym-instruction-following-calendar-v3",
                family="instruction-following",
                task_count=5673,
                notes=(
                    "Deterministic overlap and constraint check. Synthetic and templated; dedupe "
                    "against agent-calendar-v2."
                ),
                canonical_source="laion/nemotron-gym-instruction-following-calendar-v3",
                upstream_repository="laion/nemotron-gym-instruction-following-calendar-v3",
                upstream_url="https://huggingface.co/datasets/laion/nemotron-gym-instruction-following-calendar-v3",
                input_count=5673,
            ),
        ),
    )
    return [
        RlDataSource(
            metadata=metadata,
            pipeline=RlDataPipeline(
                name=name,
                source=tasktrove_source(config),
                convert=convert_calendar,
                version="1",
                environment=ShellSim(),
                intended_use=IntendedUse.TRAIN,
                rubric=rubric,
                controls=Controls(golden=calendar_golden),
                grader=GRADER_PACKAGES,
            ),
        )
        for name, config, rubric, metadata in sources
    ]
