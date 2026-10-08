# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove calendar sources: schedule a conversation's events and return the final calendar as JSON.

The archive's verifier (``tests/verifier.py``, run by ``tests/test.sh``) accepts any schedule that
meets the final event constraints. It needs only the Python 3.11 standard library, matching the
source's ``python:3.11-slim`` environment, so it runs in the executable math image. A source
witness (``solution/answer.json``) is the golden control, and the witness without its first event
is the negative.
"""

import json
import tomllib
from typing import Any

from taskcompendium.convert.answers import source_defect, unsupported
from taskcompendium.convert.delivery import rewritten_task
from taskcompendium.convert.tasktrove import archive_file, archive_resources
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    FileReward,
    PlainText,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.controls import answer_reply, wrong_reply
from taskcompendium.pipeline.models import (
    Controls,
    ImportRejection,
    IntendedUse,
    NormalizedTask,
    RawRow,
    Reply,
)
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.tasktrove import tasktrove_source
from experiments.post_training.task_curation.images import EXECUTABLE_MATH_IMAGE
from experiments.post_training.task_curation.pipeline import RlDataPipeline, ShellSim

WITNESS_PATH = "solution/answer.json"
GRADER_FILES = ("tests/test.sh", "tests/verifier.py", "tests/verifier_data.json", "task.toml")
REWARD = FileReward(files=(RewardFile(path="/logs/verifier/reward.txt", format=RewardFileFormat.NUMBER),))
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


def convert_calendar(row: RawRow) -> TaskSpec | NormalizedTask | ImportRejection:
    instruction, data = row.data["instruction"], row.data.get("verifier_data")
    if not instruction.strip() or not isinstance(data, dict):
        return source_defect("missing_input", "Instruction and calendar verifier data are required")
    defect = calendar_defect(data.get("expected_events"))
    if defect is not None:
        return source_defect("invalid_calendar", defect)
    if row.data.get("archive_links") or any(path not in row.data["files"] for path in GRADER_FILES):
        return unsupported(
            "missing_original_calendar_command",
            "The source row needs its original test script, verifier, data, and task.toml without archive links",
        )
    task_toml = archive_file(row.data, "task.toml")
    assert task_toml is not None
    archive = archive_resources(row.data)
    grader = ScriptGrader(
        argv=("bash", "/tests/test.sh"),
        cwd="/",
        environment=EXECUTABLE_MATH_IMAGE.requirements(),
        answer_path="/app/answer.txt",
        reward=REWARD,
        timeout=float(tomllib.loads(task_toml.decode())["verifier"]["timeout_sec"]),
    )
    task = TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction.replace(*DELIVERY)),)),
        environment_requirements=EnvironmentRequirements(),
        resources=ResourceGroups(worker=archive.worker, verifier=archive.verifier, oracle=archive.oracle),
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


def calendar_negative(task: TaskSpec) -> Reply:
    """The witness without its first event: a required event is missing, so no slot has to be guessed."""
    witness = witness_text(task)
    try:
        events = json.loads(witness) if witness is not None else None
    except ValueError:
        events = None
    if not isinstance(events, list) or not events or not isinstance(events[0], dict):
        return wrong_reply(task)
    return answer_reply(task, json.dumps(events[1:]))


def pipelines() -> list[RlDataPipeline]:
    sources = (
        ("tasktrove-calendar", "laion__nemotron-gym-agent-calendar-v2", CALENDAR_RUBRIC),
        ("tasktrove-if_calendar", "laion__nemotron-gym-instruction-following-calendar-v3", IF_CALENDAR_RUBRIC),
    )
    return [
        RlDataPipeline(
            name=name,
            source=tasktrove_source(config),
            convert=convert_calendar,
            version="1",
            environment=ShellSim(),
            intended_use=IntendedUse.TRAIN,
            rubric=rubric,
            controls=Controls(golden=calendar_golden, negative=calendar_negative),
            atlas_id=f"Task Trove:{config}",
        )
        for name, config, rubric in sources
    ]
