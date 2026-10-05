# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize actual TaskTrove calendar conversations and final-schedule contracts."""

import json
from pathlib import Path

from taskcompendium.grader import script_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ResourceGroups,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.datasets.grader_scripts.schedule import normalized_name, parse_time
from taskcompendium.pipeline.datasets.reasoning_tasks import snapshot_file
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    ImportRejection,
    RawRow,
    ReviewRubric,
    TaskPipeline,
    VerificationReport,
)
from taskcompendium.pipeline.verification import verify_witness
from taskcompendium.runtime.resources import inline_resource, resource_bytes

SCHEDULE_SCRIPT = (Path(__file__).with_name("grader_scripts") / "schedule.py").read_bytes()

WITNESS_PATH = "/control/calendar-answer.json"
RUBRIC = ReviewRubric(
    id="tasktrove-calendar-feasibility",
    version="1",
    criteria=(
        "Read the complete source conversation, including rescheduling, removals, and unrelated messages. "
        "The task requests the final schedule as JSON; it does not supply interactive calendar tools.",
        "Compare every private expected event with the user requests: IDs, names, durations, permanent "
        "constraints, working hours, removed events, and whether a schedule remains feasible.",
        "Check that all events fit their allowed windows without overlap. A source oracle may be wrong; "
        "the existence of one valid schedule does not establish agreement with the conversation.",
        "Any schedule satisfying the actual final-state contract is acceptable. Different valid start "
        "times are not reference conflicts. Before means ending at or before, and after means starting "
        "at or after, the named time.",
        "Flag contradictory or absent inputs rather than inventing exceptions, durations, dates, attendees, "
        "or a tool environment. Distinguish the source's final-schedule task from a full agent episode.",
    ),
)


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    instruction, data = row.data.get("instruction"), row.data.get("verifier_data")
    if not isinstance(instruction, str) or not instruction.strip() or not isinstance(data, dict):
        return ImportRejection(reason="missing_input", detail="Instruction and calendar verifier data are required")
    witness = snapshot_file(row, "solution/answer.json")
    expected_events = data.get("expected_events")
    if not isinstance(expected_events, dict) or not expected_events:
        return ImportRejection(reason="invalid_calendar", detail="Expected nonempty event constraints")
    for key, event in expected_events.items():
        try:
            int(key)
        except (ValueError, TypeError) as error:
            return ImportRejection(reason="invalid_calendar", detail=str(error))
        if not isinstance(event, dict):
            return ImportRejection(reason="invalid_calendar", detail=f"Malformed calendar event {key}")
        duration = event.get("duration")
        if (
            not isinstance(duration, int)
            or isinstance(duration, bool)
            or duration <= 0
            or normalized_name(event.get("event_name")) is None
            or parse_time(event.get("min_time")) is None
            or parse_time(event.get("max_time")) is None
        ):
            return ImportRejection(reason="invalid_calendar", detail=f"Malformed calendar event {key}")
    package = script_package(SCHEDULE_SCRIPT, {"expected_events": expected_events})
    instruction = instruction.replace(
        "write your final calendar as a JSON list to `/app/answer.txt`",
        "return your final calendar as a JSON list in the assistant response",
    )
    return TaskSpec(
        id=row.id,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=package.verifier,
        resources=ResourceGroups(
            oracle=(inline_resource(WITNESS_PATH.lstrip("/"), witness),) if witness is not None else (),
            verifier=package.resources,
        ),
        source=row.source,
    )


def verification_report(task: TaskSpec) -> VerificationReport:
    witness = next((resource for resource in task.resources.oracle if resource.path == WITNESS_PATH.lstrip("/")), None)
    if witness is None:
        return VerificationReport(
            checks=[
                CheckResult(
                    check="calendar_witness",
                    status=CheckStatus.UNSUPPORTED,
                    detail="No source-supplied solution/answer.json witness; no schedule was invented",
                )
            ]
        )
    try:
        witness_json = resource_bytes(witness).decode()
        events = json.loads(witness_json)
    except ValueError as error:
        return VerificationReport(
            checks=[CheckResult(check="calendar_witness", status=CheckStatus.FAIL, detail=str(error))]
        )
    if not isinstance(events, list) or not events or not isinstance(events[0], dict):
        return VerificationReport(
            checks=[CheckResult(check="calendar_witness", status=CheckStatus.FAIL, detail="Malformed source witness")]
        )
    # Dropping a required source event supplies a genuinely invalid schedule without a guessed slot.
    negative = json.dumps(events[1:])
    return VerificationReport(checks=verify_witness(task, witness_json, negative))


def pipeline() -> TaskPipeline:
    """Build the source normalization and review policy."""
    return TaskPipeline(
        normalize=normalize,
        rubric=RUBRIC,
        check_suite=CheckSuite(
            id="calendar-source-witness-controls", revision="1", parameters={}, run=verification_report
        ),
    )
