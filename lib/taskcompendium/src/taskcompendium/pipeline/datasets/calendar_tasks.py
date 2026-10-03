# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize actual TaskTrove calendar conversations and final-schedule contracts."""

import json

from pydantic import ValidationError

from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ResourceGroups,
    TaskSpec,
    TextMessage,
    VerifierSpec,
)
from taskcompendium.pipeline.datasets.instruction_following import REVISION
from taskcompendium.pipeline.datasets.reasoning_tasks import snapshot_file
from taskcompendium.pipeline.datasets.source_definitions import TASKTROVE_DATASET, tasktrove_inputs
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    DatasetRecipe,
    HFSource,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
    VerificationReport,
)
from taskcompendium.pipeline.verification import verify_witness
from taskcompendium.runtime.resources import inline_resource, resource_bytes
from taskcompendium.verifiers.base import VerifierKind
from taskcompendium.verifiers.schedule import ScheduleAnswerVerifier

CONFIG = "laion__nemotron-gym-agent-calendar-v2"
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
    try:
        verifier = ScheduleAnswerVerifier.model_validate({"expected_events": data.get("expected_events")})
    except (ValidationError, ValueError) as error:
        return ImportRejection(reason="invalid_calendar", detail=str(error))
    instruction = instruction.replace(
        "write your final calendar as a JSON list to `/app/answer.txt`",
        "return your final calendar as a JSON list in the assistant response",
    )
    return TaskSpec(
        id=row.id,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=VerifierSpec(kind=VerifierKind.SCHEDULE_ANSWER, parameters_json=verifier.model_dump_json()),
        resources=ResourceGroups(
            oracle=(inline_resource(WITNESS_PATH.lstrip("/"), witness),) if witness is not None else ()
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


def recipe() -> DatasetRecipe:
    return DatasetRecipe(
        name="tasktrove-calendar",
        version="tasktrove-calendar-v1",
        source=HFSource(TASKTROVE_DATASET, REVISION, CONFIG, "train"),
        inputs=tasktrove_inputs(CONFIG, REVISION),
        normalize=normalize,
        rubric=RUBRIC,
        intended_use=IntendedUse.TRAIN,
        check_suite=CheckSuite(
            id="calendar-source-witness-controls", revision="1", parameters={}, run=verification_report
        ),
    )
