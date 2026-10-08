# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize actual TaskTrove calendar conversations and final-schedule contracts."""

import json
import tomllib

from taskcompendium.datasets.reasoning_tasks import snapshot_file
from taskcompendium.datasets.source_definitions import archive_resources
from taskcompendium.grader import GraderPackage
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    ResourceGroups,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    ImportFailureKind,
    ImportRejection,
    RawRow,
    ReviewRubric,
    TaskPolicy,
    VerificationReport,
)
from taskcompendium.pipeline.verification import answer_checks, verify_witness
from taskcompendium.runtime.resources import resource_bytes

WITNESS_PATH = "solution/answer.json"
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
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="missing_input",
            detail="Instruction and calendar verifier data are required",
        )
    expected_events = data.get("expected_events")
    if not isinstance(expected_events, dict) or not expected_events:
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT, reason="invalid_calendar", detail="Expected nonempty event constraints"
        )
    for key, event in expected_events.items():
        try:
            int(key)
        except (ValueError, TypeError) as error:
            return ImportRejection(kind=ImportFailureKind.SOURCE_DEFECT, reason="invalid_calendar", detail=str(error))
        if not isinstance(event, dict):
            return ImportRejection(
                kind=ImportFailureKind.SOURCE_DEFECT, reason="invalid_calendar", detail=f"Malformed calendar event {key}"
            )
        duration = event.get("duration")
        if not isinstance(duration, int) or isinstance(duration, bool) or duration <= 0:
            return ImportRejection(
                kind=ImportFailureKind.SOURCE_DEFECT, reason="invalid_calendar", detail=f"Malformed calendar event {key}"
            )
    files = row.data.get("files")
    if (
        row.data.get("archive_links")
        or not isinstance(files, dict)
        or any(
            path not in files for path in ("tests/test.sh", "tests/verifier.py", "tests/verifier_data.json", "task.toml")
        )
    ):
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="missing_original_calendar_command",
            detail="The source row needs its original test script, verifier, data, and task.toml without archive links",
        )
    source_task = snapshot_file(row, "task.toml")
    assert source_task is not None
    source_timeout = float(tomllib.loads(source_task.decode())["verifier"]["timeout_sec"])
    archive = archive_resources(row.data)
    package = GraderPackage(
        NoGrader(
            reason="The TaskTrove calendar test script has no pinned grader image",
            contract={
                "argv": ["bash", "/tests/test.sh"],
                "cwd": "/",
                "reward_path": "/logs/verifier/reward.txt",
                "timeout": source_timeout,
            },
        ),
        archive.verifier,
    )
    instruction = instruction.replace(
        "write your final calendar as a JSON list to `/app/answer.txt`",
        "return your final calendar as a JSON list in the assistant response",
    )
    return TaskSpec(
        id=row.id,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=package.grader,
        resources=ResourceGroups(
            oracle=archive.oracle,
            worker=archive.worker,
            verifier=package.resources,
        ),
        source=row.source,
    )


def verification_report(task: TaskSpec) -> VerificationReport:
    witness = next((resource for resource in task.resources.oracle if resource.path == WITNESS_PATH), None)
    if witness is None:
        return VerificationReport(
            checks=[
                *answer_checks(task, (("empty", "", 0.0),)),
                CheckResult(
                    check="calendar_witness",
                    status=CheckStatus.SKIPPED,
                    detail="No source-supplied solution/answer.json witness; no schedule was invented",
                ),
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


def policy() -> TaskPolicy:
    """Build the source normalization and review policy."""
    return TaskPolicy(
        normalize=normalize,
        rubric=RUBRIC,
        check_suite=CheckSuite(
            id="calendar-source-witness-controls", revision="1", parameters={}, run=verification_report
        ),
    )
