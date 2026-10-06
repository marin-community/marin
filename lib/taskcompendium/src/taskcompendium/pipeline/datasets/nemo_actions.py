# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Next-action prediction using the existing NeMo importer and comparator."""

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from verifyit.spec import PredictedActionSpec

from taskcompendium.grading import resolve_verifier
from taskcompendium.importers.nemo_predicted_action import canonical_sha256, import_row
from taskcompendium.models import TaskSpec
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
from taskcompendium.pipeline.verification import verify_task


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    data = dict(row.data)
    try:
        task, _ = import_row(data, expected_sha256=canonical_sha256(data))
    except ValueError as error:
        return ImportRejection(reason="unsupported_action", detail=str(error))
    return task.model_copy(update={"id": row.id, "source": row.source})


def verification_report(task: TaskSpec) -> VerificationReport:
    """Check reference arguments against public schemas before comparator controls."""
    functions = {function.name: function for function in task.final_tools}
    verifier = resolve_verifier(task.verifier)
    assert isinstance(verifier, PredictedActionSpec)
    checks = []
    for call in verifier.expected_calls:
        schema = functions[call.name].parameters
        validator = validator_for(schema)
        try:
            validator.check_schema(schema)
        except SchemaError as error:
            checks.append(CheckResult(check="tool_schema", status=CheckStatus.FAIL, detail=str(error)))
            continue
        errors = list(validator(schema).iter_errors(call.arguments))
        checks.append(
            CheckResult(
                check="reference_arguments",
                status=CheckStatus.FAIL if errors else CheckStatus.PASS,
                detail="; ".join(error.message for error in errors) or f"Arguments satisfy {call.name}'s public schema",
            )
        )
    return VerificationReport(checks=[*checks, *verify_task(task)])


RUBRIC = ReviewRubric(
    "next-action",
    "2",
    (
        "Judge the next action from the complete conversation and advertised tool schemas.",
        "Check that required arguments are grounded in the conversation, without guessing hidden values.",
        "Flag a reference that chooses one of several equally defensible actions under an exact-call grader.",
        "Historical tool results are context. This task predicts a call; it does not execute it.",
        "Check that each reference argument satisfies the advertised tool's parameter schema and that "
        "authentication, consent, and ordering prerequisites are grounded in the visible history.",
        "Do not treat a plausible guessed identifier, date, or location as grounded. Flag absent inputs "
        "and contradictory tool outputs while allowing explicit relative dates tied to a supplied date.",
    ),
)


def pipeline() -> TaskPipeline:
    return TaskPipeline(
        normalize=normalize,
        rubric=RUBRIC,
        check_suite=CheckSuite(
            id="next-action-schema-and-controls", revision="1", parameters={}, run=verification_report
        ),
    )
