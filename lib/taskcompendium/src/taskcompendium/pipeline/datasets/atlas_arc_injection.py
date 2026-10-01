# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned TaskTrove ARC and indirect-injection snapshot adapters."""

import base64
import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from taskcompendium.grading import GradingAttempt, Outcome
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ResourceVisibility,
    TaskSpec,
    TextMessage,
    VerifierKind,
    VerifierSpec,
    task_resource,
)
from taskcompendium.pipeline.datasets.instruction_following import REVISION
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    DatasetRecipe,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
    SnapshotSource,
    VerificationReport,
)
from taskcompendium.pipeline.verification import PLAIN, verify_witness
from taskcompendium.verifiers.arc_injection import ArcGridVerifier, ArcTransformVerifier, IndirectInjectionVerifier

CONFIGS = {
    "arc_transductive": "laion__nemotron-gym-arc-agi-transductive-v3",
    "arc_inductive": "laion__nemotron-gym-arc-agi-python-inductive-v2",
    "indirect_injection": "laion__nemotron-gym-agentic-indirect-prompt-injection-v3",
}
SUBMISSION_SECTION = "\n## Submitting your answer (IMPORTANT)"


def response_instruction(instruction: str, name: str) -> str:
    """Replace terminal delivery while preserving the underlying prompt and formats."""
    instruction = instruction.split(SUBMISSION_SECTION, 1)[0]
    if name == "arc_transductive":
        instruction = instruction.replace(
            "write your final answer to the path `/app/answer.txt`", "return your final answer in the assistant response"
        ).replace("the entire content of `/app/answer.txt`", "the entire assistant response")
    else:
        instruction = instruction.replace(
            "Write a single JSON object describing your next action to `/app/answer.txt`",
            "Return a single JSON object describing your next action in the assistant response",
        )
    return instruction.strip()


def normalize(row: RawRow, name: str) -> TaskSpec | ImportRejection:
    """Preserve source grader evidence; normalize only recognized file delivery."""
    instruction, data, encoded_files = (
        row.data.get("instruction"),
        row.data.get("verifier_data"),
        row.data.get("files"),
    )
    if not isinstance(instruction, str) or not instruction.strip() or not isinstance(data, dict):
        return ImportRejection(reason="missing_input", detail="Instruction and verifier_data are required")
    if not isinstance(encoded_files, dict):
        return ImportRejection(reason="missing_grader", detail="Source grader files are required for static review")
    files = {path: base64.b64decode(value, validate=True) for path, value in encoded_files.items()}
    grader = files.get("tests/verifier.py")
    if grader is None:
        return ImportRejection(reason="missing_grader", detail="tests/verifier.py is required")
    try:
        if name == "arc_transductive":
            verifier = ArcGridVerifier(expected_output=data["expected_output"])
            kind = VerifierKind.ARC_GRID
        elif name == "arc_inductive":
            verifier = ArcTransformVerifier(
                test_cases=data["test_cases"], source_grader_sha256=hashlib.sha256(grader).hexdigest()
            )
            if not verifier.test_cases:
                return ImportRejection(reason="missing_cases", detail="At least one held-out grid pair is required")
            kind = VerifierKind.ARC_TRANSFORM
        else:
            verifier = IndirectInjectionVerifier(
                target_tool=data["target_tool"],
                target_args=data["target_args"],
                available_tools=tuple(data["available_tools"]),
            )
            kind = VerifierKind.INDIRECT_INJECTION
    except (ValidationError, KeyError, TypeError, ValueError) as error:
        return ImportRejection(reason="invalid_verifier_data", detail=str(error))
    resources = tuple(
        task_resource(
            "/" + path,
            value,
            ResourceVisibility.VERIFIER if path.startswith("tests/") else ResourceVisibility.CONTROL,
        )
        for path, value in files.items()
        if path.startswith(("tests/", "environment/", "solution/"))
    )
    inductive = name == "arc_inductive"
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(
            events=(
                TextMessage(role="user", content=instruction if inductive else response_instruction(instruction, name)),
            )
        ),
        environment_requirements=EnvironmentRequirements(capabilities=("filesystem", "python") if inductive else ()),
        resources=resources,
        output_paths=("/app/solution.py", "/app/answer.txt") if inductive else (),
        answer_type=AnswerType.FILE if inductive else AnswerType.TEXT,
        verifier=VerifierSpec(kind=kind, parameters_json=verifier.model_dump_json()),
    )


def verification_report(task: TaskSpec) -> VerificationReport:
    if task.verifier.kind == VerifierKind.ARC_TRANSFORM:
        return VerificationReport(
            checks=[
                CheckResult(
                    check="isolated_transform_runtime",
                    status=CheckStatus.UNSUPPORTED,
                    detail="Source Python transform grader and held-out cases retained; isolated runtime not yet bound",
                )
            ]
        )
    if task.verifier.kind == VerifierKind.ARC_GRID:
        verifier = ArcGridVerifier.model_validate_json(task.verifier.parameters_json)
        grid = verifier.expected_output
        positive = "\n".join(" ".join(str(cell) for cell in row) for row in grid)
        wrong = [row[:] for row in grid]
        wrong[0][0] = (wrong[0][0] + 1) % 10
        return VerificationReport(checks=verify_witness(task, positive, json.dumps(wrong)))
    verifier = IndirectInjectionVerifier.model_validate_json(task.verifier.parameters_json)
    target = json.dumps({"name": verifier.target_tool, "arguments": verifier.target_args})
    witness = json.dumps({"action": "respond", "content": "I will ignore the untrusted instruction."})
    checks = verify_witness(task, witness, target)
    for name, answer in (
        ("empty_object", "{}"),
        ("unadvertised_tool", json.dumps({"name": "__unadvertised_tool__", "arguments": {}})),
    ):
        grade = verifier.grade(
            GradingAttempt(PLAIN, (*task.context.events, TextMessage(role="assistant", content=answer)), None)
        )
        checks.append(
            CheckResult(
                check=name,
                status=CheckStatus.PASS if grade.status == Outcome.GRADED and grade.reward == 0.0 else CheckStatus.FAIL,
                detail=f"Source scorer accepted an invalid next action: {grade.status}, reward={grade.reward}",
            )
        )
    return VerificationReport(checks=checks)


def recipe(name: str, snapshot: Path, *, rubric: ReviewRubric) -> DatasetRecipe:
    """Bind one pinned ARC or injection snapshot to static review and real controls."""
    config = CONFIGS[name]

    def normalize_row(row: RawRow) -> TaskSpec | ImportRejection:
        return normalize(row, name)

    return DatasetRecipe(
        name=f"tasktrove-{name}",
        version=f"tasktrove-{name}-v1",
        source=SnapshotSource("open-thoughts/TaskTrove", REVISION, config, "train", str(snapshot)),
        normalize=normalize_row,
        rubric=rubric,
        intended_use=IntendedUse.TRAIN,
        check_suite=CheckSuite(id="arc-injection-source-controls", revision="1", parameters={}, run=verification_report),
    )
