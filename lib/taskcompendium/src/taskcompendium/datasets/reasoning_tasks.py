# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize TaskTrove Reasoning Gym and all-puzzles task contracts."""

import base64
import json

from verifyit.spec import ExactSpec, MathSpec, MathType

from taskcompendium.datasets.direct_contracts import source_contract_package
from taskcompendium.grader import GraderPackage, verifyit_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
    verifyit_spec,
)
from taskcompendium.pipeline.models import (
    CheckSuite,
    ImportFailureKind,
    ImportRejection,
    RawRow,
    ReviewRubric,
    TaskPolicy,
    VerificationReport,
)
from taskcompendium.pipeline.verification import verify_task, verify_witness

REASONING_RUBRIC = ReviewRubric(
    id="reasoning-gym-answerability",
    version="2",
    criteria=(
        "The whole problem must be comprehensible and provide every grid, sequence, statement, or rule needed "
        "to answer it. Procedural generation and unfamiliar puzzles are not defects by themselves.",
        "Independently work out the answer where feasible; check that the reference actually follows the public "
        "problem. Passing a reference through its scorer tests mechanics, not its truth.",
        "Compare the public answer format with the named upstream Reasoning Gym scorer. The original grader "
        "thresholds its native reward at 0.5 into a binary verdict. Native parsing is preserved. The original "
        "fallback is reachable only when its source guard and validation permit it.",
        "Flag multiple defensible answers when the named scorer rejects them, hidden assumptions, "
        "underspecified transformations, or a private question different from the public problem.",
    ),
)
PUZZLE_RUBRIC = ReviewRubric(
    id="puzzle-answerability",
    version="1",
    criteria=(
        "Identify the actual puzzle, givens, requested result, and answer format. Reject absent diagrams or "
        "inputs rather than guessing them. Difficult reasoning alone is not a defect.",
        "Solve enough of the puzzle to check the private key, including order direction, case-sensitive "
        "ASCII/Unicode sorting, rounding, coordinate axes, units, and indexing conventions.",
        "Ordered-list scoring preserves order but normalizes case and whitespace. Number and coordinate "
        "answers use the cleanup's existing symbolic math checker, not an invented tolerance.",
        "Flag ambiguous puzzles or multiple valid solutions that the single reference cannot represent. "
        "The reference grading successfully does not certify its mathematical correctness.",
    ),
)


def snapshot_file(row: RawRow, path: str) -> bytes | None:
    files = row.data.get("files")
    value = files.get(path) if isinstance(files, dict) else None
    return base64.b64decode(value, validate=True) if isinstance(value, str) else None


def direct_instruction(instruction: str) -> str:
    """Adapt recognized puzzle file delivery to a terminal response."""
    return (
        instruction.replace(
            "write your final answer to `/app/answer.txt`", "return your final answer in the assistant response"
        )
        .replace(
            "Write ONLY your final answer to **`/app/answer.txt`**",
            "Return ONLY your final answer in the assistant response",
        )
        .replace("The verifier reads that file", "The verifier reads the assistant response")
    )


def normalize_reasoning(row: RawRow) -> TaskSpec | ImportRejection:
    instruction, data = row.data.get("instruction"), row.data.get("verifier_data")
    if not isinstance(instruction, str) or not instruction.strip() or not isinstance(data, dict):
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="missing_input",
            detail="Instruction and entry data are required",
        )
    metadata = data.get("metadata")
    dataset = metadata.get("source_dataset") if isinstance(metadata, dict) else None
    if not isinstance(dataset, str) or not dataset:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED, reason="missing_scorer", detail="metadata.source_dataset is required"
        )
    if not isinstance(data.get("answer"), str):
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT, reason="invalid_entry", detail="Entry answer must be a string"
        )
    package = source_contract_package(
        "TaskTrove:tests/verifier.py",
        row.source.revision,
        {"dataset": dataset, "entry": data, "original_instruction": instruction},
        ("Original TaskTrove verifier and validation with reasoning-gym==0.1.20",),
    )
    public_instruction = direct_instruction(instruction)
    if public_instruction == instruction and "/app/answer.txt" in instruction:
        public_instruction += "\nThe runtime writes your final assistant response to /app/answer.txt."
    return _task(row, public_instruction, package)


def normalize_puzzle(row: RawRow) -> TaskSpec | ImportRejection:
    instruction = row.data.get("instruction")
    gold_file = snapshot_file(row, "tests/gold.json")
    if not isinstance(instruction, str) or not instruction.strip() or gold_file is None:
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="missing_input",
            detail="Instruction and tests/gold.json are required",
        )
    try:
        data = json.loads(gold_file)
        expected = data["gold"]
        answer_type = data["answer_type"]
        if (
            not isinstance(expected, str)
            or not expected.strip()
            or answer_type not in {"choice", "exact", "ordered_list", "number", "coords"}
        ):
            raise ValueError("Puzzle reference must contain a nonempty answer and supported answer type")
    except (ValueError, KeyError, TypeError) as error:
        return ImportRejection(kind=ImportFailureKind.UNSUPPORTED, reason="invalid_puzzle_key", detail=str(error))
    spec = (
        MathSpec(expected=expected, math_type=MathType.SCALAR)
        if answer_type in {"number", "coords"}
        else ExactSpec(
            expected=(
                tuple(item.strip() for item in expected.split(",") if item.strip())
                if answer_type == "ordered_list"
                else (expected,)
            )
        )
    )
    return _task(row, direct_instruction(instruction), verifyit_package(spec))


def _task(row: RawRow, instruction: str, package: GraderPackage) -> TaskSpec:
    return TaskSpec(
        id=row.id,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=package.grader,
        resources=ResourceGroups(verifier=package.resources),
        source=row.source,
    )


def reasoning_checks(task: TaskSpec) -> VerificationReport:
    return VerificationReport(checks=verify_task(task))


def puzzle_checks(task: TaskSpec) -> VerificationReport:
    assert isinstance(task.grader, VerifyitGrader)
    spec = verifyit_spec(task.grader)
    assert isinstance(spec, (ExactSpec, MathSpec))
    expected = ", ".join(spec.expected) if isinstance(spec, ExactSpec) else spec.expected
    return VerificationReport(checks=verify_witness(task, expected, "__incorrect_puzzle_answer__"))


def reasoning_policy() -> TaskPolicy:
    """Build the source normalization and review policy."""
    return TaskPolicy(
        normalize=normalize_reasoning,
        rubric=REASONING_RUBRIC,
        check_suite=CheckSuite(id="reasoning-gym-reference-controls", revision="1", parameters={}, run=reasoning_checks),
    )


def puzzle_policy() -> TaskPolicy:
    """Build the source normalization and review policy."""
    return TaskPolicy(
        normalize=normalize_puzzle,
        rubric=PUZZLE_RUBRIC,
        check_suite=CheckSuite(id="puzzle-reference-controls", revision="1", parameters={}, run=puzzle_checks),
    )
