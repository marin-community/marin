# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize TaskTrove Reasoning Gym and all-puzzles task contracts."""

import base64
import json

from verifyit.spec import ExactSpec, MathSpec

from taskcompendium.grader import grader_package
from taskcompendium.grading import resolve_verifier
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ResourceGroups,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.datasets.direct_contracts import source_contract_package
from taskcompendium.pipeline.datasets.grader_scripts.puzzle import puzzle_spec
from taskcompendium.pipeline.models import (
    CheckSuite,
    ImportRejection,
    RawRow,
    ReviewRubric,
    TaskPipeline,
    VerificationReport,
)
from taskcompendium.pipeline.verification import verify_task, verify_witness

UNSCORABLE_DATASETS = frozenset({"arc_agi", "rearc"})
REASONING_RUBRIC = ReviewRubric(
    id="reasoning-gym-answerability",
    version="1",
    criteria=(
        "The whole problem must be comprehensible and provide every grid, sequence, statement, or rule needed "
        "to answer it. Procedural generation and unfamiliar puzzles are not defects by themselves.",
        "Independently work out the answer where feasible; check that the reference actually follows the public "
        "problem. Passing a reference through its scorer tests mechanics, not its truth.",
        "Compare the public answer format with the named upstream Reasoning Gym scorer. Partial credit and "
        "dataset-specific parsing are preserved. There is no substring or exact-match fallback.",
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
    """Change only recognized file delivery and the obsolete scorer fallback."""
    return (
        instruction.replace(
            "write your final answer to `/app/answer.txt`", "return your final answer in the assistant response"
        )
        .replace(
            "Write ONLY your final answer to **`/app/answer.txt`**",
            "Return ONLY your final answer in the assistant response",
        )
        .replace("The verifier reads that file", "The verifier reads the assistant response")
        .replace(
            "The verifier will try the upstream Reasoning Gym scorer first, then fall back to normalized exact-match.",
            "The verifier uses the upstream Reasoning Gym scorer, with no fallback.",
        )
    )


def normalize_reasoning(row: RawRow) -> TaskSpec | ImportRejection:
    instruction, data = row.data.get("instruction"), row.data.get("verifier_data")
    if not isinstance(instruction, str) or not instruction.strip() or not isinstance(data, dict):
        return ImportRejection(reason="missing_input", detail="Instruction and entry data are required")
    metadata = data.get("metadata")
    dataset = metadata.get("source_dataset") if isinstance(metadata, dict) else None
    if not isinstance(dataset, str) or not dataset:
        return ImportRejection(reason="missing_scorer", detail="metadata.source_dataset is required")
    if dataset in UNSCORABLE_DATASETS:
        return ImportRejection(
            reason="known_broken_scorer", detail=f"Cleanup identified {dataset!r} as unable to score its own reference"
        )
    if not isinstance(data.get("answer"), str) or not data["answer"].strip():
        return ImportRejection(reason="invalid_entry", detail="Entry answer must be a nonempty string")
    package = source_contract_package(
        "reasoning_gym.get_score_answer_fn",
        row.source.revision,
        {"dataset": dataset, "entry": data},
        ("Pinned isolated Reasoning Gym scorer runtime",),
    )
    return _task(row, direct_instruction(instruction), package)


def normalize_puzzle(row: RawRow) -> TaskSpec | ImportRejection:
    instruction = row.data.get("instruction")
    gold_file = snapshot_file(row, "tests/gold.json")
    if not isinstance(instruction, str) or not instruction.strip() or gold_file is None:
        return ImportRejection(reason="missing_input", detail="Instruction and tests/gold.json are required")
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
        return ImportRejection(reason="invalid_puzzle_key", detail=str(error))
    return _task(row, direct_instruction(instruction), grader_package(puzzle_spec(expected, answer_type)))


def _task(row: RawRow, instruction: str, package) -> TaskSpec:
    return TaskSpec(
        id=row.id,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=package.verifier,
        resources=ResourceGroups(verifier=package.resources),
        source=row.source,
    )


def reasoning_checks(task: TaskSpec) -> VerificationReport:
    return VerificationReport(checks=verify_task(task))


def puzzle_checks(task: TaskSpec) -> VerificationReport:
    spec = resolve_verifier(task.verifier)
    assert isinstance(spec, (ExactSpec, MathSpec))
    expected = ", ".join(spec.expected) if isinstance(spec, ExactSpec) else spec.expected
    return VerificationReport(checks=verify_witness(task, expected, "__incorrect_puzzle_answer__"))


def reasoning_pipeline() -> TaskPipeline:
    """Build the source normalization and review policy."""
    return TaskPipeline(
        normalize=normalize_reasoning,
        rubric=REASONING_RUBRIC,
        check_suite=CheckSuite(id="reasoning-gym-reference-controls", revision="1", parameters={}, run=reasoning_checks),
    )


def puzzle_pipeline() -> TaskPipeline:
    """Build the source normalization and review policy."""
    return TaskPipeline(
        normalize=normalize_puzzle,
        rubric=PUZZLE_RUBRIC,
        check_suite=CheckSuite(id="puzzle-reference-controls", revision="1", parameters={}, run=puzzle_checks),
    )
