# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize pinned TaskTrove Reasoning Gym and all-puzzles sources."""

import base64
import json

from pydantic import ValidationError

from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    TaskSpec,
    TextMessage,
    VerifierKind,
    VerifierSpec,
)
from taskcompendium.pipeline.datasets.instruction_following import REVISION
from taskcompendium.pipeline.datasets.source_definitions import tasktrove_inputs
from taskcompendium.pipeline.models import (
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
from taskcompendium.verifiers.reasoning import PuzzleAnswerVerifier, ReasoningGymVerifier

REASONING_CONFIG = "laion__nemotron-gym-reasoning-gym-v2"
PUZZLE_CONFIG = "laion__all-puzzles-v2"
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
    try:
        verifier = ReasoningGymVerifier(dataset=dataset, entry=data)
    except ValidationError as error:
        return ImportRejection(reason="invalid_entry", detail=str(error))
    return _task(row, direct_instruction(instruction), VerifierKind.REASONING_GYM, verifier.model_dump_json())


def normalize_puzzle(row: RawRow) -> TaskSpec | ImportRejection:
    instruction = row.data.get("instruction")
    gold_file = snapshot_file(row, "tests/gold.json")
    if not isinstance(instruction, str) or not instruction.strip() or gold_file is None:
        return ImportRejection(reason="missing_input", detail="Instruction and tests/gold.json are required")
    try:
        data = json.loads(gold_file)
        verifier = PuzzleAnswerVerifier(expected=data["gold"], answer_type=data["answer_type"])
    except (ValidationError, ValueError, KeyError, TypeError) as error:
        return ImportRejection(reason="invalid_puzzle_key", detail=str(error))
    return _task(row, direct_instruction(instruction), VerifierKind.PUZZLE_ANSWER, verifier.model_dump_json())


def _task(row: RawRow, instruction: str, kind: VerifierKind, parameters_json: str) -> TaskSpec:
    return TaskSpec(
        id=row.id,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=VerifierSpec(kind=kind, parameters_json=parameters_json),
        source=row.source,
    )


def reasoning_checks(task: TaskSpec) -> VerificationReport:
    verifier = ReasoningGymVerifier.model_validate_json(task.verifier.parameters_json)
    answer = verifier.entry["answer"]
    assert isinstance(answer, str)
    return VerificationReport(checks=verify_witness(task, answer, "__incorrect_reasoning_answer__"))


def puzzle_checks(task: TaskSpec) -> VerificationReport:
    verifier = PuzzleAnswerVerifier.model_validate_json(task.verifier.parameters_json)
    return VerificationReport(checks=verify_witness(task, verifier.expected, "__incorrect_puzzle_answer__"))


def reasoning_recipe() -> DatasetRecipe:
    return DatasetRecipe(
        name="tasktrove-reasoning-gym",
        version="tasktrove-reasoning-gym-v1",
        source=HFSource("open-thoughts/TaskTrove", REVISION, REASONING_CONFIG, "train"),
        inputs=tasktrove_inputs(REASONING_CONFIG, REVISION),
        normalize=normalize_reasoning,
        rubric=REASONING_RUBRIC,
        intended_use=IntendedUse.TRAIN,
        check_suite=CheckSuite(id="reasoning-gym-reference-controls", revision="1", parameters={}, run=reasoning_checks),
    )


def puzzle_recipe() -> DatasetRecipe:
    return DatasetRecipe(
        name="tasktrove-puzzles",
        version="tasktrove-puzzles-v1",
        source=HFSource("open-thoughts/TaskTrove", REVISION, PUZZLE_CONFIG, "train"),
        inputs=tasktrove_inputs(PUZZLE_CONFIG, REVISION),
        normalize=normalize_puzzle,
        rubric=PUZZLE_RUBRIC,
        intended_use=IntendedUse.TRAIN,
        check_suite=CheckSuite(id="puzzle-reference-controls", revision="1", parameters={}, run=puzzle_checks),
    )
