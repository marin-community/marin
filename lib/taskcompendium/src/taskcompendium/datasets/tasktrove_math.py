# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize typed TaskTrove math tasks without claiming source scorer parity."""

from taskcompendium.datasets import atlas_math_qa
from taskcompendium.pipeline.models import (
    CheckSuite,
    ImportFailureKind,
    ImportRejection,
    NormalizationChange,
    NormalizedTask,
    RawRow,
    ReviewRubric,
    TaskPolicy,
)

MATH_CRITERIA = (
    "Require a complete mathematical problem, supplied givens, notation, units, and requested result.",
    "Check private reference consistency; difficulty alone is not a defect and a failed control does "
    "not prove the problem is bad.",
    "Original source grader code and data remain private. Assess its exact extraction, typed comparison and "
    "error behavior; distinguish content quality from grading readiness.",
)

SUBMISSION = "\n## Submitting the answer\n"
TERMINAL_SUBMISSION = "\n## Submitting your answer (IMPORTANT)\n"
DELIVERY = {
    "Provide your answer in the file answer.txt": "Return your final answer in the assistant response.",
    "Please place your final answer in a file named `/app/solution.txt`.": (
        "Return your final answer in the assistant response."
    ),
    "write your final answer at the path `/app/answer.txt`": "return your final answer in the assistant response",
    "the answer file": "the assistant response",
}


def normalize(row: RawRow) -> NormalizedTask | ImportRejection:
    instruction = row.data.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="missing_instruction",
            detail="Public mathematical instruction is required",
        )
    public = instruction.partition(SUBMISSION)[0].partition(TERMINAL_SUBMISSION)[0]
    for original, replacement in DELIVERY.items():
        public = public.replace(original, replacement)
    public = public.strip()
    prepared = RawRow(row.id, row.source, {**row.data, "instruction": public})
    task = atlas_math_qa.normalize(prepared, "math_openreasoning")
    if isinstance(task, ImportRejection):
        return task
    changes = (
        ()
        if public == instruction
        else (
            NormalizationChange(
                field="instruction",
                reason="Replace observed source answer-file delivery with the assistant response convention",
                original=instruction,
                replacement=public,
            ),
        )
    )
    return NormalizedTask(task, changes)


RUBRICS: dict[str, ReviewRubric] = {
    "math_gym": ReviewRubric(
        id="math_gym-answerability",
        version="2",
        criteria=(
            *MATH_CRITERIA,
            "Check complete contest statements and exact final-answer format; independently verify feasible "
            "calculations and flag private references answering a different quantity.",
        ),
    ),
    "math_oracle": ReviewRubric(
        id="math_oracle-answerability",
        version="2",
        criteria=(
            *MATH_CRITERIA,
            "Check that oracle-filtered references solve the public problem; source oracle existence is evidence of "
            "grader compatibility, not proof of mathematical correctness.",
        ),
    ),
    "math_prism": ReviewRubric(
        id="math_prism-answerability",
        version="2",
        criteria=(
            *MATH_CRITERIA,
            "Check symbolic olympiad statements, quantifiers, strict versus attained extrema, and whether escaped "
            "LaTeX keys express the requested quantity.",
        ),
    ),
    "math_stack": ReviewRubric(
        id="math_stack-answerability",
        version="2",
        criteria=(
            *MATH_CRITERIA,
            "Check mathematical questions for missing prior context, definitions, diagrams, or truncated "
            "expressions; "
            "a plausible private answer cannot fill absent public premises.",
        ),
    ),
}


def policy(name: str) -> TaskPolicy:
    """Build typed TaskTrove math normalization and source controls."""
    return TaskPolicy(
        normalize=normalize,
        rubric=RUBRICS[name],
        check_suite=CheckSuite(
            id=f"{name}-typed-math-controls", revision="1", parameters={}, run=atlas_math_qa.verification_report
        ),
    )
