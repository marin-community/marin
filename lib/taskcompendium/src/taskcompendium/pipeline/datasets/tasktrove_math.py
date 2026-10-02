# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind remaining typed TaskTrove math sources without claiming source scorer parity."""

from pathlib import Path

from taskcompendium.pipeline.datasets import atlas_math_qa
from taskcompendium.pipeline.models import (
    CheckSuite,
    DatasetRecipe,
    ImportRejection,
    IntendedUse,
    NormalizationChange,
    NormalizedTask,
    RawRow,
    ReviewRubric,
    SnapshotSource,
)

MATH_CRITERIA = (
    "Require a complete mathematical problem, supplied givens, notation, units, and requested result.",
    "Check private reference consistency; difficulty alone is not a defect and a failed control does "
    "not prove the problem is bad.",
    "Original source grader code and data remain private. Cleanup comparator parity is unsupported; "
    "distinguish content quality from grading readiness.",
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
        return ImportRejection(reason="missing_instruction", detail="Public mathematical instruction is required")
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


def recipe(name: str, snapshot: Path, *, config: str, revision: str, rubric: ReviewRubric) -> DatasetRecipe:
    return DatasetRecipe(
        name=f"tasktrove-{name}",
        version=f"tasktrove-{name}-v1",
        source=SnapshotSource("open-thoughts/TaskTrove", revision, config, "train", str(snapshot)),
        normalize=normalize,
        rubric=rubric,
        intended_use=IntendedUse.TRAIN,
        check_suite=CheckSuite(
            id=f"{name}-typed-math-controls",
            revision="1",
            parameters={},
            run=atlas_math_qa.verification_report,
        ),
    )
