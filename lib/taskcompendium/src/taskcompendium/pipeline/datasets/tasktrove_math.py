# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind remaining typed TaskTrove math sources without claiming source scorer parity."""

from taskcompendium.pipeline.datasets import atlas_math_qa
from taskcompendium.pipeline.datasets.instruction_following import REVISION
from taskcompendium.pipeline.datasets.source_definitions import TASKTROVE_DATASET, tasktrove_inputs, tasktrove_source
from taskcompendium.pipeline.models import (
    CheckSuite,
    DatasetRecipe,
    HFSource,
    ImportRejection,
    IntendedUse,
    NormalizationChange,
    NormalizedTask,
    RawRow,
    ReviewRubric,
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


def recipe(name: str, *, config: str, revision: str, rubric: ReviewRubric) -> DatasetRecipe:
    return DatasetRecipe(
        name=f"tasktrove-{name}",
        version=f"tasktrove-{name}-v1",
        source=HFSource(TASKTROVE_DATASET, revision, config, "train"),
        inputs=tasktrove_inputs(config, revision),
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


SOURCES = {
    "math_gym": tasktrove_source(
        config="laion__nemotron-gym-math-v5",
        revision=REVISION,
        rubric=ReviewRubric(
            id="math_gym-answerability",
            version="1",
            criteria=(
                *MATH_CRITERIA,
                "Check complete contest statements and exact final-answer format; independently verify feasible "
                "calculations and flag private references answering a different quantity.",
            ),
        ),
    ),
    "math_oracle": tasktrove_source(
        config="SankalpKJ__nemotron-math-oracle-filtered-v2",
        revision=REVISION,
        rubric=ReviewRubric(
            id="math_oracle-answerability",
            version="1",
            criteria=(
                *MATH_CRITERIA,
                "Check that oracle-filtered references solve the public problem; source oracle existence is evidence of "
                "grader compatibility, not proof of mathematical correctness.",
            ),
        ),
    ),
    "math_prism": tasktrove_source(
        config="laion__nemo-prism-math-v3",
        revision=REVISION,
        rubric=ReviewRubric(
            id="math_prism-answerability",
            version="1",
            criteria=(
                *MATH_CRITERIA,
                "Check symbolic olympiad statements, quantifiers, strict versus attained extrema, and whether escaped "
                "LaTeX keys express the requested quantity.",
            ),
        ),
    ),
    "math_stack": tasktrove_source(
        config="laion__nemotron-gym-math-stack-overflow-v3",
        revision=REVISION,
        rubric=ReviewRubric(
            id="math_stack-answerability",
            version="1",
            criteria=(
                *MATH_CRITERIA,
                "Check mathematical questions for missing prior context, definitions, diagrams, or truncated "
                "expressions; "
                "a plausible private answer cannot fill absent public premises.",
            ),
        ),
    ),
}


def recipe_for_source(
    name: str,
) -> DatasetRecipe:
    source = SOURCES[name]
    return recipe(name, config=source.config, revision=source.revision, rubric=source.rubric)
