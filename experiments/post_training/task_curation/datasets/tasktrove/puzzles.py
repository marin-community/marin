# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove logic puzzles with one reference answer, graded in process by verifyit.

``tests/gold.json`` holds the reference and its answer type. Ordered lists are compared item by
item, numbers and coordinates with the symbolic math comparator, and other answers as text.
"""

import json

from taskcompendium.convert.answers import answer_task, source_defect, unsupported
from taskcompendium.convert.delivery import replace_phrases, rewritten_task
from taskcompendium.convert.tasktrove import archive_file
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.controls import reference_reply
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import Controls, ImportRejection, IntendedUse, NormalizedTask, RawRow
from verifyit.spec import ExactSpec, MathSpec, MathType, Spec

from experiments.post_training.task_curation.datasets.tasktrove.archives import (
    ANSWER_FILE_DELIVERY,
    TaskTroveConverter,
    tasktrove_source,
)
from experiments.post_training.task_curation.pipeline import RlDataPipeline, ShellSim
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

ANSWER_TYPES = frozenset({"choice", "exact", "ordered_list", "number", "coords"})
MATH_ANSWER_TYPES = frozenset({"number", "coords"})
REWRITE_REASON = "Adapt the puzzle's answer-file delivery to the assistant response"

RUBRIC = """
Identify the actual puzzle, givens, requested result, and answer format. Reject absent diagrams or inputs rather than
guessing them. Difficult reasoning alone is not a defect.

Solve enough of the puzzle to check the hidden key, including order direction, case-sensitive ASCII/Unicode sorting,
rounding, coordinate axes, units, and indexing conventions.

Ordered-list scoring preserves order but normalizes case and whitespace. Number and coordinate answers use the
symbolic math checker, not an invented tolerance.

Flag ambiguous puzzles or multiple valid solutions that the single reference cannot represent. The reference grading
successfully does not certify its mathematical correctness.
"""


def puzzle_spec(expected: str, answer_type: str) -> Spec:
    if answer_type in MATH_ANSWER_TYPES:
        return MathSpec(expected=expected, math_type=MathType.SCALAR)
    if answer_type == "ordered_list":
        return ExactSpec(expected=tuple(item.strip() for item in expected.split(",") if item.strip()))
    return ExactSpec(expected=(expected,))


def convert_puzzle(row: RawRow, _context: ConversionContext) -> TaskSpec | NormalizedTask | ImportRejection:
    instruction, gold = row.data["instruction"], archive_file(row.data, "tests/gold.json")
    if not instruction.strip() or gold is None:
        return source_defect("missing_input", "Instruction and tests/gold.json are required")
    try:
        data = json.loads(gold)
        expected, answer_type = data["gold"], data["answer_type"]
        if not isinstance(expected, str) or not expected.strip() or answer_type not in ANSWER_TYPES:
            raise ValueError("Puzzle reference must contain a nonempty answer and supported answer type")
    except (ValueError, KeyError, TypeError) as error:
        return unsupported("invalid_puzzle_key", str(error))
    task = answer_task(
        row, prompt=replace_phrases(instruction, ANSWER_FILE_DELIVERY), spec=puzzle_spec(expected, answer_type)
    )
    return rewritten_task(task, original=instruction, reason=REWRITE_REASON)


def sources() -> list[RlDataSource]:
    return [
        RlDataSource(
            info=SourceInfo(
                id="Task Trove:laion__all-puzzles-v2",
                title="laion/all-puzzles-v2",
                origin="Task Trove",
                family="math-answer",
                tags=("agentic", "multi-turn"),
                count=6926,
                notes=(
                    "Numeric compare on real puzzles. Gold lives in /tests, which the harness must hide; "
                    "fine under the new grader."
                ),
            ),
            pipeline=RlDataPipeline(
                name="tasktrove-puzzles",
                source=tasktrove_source("laion__all-puzzles-v2"),
                convert=TaskTroveConverter("laion__all-puzzles-v2", convert_puzzle),
                version="1",
                environment=ShellSim(),
                intended_use=IntendedUse.TRAIN,
                rubric=RUBRIC,
                controls=Controls(golden=reference_reply),
            ),
        )
    ]
