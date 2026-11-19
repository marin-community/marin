# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove logic puzzles in the standard non-agentic QA format, repaired from the original oracle.

The puzzle archives ship an original oracle (``solution/solve.sh``) that writes its answer to
``/app/answer.txt``. This repair converts each task to the standard QA TaskSpec: the prompt goes to
the model, and the model's answer is checked by the task's pinned Verifyit grader (exact or math,
exactly as the source declares) against a reference answer mechanically extracted from that oracle.

The reference is never written, guessed or substituted: it is extracted only when the whole script
is the single-printf literal template every archive in this source uses. Any other script is
rejected with a typed ``oracle_answer_not_extractable`` outcome and its bytes as evidence.

The answer's typed shape follows the source's own declared contract: the grader mode and its
declared expected-member shape, exactly as the canonical TaskSpecs record them. The extraction is
cross-checked against that declaration; a mismatch is a source defect, never a silent fallback.

Each task's labels carry its lineage back to the canonical conversion this repair derives from:
the original task ID, the oracle script digest and the extraction method. Labels stay out of the
contract identity, so archives repeated across rows still deduplicate to the canonical cohort.
"""

import hashlib
import json
import re

from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.convert.answers import answer_task, source_defect, unsupported
from taskcompendium.convert.delivery import replace_phrases, rewritten_task
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.controls import reference_reply
from taskcompendium.pipeline.models import Controls, ImportRejection, NormalizedTask, RawRow
from verifyit.spec import ExactSpec, MathSpec, MathType, Spec

from experiments.post_training.task_curation.datasets.tasktrove.archives import (
    ANSWER_FILE_DELIVERY,
    TaskTroveConverter,
    tasktrove_source,
)
from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import SOLVE_SH, archive_file
from experiments.post_training.task_curation.datasets.tasktrove.puzzles import (
    ANSWER_TYPES,
    MATH_ANSWER_TYPES,
    PUZZLES_CONFIG,
    RUBRIC,
)
from experiments.post_training.task_curation.pipeline import CurationRecipe, IntendedUse, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

PUZZLES_QA_CONFIG = PUZZLES_CONFIG

REWRITE_REASON = "Adapt the puzzle's answer-file delivery to the assistant response"

# The laion v2 puzzle oracle template: create /app, then a single printf of a single-quoted
# literal into /app/answer.txt. Anything else is not mechanically extractable.
LITERAL_TEMPLATE = re.compile(
    r"^#!/usr/bin/env bash\n" r"mkdir -p /app\n" r"printf '%s\\n' '(.*)' > /app/answer.txt\n$",
    re.DOTALL,
)

# The canonical laion__all-puzzles-v2 conversion this repair derives its task identities from.
ORIGINAL_RECIPE_NAME = "tasktrove-puzzles"
ORIGINAL_RECIPE_VERSION = "2"


def extract_oracle_answer(script: bytes) -> str | None:
    """The answer a literal-template oracle writes, or ``None`` when it computes one."""
    match = LITERAL_TEMPLATE.match(script.decode("utf-8", errors="replace"))
    return None if match is None else match.group(1)


def original_task_id(row: RawRow) -> str:
    """The canonical cohort's task ID for the same source row."""
    origin = row.source.model_copy(update={"importer_revision": ORIGINAL_RECIPE_VERSION})
    return f"{ORIGINAL_RECIPE_NAME}-{canonical_sha256(origin.model_dump())}"


def lineage_labels(row: RawRow, script: bytes) -> dict[str, str]:
    """Per-task lineage from the published task back to its original oracle."""
    return {
        "original_task_id": original_task_id(row),
        "original_solve_sh_sha256": hashlib.sha256(script).hexdigest(),
        "oracle_extraction": "literal_printf_template",
    }


def qa_spec(answer: str, declared_type: str) -> Spec:
    """Build the pinned Verifyit spec for an extracted answer under the source's declared mode."""
    if declared_type in MATH_ANSWER_TYPES:
        return MathSpec(expected=answer, math_type=MathType.SCALAR)
    if declared_type == "ordered_list":
        return ExactSpec(expected=tuple(item.strip() for item in answer.split(",") if item.strip()))
    return ExactSpec(expected=(answer,))


def convert_puzzle_qa(row: RawRow, _context) -> TaskSpec | NormalizedTask | ImportRejection:
    instruction = row.data["instruction"]
    script = archive_file(row.data, SOLVE_SH)
    gold = archive_file(row.data, "tests/gold.json")
    if not instruction.strip() or gold is None or script is None:
        return source_defect(
            "missing_input", "Instruction, tests/gold.json and the original solution/solve.sh are required"
        )
    answer = extract_oracle_answer(script)
    if answer is None or not answer.strip():
        return unsupported(
            "oracle_answer_not_extractable",
            "The original oracle does not write a single-quoted literal answer to /app/answer.txt",
        )
    try:
        data = json.loads(gold)
        declared_answer, declared_type = data["gold"], data["answer_type"]
        if not isinstance(declared_answer, str) or not declared_answer.strip() or declared_type not in ANSWER_TYPES:
            raise ValueError("Puzzle reference must contain a nonempty answer and supported answer type")
    except (ValueError, KeyError, TypeError) as error:
        return unsupported("invalid_puzzle_key", str(error))
    # The source's declared contract types the grader; the extracted reference must agree with it
    # member-for-member. A disagreement is a source defect, never resolved by preferring one side.
    if declared_type in MATH_ANSWER_TYPES:
        if answer != declared_answer:
            return source_defect(
                "oracle_answer_disagrees_with_declared_key",
                f"solve.sh writes {answer!r}; the declared gold answer is {declared_answer!r}",
            )
    elif [item.strip() for item in answer.split(",")] != [item.strip() for item in declared_answer.split(",")]:
        return source_defect(
            "oracle_answer_disagrees_with_declared_key",
            f"solve.sh writes {answer!r}; the declared gold answer is {declared_answer!r}",
        )
    task = answer_task(
        row,
        prompt=replace_phrases(instruction, ANSWER_FILE_DELIVERY),
        spec=qa_spec(answer, declared_type),
    )
    puzzle_type = str(data.get("ptype", "")).strip().lower().replace("_", "-") or "unknown"
    task = task.model_copy(update={"tags": ("puzzle", "laion", puzzle_type), "labels": lineage_labels(row, script)})
    return rewritten_task(task, original=instruction, reason=REWRITE_REASON)


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            pipeline=process_rows,
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                id="TaskTrove:tasktrove-puzzles-qa",
                upstream_dataset="tasktrove-puzzles-qa",
                upstream_name="tasktrove-puzzles-qa",
                subset=None,
                build_path="TaskTrove",
                origin=SourceOrigin.TASKTROVE,
                family="math-answer",
                tags=("rlvr", "single-turn", "qa-format-repair", "converted"),
                count=6921,
                notes=(
                    "QA-format repair of TaskTrove:laion__all-puzzles-v2: non-agentic one-message "
                    "conversation tasks graded in process by the source's pinned Verifyit exact/math "
                    "grader against the reference mechanically extracted from each task's original "
                    "solution/solve.sh oracle. Grader mode and instruction text unchanged from the "
                    "canonical conversion; task labels carry the original task ID and oracle digest."
                ),
                task_format=TaskFormat.TASKCOMPENDIUM,
            ),
            config=CurationRecipe(
                name="tasktrove-puzzles-qa",
                source=tasktrove_source(PUZZLES_QA_CONFIG),
                convert=TaskTroveConverter(PUZZLES_QA_CONFIG, convert_puzzle_qa),
                version="1",
                intended_use=IntendedUse.TRAIN,
                rubric=RUBRIC,
                controls=Controls(golden=reference_reply),
            ),
        )
    ]
