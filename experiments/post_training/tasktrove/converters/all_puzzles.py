# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Laion all-puzzles: a baked gold string compared to ``/app/answer.txt`` after normalization.

``tests/compare_answer.py`` reads ``tests/gold.json`` (``{"gold": ..., "answer_type": ...,
"ptype": ...}``), normalizes the agent's answer per ``answer_type``, and compares. Four
``answer_type`` values appear in the source: ``choice`` and ``ordered_list`` are string
normalization (casefold, collapse whitespace, and for ``ordered_list`` split on commas and
compare in order) and map onto :class:`ExactSpec`. ``number`` and ``coords`` are numeric and map
onto :class:`MathSpec`: math-verify's LaTeX extraction reads a plain ``(x, y)`` pair as an
interval, which still requires both components to match in order, so a coordinate task grades
soundly under the same mode as a bare number. A fifth type the grader supports, ``exact``, never
appears in the source but is handled identically to ``choice``.
"""

import json
import re

from tasktrove_verify.spec import ExactSpec, MathSpec, MathType, Spec

from experiments.post_training.tasktrove.contract import drop_dockerfile_lines
from experiments.post_training.tasktrove.converters.answer_solution import answer_solution
from experiments.post_training.tasktrove.converters.converted_task import (
    ConvertedTask,
    Converter,
    ConverterKey,
    ConvertStatus,
    Rejected,
)
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, TaskFiles

GOLD_FILE = "tests/gold.json"
_STRING_ANSWER_TYPES = frozenset({"choice", "exact", "ordered_list"})
_NUMERIC_ANSWER_TYPES = frozenset({"number", "coords"})
_OLD_GRADER_INSTALL = re.compile(r"pip install .*\bpytest\b")


def _exact_spec(answer_type: str, gold: str) -> ExactSpec | None:
    if answer_type == "ordered_list":
        items = tuple(item.strip() for item in gold.split(",") if item.strip())
        return ExactSpec(expected=items) if items else None
    return ExactSpec(expected=(gold,)) if gold else None


def _math_spec(gold: str) -> MathSpec | None:
    return MathSpec(expected=gold, math_type=MathType.SCALAR) if gold else None


def _solution_files(task: TaskFiles, spec: Spec) -> dict[str, bytes]:
    return task.under("solution/") or answer_solution(spec)


def convert_all_puzzles(task: TaskFiles) -> ConvertedTask | Rejected:
    raw = task.get_text(GOLD_FILE)
    if raw is None:
        return Rejected(ConvertStatus.NULL_GRADER, f"missing {GOLD_FILE}")
    data = json.loads(raw)
    answer_type = str(data.get("answer_type", ""))
    gold = str(data.get("gold", "")).strip()
    ptype = str(data.get("ptype", "")).strip().lower().replace("_", "-") or "unknown"

    if answer_type in _STRING_ANSWER_TYPES:
        spec = _exact_spec(answer_type, gold)
    elif answer_type in _NUMERIC_ANSWER_TYPES:
        spec = _math_spec(gold)
    else:
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, f"unhandled answer_type {answer_type!r}")
    if spec is None:
        return Rejected(ConvertStatus.NULL_GRADER, f"empty gold for answer_type {answer_type!r}")

    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        spec=spec,
        dockerfile=drop_dockerfile_lines(task.text(DOCKERFILE), _OLD_GRADER_INSTALL),
        tags=("puzzle", "laion", ptype),
        solution_files=_solution_files(task, spec),
        metadata={"ptype": ptype, "answer_type": answer_type},
    )


CONVERTER = Converter(
    name="all_puzzles",
    keys=(ConverterKey("math-answer", frozenset({"tests/test.sh", "tests/compare_answer.py"})),),
    convert=convert_all_puzzles,
)
