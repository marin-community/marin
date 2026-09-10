# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TACO stdin/stdout problems with oracle solutions.

Every task ships one file the template varies: ``solution/solution.py``, run against
``tests/inputs/input_<n>.txt`` / ``tests/outputs/output_<n>.txt`` pairs by the old grader
(``python3 /app/solution.py < input_n.txt``, output compared after squashing whitespace). That
maps directly onto ``stdio``, token comparison matching the old grader's whitespace squash, except
for the rare prompt that promises a numeric tolerance instead of an exact match.

A meaningful slice of TACO (LeetCode- and Codewars-derived problems) is written as a bare function
definition — ``class Solution: def foo(self, ...)`` or a top-level ``def foo(...)`` with no driver
that reads stdin or prints a result. The old grader ran those against stdin fixtures anyway, so
their stdout was always empty; they cannot be graded by ``stdio`` and are rejected rather than
converted.
"""

import re

from tasktrove_verify.spec import Compare, StdioSpec

from experiments.post_training.tasktrove.converters.converted_task import (
    ConvertedTask,
    Converter,
    ConverterKey,
    ConvertStatus,
    Rejected,
)
from experiments.post_training.tasktrove.converters.nemotron_data import metadata
from experiments.post_training.tasktrove.converters.stdio_cases import case_files_from_dirs, hidden_case_rejection
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, SOLVE_SH, TaskFiles

COMMAND = "python3 /app/solution.py"

SOLUTION_PY = "solution/solution.py"
_SOLVE_SCRIPT = "#!/bin/bash\nset -e\ncp /solution/solution.py /app/solution.py\n"

_STDIN_READ_RE = re.compile(r"\b(input\s*\(|sys\.stdin|raw_input\s*\(|fileinput\.|open\(0\))")
_FLOAT_TOLERANCE_RE = re.compile(r"absolute (?:or relative )?error|relative error|error does not exceed", re.IGNORECASE)
"""A handful of problems promise a numeric tolerance (``absolute or relative error <= 1e-6``) in
the prompt itself; token comparison would reject a correctly rounded answer that differs in its
last digit, so those get float comparison instead of the source's exact token match."""


def _oracle_solution_files(solution: str) -> dict[str, bytes]:
    """``solve.sh`` copies the oracle into place at the path ``COMMAND`` runs."""
    return {SOLVE_SH: _SOLVE_SCRIPT.encode(), SOLUTION_PY: solution.encode()}


def _compare(instruction: str) -> Compare:
    return Compare.FLOAT if _FLOAT_TOLERANCE_RE.search(instruction) else Compare.TOKENS


def convert_taco(task: TaskFiles) -> ConvertedTask | Rejected:
    solution = task.get_text(SOLUTION_PY)
    if not solution or not solution.strip():
        return Rejected(ConvertStatus.NULL_GRADER, f"{SOLUTION_PY} missing or empty")
    if not _STDIN_READ_RE.search(solution):
        return Rejected(
            ConvertStatus.UNSUPPORTED_VARIANT, "oracle solution never reads stdin: function-call style problem"
        )
    try:
        cases = case_files_from_dirs(task)
    except ValueError as error:
        return Rejected(ConvertStatus.NULL_GRADER, str(error))
    instruction = task.text(INSTRUCTION)
    rejection = hidden_case_rejection(cases, instruction)
    if rejection is not None:
        return rejection
    return ConvertedTask(
        instruction=instruction,
        spec=StdioSpec(command=COMMAND, compare=_compare(instruction)),
        dockerfile=task.text(DOCKERFILE),
        tags=("code", "competitive-programming", "stdio", "taco"),
        language="python",
        data_files=cases,
        solution_files=_oracle_solution_files(solution),
        metadata=metadata(task),
    )


CONVERTER = Converter(
    name="taco",
    keys=(ConverterKey("stdin-stdout", frozenset({"tests/test.sh"})),),
    convert=convert_taco,
)
