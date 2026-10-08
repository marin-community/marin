# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checks and grading settings for code tasks whose hidden tests use the LiveCodeBench test-case layout."""

import ast
import re
from collections.abc import Mapping
from typing import Any

CODE_GRADER_MEMORY_MB = 5120
"""The LiveCodeBench child process may use 4 GiB; the grading machine also needs room for its runtime."""
THREAD_ENVIRONMENT = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
"""Run each test's numeric libraries on one thread."""
FAILING_PROGRAM = '```python\nraise RuntimeError("negative control")\n```'
CODE_BLOCK = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)
"""A fenced code block as the source scorers match one; they run the last block in a reply."""


def validate_code_cases(cases: Mapping[str, Any] | list[Any]) -> None:
    """Reject decoded test layouts the LiveCodeBench scorer cannot normalize."""
    if isinstance(cases, Mapping) and "test_cases" in cases:
        if cases.get("language") not in (None, "python"):
            raise ValueError("LCB supports only Python verification specs")
        cases = cases["test_cases"]
    elif isinstance(cases, Mapping):
        inputs, outputs = cases.get("inputs"), cases.get("outputs")
        if not isinstance(inputs, list) or not isinstance(outputs, list) or not inputs or len(inputs) != len(outputs):
            raise ValueError("LCB requires aligned nonempty input and output lists")
        kind = "functional" if cases.get("fn_name") is not None else "stdin"
        cases = [
            {"input": item, "output": result, "testtype": kind, "fn_name": cases.get("fn_name")}
            for item, result in zip(inputs, outputs, strict=True)
        ]
    if not isinstance(cases, list) or not cases or not all(isinstance(case, Mapping) for case in cases):
        raise ValueError("LCB requires a nonempty list of test cases")
    modes = set()
    names = set()
    for case in cases:
        kind = case.get("testtype", case.get("type"))
        if kind in ("stdin", "stdin_stdout"):
            if not isinstance(case.get("input"), str) or not isinstance(case.get("output"), str):
                raise ValueError("Standard-input LCB cases require string input and output")
            modes.add("stdin")
        elif kind in ("functional", "call_based"):
            metadata = case.get("metadata")
            name = case.get("fn_name") or (metadata.get("func_name") if isinstance(metadata, Mapping) else None)
            if not isinstance(name, str) or not name:
                raise ValueError("Functional LCB cases require a function name")
            if not isinstance(case.get("input"), str | list):
                raise ValueError("Functional LCB input must be a list or encoded string")
            modes.add("functional")
            names.add(name)
        else:
            raise ValueError("Unsupported LCB test-case type")
    if len(modes) != 1 or len(names) > 1:
        raise ValueError("LCB cases must share one execution mode and function name")


def has_code_block(response: str) -> bool:
    """Match a fenced code block the way the source scorers extract one."""
    return CODE_BLOCK.search(response) is not None


def python_reply(solution: Any) -> str | None:
    """A source solution as a reply the scorers extract code from.

    Returns ``None`` when there is no solution or when the program the scorers would run is not
    Python 3. Some sources keep Python 2 solutions; the graders run Python 3, so such a solution
    cannot serve as a known-correct submission.
    """
    if not isinstance(solution, str) or not solution.strip():
        return None
    reply = solution if has_code_block(solution) else f"```python\n{solution}\n```"
    try:
        ast.parse(CODE_BLOCK.findall(reply)[-1].strip())
    except SyntaxError:  # Includes TabError from Python 2 tab indentation.
        return None
    return reply
