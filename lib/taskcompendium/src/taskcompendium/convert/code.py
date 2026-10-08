# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checks for code tasks whose hidden tests use the LiveCodeBench test-case layout."""

import json
import re
from collections.abc import Mapping


def validate_code_cases(ground_truth: str | Mapping | list) -> None:
    """Reject test layouts the LiveCodeBench scorer cannot normalize."""
    cases = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
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
    return re.search(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL) is not None
