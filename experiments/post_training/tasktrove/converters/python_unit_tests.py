# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Self-contained Python implementation tasks graded by one pytest file."""

import ast

from tasktrove_verify.spec import PytestSpec

from experiments.post_training.tasktrove.converters.converted_task import (
    ConvertedTask,
    Converter,
    ConverterKey,
    ConvertStatus,
    Rejected,
)
from experiments.post_training.tasktrove.task_format import TESTS_MOUNT
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, SOLUTION_DIR, SOLVE_SH, TaskFiles

TEST_FILES = (
    "tests/test_curriculum.py",
    "tests/test_multifile.py",
    "tests/test_solution.py",
)
ORACLE_FILE = "solution/solution.py"
ORACLE_SCRIPT = """#!/bin/bash
set -e
cp /solution/solution.py /app/solution.py
"""
PYTEST_ENV = "/opt/tasktrove-pytest"
PYTEST_PYTHON = f"{PYTEST_ENV}/bin/python"
PYTEST_INSTALL = (
    f"RUN python3 -m venv --system-site-packages {PYTEST_ENV}"
    f" && {PYTEST_ENV}/bin/pip install --no-cache-dir pytest pytest-json-report\n"
)


def _test_file(task: TaskFiles) -> str | Rejected:
    candidates = [path for path in TEST_FILES if path in task.files]
    if len(candidates) != 1:
        return Rejected(
            ConvertStatus.UNSUPPORTED_VARIANT,
            f"expected one supported pytest file, found {candidates}",
        )
    path = candidates[0]
    try:
        tree = ast.parse(task.text(path))
    except SyntaxError as error:
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, f"{path} is not valid Python: {error.msg}")
    has_test = any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test_")
        for node in ast.walk(tree)
    )
    if not has_test:
        return Rejected(ConvertStatus.NULL_GRADER, f"{path} defines no local test function")
    return path


def _solution_files(task: TaskFiles) -> dict[str, bytes] | Rejected:
    files = task.under(SOLUTION_DIR)
    if not files:
        return {}
    if set(files) != {ORACLE_FILE}:
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, f"unsupported oracle files: {sorted(files)}")
    try:
        ast.parse(task.text(ORACLE_FILE))
    except SyntaxError as error:
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, f"{ORACLE_FILE} is not valid Python: {error.msg}")
    return {**files, SOLVE_SH: ORACLE_SCRIPT.encode()}


def convert(task: TaskFiles) -> ConvertedTask | Rejected:
    """Convert one self-contained Python task without preserving its legacy shell grader."""
    test_file = _test_file(task)
    if isinstance(test_file, Rejected):
        return test_file
    solution_files = _solution_files(task)
    if isinstance(solution_files, Rejected):
        return solution_files

    data_files = {test_file: task.files[test_file], **task.under("setup_files/")}
    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        spec=PytestSpec(paths=(f"{TESTS_MOUNT}/{test_file.removeprefix('tests/')}",), python=PYTEST_PYTHON),
        dockerfile=task.text(DOCKERFILE).rstrip() + "\n" + PYTEST_INSTALL,
        tags=("code", "python", "unit-test", "kata"),
        language="python",
        data_files=data_files,
        solution_files=solution_files,
    )


CONVERTER = Converter(
    name="python_unit_tests",
    keys=(ConverterKey("unit-test-gen", frozenset({"tests/test.sh"})),),
    convert=convert,
)
