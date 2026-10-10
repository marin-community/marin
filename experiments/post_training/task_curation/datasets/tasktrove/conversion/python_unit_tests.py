# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Self-contained Python implementation tasks graded by one pytest file."""

import ast

from verifyit.spec import PytestSpec

from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import DOCKERFILE, INSTRUCTION, SOLUTION_DIR, SOLVE_SH, TESTS_MOUNT, TaskFiles
from experiments.post_training.task_curation.datasets.tasktrove.conversion.result import (
    ConvertedTask,
    ConvertStatus,
    Rejected,
)

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
    f" && {PYTEST_ENV}/bin/pip install --no-cache-dir pytest pytest-json-report{{dependencies}}\n"
)


def _test_module(task: TaskFiles, test_files: tuple[str, ...]) -> tuple[str, ast.Module] | Rejected:
    candidates = [path for path in test_files if path in task.files]
    if len(candidates) != 1:
        return Rejected(
            ConvertStatus.UNSUPPORTED_VARIANT,
            f"expected one supported pytest file, found {candidates}",
        )
    path = candidates[0]
    try:
        tree = ast.parse(task.files[path])
    except SyntaxError as error:
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, f"{path} is not valid Python: {error.msg}")
    has_test = any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test_")
        for node in ast.walk(tree)
    )
    if not has_test:
        return Rejected(ConvertStatus.NULL_GRADER, f"{path} defines no local test function")
    return path, tree


def _solution_files(task: TaskFiles) -> dict[str, bytes] | Rejected:
    files = task.under(SOLUTION_DIR)
    if not files:
        return {}
    if set(files) != {ORACLE_FILE}:
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, f"unsupported oracle files: {sorted(files)}")
    try:
        ast.parse(task.files[ORACLE_FILE])
    except SyntaxError as error:
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, f"{ORACLE_FILE} is not valid Python: {error.msg}")
    return {**files, SOLVE_SH: ORACLE_SCRIPT.encode()}


def pytest_dockerfile(dockerfile: str, tree: ast.Module) -> str:
    """Install the legacy isolated pytest interpreter and its test-import dependencies."""
    modules = {
        alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    } | {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    dependencies = " mock" if "mock" in modules else ""
    return dockerfile.rstrip() + "\n" + PYTEST_INSTALL.format(dependencies=dependencies)


def convert(task: TaskFiles, *, test_files: tuple[str, ...] = TEST_FILES) -> ConvertedTask | Rejected:
    """Convert one self-contained Python task without preserving its legacy shell grader."""
    module = _test_module(task, test_files)
    if isinstance(module, Rejected):
        return module
    test_file, tree = module
    solution_files = _solution_files(task)
    if isinstance(solution_files, Rejected):
        return solution_files

    data_files = {test_file: task.files[test_file], **task.under("setup_files/")}
    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        spec=PytestSpec(
            paths=(f"{TESTS_MOUNT}/{test_file.removeprefix('tests/')}",),
            python=PYTEST_PYTHON,
        ),
        dockerfile=pytest_dockerfile(task.text(DOCKERFILE), tree),
        tags=("code", "python", "unit-test", "kata"),
        language="python",
        data_files=data_files,
        solution_files=solution_files,
    )
