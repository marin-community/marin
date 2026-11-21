# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Self-contained Python implementation tasks graded by one pytest file."""

import ast
import re

from verifyit.spec import PytestSpec

from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import (
    DOCKERFILE,
    INSTRUCTION,
    SOLUTION_DIR,
    SOLVE_SH,
    TESTS_MOUNT,
    TaskFiles,
)
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
PUBLIC_MODULE_PATH = re.compile(r"/app/([A-Za-z_]\w*)(?:/|\.py\b)")
PUBLIC_MODULE_FILE = re.compile(r"(?<![\w/])([A-Za-z_]\w*)\.py\b")
PUBLIC_APP_PATH = re.compile(r"/app/((?:[A-Za-z_]\w*/)*[A-Za-z_]\w*(?:\.py)?)")
PUBLIC_APP_IMPORT = re.compile(r"\bfrom\s+app\s+import\b")


def public_candidate_modules(instruction: str) -> tuple[str, ...]:
    """Name modules and packages explicitly requested in the ``/app`` workspace."""
    relative_files = set(PUBLIC_MODULE_FILE.findall(instruction))
    modules = set(PUBLIC_MODULE_PATH.findall(instruction)) | relative_files
    modules.update("app." + module for module in relative_files)
    if PUBLIC_APP_IMPORT.search(instruction):
        modules.add("app")
    for path in PUBLIC_APP_PATH.findall(instruction):
        parts = path.removesuffix(".py").split("/")
        modules.update("app." + ".".join(parts[:index]) for index in range(1, len(parts) + 1))
    return tuple(sorted(modules))


def _validated_test_file(task: TaskFiles, test_files: tuple[str, ...]) -> str | Rejected:
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
    return path


def _validated_oracle_files(task: TaskFiles) -> dict[str, bytes] | Rejected:
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


def convert(task: TaskFiles, *, test_files: tuple[str, ...] = TEST_FILES) -> ConvertedTask | Rejected:
    """Convert one self-contained Python task without preserving its legacy shell grader."""
    test_file = _validated_test_file(task, test_files)
    if isinstance(test_file, Rejected):
        return test_file
    solution_files = _validated_oracle_files(task)
    if isinstance(solution_files, Rejected):
        return solution_files

    data_files = {test_file: task.files[test_file], **task.under("setup_files/")}
    instruction = task.text(INSTRUCTION)
    return ConvertedTask(
        instruction=instruction,
        spec=PytestSpec(
            paths=(f"{TESTS_MOUNT}/{test_file.removeprefix('tests/')}",),
            python="python3",
            candidate_modules=public_candidate_modules(instruction),
        ),
        dockerfile=task.text(DOCKERFILE),
        tags=("code", "python", "unit-test", "kata"),
        language="python",
        data_files=data_files,
        solution_files=solution_files,
    )
