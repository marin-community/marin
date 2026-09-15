# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys

from tasktrove_verify.modes import grade_pytest
from tasktrove_verify.spec import PytestSpec, parse_spec

from experiments.post_training.tasktrove.convert import convert_one
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.converters.registry import converter_index
from experiments.post_training.tasktrove.dataset import SourceInfo, SourceVerdict
from experiments.post_training.tasktrove.task_format import VERIFIER_TOML, VERIFY_TEST_SH
from experiments.post_training.tasktrove.taskbinary import (
    DOCKERFILE,
    INSTRUCTION,
    TEST_SH,
    TaskFiles,
    read_task_binary,
    write_task_binary,
)

TOOL_REF = "0123abc"
SOURCE = "DCAgent__exp_rpt_unitsyn-python-large-v2"
FAMILY = "unit-test-gen"
TEST_FILE = "tests/test_solution.py"
VALID_TEST = """from solution import add


def test_add():
    assert add(2, 3) == 5
"""


def _task(test: str = VALID_TEST, solution: str | None = "def add(left, right):\n    return left + right\n") -> bytes:
    files = {
        INSTRUCTION: b"Implement add in /app/solution.py.",
        DOCKERFILE: b"FROM python:3.12-slim\nWORKDIR /app\nRUN pip install pytest\n",
        TEST_SH: b"#!/bin/bash\npytest /tests/test_solution.py\n",
        TEST_FILE: test.encode(),
        "task.toml": b'verifier = "pytest"\n',
    }
    if solution is not None:
        files["solution/solution.py"] = solution.encode()
    return write_task_binary(TaskFiles(files))


def _convert(blob: bytes | None = None):
    info = SourceInfo(SOURCE, SourceVerdict.KEEP, FAMILY, "")
    return convert_one(info, "task.tar.gz", blob if blob is not None else _task(), converter_index(), TOOL_REF)


def test_python_unit_task_converts_to_pytest_with_kata_tag_and_oracle():
    record = _convert()
    assert record.status == ConvertStatus.CONVERTED
    assert record.converter == "python_unit_tests"
    assert record.mode == "pytest"
    assert record.tags == ["code", "python", "unit-test", "kata"]
    assert record.language == "python"

    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, PytestSpec)
    assert spec.paths == ("/tests/test_solution.py",)
    assert spec.python == "/opt/tasktrove-pytest/bin/python"
    assert task.text(TEST_SH) == VERIFY_TEST_SH
    assert TEST_FILE in task.files
    assert "pytest-json-report" in task.text(DOCKERFILE)

    assert record.solution_binary is not None
    solution = read_task_binary(record.solution_binary)
    assert solution.text("solution/solution.py").startswith("def add")
    assert "cp /solution/solution.py /app/solution.py" in solution.text("solution/solve.sh")


def test_task_without_shipped_oracle_still_converts():
    record = _convert(_task(solution=None))
    assert record.status == ConvertStatus.CONVERTED
    assert record.solution_binary is None


def test_invalid_test_is_rejected():
    record = _convert(_task(test="def test_broken(:\n"))
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT
    assert "not valid Python" in record.error


def test_file_without_local_test_function_is_rejected_as_null_grader():
    record = _convert(_task(test="def helper():\n    return True\n"))
    assert record.status == ConvertStatus.NULL_GRADER
    assert "defines no local test function" in record.error


def test_pytest_mode_rejects_empty_implementation_and_accepts_oracle(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    test_file = tests / "test_solution.py"
    test_file.write_text(VALID_TEST)
    workspace = tmp_path / "app"
    workspace.mkdir()
    solution = workspace / "solution.py"
    spec = PytestSpec(paths=(str(test_file),), python=sys.executable)

    solution.write_text("")
    assert grade_pytest.grade(spec, tests, workspace).reward == 0.0

    solution.write_text("def add(left, right):\n    return left + right\n")
    assert grade_pytest.grade(spec, tests, workspace).reward == 1.0
