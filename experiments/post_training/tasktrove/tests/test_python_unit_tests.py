# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from verifyit.grade import Status, run
from verifyit.spec import PytestSpec, parse_spec, render_spec

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


def test_pytest_mode_distinguishes_collection_failure_wrong_answer_and_oracle(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    test_file = tests / "test_solution.py"
    test_file.write_text(VALID_TEST)
    workspace = tmp_path / "app"
    workspace.mkdir()
    solution = workspace / "solution.py"
    spec = PytestSpec(paths=(str(test_file),), python=sys.executable)

    spec_path = tests / "verifier.toml"
    spec_path.write_text(render_spec(spec))

    solution.write_text("")
    missing_implementation = run(spec_path, workspace)
    assert (missing_implementation.status, missing_implementation.reward) == (Status.SCORED, 0.0)

    solution.write_text("def add(left, right):\n    return left - right - 1\n")
    wrong_answer = run(spec_path, workspace)
    assert (wrong_answer.status, wrong_answer.reward) == (Status.SCORED, 0.0)

    solution.write_text("def add(left, right):\n    return left + right\n")
    oracle = run(spec_path, workspace)
    assert (oracle.status, oracle.reward) == (Status.SCORED, 1.0)


@pytest.mark.docker
@pytest.mark.timeout(300)
@pytest.mark.parametrize("module", ["inventory", "calculator", "factorial"])
def test_e2egit_contracts_accept_valid_implementations_and_reject_reported_defects(tmp_path, module):
    fixtures = Path(__file__).parents[1] / "fixtures"
    info = SourceInfo("DCAgent__exp_rpt_e2egit-v2", SourceVerdict.KEEP, FAMILY, "")
    record = convert_one(
        info, module, (fixtures / f"e2egit_{module}.tar.gz").read_bytes(), converter_index(), "e9859a4e80"
    )
    assert record.status == ConvertStatus.CONVERTED
    task = read_task_binary(record.task_binary)
    task.write_to(tmp_path)
    (tmp_path / "Dockerfile").write_text(task.text(DOCKERFILE) + "\nCOPY tests /tests\n")
    program = (fixtures / f"{module}_control.py").read_text()
    controls = [("correct", program, 1.0)]
    if module == "inventory":
        controls += [
            ("no-op", program.replace('raise ValueError("underflow")', "return"), 1.0),
            ("clamps", program.replace('raise ValueError("underflow")', "self.quantity = 0; return"), 0.0),
        ]
    elif module == "calculator":
        controls += [
            ("wrong-multiply", program.replace("return float(a * b)", "return 0.0"), 0.0),
            ("wrong-divide", program.replace("return float(a / b)", "return 0.0"), 0.0),
            ("no-docstrings", "\n".join(line for line in program.splitlines() if '"""' not in line), 0.0),
        ]
    else:
        controls += [
            (
                "iterative",
                program.replace(
                    "return number * calculate_factorial(number - 1)",
                    "result = 1\n    for value in range(1, number + 1):\n        result *= value\n    return result",
                ),
                0.0,
            ),
            ("no-zero-test", program, 0.0),
            ("no-student-tests", program, 0.0),
        ]
    image = f"atlas-e2egit-regression:{uuid.uuid4().hex}"
    subprocess.run(["docker", "build", "-t", image, str(tmp_path)], check=True, capture_output=True, text=True)
    try:
        # Benchmark tests execute only inside this owned container; controls are authored fixtures.
        for name, candidate, expected in controls:
            workspace = tmp_path / name
            workspace.mkdir()
            (workspace / f"{module}.py").write_text(candidate)
            if module == "factorial" and name != "no-student-tests":
                student_tests = (fixtures / "factorial_student_tests.py").read_text()
                if name == "no-zero-test":
                    student_tests = student_tests.replace("(0, 1),", "")
                (workspace / "tests").mkdir()
                (workspace / "tests/test_factorial.py").write_text(student_tests)
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--cpus",
                    "1",
                    "--memory",
                    "512m",
                    "--pids-limit",
                    "128",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "-v",
                    f"{workspace}:/app:ro",
                    "-e",
                    "PYTHONPATH=/app",
                    "-e",
                    "PYTHONDONTWRITEBYTECODE=1",
                    image,
                    "bash",
                    "-c",
                    "verifyit /tests/verifier.toml && cat /logs/verifier/reward.json",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
            assert json.loads(result.stdout)["reward"] == expected, name
    finally:
        subprocess.run(["docker", "image", "rm", image], check=True, capture_output=True)
