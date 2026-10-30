# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove archive conversion types converter failures and keeps the archive's oracle."""

import base64
import json
import sys
from dataclasses import replace

import pytest
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection
from verifyit.grade import Status
from verifyit.modes import grade_pytest
from verifyit.spec import PytestSpec

from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import INSTRUCTION, SOLVE_SH
from experiments.post_training.task_curation.datasets.tasktrove.conversion.code_contests import convert_code_contests
from experiments.post_training.task_curation.datasets.tasktrove.conversion.nl2bash import convert_nl2bash
from experiments.post_training.task_curation.datasets.tasktrove.conversion.python_unit_tests import (
    convert as convert_python,
)
from experiments.post_training.task_curation.datasets.tasktrove.conversion.result import (
    ConvertedTask,
    archive_conversion,
)
from experiments.post_training.task_curation.datasets.tasktrove.conversion.structured_outputs import (
    convert_nemotron_structured_outputs,
)
from experiments.post_training.task_curation.datasets.tasktrove.conversion.taco import convert_taco

DOCKERFILE = b"FROM python:3.12-slim\n"
SUM_PROMPT = b"Read two integers and print their sum to stdout in /app/solution.py."
ARCHIVE_ORACLE = b"#!/bin/bash\ncp /solution/solution.py /app/solution.py\n"
SUM_SOLUTION = b"a, b = map(int, input().split())\nprint(a + b)\n"


def archive_data(files: dict[str, bytes]) -> dict:
    """An archive row as ``unpack_task_binary`` decodes it."""
    files = {INSTRUCTION: SUM_PROMPT, "environment/Dockerfile": DOCKERFILE, **files}
    return {
        "instruction": files[INSTRUCTION].decode(),
        "files": {path: base64.b64encode(data).decode() for path, data in files.items()},
    }


def verifier_data(value: dict) -> dict[str, bytes]:
    return {"tests/verifier_data.json": json.dumps(value).encode()}


@pytest.mark.parametrize(
    "convert, files, kind, reason",
    [
        (convert_nl2bash, verifier_data({"expected_output": None}), ImportFailureKind.SOURCE_DEFECT, "null_grader"),
        (convert_nl2bash, {}, ImportFailureKind.CONVERTER_ERROR, "converter_error"),
        (
            convert_nemotron_structured_outputs,
            verifier_data({"schema": {"type": "object"}, "schema_type": "ini"}),
            ImportFailureKind.UNSUPPORTED,
            "unsupported_variant",
        ),
        (
            convert_code_contests,
            {
                INSTRUCTION: b"Read two integers, such as 3 4, and print their sum.",
                "tests/test_data.json": json.dumps({"inputs": ["3 4"], "outputs": ["7"]}).encode(),
            },
            ImportFailureKind.SOURCE_DEFECT,
            "gold_in_instruction",
        ),
    ],
    ids=["missing_expected_output", "missing_verifier_data", "unknown_schema_type", "samples_only"],
)
def test_converter_failures_become_typed_rejections(convert, files, kind, reason):
    rejection = archive_conversion(archive_data(files), convert)
    assert isinstance(rejection, ImportRejection)
    assert (rejection.kind, rejection.reason) == (kind, reason)
    assert rejection.detail


def test_archive_solution_files_are_the_oracle_when_the_converter_supplies_none():
    files = {
        "tests/test_data.json": json.dumps({"inputs": ["10 20\n"], "outputs": ["30\n"]}).encode(),
        SOLVE_SH: ARCHIVE_ORACLE,
        "solution/solution.py": SUM_SOLUTION,
    }
    converted = archive_conversion(archive_data(files), convert_code_contests)
    assert isinstance(converted, ConvertedTask)
    assert converted.solution_files == {SOLVE_SH: ARCHIVE_ORACLE, "solution/solution.py": SUM_SOLUTION}


def test_converter_oracle_replaces_the_archive_oracle():
    files = {
        "tests/inputs/input_0.txt": b"10 20\n",
        "tests/outputs/output_0.txt": b"30\n",
        SOLVE_SH: b"#!/bin/bash\nexit 1\n",
        "solution/solution.py": SUM_SOLUTION,
    }
    converted = archive_conversion(archive_data(files), convert_taco)
    assert isinstance(converted, ConvertedTask)
    assert converted.solution_files[SOLVE_SH] != files[SOLVE_SH]
    assert converted.solution_files["solution/solution.py"] == SUM_SOLUTION


@pytest.mark.parametrize(
    "module, location",
    [
        ("calc", "calc.py"),
        ("arithmetic.calc", "/app/arithmetic/calc.py"),
        ("app.practical", "/app/practical.py"),
        ("calculator", "calculator.py"),
        ("app.calculator", "calculator.py"),
    ],
)
def test_python_archive_missing_public_module_scores_zero_then_implementation_passes(tmp_path, module, location):
    tests = f"from {module} import add\ndef test_add():\n    assert add(2, 3) == 5\n".encode()
    converted = archive_conversion(
        archive_data(
            {INSTRUCTION: f"Implement add(a, b) in {location} under /app.".encode(), "tests/test_solution.py": tests}
        ),
        convert_python,
    )
    assert isinstance(converted, ConvertedTask)
    assert isinstance(converted.spec, PytestSpec)
    workspace = tmp_path / "app"
    workspace.mkdir()
    test = tmp_path / "test_solution.py"
    test.write_bytes(converted.data_files["tests/test_solution.py"])
    spec = replace(converted.spec, paths=(str(test),), python=sys.executable)
    empty = grade_pytest.grade(spec, tmp_path, workspace)
    assert (empty.status, empty.reward) == (Status.SCORED, 0.0)
    implementation = workspace.joinpath(*module.removeprefix("app.").split(".")).with_suffix(".py")
    implementation.parent.mkdir(parents=True, exist_ok=True)
    implementation.write_text("def add(a, b):\n    return a + b\n")
    passing = grade_pytest.grade(spec, tmp_path, workspace)
    assert (passing.status, passing.reward) == (Status.SCORED, 1.0)


def test_python_archive_app_export_failure_scores_zero_then_package_passes(tmp_path):
    converted = archive_conversion(
        archive_data(
            {
                INSTRUCTION: b"Implement create_app in /app/app.py, or re-export it from /app/__init__.py.",
                "tests/test_solution.py": (
                    b"from app import create_app\ndef test_app():\n    assert create_app() == 'ready'\n"
                ),
            }
        ),
        convert_python,
    )
    assert isinstance(converted, ConvertedTask)
    assert isinstance(converted.spec, PytestSpec)
    assert {"app", "app.app"}.issubset(converted.spec.candidate_modules)

    workspace = tmp_path / "app"
    workspace.mkdir()
    test = tmp_path / "test_solution.py"
    test.write_bytes(converted.data_files["tests/test_solution.py"])
    spec = replace(converted.spec, paths=(str(test),), python=sys.executable)
    missing = grade_pytest.grade(spec, tmp_path, workspace)
    assert (missing.status, missing.reward) == (Status.SCORED, 0.0)

    (workspace / "__init__.py").write_text("def create_app():\n    return 'ready'\n")
    passing = grade_pytest.grade(spec, tmp_path, workspace)
    assert (passing.status, passing.reward) == (Status.SCORED, 1.0)

    (workspace / "__init__.py").unlink()
    (workspace / "app.py").write_text("def create_app():\n    return 'ready'\n")
    passing = grade_pytest.grade(spec, tmp_path, workspace)
    assert (passing.status, passing.reward) == (Status.SCORED, 1.0)


def test_python_archive_does_not_infer_unstated_module_names():
    converted = archive_conversion(
        archive_data(
            {
                INSTRUCTION: b"Implement a Calculator in the /app directory.",
                "tests/test_solution.py": (
                    b"from calculator import Calculator\ndef test_add():\n    assert Calculator().add(2, 3) == 5\n"
                ),
            }
        ),
        convert_python,
    )
    assert isinstance(converted, ConvertedTask)
    assert isinstance(converted.spec, PytestSpec)
    assert "calculator" not in converted.spec.candidate_modules
    assert "app.calculator" not in converted.spec.candidate_modules
