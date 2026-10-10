# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove archive conversion types converter failures and keeps the archive's oracle."""

import base64
import json

import pytest

from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import INSTRUCTION, SOLVE_SH
from experiments.post_training.task_curation.datasets.tasktrove.conversion.code_contests import convert_code_contests
from experiments.post_training.task_curation.datasets.tasktrove.conversion.result import ConvertedTask, archive_conversion
from experiments.post_training.task_curation.datasets.tasktrove.conversion.structured_outputs import convert_nemotron_structured_outputs
from experiments.post_training.task_curation.datasets.tasktrove.conversion.nl2bash import convert_nl2bash
from experiments.post_training.task_curation.datasets.tasktrove.conversion.taco import convert_taco
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection

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
