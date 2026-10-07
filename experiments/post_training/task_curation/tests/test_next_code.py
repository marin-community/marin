# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CodeNet conversion preserves source token comparison and private source bytes."""

import base64
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from verifyit.grade import grade
from verifyit.spec import Compare, StdioSpec, spec_from_table

from experiments.post_training.task_curation.datasets.tasktrove.conversion import converted_row
from experiments.post_training.tasktrove.converters.codeforces import convert_codeforces
from experiments.post_training.tasktrove.converters.converted_task import ConvertedTask, ConvertStatus, Rejected
from experiments.post_training.tasktrove.converters.stdio_cases import SOLUTION_COMMAND
from experiments.post_training.tasktrove.taskbinary import TaskFiles, read_task_binary


def convert_codenet(task: TaskFiles) -> ConvertedTask | Rejected:
    """Extract CodeNet cases and bind whitespace-token output comparison."""
    converted = convert_codeforces(task)
    if isinstance(converted, Rejected):
        return converted
    cases = sum(path.startswith("tests/cases/input_") for path in converted.data_files)
    if cases < 2:
        return Rejected(ConvertStatus.NULL_GRADER, "CodeNet source requires at least two input/output pairs")
    return replace(
        converted,
        spec=StdioSpec(command=SOLUTION_COMMAND, compare=Compare.TOKENS, per_case_timeout=30.0, min_cases=2),
        tags=("code", "competitive-programming", "stdio", "codenet"),
    )


def test_codenet_token_grader_accepts_valid_multiline_output(tmp_path):
    source = {
        "instruction.md": b"Read an integer N and print N and its successor. Write /app/solution.py.",
        "environment/Dockerfile": b"FROM python:3.12\n",
        "tests/inputs/input_0.txt": b"8\n",
        "tests/outputs/output_0.txt": b"8 9\n",
        "tests/inputs/input_1.txt": b"11\n",
        "tests/outputs/output_1.txt": b"11 12\n",
        "solution/solve.sh": b"#!/bin/bash\ntrue\n",
        "metadata.json": b'{"origin":"codenet"}',
    }
    raw = {
        "instruction": source["instruction.md"].decode(),
        "files": {path: base64.b64encode(content).decode() for path, content in source.items()},
    }
    row = converted_row(raw, converter=convert_codenet)
    assert {path: base64.b64decode(content) for path, content in row["files"].items()} == source
    converted = row["converted"]
    for path, encoded in converted["data_files"].items():
        output = tmp_path / path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(base64.b64decode(encoded))
    program = tmp_path / "solution.py"
    program.write_text("n=int(input()); print(n); print(n+1)\n")
    spec = spec_from_table(converted["grader_spec"])
    assert isinstance(spec, StdioSpec)
    local = replace(spec, command=f"python3 {program}", workspace=str(tmp_path))
    assert grade(local, tmp_path / "tests", tmp_path).reward == 1.0
    program.write_text("print(0)\n")
    assert grade(local, tmp_path / "tests", tmp_path).reward == 0.0
    assert base64.b64decode(converted["control_files"]["solution/solve.sh"]) == source["solution/solve.sh"]


@pytest.mark.parametrize("exit_code, expected_reward", [(0, 1.0), (7, 0.0)])
def test_codenet_original_runner_and_converted_grader_agree_on_matching_stdout_exit(
    tmp_path, exit_code, expected_reward
):
    # Original pinned TaskTrove row; this table-driven submission probes the
    # scorer's exit contract and is not offered as a source golden.
    original = read_task_binary((Path(__file__).parent / "fixtures/codenet.tar.gz").read_bytes())
    converted = convert_codenet(original)
    assert isinstance(converted, ConvertedTask) and isinstance(converted.spec, StdioSpec)
    for path, data in {**original.under("tests/"), **converted.data_files}.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    answers = {
        data.decode(): original.files[path.replace("/inputs/input_", "/outputs/output_")].decode()
        for path, data in original.files.items()
        if path.startswith("tests/inputs/input_")
    }
    workspace = tmp_path / "app"
    workspace.mkdir()
    solution = workspace / "solution.py"
    solution.write_text(
        f"import sys\nanswers={answers!r}\nsys.stdout.write(answers[sys.stdin.read()])\nsys.exit({exit_code})\n"
    )
    source_runner = original.text("tests/test.sh")
    for old, new in (
        ("/logs/verifier", str(tmp_path / "logs")),
        ("/tmp/codenet-actual.txt", str(tmp_path / "actual.txt")),
        ("/tests", str(tmp_path / "tests")),
        ("/app", str(workspace)),
    ):
        source_runner = source_runner.replace(old, new)
    subprocess.run(["bash", "-c", source_runner], check=False, capture_output=True)
    assert float((tmp_path / "logs/reward.txt").read_text()) == expected_reward
    local = replace(converted.spec, command=f"python3 {solution}", workspace=str(workspace))
    reward = grade(local, tmp_path / "tests", workspace)
    assert reward.reward == expected_reward
