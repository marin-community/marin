# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CodeNet conversion preserves source token comparison and private source bytes."""

import base64
from dataclasses import replace

from verifyit.grade import grade
from verifyit.spec import StdioSpec, spec_from_table

from experiments.post_training.task_curation.executable import converted_row
from experiments.post_training.task_curation.source_bindings import convert_codenet


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
    row = converted_row(raw, "codenet", converter=convert_codenet)
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
