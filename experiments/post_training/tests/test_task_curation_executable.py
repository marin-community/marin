# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Legacy conversion outputs run with the standalone verifier environment."""

import base64
import json
from dataclasses import replace
from pathlib import Path

from verifyit.grade import grade
from verifyit.spec import ScriptSpec, spec_from_table

from experiments.post_training.task_curation_executable import converted_row
from experiments.post_training.tasktrove.taskbinary import read_task_binary


def test_nl2bash_checker_grades_capture_and_preserves_conversion_provenance(tmp_path):
    fixture = Path(__file__).parents[1] / "tasktrove/fixtures/nl2bash.tar.gz"
    source = read_task_binary(fixture.read_bytes()).files
    raw = {
        "instruction": source["instruction.md"].decode(),
        "files": {path: base64.b64encode(content).decode() for path, content in source.items()},
    }
    row = converted_row(raw, "nl2bash")
    assert row["files"] == raw["files"]
    converted = row["converted"]
    for path, encoded in converted["data_files"].items():
        output = tmp_path / path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(base64.b64decode(encoded))
    change = next(
        item for item in converted["normalization_changes"] if item["field"] == "data_files/tests/nl2bash_check.py"
    )
    original_checker = base64.b64decode(change["original"])
    assert b"TASKTROVE_TESTS_DIR" in original_checker
    assert base64.b64decode(change["replacement"]) == (tmp_path / "tests/nl2bash_check.py").read_bytes()
    spec = spec_from_table(converted["grader_spec"])
    assert isinstance(spec, ScriptSpec)
    capture = tmp_path / "capture.txt"
    local = replace(spec, args=(str(capture),), workspace=str(tmp_path))
    expected = json.loads((tmp_path / "tests/nl2bash_expected.json").read_text())["expected_output"]
    capture.write_text(expected)
    assert grade(local, tmp_path / "tests", tmp_path).reward == 1.0
    capture.write_text("unexpected error: missing input\n")
    assert grade(local, tmp_path / "tests", tmp_path).reward == 0.0
