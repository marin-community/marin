# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Legacy conversion outputs run with the standalone verifier environment."""

import base64
import json
from dataclasses import replace
from pathlib import Path

import pytest
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection
from verifyit.grade import grade
from verifyit.spec import ScriptSpec, spec_from_table

from experiments.post_training.task_curation.datasets.tasktrove.conversion import converted_row
from experiments.post_training.task_curation.sources import rl_data_pipelines
from experiments.post_training.tasktrove.converters.nl2bash import convert_nl2bash
from experiments.post_training.tasktrove.taskbinary import read_task_binary

SOURCES = {source.source_key: source for source in rl_data_pipelines().values()}


@pytest.mark.parametrize("kind", [ImportFailureKind.SOURCE_DEFECT, ImportFailureKind.CONVERTER_ERROR])
def test_converter_failures_preserve_raw_evidence_and_typed_cause(kind):
    fixture = Path(__file__).parents[2] / "tasktrove/fixtures/nl2bash.tar.gz"
    files = dict(read_task_binary(fixture.read_bytes()).files)
    if kind is ImportFailureKind.SOURCE_DEFECT:
        data = json.loads(files["tests/verifier_data.json"])
        data["expected_output"] = None
        files["tests/verifier_data.json"] = json.dumps(data).encode()
    else:
        del files["tests/verifier_data.json"]
    raw = {
        "instruction": files["instruction.md"].decode(),
        "files": {path: base64.b64encode(content).decode() for path, content in files.items()},
    }
    result = converted_row(raw, converter=convert_nl2bash)
    failure = ImportRejection.model_validate(result["conversion_rejection"])
    assert failure.kind is kind
    assert failure.detail
    assert "converted" not in result
    assert result["files"] == raw["files"] and result["instruction"] == raw["instruction"]


def test_nl2bash_checker_grades_capture_and_preserves_conversion_provenance(tmp_path):
    fixture = Path(__file__).parents[2] / "tasktrove/fixtures/nl2bash.tar.gz"
    source = read_task_binary(fixture.read_bytes()).files
    raw = {
        "instruction": source["instruction.md"].decode(),
        "files": {path: base64.b64encode(content).decode() for path, content in source.items()},
    }
    row = converted_row(raw, converter=convert_nl2bash)
    assert row["files"] == raw["files"]
    converted = row["converted"]
    for path, encoded in converted["data_files"].items():
        output = tmp_path / path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(base64.b64decode(encoded))
    spec = spec_from_table(converted["grader_spec"])
    assert isinstance(spec, ScriptSpec)
    capture = tmp_path / "capture.txt"
    local = replace(spec, args=(str(capture),), workspace=str(tmp_path))
    expected = json.loads((tmp_path / "tests/nl2bash_expected.json").read_text())["expected_output"]
    capture.write_text(expected)
    assert grade(local, tmp_path / "tests", tmp_path).reward == 1.0
    capture.write_text("unexpected error: missing input\n")
    assert grade(local, tmp_path / "tests", tmp_path).reward == 0.0
