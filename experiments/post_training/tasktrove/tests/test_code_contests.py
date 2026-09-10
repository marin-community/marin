# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converter behaviour on the ``code_contests`` exemplar."""

import json
from pathlib import Path

from tasktrove_verify.spec import Compare, StdioSpec, parse_spec

from experiments.post_training.tasktrove.contract import INSTALL_MARKER, VERIFIER_TOML, VERIFY_TEST_SH
from experiments.post_training.tasktrove.convert import convert_one
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.converters.registry import converter_index
from experiments.post_training.tasktrove.sources import SourceInfo, SourceVerdict
from experiments.post_training.tasktrove.taskbinary import (
    DOCKERFILE,
    TEST_SH,
    read_task_binary,
    write_task_binary,
)
from experiments.post_training.tasktrove.verify import verify_task

FIXTURES = Path(__file__).parents[1] / "fixtures"
TOOL_REF = "0123abc"


def _fixture(name: str = "code_contests") -> bytes:
    return (FIXTURES / f"{name}.tar.gz").read_bytes()


def _info() -> SourceInfo:
    return SourceInfo("DCAgent__code-contests-noblock", SourceVerdict.KEEP, "competitive-programming", "")


def test_code_contests_exemplar_converts_to_stdio_spec_with_expected_tags():
    record = convert_one(_info(), "t.tar.gz", _fixture(), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.CONVERTED
    assert record.converter == "code_contests"
    assert record.mode == "stdio"
    assert record.tags == ["code", "competitive-programming", "stdio", "code-contests"]
    assert record.language == "python"

    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, StdioSpec)
    assert spec.command == "python3 /app/solution.py"
    assert spec.compare == Compare.EXACT

    assert task.text(TEST_SH) == VERIFY_TEST_SH
    assert "tests/test_state.py" not in task.files, "old grader code must not ship"
    assert "tests/test_data.json" not in task.files, "raw per-task data must not ship as-is"

    cases = sorted(p for p in task.files if p.startswith("tests/cases/"))
    assert len(cases) == 10  # 5 cases, input + output each
    assert task.text("tests/cases/input_0.txt") == "3\n((()))\n(())()\n()(()"
    assert task.text("tests/cases/output_0.txt") == "YES\nYES\nNO"

    dockerfile = task.text(DOCKERFILE)
    assert dockerfile.startswith("FROM ubuntu:24.04") and INSTALL_MARKER in dockerfile and TOOL_REF in dockerfile
    assert record.solution_binary is None and not record.has_solution


def test_code_contests_exemplar_passes_verification():
    record = convert_one(_info(), "t.tar.gz", _fixture(), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.CONVERTED
    assert verify_task(record.task_binary) is None


def test_two_hidden_cases_still_convert():
    task = read_task_binary(_fixture())
    data = json.loads(task.text("tests/test_data.json"))
    data["inputs"], data["outputs"] = data["inputs"][:2], data["outputs"][:2]
    task.files["tests/test_data.json"] = json.dumps(data).encode()
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.CONVERTED


def test_cases_that_are_all_prompt_samples_are_rejected():
    task = read_task_binary(_fixture())
    data = json.loads(task.text("tests/test_data.json"))
    data["inputs"], data["outputs"] = data["inputs"][:1], data["outputs"][:1]
    task.files["tests/test_data.json"] = json.dumps(data).encode()
    task.files["instruction.md"] = (task.text("instruction.md") + "\n\nSample input:\n" + data["inputs"][0]).encode()
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.GOLD_IN_INSTRUCTION and record.task_binary is None


def test_mismatched_input_output_counts_is_rejected_as_null_grader():
    task = read_task_binary(_fixture())
    data = json.loads(task.text("tests/test_data.json"))
    data["outputs"] = data["outputs"][:-1]
    task.files["tests/test_data.json"] = json.dumps(data).encode()
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.NULL_GRADER and record.task_binary is None


def test_missing_test_data_is_rejected_as_null_grader():
    task = read_task_binary(_fixture())
    del task.files["tests/test_data.json"]
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.NULL_GRADER and record.task_binary is None
