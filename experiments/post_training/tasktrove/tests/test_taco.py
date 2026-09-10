# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converter behaviour on the ``taco`` exemplar."""

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
    TaskFiles,
    read_task_binary,
    write_task_binary,
)
from experiments.post_training.tasktrove.verify import verify_task

FIXTURES = Path(__file__).parents[1] / "fixtures"
TOOL_REF = "0123abc"
FAMILY = "stdin-stdout"
SOURCE = "laion__exp_rpt_taco-v2"


def _fixture() -> bytes:
    return (FIXTURES / "taco.tar.gz").read_bytes()


def _info() -> SourceInfo:
    return SourceInfo(SOURCE, SourceVerdict.KEEP, FAMILY, "")


def _convert(blob: bytes):
    return convert_one(_info(), "t.tar.gz", blob, converter_index(), TOOL_REF)


def test_taco_exemplar_converts_to_stdio_spec_with_expected_tags():
    record = _convert(_fixture())
    assert record.status == ConvertStatus.CONVERTED
    assert record.converter == "taco"
    assert record.mode == "stdio"
    assert record.tags == ["code", "competitive-programming", "stdio", "taco"]
    assert record.language == "python"

    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, StdioSpec)
    assert spec.command == "python3 /app/solution.py"
    assert spec.compare == Compare.TOKENS

    assert task.text(TEST_SH) == VERIFY_TEST_SH
    assert "tests/inputs/input_0.txt" not in task.files, "old layout must not ship as-is"
    assert "tests/outputs/output_0.txt" not in task.files, "old layout must not ship as-is"

    cases = [p for p in task.files if p.startswith("tests/cases/input_")]
    assert len(cases) >= 2
    assert task.text("tests/cases/input_0.txt") == "5 3 10\n1 2 3 4 5\nRGBRR\n"
    assert task.text("tests/cases/output_0.txt") == "4\n"

    dockerfile = task.text(DOCKERFILE)
    assert dockerfile.startswith("FROM python:3.10-slim")
    assert INSTALL_MARKER in dockerfile and TOOL_REF in dockerfile

    assert record.has_solution and record.solution_binary is not None
    solution = read_task_binary(record.solution_binary)
    assert "solution/solve.sh" in solution.files
    assert "solution/solution.py" in solution.files
    assert "def main" in solution.text("solution/solution.py")


def test_taco_exemplar_passes_verification():
    record = _convert(_fixture())
    assert record.status == ConvertStatus.CONVERTED
    assert verify_task(record.task_binary) is None


def test_only_the_prompt_sample_as_a_case_is_rejected():
    task = read_task_binary(_fixture())
    keep = {"0"}
    for path in list(task.files):
        if path.startswith("tests/inputs/input_") or path.startswith("tests/outputs/output_"):
            number = path.rsplit("_", 1)[-1].removesuffix(".txt")
            if number not in keep:
                del task.files[path]
    record = _convert(write_task_binary(task))
    assert record.status == ConvertStatus.GOLD_IN_INSTRUCTION and record.task_binary is None


def test_mismatched_input_output_counts_is_rejected_as_null_grader():
    task = read_task_binary(_fixture())
    del task.files["tests/outputs/output_0.txt"]
    record = _convert(write_task_binary(task))
    assert record.status == ConvertStatus.NULL_GRADER and record.task_binary is None


def test_missing_solution_is_rejected_as_null_grader():
    task = read_task_binary(_fixture())
    del task.files["solution/solution.py"]
    record = _convert(write_task_binary(task))
    assert record.status == ConvertStatus.NULL_GRADER and record.task_binary is None


def test_function_call_style_solution_is_rejected_as_unsupported_variant():
    """A LeetCode/Codewars-style solution defines a function but never reads stdin."""
    task = read_task_binary(_fixture())
    task.files["solution/solution.py"] = b"class Solution:\n    def foo(self, x):\n        return x + 1\n"
    record = _convert(write_task_binary(task))
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT and record.task_binary is None


def test_float_tolerance_instruction_uses_float_compare():
    task = read_task_binary(_fixture())
    original = task.text("instruction.md")
    task.files["instruction.md"] = (
        original + "\nThe result will be considered correct if the absolute or relative error does not exceed 1e-6.\n"
    ).encode()
    record = _convert(write_task_binary(task))
    assert record.status == ConvertStatus.CONVERTED
    spec = parse_spec(read_task_binary(record.task_binary).text(VERIFIER_TOML))
    assert isinstance(spec, StdioSpec)
    assert spec.compare == Compare.FLOAT


def test_dockerfile_passthrough_leaves_pytest_install():
    """No rewardkit/litellm lines to drop for taco: the Dockerfile ships unmodified."""
    record = _convert(_fixture())
    dockerfile = TaskFiles(read_task_binary(record.task_binary).files).text(DOCKERFILE)
    assert "pip install --no-cache-dir pytest" in dockerfile
