# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converter behaviour on the checked-in ``nemotron_competitive`` exemplar.

The checked-in exemplar is a real task pulled straight from the source template and carries a
single stdin/stdout case that is the sample printed in the problem statement, a genuine
``gold_in_instruction`` rejection. The accepted-path tests add one hidden case, since ``stdio`` has
no grading probe (:mod:`tasktrove_verify.grade` only knows output-file modes) and an extra case
does not change what :func:`verify_task` checks.
"""

import json
from pathlib import Path

from tasktrove_verify.spec import Compare, StdioSpec, parse_spec

from experiments.post_training.tasktrove.convert import convert_one
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.converters.registry import converter_index
from experiments.post_training.tasktrove.dataset import SourceInfo, SourceVerdict
from experiments.post_training.tasktrove.task_format import INSTALL_MARKER, VERIFIER_TOML, VERIFY_TEST_SH
from experiments.post_training.tasktrove.taskbinary import (
    DOCKERFILE,
    TEST_SH,
    read_task_binary,
    write_task_binary,
)
from experiments.post_training.tasktrove.verify import verify_task

FIXTURES = Path(__file__).parents[1] / "fixtures"
TOOL_REF = "0123abc"
FAMILY = "competitive-programming"
SOURCE = "laion__nemotron-gym-competitive-coding-v2"


def _fixture() -> bytes:
    return (FIXTURES / "nemotron_competitive.tar.gz").read_bytes()


def _info() -> SourceInfo:
    return SourceInfo(SOURCE, SourceVerdict.KEEP, FAMILY, "")


HIDDEN_CASES = 2


def _padded_fixture() -> bytes:
    """The real exemplar's sample case plus one case the prompt does not show."""
    task = read_task_binary(_fixture())
    data = json.loads(task.text("tests/verifier_data.json"))
    data["inputs"].append("2\n1 2\n1\n1\n")
    data["outputs"].append("2\n")
    task.files["tests/verifier_data.json"] = json.dumps(data).encode()
    return write_task_binary(task)


def test_exemplar_with_only_the_prompt_sample_is_rejected():
    record = convert_one(_info(), "t.tar.gz", _fixture(), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.GOLD_IN_INSTRUCTION and record.task_binary is None


def test_padded_exemplar_converts_to_stdio_spec_with_expected_tags():
    record = convert_one(_info(), "t.tar.gz", _padded_fixture(), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.CONVERTED
    assert record.mode == "stdio"
    assert record.tags == ["code", "competitive-programming", "stdio", "nemotron"]
    assert record.language == "python"

    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, StdioSpec)
    assert spec.command == "python3 /app/solution.py"
    assert spec.compare == Compare.EXACT

    assert task.text(TEST_SH) == VERIFY_TEST_SH
    for old_grader_file in ("tests/verifier.py", "tests/validate_verifier_data.py"):
        assert old_grader_file not in task.files, "old grader code must not ship"

    dockerfile = task.text(DOCKERFILE)
    assert INSTALL_MARKER in dockerfile and TOOL_REF in dockerfile

    for index in range(HIDDEN_CASES):
        assert f"tests/cases/input_{index}.txt" in task.files
        assert f"tests/cases/output_{index}.txt" in task.files

    assert record.solution_binary is None and not record.has_solution


def test_padded_exemplar_passes_verification():
    record = convert_one(_info(), "t.tar.gz", _padded_fixture(), converter_index(), TOOL_REF)
    assert verify_task(record.task_binary) is None


def test_mismatched_inputs_and_outputs_are_rejected_as_null_grader():
    task = read_task_binary(_fixture())
    data = json.loads(task.text("tests/verifier_data.json"))
    data["outputs"] = data["outputs"] + ["extra"]
    task.files["tests/verifier_data.json"] = json.dumps(data).encode()
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.NULL_GRADER and record.task_binary is None
