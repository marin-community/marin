# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converter behaviour on checked-in exemplar tasks."""

import json
from pathlib import Path

from tasktrove_verify.spec import MathSpec, McqSpec, parse_spec

from experiments.post_training.tasktrove.convert import convert_one
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.converters.registry import converter_index
from experiments.post_training.tasktrove.dataset import SourceInfo, SourceVerdict, load_source_verdicts
from experiments.post_training.tasktrove.task_format import INSTALL_MARKER, VERIFIER_TOML, VERIFY_TEST_SH
from experiments.post_training.tasktrove.taskbinary import (
    DOCKERFILE,
    TEST_SH,
    TaskFiles,
    read_task_binary,
    write_task_binary,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"
TOOL_REF = "0123abc"


def _fixture(name: str) -> bytes:
    return (FIXTURES / f"{name}.tar.gz").read_bytes()


def _info(source: str, family: str) -> SourceInfo:
    return SourceInfo(source, SourceVerdict.KEEP, family, "")


def test_mcqa_exemplar_converts_to_mcq_spec():
    record = convert_one(
        _info("mcqa", "qa-short-answer"), "t.tar.gz", _fixture("nemotron_mcqa"), converter_index(), TOOL_REF
    )
    assert record.status == ConvertStatus.CONVERTED
    assert record.converter == "nemotron_mcqa" and record.mode == "mcq" and record.tags == ["qa", "mcq", "nemotron"]
    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, McqSpec) and spec.expected.isalpha() and spec.options >= 2
    assert task.text(TEST_SH) == VERIFY_TEST_SH
    assert "tests/verifier.py" not in task.files, "old grader code must not ship"
    dockerfile = task.text(DOCKERFILE)
    assert dockerfile.startswith("# DO NOT EDIT") and INSTALL_MARKER in dockerfile and TOOL_REF in dockerfile
    assert record.has_solution and record.solution_binary is not None
    solve = read_task_binary(record.solution_binary).text("solution/solve.sh")
    assert f"Answer: {spec.expected}" in solve and spec.output in solve


def test_math_exemplar_strips_solution_into_its_own_column():
    record = convert_one(
        _info("math", "math-answer"), "t.tar.gz", _fixture("nemotron_math"), converter_index(), TOOL_REF
    )
    assert record.status == ConvertStatus.CONVERTED
    task = read_task_binary(record.task_binary)
    assert isinstance(parse_spec(task.text(VERIFIER_TOML)), MathSpec)
    assert not task.has_solution and record.has_solution
    assert "solution/solve.sh" in read_task_binary(record.solution_binary).files
    assert "[answer]" in task.text(DOCKERFILE) or "tasktrove-verify[answer]" in task.text(DOCKERFILE)


def test_math_instruction_names_only_the_graded_answer_file():
    record = convert_one(
        _info("math", "math-answer"), "t.tar.gz", _fixture("nemotron_math"), converter_index(), TOOL_REF
    )
    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, MathSpec)
    instruction = task.text("instruction.md")
    assert "/app/solution.txt" not in instruction
    assert spec.output in instruction


def test_multi_letter_gold_is_rejected_not_converted():
    task = read_task_binary(_fixture("nemotron_mcqa"))
    data = json.loads(task.text("tests/verifier_data.json"))
    data["expected_answer"] = "AB"
    task.files["tests/verifier_data.json"] = json.dumps(data).encode()
    record = convert_one(
        _info("mcqa", "qa-short-answer"), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF
    )
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT and record.task_binary is None


def test_dropped_source_passes_through_unconverted():
    info = SourceInfo("mcqa", SourceVerdict.DROP, "qa-short-answer", "")
    record = convert_one(info, "t.tar.gz", _fixture("nemotron_mcqa"), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.DROPPED_SOURCE and record.task_binary is None


def test_reviewed_bad_row_is_recorded_before_conversion():
    record = convert_one(
        _info("DCAgent__code-contests-noblock", "competitive-programming"),
        "code_contests-4395",
        _fixture("code_contests"),
        converter_index(),
        TOOL_REF,
    )

    assert record.status == ConvertStatus.REVIEWED_DEFECT
    assert record.task_binary is None
    assert "constructive" in record.error


def test_prompt_injection_source_is_dropped():
    info = load_source_verdicts()["laion__nemotron-gym-agentic-indirect-prompt-injection-v3"]

    record = convert_one(info, "task.tar.gz", _fixture("prompt_injection"), converter_index(), TOOL_REF)

    assert record.status == ConvertStatus.DROPPED_SOURCE


def test_unknown_key_is_reported_not_guessed():
    record = convert_one(
        _info("mcqa", "no-such-family"), "t.tar.gz", _fixture("nemotron_mcqa"), converter_index(), TOOL_REF
    )
    assert record.status == ConvertStatus.NO_CONVERTER


def test_dockerfile_edit_is_idempotent_across_tasks():
    index = converter_index()
    a = convert_one(_info("mcqa", "qa-short-answer"), "a.tar.gz", _fixture("nemotron_mcqa"), index, TOOL_REF)
    b = convert_one(_info("mcqa", "qa-short-answer"), "b.tar.gz", _fixture("nemotron_mcqa"), index, TOOL_REF)
    assert a.dockerfile_id == b.dockerfile_id
    assert TaskFiles(read_task_binary(a.task_binary).files).text(DOCKERFILE) == read_task_binary(b.task_binary).text(
        DOCKERFILE
    )
