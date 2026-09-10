# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron open-QA converter behaviour on the checked-in exemplar."""

import json
from pathlib import Path

from tasktrove_verify.spec import JudgeSpec, parse_spec

from experiments.post_training.tasktrove.contract import VERIFIER_TOML
from experiments.post_training.tasktrove.convert import convert_one
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.converters.registry import converter_index
from experiments.post_training.tasktrove.sources import SourceInfo, SourceVerdict
from experiments.post_training.tasktrove.taskbinary import (
    DOCKERFILE,
    read_task_binary,
    write_task_binary,
)
from experiments.post_training.tasktrove.verify import verify_task

FIXTURES = Path(__file__).parents[1] / "fixtures"
TOOL_REF = "0123abc"


def _fixture(name: str) -> bytes:
    return (FIXTURES / f"{name}.tar.gz").read_bytes()


def _info(source: str, family: str) -> SourceInfo:
    return SourceInfo(source, SourceVerdict.KEEP, family, "")


def _convert(family: str = "qa-short-answer"):
    return convert_one(
        _info("knowledge-openqa", family), "t.tar.gz", _fixture("nemotron_openqa"), converter_index(), TOOL_REF
    )


def test_exemplar_converts_to_judge_spec_with_exact_gate():
    record = _convert()
    assert record.status == ConvertStatus.CONVERTED
    assert record.converter == "nemotron_openqa" and record.mode == "judge"
    assert record.tags == ["qa", "openqa", "judge", "reference", "nemotron", "knowledge"]
    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, JudgeSpec)
    assert spec.exact_gate is True
    assert len(spec.references) == 1
    assert spec.references[0].startswith("Yes, if the non-state actor")
    assert spec.references[0].endswith("suppress the threat.")
    assert spec.output == "/app/response.txt"
    assert "Under the narrow interpretation of Article 51" in spec.question


def test_exemplar_drops_old_grader_files_and_dependency():
    record = _convert()
    task = read_task_binary(record.task_binary)
    for path in ("tests/exact_gate", "tests/sitecustomize.py", "tests/judge.toml", "tests/verifier_data.json"):
        assert path not in task.files, f"old grader file {path} must not ship"
    dockerfile = task.text(DOCKERFILE)
    assert "rewardkit" not in dockerfile.lower()
    assert "litellm" not in dockerfile.lower()
    assert "tasktrove-verify[judge]" in dockerfile


def test_second_registered_key_routes_to_the_same_converter():
    record = _convert(family="llm-judge-freeform")
    assert record.status == ConvertStatus.CONVERTED
    assert record.converter == "nemotron_openqa" and record.mode == "judge"


def test_verify_task_passes_the_real_grader_in_process():
    record = _convert()
    assert verify_task(record.task_binary) is None


def test_reference_answer_shape_routes_to_the_science_tag():
    task = read_task_binary(_fixture("nemotron_openqa"))
    data = json.loads(task.text("tests/verifier_data.json"))
    del data["expected_answers"]
    data["reference_answer"] = "Ohmic heating in the transformer windings reduces secondary voltage under load."
    task.files["tests/verifier_data.json"] = json.dumps(data).encode()
    record = convert_one(
        _info("science-openq", "llm-judge-freeform"), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF
    )
    assert record.status == ConvertStatus.CONVERTED
    assert record.tags == ["qa", "openqa", "judge", "reference", "nemotron", "science"]
    spec = parse_spec(read_task_binary(record.task_binary).text(VERIFIER_TOML))
    assert isinstance(spec, JudgeSpec)
    assert spec.references == ("Ohmic heating in the transformer windings reduces secondary voltage under load.",)


def test_markdown_emphasis_around_a_reference_is_trimmed():
    task = read_task_binary(_fixture("nemotron_openqa"))
    data = json.loads(task.text("tests/verifier_data.json"))
    data["expected_answers"] = ["** The windings heat up. **", "**"]
    task.files["tests/verifier_data.json"] = json.dumps(data).encode()
    record = convert_one(
        _info("science-openq", "llm-judge-freeform"), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF
    )
    assert record.status == ConvertStatus.CONVERTED
    spec = parse_spec(read_task_binary(record.task_binary).text(VERIFIER_TOML))
    assert isinstance(spec, JudgeSpec)
    assert spec.references == ("The windings heat up.",)


def test_no_non_empty_reference_answers_is_rejected():
    task = read_task_binary(_fixture("nemotron_openqa"))
    data = json.loads(task.text("tests/verifier_data.json"))
    data["expected_answers"] = ["", "   "]
    task.files["tests/verifier_data.json"] = json.dumps(data).encode()
    record = convert_one(
        _info("knowledge-openqa", "qa-short-answer"), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF
    )
    assert record.status == ConvertStatus.NULL_GRADER and record.task_binary is None
