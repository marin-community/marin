# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converter behaviour on the ``nemotron_ifeval`` exemplar."""

import json
from pathlib import Path

from tasktrove_verify.spec import IfevalSpec, parse_spec

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


def _fixture(name: str = "nemotron_ifeval") -> bytes:
    return (FIXTURES / f"{name}.tar.gz").read_bytes()


def _info() -> SourceInfo:
    return SourceInfo("laion__nemotron-gym-instruction-following-v3", SourceVerdict.KEEP, "instruction-following", "")


def test_ifeval_exemplar_converts_to_ifeval_spec_with_expected_tags():
    record = convert_one(_info(), "t.tar.gz", _fixture(), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.CONVERTED
    assert record.converter == "nemotron_ifeval"
    assert record.mode == "ifeval"
    assert record.tags == ["instruction-following", "ifeval", "nemotron"]

    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, IfevalSpec)
    names = {c.name for c in spec.constraints}
    assert names == {"length_constraints:nth_paragraph_first_word", "last_word:last_word_answer"}
    by_name = {c.name: c.params for c in spec.constraints}
    assert by_name["last_word:last_word_answer"] == {"last_word": "contest"}

    assert task.text(TEST_SH) == VERIFY_TEST_SH
    assert "tests/verifier.py" not in task.files, "old grader code must not ship"
    assert "tests/verifier_data.json" not in task.files, "raw grader data must not ship"

    dockerfile = task.text(DOCKERFILE)
    assert dockerfile.startswith("# DO NOT EDIT") and INSTALL_MARKER in dockerfile and TOOL_REF in dockerfile
    assert record.solution_binary is None and not record.has_solution


def test_ifeval_exemplar_passes_verification():
    record = convert_one(_info(), "t.tar.gz", _fixture(), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.CONVERTED
    assert verify_task(record.task_binary) is None


def test_unsupported_constraint_name_is_rejected_not_converted():
    task = read_task_binary(_fixture())
    data = json.loads(task.text("tests/verifier_data.json"))
    data["instruction_id_list"].append("nemotron_gym:not_a_real_constraint")
    data["kwargs"].append({})
    task.files["tests/verifier_data.json"] = json.dumps(data).encode()
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT and record.task_binary is None
    assert "not_a_real_constraint" in record.error


def test_empty_instruction_list_is_rejected_as_null_grader():
    task = read_task_binary(_fixture())
    data = json.loads(task.text("tests/verifier_data.json"))
    data["instruction_id_list"] = []
    data["kwargs"] = []
    task.files["tests/verifier_data.json"] = json.dumps(data).encode()
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.NULL_GRADER and record.task_binary is None


def test_dockerfile_edit_is_idempotent_across_tasks():
    index = converter_index()
    a = convert_one(_info(), "a.tar.gz", _fixture(), index, TOOL_REF)
    b = convert_one(_info(), "b.tar.gz", _fixture(), index, TOOL_REF)
    assert a.dockerfile_id == b.dockerfile_id
    assert TaskFiles(read_task_binary(a.task_binary).files).text(DOCKERFILE) == read_task_binary(b.task_binary).text(
        DOCKERFILE
    )
