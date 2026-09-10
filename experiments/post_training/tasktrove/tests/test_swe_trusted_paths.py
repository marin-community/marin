# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converter behaviour on the checked-in ``swe_trusted_paths`` exemplar."""

import json
from pathlib import Path

from tasktrove_verify.spec import PytestSpec, parse_spec

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
FAMILY = "swe-repo"
SOURCE = "laion__swesmith-oracle-filtered-v2"

OLD_GRADER_FILES = ("tests/config.json", "tests/test_state.py", "tests/install_trusted_test_paths.sh")


def _fixture() -> bytes:
    return (FIXTURES / "swe_trusted_paths.tar.gz").read_bytes()


def _info() -> SourceInfo:
    return SourceInfo(SOURCE, SourceVerdict.KEEP, FAMILY, "")


def _edited(**config_overrides: object) -> bytes:
    task = read_task_binary(_fixture())
    config = json.loads(task.text("tests/config.json"))
    config.update(config_overrides)
    task.files["tests/config.json"] = json.dumps(config).encode()
    return write_task_binary(task)


def test_exemplar_converts_to_pytest_spec_with_expected_tags():
    record = convert_one(_info(), "t.tar.gz", _fixture(), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.CONVERTED
    assert record.mode == "pytest"
    assert record.tags == ["code", "swe", "swe-repo", "trusted-test-paths"]
    assert record.language == "python"

    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, PytestSpec)
    assert spec.workspace == "/testbed"
    assert len(spec.must_pass) == 18
    assert len(spec.must_not_break) == 655
    assert all(path.startswith("tests/") for path in spec.paths)
    assert "3b1e6ec37ffeacc5bed0e55e287f86166b4db21c" in spec.setup
    assert "614b1348ef893b4fa90e76f56df44ee64f5d0222" in spec.setup

    assert task.text(TEST_SH) == VERIFY_TEST_SH
    for old_grader_file in OLD_GRADER_FILES:
        assert old_grader_file not in task.files, "old grader code must not ship"
    assert "tests/trusted_test_paths.txt" in task.files

    dockerfile = task.text(DOCKERFILE)
    assert INSTALL_MARKER in dockerfile and TOOL_REF in dockerfile
    assert "pytest-json-report" in dockerfile

    assert record.has_solution and record.solution_binary is not None
    assert "solution/solve.sh" in read_task_binary(record.solution_binary).files


def test_exemplar_passes_verification():
    record = convert_one(_info(), "t.tar.gz", _fixture(), converter_index(), TOOL_REF)
    assert verify_task(record.task_binary) is None


def test_missing_config_json_is_rejected_as_unsupported_variant():
    task = read_task_binary(_fixture())
    del task.files["tests/config.json"]
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT and record.task_binary is None


def test_empty_fail_and_pass_to_pass_is_rejected_as_null_grader():
    record = convert_one(_info(), "t.tar.gz", _edited(FAIL_TO_PASS=[], PASS_TO_PASS=[]), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.NULL_GRADER and record.task_binary is None


def test_test_sh_without_install_invocation_is_rejected_as_unsupported_variant():
    task = read_task_binary(_fixture())
    task.files["tests/test.sh"] = b"#!/bin/bash\necho 0 > /logs/verifier/reward.txt\n"
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT and record.task_binary is None


def test_graded_file_missing_from_manifest_is_rejected_as_unsupported_variant():
    record = convert_one(
        _info(),
        "t.tar.gz",
        _edited(FAIL_TO_PASS=["tests/not_in_manifest.py::TestX::test_y"]),
        converter_index(),
        TOOL_REF,
    )
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT and record.task_binary is None


def test_fail_to_pass_outside_a_python_file_is_rejected():
    record = convert_one(
        _info(), "t.tar.gz", _edited(FAIL_TO_PASS=["tests/tests.md::tests.md"]), converter_index(), TOOL_REF
    )
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT
    assert "tests/tests.md::tests.md" in record.error


def test_pass_to_pass_outside_a_python_file_is_dropped_from_the_spec():
    record = convert_one(
        _info(),
        "t.tar.gz",
        _edited(PASS_TO_PASS=["tests/tests.md::tests.md", "tests/test_common.py::test_other"]),
        converter_index(),
        TOOL_REF,
    )
    assert record.status == ConvertStatus.CONVERTED
    spec = parse_spec(read_task_binary(record.task_binary).text(VERIFIER_TOML))
    assert spec.must_not_break == ("tests/test_common.py::test_other",)
    assert all(path.endswith(".py") for path in spec.paths)


def test_json_encoded_string_fail_to_pass_is_accepted():
    record = convert_one(
        _info(),
        "t.tar.gz",
        _edited(FAIL_TO_PASS=json.dumps(["tests/test_common.py::test_something"])),
        converter_index(),
        TOOL_REF,
    )
    assert record.status == ConvertStatus.CONVERTED
    spec = parse_spec(read_task_binary(record.task_binary).text(VERIFIER_TOML))
    assert spec.must_pass == ("tests/test_common.py::test_something",)


def test_dockerfile_reuses_existing_pip_install_line_instead_of_adding_a_new_run():
    record = convert_one(_info(), "t.tar.gz", _fixture(), converter_index(), TOOL_REF)
    dockerfile = read_task_binary(record.task_binary).text(DOCKERFILE)
    body = dockerfile.split(INSTALL_MARKER)[0]
    assert body.count("pytest-json-report") == 1
    assert "RUN pip install --upgrade pip uv pytest pytest-json-report" in body
