# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converter behaviour on the checked-in ``nl2bash`` exemplar."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from tasktrove_verify.spec import ScriptSpec, parse_spec

from experiments.post_training.tasktrove.convert import convert_one
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.converters.nl2bash import CHECKER_NAME, DATA_NAME
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
SOURCE = "DCAgent2__nl2bash-tasks-cleaned-oracle-v2"


def _fixture() -> bytes:
    return (FIXTURES / "nl2bash.tar.gz").read_bytes()


def _info() -> SourceInfo:
    return SourceInfo(SOURCE, SourceVerdict.KEEP, "shell-cmd", "")


def _run_checker(checker_source: str, tests_dir: Path, output_file: Path) -> dict:
    """Run the shipped checker exactly as the container would, naming the capture file to read."""
    script = tests_dir / CHECKER_NAME
    script.write_text(checker_source)
    logs_dir = tests_dir.parent / "logs"
    env = {
        **os.environ,
        "TASKTROVE_TESTS_DIR": str(tests_dir),
        "TASKTROVE_LOGS_DIR": str(logs_dir),
    }
    proc = subprocess.run(
        [sys.executable, str(script), str(output_file)], cwd=tests_dir.parent, env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads((logs_dir / "reward.json").read_text())


def test_exemplar_converts_to_script_spec_with_expected_tags():
    record = convert_one(_info(), "t.tar.gz", _fixture(), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.CONVERTED
    assert record.converter == "nl2bash"
    assert record.mode == "script"
    assert record.tags == ["shell", "bash", "nl2bash", "terminal", "dcagent2"]
    assert record.language == "bash"

    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, ScriptSpec) and spec.path == CHECKER_NAME and spec.workspace == "/workspace"

    assert task.text(TEST_SH) == VERIFY_TEST_SH
    assert "tests/verifier.py" not in task.files, "old grader code must not ship"
    assert "tests/verifier_data.json" not in task.files, "old grader data must not ship under its old name"
    assert f"tests/{CHECKER_NAME}" in task.files
    assert f"tests/{DATA_NAME}" in task.files
    expected = json.loads(task.text(f"tests/{DATA_NAME}"))
    assert expected["expected_output"] == "./dir1/fileName.txt\x00./dir2/fileName_extra.TXT\x00"

    # Fixtures the agent's own "Environment Setup" step needs, and their oracle-side mirror.
    assert "setup_files/setup_seeds.sh" in task.files
    assert "setup_files/seeds/dir1/fileName.txt" in task.files
    assert "tests/setup_files/setup_seeds.sh" in task.files

    dockerfile = task.text(DOCKERFILE)
    assert dockerfile.startswith("FROM ubuntu:24.04") and INSTALL_MARKER in dockerfile and TOOL_REF in dockerfile

    assert record.has_solution and record.solution_binary is not None
    solution = read_task_binary(record.solution_binary)
    solve = solution.text("solution/solve.sh")
    assert "bash /tests/setup_seeds.sh" not in solve, "the oracle's broken seed path must be fixed"
    assert "bash /tests/setup_files/setup_seeds.sh" in solve


def test_exemplar_passes_verification():
    record = convert_one(_info(), "t.tar.gz", _fixture(), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.CONVERTED
    assert verify_task(record.task_binary) is None


def test_checker_scores_empty_oracle_and_tolerates_a_harmless_extra_line():
    """``verify.verify_task`` has no automatic grading probe for ``script`` mode, so the checker's
    own comparison logic is exercised directly here, running the shipped script exactly as the
    container would (only its fixed ``/output`` path is redirected to a temp file)."""
    record = convert_one(_info(), "t.tar.gz", _fixture(), converter_index(), TOOL_REF)
    task = read_task_binary(record.task_binary)
    checker_source = task.text(f"tests/{CHECKER_NAME}")
    expected = json.loads(task.text(f"tests/{DATA_NAME}"))["expected_output"]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tests_dir = root / "tests"
        tests_dir.mkdir()
        (tests_dir / DATA_NAME).write_text(json.dumps({"expected_output": expected}))
        output_file = root / "command_capture.txt"

        reward = _run_checker(checker_source, tests_dir, output_file)
        assert reward["reward"] == 0

        output_file.write_text(expected)
        reward = _run_checker(checker_source, tests_dir, output_file)
        assert reward["reward"] == 1

        output_file.write_text(expected + "\nan unrelated extra line\n")
        reward = _run_checker(checker_source, tests_dir, output_file)
        assert reward["reward"] == 1

        output_file.write_text("totally different output\n")
        reward = _run_checker(checker_source, tests_dir, output_file)
        assert reward["reward"] == 0


def test_missing_expected_output_is_rejected_as_null_grader():
    task = read_task_binary(_fixture())
    task.files["tests/verifier_data.json"] = json.dumps({}).encode()
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.NULL_GRADER and record.task_binary is None


def test_missing_oracle_solution_is_rejected_as_null_grader():
    task = read_task_binary(_fixture())
    del task.files["solution/solve.sh"]
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.NULL_GRADER and record.task_binary is None
