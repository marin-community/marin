# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converter behaviour on the checked-in ``agent_calendar`` exemplar."""

import json
import tempfile
from pathlib import Path

from tasktrove_verify.grade import Status, grade
from tasktrove_verify.spec import ScriptSpec, parse_spec

from experiments.post_training.tasktrove.convert import convert_one
from experiments.post_training.tasktrove.converters.agent_calendar import CHECKER_NAME, DATA_NAME
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
SOURCE = "laion__nemotron-gym-agent-calendar-v2"


def _fixture() -> bytes:
    return (FIXTURES / "agent_calendar.tar.gz").read_bytes()


def _info() -> SourceInfo:
    return SourceInfo(SOURCE, SourceVerdict.KEEP, "tool-use", "")


def _materialize(tests_dir: Path, data_files: dict[str, bytes]) -> None:
    for path, data in data_files.items():
        target = tests_dir.parent / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def test_exemplar_converts_to_script_spec_with_expected_tags():
    record = convert_one(_info(), "t.tar.gz", _fixture(), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.CONVERTED
    assert record.converter == "agent_calendar"
    assert record.mode == "script"
    assert record.tags == ["tool-use", "calendar", "scheduling", "state-tracking", "nemotron"]

    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, ScriptSpec) and spec.path == CHECKER_NAME

    assert task.text(TEST_SH) == VERIFY_TEST_SH
    assert "tests/verifier.py" not in task.files, "old grader code must not ship"
    assert f"tests/{CHECKER_NAME}" in task.files
    assert f"tests/{DATA_NAME}" in task.files
    expected = json.loads(task.text(f"tests/{DATA_NAME}"))
    assert expected and all("event_name" in spec for spec in expected.values())

    dockerfile = task.text(DOCKERFILE)
    assert dockerfile.startswith("# DO NOT EDIT") and INSTALL_MARKER in dockerfile and TOOL_REF in dockerfile
    assert record.has_solution and record.solution_binary is not None
    solution = read_task_binary(record.solution_binary)
    assert "solution/answer.json" in solution.files and "solution/solve.sh" in solution.files


def test_exemplar_passes_verification():
    record = convert_one(_info(), "t.tar.gz", _fixture(), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.CONVERTED
    assert verify_task(record.task_binary) is None


def test_checker_scores_empty_one_oracle_and_a_perturbation():
    """``verify.verify_task`` has no automatic grading probe for ``script`` mode (only output-file
    modes with a known candidate shape do), so the checker's own correctness is exercised directly
    here: an empty workspace must score 0, the oracle answer 1, and a perturbed oracle 0."""
    record = convert_one(_info(), "t.tar.gz", _fixture(), converter_index(), TOOL_REF)
    task = read_task_binary(record.task_binary)
    solution = read_task_binary(record.solution_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tests_dir = root / "tests"
        tests_dir.mkdir()
        _materialize(tests_dir, task.under("tests/"))
        workspace = root / "app"
        workspace.mkdir()

        reward = grade(spec, tests_dir, workspace)
        assert reward.status == Status.SCORED and reward.reward == 0.0

        answer = json.loads(solution.text("solution/answer.json"))
        (workspace / "answer.txt").write_text(json.dumps(answer))
        reward = grade(spec, tests_dir, workspace)
        assert reward.status == Status.SCORED and reward.reward == 1.0

        answer[0]["start_time"] = "23:59"
        (workspace / "answer.txt").write_text(json.dumps(answer))
        reward = grade(spec, tests_dir, workspace)
        assert reward.status == Status.SCORED and reward.reward == 0.0


def test_missing_expected_events_is_rejected_as_null_grader():
    task = read_task_binary(_fixture())
    task.files["tests/verifier_data.json"] = json.dumps({}).encode()
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.NULL_GRADER and record.task_binary is None


def test_malformed_event_spec_is_rejected_as_null_grader():
    task = read_task_binary(_fixture())
    data = json.loads(task.text("tests/verifier_data.json"))
    del data["expected_events"]["0"]["duration"]
    task.files["tests/verifier_data.json"] = json.dumps(data).encode()
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.NULL_GRADER and record.task_binary is None
