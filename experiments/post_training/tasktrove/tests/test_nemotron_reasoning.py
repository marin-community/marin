# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converter behaviour on the ``nemotron_reasoning`` exemplar and its two sibling shapes.

The checked-in exemplar is the ``reasoning-gym`` source. The ``grid-transform`` (ARC-AGI
python-inductive) and ``grid-match`` (ARC-AGI transductive) shapes are exercised by swapping the
exemplar's ``tests/verifier_data.json`` for a small synthetic grid, since routing depends only on
that file's keys and the other two real sources are not checked in separately.
"""

import json
import tempfile
from pathlib import Path

from tasktrove_verify.grade import grade
from tasktrove_verify.spec import ExactSpec, ReasoningGymSpec, ScriptSpec, parse_spec

from experiments.post_training.tasktrove.convert import convert_one
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.converters.nemotron_reasoning import CASES_FILE, TRANSFORM_SCRIPT
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
FAMILY = "other"

# A trivial 2x2 transform: swap rows. Small enough to keep the fixture readable while still
# exercising a real held-out case.
GRID_TRANSFORM_CASES = [{"input": [[1, 2], [3, 4]], "output": [[3, 4], [1, 2]]}]
GRID_MATCH_OUTPUT = [[1, 2], [3, 4]]


def _fixture() -> bytes:
    return (FIXTURES / "nemotron_reasoning.tar.gz").read_bytes()


def _info(source: str = "laion__nemotron-gym-reasoning-gym-v2") -> SourceInfo:
    return SourceInfo(source, SourceVerdict.KEEP, FAMILY, "")


def _with_verifier_data(data: dict) -> bytes:
    """The checked-in exemplar's other files, with ``tests/verifier_data.json`` replaced."""
    task = read_task_binary(_fixture())
    task.files["tests/verifier_data.json"] = json.dumps(data).encode()
    return write_task_binary(task)


def test_reasoning_gym_exemplar_converts_with_expected_mode_and_tags():
    record = convert_one(_info(), "t.tar.gz", _fixture(), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.CONVERTED
    assert record.converter == "nemotron_reasoning"
    assert record.mode == "reasoning-gym"
    assert record.tags == ["reasoning", "reasoning-gym", "needle-haystack", "nemotron"]
    assert record.language == ""

    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, ReasoningGymSpec) and spec.dataset == "needle_haystack"
    assert task.text(TEST_SH) == VERIFY_TEST_SH

    for old_grader_file in ("tests/verifier.py", "tests/validate_verifier_data.py", "tests/verifier_data.json"):
        assert old_grader_file not in task.files, "old grader code must not ship"
    assert "tests/entry.json" in task.files
    entry = json.loads(task.text("tests/entry.json"))
    assert entry["answer"] == "Richard" and entry["metadata"]["source_dataset"] == "needle_haystack"

    dockerfile = task.text(DOCKERFILE)
    assert dockerfile.startswith("# DO NOT EDIT") and INSTALL_MARKER in dockerfile and TOOL_REF in dockerfile
    assert "reasoning-gym==0.1.20" not in dockerfile, "the old grader's own reasoning-gym install must be dropped"
    assert record.has_solution is False and record.solution_binary is None


def test_reasoning_gym_exemplar_passes_verification():
    record = convert_one(_info(), "t.tar.gz", _fixture(), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.CONVERTED
    assert verify_task(record.task_binary) is None


def test_unscorable_reasoning_gym_dataset_is_rejected_as_unsupported_variant():
    """``arc_agi`` and ``rearc`` can't score even their own gold answer; see the converter comment."""
    data = {"answer": "5 5\n5 5", "metadata": {"source_dataset": "arc_agi"}, "question": "q"}
    record = convert_one(_info(), "t.tar.gz", _with_verifier_data(data), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT and record.task_binary is None
    assert "arc_agi" in record.error


def test_missing_source_dataset_is_rejected_as_null_grader():
    data = {"answer": "Richard", "metadata": {}, "question": "q"}
    record = convert_one(_info(), "t.tar.gz", _with_verifier_data(data), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.NULL_GRADER and record.task_binary is None


def test_unrecognized_verifier_data_shape_is_rejected_as_unsupported_variant():
    data = _with_verifier_data({"nothing": "recognizable"})
    record = convert_one(_info(), "t.tar.gz", data, converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT and record.task_binary is None


def test_grid_transform_shape_converts_to_script_mode_and_grades_the_held_out_case():
    data = {"test_cases": GRID_TRANSFORM_CASES}
    record = convert_one(
        _info("laion__nemotron-gym-arc-agi-python-inductive-v2"),
        "t.tar.gz",
        _with_verifier_data(data),
        converter_index(),
        TOOL_REF,
    )
    assert record.status == ConvertStatus.CONVERTED
    assert record.mode == "script"
    assert record.tags == ["reasoning", "arc-agi", "grid-transform", "code", "nemotron"]
    assert record.language == "python"

    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, ScriptSpec) and spec.path == TRANSFORM_SCRIPT
    assert f"tests/{TRANSFORM_SCRIPT}" in task.files and f"tests/{CASES_FILE}" in task.files
    assert json.loads(task.text(f"tests/{CASES_FILE}")) == GRID_TRANSFORM_CASES
    for old_grader_file in ("tests/verifier.py", "tests/validate_verifier_data.py", "tests/verifier_data.json"):
        assert old_grader_file not in task.files, "old grader code must not ship"

    # ScriptSpec has no probe in tasktrove_verify.grade, so verify_task cannot exercise the real
    # grader; check_grading only skips it, so we run it ourselves the way check_grading would.
    assert verify_task(record.task_binary) is None
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tests_dir = root / "tests"
        for path, blob in task.under("tests/").items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
        workspace = root / "app"
        workspace.mkdir()

        empty = grade(spec, tests_dir, workspace)
        assert empty.reward == 0.0

        (workspace / "solution.py").write_text("def transform(grid):\n    return list(reversed(grid))\n")
        correct = grade(spec, tests_dir, workspace)
        assert correct.reward == 1.0

        (workspace / "solution.py").write_text("def transform(grid):\n    return grid\n")
        wrong = grade(spec, tests_dir, workspace)
        assert wrong.reward == 0.0


def test_grid_transform_with_no_held_out_cases_is_rejected_as_too_few_cases():
    record = convert_one(
        _info("laion__nemotron-gym-arc-agi-python-inductive-v2"),
        "t.tar.gz",
        _with_verifier_data({"test_cases": []}),
        converter_index(),
        TOOL_REF,
    )
    assert record.status == ConvertStatus.TOO_FEW_CASES and record.task_binary is None


def test_grid_match_shape_converts_to_exact_mode_and_passes_verification():
    record = convert_one(
        _info("laion__nemotron-gym-arc-agi-transductive-v3"),
        "t.tar.gz",
        _with_verifier_data({"expected_output": GRID_MATCH_OUTPUT}),
        converter_index(),
        TOOL_REF,
    )
    assert record.status == ConvertStatus.CONVERTED
    assert record.mode == "exact"
    assert record.tags == ["reasoning", "arc-agi", "grid-match", "nemotron"]

    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, ExactSpec) and spec.expected == ("1 2\n3 4",)
    for old_grader_file in ("tests/verifier.py", "tests/validate_verifier_data.py", "tests/verifier_data.json"):
        assert old_grader_file not in task.files, "old grader code must not ship"

    # ExactSpec has a probe, so this exercises the real grader on the empty, expected, and (for a
    # multi-entry expected) perturbed candidate in-process.
    assert verify_task(record.task_binary) is None


def test_grid_match_with_out_of_range_cell_is_rejected_as_unsupported_variant():
    record = convert_one(
        _info("laion__nemotron-gym-arc-agi-transductive-v3"),
        "t.tar.gz",
        _with_verifier_data({"expected_output": [[1, 10], [3, 4]]}),
        converter_index(),
        TOOL_REF,
    )
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT and record.task_binary is None


def test_grid_match_with_empty_expected_output_is_rejected_as_null_grader():
    record = convert_one(
        _info("laion__nemotron-gym-arc-agi-transductive-v3"),
        "t.tar.gz",
        _with_verifier_data({"expected_output": []}),
        converter_index(),
        TOOL_REF,
    )
    assert record.status == ConvertStatus.NULL_GRADER and record.task_binary is None
