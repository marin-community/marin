# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converter behaviour on the ``all_puzzles`` exemplar (laion all-puzzles-v2)."""

import json
import tempfile
from pathlib import Path

from tasktrove_verify.grade import grade
from tasktrove_verify.spec import ExactSpec, MathSpec, MathType, parse_spec

from experiments.post_training.tasktrove.convert import convert_one
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.converters.registry import converter_index
from experiments.post_training.tasktrove.dataset import SourceInfo, SourceVerdict
from experiments.post_training.tasktrove.task_format import INSTALL_MARKER, VERIFIER_TOML, VERIFY_TEST_SH
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, TEST_SH, read_task_binary, write_task_binary
from experiments.post_training.tasktrove.verify import verify_task

FIXTURES = Path(__file__).parents[1] / "fixtures"
TOOL_REF = "0123abc"


def _fixture() -> bytes:
    return (FIXTURES / "all_puzzles.tar.gz").read_bytes()


def _info() -> SourceInfo:
    return SourceInfo("laion__all-puzzles-v2", SourceVerdict.KEEP, "math-answer", "")


def _convert(blob: bytes | None = None):
    return convert_one(_info(), "t.tar.gz", blob if blob is not None else _fixture(), converter_index(), TOOL_REF)


def test_exemplar_converts_to_exact_spec_with_expected_tags():
    record = _convert()
    assert record.status == ConvertStatus.CONVERTED
    assert record.converter == "all_puzzles"
    assert record.mode == "exact"
    assert record.tags == ["puzzle", "laion", "alphabetical-sorting"]

    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, ExactSpec)
    assert spec.expected == ("Defect", "Salt", "chair", "donate")
    assert spec.ignore_case and spec.ignore_whitespace and spec.ordered

    assert task.text(TEST_SH) == VERIFY_TEST_SH
    dockerfile = task.text(DOCKERFILE)
    assert INSTALL_MARKER in dockerfile and TOOL_REF in dockerfile


def test_exemplar_drops_old_grader_files_and_dependency():
    record = _convert()
    task = read_task_binary(record.task_binary)
    for path in ("tests/compare_answer.py", "tests/gold.json", "tests/test_state.py"):
        assert path not in task.files, f"old grader file {path} must not ship"
    dockerfile = task.text(DOCKERFILE)
    assert "pytest" not in dockerfile.lower()


def test_exemplar_passes_verification():
    record = _convert()
    assert record.status == ConvertStatus.CONVERTED
    assert verify_task(record.task_binary) is None


def test_solution_files_carry_the_baked_gold_answer():
    record = _convert()
    assert record.has_solution and record.solution_binary is not None
    solve = read_task_binary(record.solution_binary).text("solution/solve.sh")
    assert "Defect, Salt, chair, donate" in solve


def test_number_answer_type_converts_to_math_spec():
    task = read_task_binary(_fixture())
    task.files["tests/gold.json"] = json.dumps(
        {"gold": "-2", "answer_type": "number", "ptype": "basic_arithmetic"}
    ).encode()
    record = _convert(write_task_binary(task))
    assert record.status == ConvertStatus.CONVERTED and record.mode == "math"
    spec = parse_spec(read_task_binary(record.task_binary).text(VERIFIER_TOML))
    assert isinstance(spec, MathSpec)
    assert spec.expected == "-2" and spec.math_type == MathType.SCALAR
    assert verify_task(record.task_binary) is None


def test_coords_answer_type_converts_to_math_spec_and_grades_both_components():
    task = read_task_binary(_fixture())
    task.files["tests/gold.json"] = json.dumps(
        {"gold": "(5.545, 7.545)", "answer_type": "coords", "ptype": "advanced_geometry"}
    ).encode()
    task.files["solution/solve.sh"] = (
        b"#!/usr/bin/env bash\nmkdir -p /app\nprintf '%s\\n' '(5.545, 7.545)' > /app/answer.txt\n"
    )
    record = _convert(write_task_binary(task))
    assert record.status == ConvertStatus.CONVERTED and record.mode == "math"
    assert verify_task(record.task_binary) is None

    spec = parse_spec(read_task_binary(record.task_binary).text(VERIFIER_TOML))
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        # Coordinate order is significant, so a swapped candidate scores zero.
        (workspace / "answer.txt").write_text("(7.545, 5.545)")
        reward = grade(spec, tests_dir=workspace, workspace=workspace)
        assert reward.reward == 0.0


def test_choice_answer_type_converts_to_single_string_exact_spec():
    task = read_task_binary(_fixture())
    task.files["tests/gold.json"] = json.dumps({"gold": "off", "answer_type": "choice", "ptype": "acre"}).encode()
    record = _convert(write_task_binary(task))
    assert record.status == ConvertStatus.CONVERTED and record.mode == "exact"
    spec = parse_spec(read_task_binary(record.task_binary).text(VERIFIER_TOML))
    assert isinstance(spec, ExactSpec)
    assert spec.expected == ("off",)
    assert verify_task(record.task_binary) is None


def test_unrecognized_answer_type_is_rejected_not_guessed():
    task = read_task_binary(_fixture())
    task.files["tests/gold.json"] = json.dumps({"gold": "42", "answer_type": "interval", "ptype": "?"}).encode()
    record = _convert(write_task_binary(task))
    assert record.status == ConvertStatus.UNSUPPORTED_VARIANT and record.task_binary is None


def test_empty_gold_is_rejected_as_null_grader():
    task = read_task_binary(_fixture())
    task.files["tests/gold.json"] = json.dumps({"gold": "", "answer_type": "number", "ptype": "?"}).encode()
    record = _convert(write_task_binary(task))
    assert record.status == ConvertStatus.NULL_GRADER and record.task_binary is None
