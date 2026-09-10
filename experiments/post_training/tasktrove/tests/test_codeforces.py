# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converter behaviour on the ``codeforces`` exemplar, plus the stdio wiring it depends on."""

from pathlib import Path

from tasktrove_verify.modes import stdio
from tasktrove_verify.reward import Status
from tasktrove_verify.spec import Compare, StdioSpec, parse_spec

from experiments.post_training.tasktrove.contract import INSTALL_MARKER, VERIFIER_TOML, VERIFY_TEST_SH
from experiments.post_training.tasktrove.convert import convert_one
from experiments.post_training.tasktrove.converters.codeforces import (
    _BUILD,
    _COMMAND,
    _JUDGE_PY,
    CHECKER_PATH,
)
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.converters.registry import converter_index
from experiments.post_training.tasktrove.sources import SourceInfo, SourceVerdict
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, TEST_SH, read_task_binary, write_task_binary
from experiments.post_training.tasktrove.verify import verify_task

FIXTURES = Path(__file__).parents[1] / "fixtures"
TOOL_REF = "0123abc"


def _fixture() -> bytes:
    return (FIXTURES / "codeforces.tar.gz").read_bytes()


def _info() -> SourceInfo:
    return SourceInfo("laion__codeforces-v3", SourceVerdict.KEEP, "competitive-programming", "")


def _drop_cases(blob: bytes, keep: int):
    task = read_task_binary(blob)
    for number in range(keep, 20):
        del task.files[f"tests/inputs/input_{number}.txt"]
        del task.files[f"tests/outputs/output_{number}.txt"]
    return task


def test_codeforces_exemplar_converts_to_stdio_spec_with_expected_tags():
    record = convert_one(_info(), "t.tar.gz", _fixture(), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.CONVERTED
    assert record.converter == "codeforces"
    assert record.mode == "stdio"
    assert record.tags == ["code", "competitive-programming", "stdio", "codeforces"]
    assert record.language == "python"

    task = read_task_binary(record.task_binary)
    spec = parse_spec(task.text(VERIFIER_TOML))
    assert isinstance(spec, StdioSpec)
    assert spec.command == _COMMAND
    assert spec.build == _BUILD
    assert spec.compare == Compare.TOKENS
    assert spec.special_judge is None

    assert task.text(TEST_SH) == VERIFY_TEST_SH
    assert "tests/test_state.py" not in task.files, "old grader code must not ship"
    assert "tests/inputs/input_0.txt" not in task.files, "raw per-task data must not ship as-is"

    cases = sorted(p for p in task.files if p.startswith("tests/cases/"))
    assert len(cases) == 40  # 20 cases, input + output each
    assert task.text("tests/cases/input_0.txt") == "3\n((()))\n(())()\n()(()"
    assert task.text("tests/cases/output_0.txt") == "YES\nYES\nNO"

    dockerfile = task.text(DOCKERFILE)
    assert dockerfile.startswith("FROM python:3.10-slim") and INSTALL_MARKER in dockerfile and TOOL_REF in dockerfile
    assert record.solution_binary is None and not record.has_solution


def test_codeforces_exemplar_passes_verification():
    record = convert_one(_info(), "t.tar.gz", _fixture(), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.CONVERTED
    assert verify_task(record.task_binary) is None


def test_a_few_hidden_cases_still_convert():
    task = _drop_cases(_fixture(), keep=4)
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.CONVERTED


def test_only_the_prompt_sample_as_a_case_is_rejected():
    """``input_0`` is the sample printed in the problem statement; alone it grades nothing hidden."""
    task = _drop_cases(_fixture(), keep=1)
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.GOLD_IN_INSTRUCTION and record.task_binary is None


def test_zero_cases_is_rejected_as_null_grader():
    """The exemplar's sibling template ships no ``tests/inputs``/``tests/outputs`` at all: the old
    grader gave full credit for a solution that merely ran without crashing."""
    task = _drop_cases(_fixture(), keep=0)
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.NULL_GRADER and record.task_binary is None


def test_task_with_checker_uses_special_judge_and_ships_both_files():
    task = read_task_binary(_fixture())
    task.files[CHECKER_PATH] = b"def main(input_path, expected_path, got_path):\n    print(1)\n"
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.CONVERTED
    assert record.tags == ["code", "competitive-programming", "stdio", "codeforces", "special-judge"]

    converted = read_task_binary(record.task_binary)
    spec = parse_spec(converted.text(VERIFIER_TOML))
    assert isinstance(spec, StdioSpec)
    assert spec.special_judge == "judge.py"
    assert converted.text("tests/judge.py") == _JUDGE_PY
    assert converted.text(CHECKER_PATH) == "def main(input_path, expected_path, got_path):\n    print(1)\n"
    assert verify_task(record.task_binary) is None


def test_all_empty_outputs_without_checker_is_null_grader():
    task = read_task_binary(_fixture())
    for number in range(20):
        task.files[f"tests/outputs/output_{number}.txt"] = b"  \n"
    record = convert_one(_info(), "t.tar.gz", write_task_binary(task), converter_index(), TOOL_REF)
    assert record.status == ConvertStatus.NULL_GRADER and record.task_binary is None


# --- The build/command/judge wiring, exercised against the real in-process grader (no Docker). ---


def _task(tmp_path: Path, cases: int = 5) -> tuple[Path, Path]:
    tests_dir = tmp_path / "tests"
    cases_dir = tests_dir / "cases"
    cases_dir.mkdir(parents=True)
    for index in range(cases):
        (cases_dir / f"input_{index}.txt").write_text(f"{index}\n")
        (cases_dir / f"output_{index}.txt").write_text(f"{index * 2}\n")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return tests_dir, workspace


DOUBLE_PY = "import sys\nfor line in sys.stdin:\n    print(int(line.strip()) * 2)\n"
DOUBLE_CPP = '#include <iostream>\nint main() { int x; while (std::cin >> x) std::cout << x * 2 << "\\n"; }\n'


def _spec(**overrides) -> StdioSpec:
    return StdioSpec(command=_COMMAND, build=_BUILD, compare=Compare.TOKENS, min_cases=1, **overrides)


def test_command_runs_a_python_solution(tmp_path):
    tests_dir, workspace = _task(tmp_path)
    (workspace / "solution.py").write_text(DOUBLE_PY)
    reward = stdio.grade(_spec(), tests_dir, workspace)
    assert (reward.reward, reward.status) == (1.0, Status.SCORED)


def test_build_compiles_and_command_runs_a_cpp_solution(tmp_path):
    tests_dir, workspace = _task(tmp_path)
    (workspace / "solution.cpp").write_text(DOUBLE_CPP)
    reward = stdio.grade(_spec(), tests_dir, workspace)
    assert (reward.reward, reward.status) == (1.0, Status.SCORED)


def test_empty_workspace_fails_the_build_and_scores_zero(tmp_path):
    tests_dir, workspace = _task(tmp_path)
    reward = stdio.grade(_spec(), tests_dir, workspace)
    assert (reward.reward, reward.status) == (0.0, Status.SCORED)
    assert reward.detail["reason"] == "build_failed"


def _judge_task(tmp_path: Path, checker_source: str) -> tuple[Path, Path]:
    tests_dir = tmp_path / "tests"
    cases_dir = tests_dir / "cases"
    cases_dir.mkdir(parents=True)
    (cases_dir / "input_0.txt").write_text("x\n")
    (cases_dir / "output_0.txt").write_text("irrelevant to the checker\n")
    (tests_dir / "judge.py").write_text(_JUDGE_PY)
    (tests_dir / CHECKER_PATH.removeprefix("tests/")).write_text(checker_source)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "solution.py").write_text("print('anything')\n")
    return tests_dir, workspace


def test_judge_py_accepts_via_a_positional_checker(tmp_path):
    tests_dir, workspace = _judge_task(tmp_path, "def main(input_path, expected_path, got_path):\n    print(1)\n")
    reward = stdio.grade(_spec(special_judge="judge.py"), tests_dir, workspace)
    assert reward.reward == 1.0


def test_judge_py_rejects_via_a_positional_checker(tmp_path):
    tests_dir, workspace = _judge_task(tmp_path, "def main(input_path, expected_path, got_path):\n    print(0)\n")
    reward = stdio.grade(_spec(special_judge="judge.py"), tests_dir, workspace)
    assert reward.reward == 0.0


def test_judge_py_falls_back_to_normalized_match_for_an_argv_style_checker(tmp_path):
    """About half of the shipped checkers define ``main()`` with no parameters; calling them
    positionally raises, and the launcher falls back to the normalized match, same as upstream."""
    tests_dir, workspace = _judge_task(tmp_path, "def main():\n    pass\n")
    (tests_dir / "cases" / "output_0.txt").write_text("anything\n")
    reward = stdio.grade(_spec(special_judge="judge.py"), tests_dir, workspace)
    assert reward.reward == 1.0  # "anything" normalized-matches the solution's stdout
