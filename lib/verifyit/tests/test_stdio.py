# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest
from verifyit.grade import InvalidTask, Status, grade
from verifyit.modes import grade_stdio
from verifyit.spec import Compare, StdioSpec

DOUBLE = "import sys\nfor line in sys.stdin:\n    print(int(line.strip()) * 2)\n"
OFF_BY_ONE = "import sys\nfor line in sys.stdin:\n    print(int(line.strip()) * 2 + 1)\n"
SLEEPER = "import time\ntime.sleep(30)\n"


def _task(tmp_path: Path, program: str, cases: int = 5, *, wrong_case: int | None = None) -> tuple[Path, Path]:
    tests_dir = tmp_path / "tests"
    cases_dir = tests_dir / "cases"
    cases_dir.mkdir(parents=True)
    for index in range(cases):
        (cases_dir / f"input_{index}.txt").write_text(f"{index}\n")
        expected = index * 2 + (1 if index == wrong_case else 0)
        (cases_dir / f"output_{index}.txt").write_text(f"{expected}\n")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "solution.py").write_text(program)
    return tests_dir, workspace


def _spec(**overrides) -> StdioSpec:
    return StdioSpec(command=f"{sys.executable} solution.py", **overrides)


def test_stdio_all_cases_correct_scores_one(tmp_path):
    tests_dir, workspace = _task(tmp_path, DOUBLE)
    reward = grade_stdio.grade(_spec(), tests_dir, workspace)
    assert (reward.reward, reward.status) == (1.0, Status.SCORED)
    assert reward.detail == {"passed": 5, "total": 5}


def test_stdio_wrong_output_reports_first_failing_case(tmp_path):
    tests_dir, workspace = _task(tmp_path, DOUBLE, wrong_case=2)
    reward = grade_stdio.grade(_spec(), tests_dir, workspace)
    assert reward.reward == 0.0
    assert reward.detail["first_failure"] == "2"
    assert reward.detail["passed"] == 2


def test_stdio_cases_run_in_numeric_order(tmp_path):
    # Case 2 is the third case by number but the fifth by filename sort, so the count of cases that
    # ran before the failure pins the ordering.
    tests_dir, workspace = _task(tmp_path, DOUBLE, cases=12, wrong_case=2)
    reward = grade_stdio.grade(_spec(), tests_dir, workspace)
    assert (reward.detail["first_failure"], reward.detail["passed"]) == ("2", 2)


def test_stdio_hanging_program_times_out(tmp_path):
    tests_dir, workspace = _task(tmp_path, SLEEPER)
    reward = grade_stdio.grade(_spec(per_case_timeout=0.5), tests_dir, workspace)
    assert reward.reward == 0.0
    assert reward.detail["reason"] == "timeout"


def test_stdio_missing_program_scores_zero_without_infra_error(tmp_path):
    tests_dir, workspace = _task(tmp_path, DOUBLE)
    spec = StdioSpec(command="/nonexistent/interpreter solution.py")
    reward = grade(spec, tests_dir, workspace)
    assert (reward.reward, reward.status) == (0.0, Status.SCORED)
    assert reward.detail["reason"] == "command_failed"


def test_stdio_without_cases_is_invalid_task(tmp_path):
    tests_dir, workspace = _task(tmp_path, DOUBLE, cases=0)
    reward = grade(_spec(), tests_dir, workspace)
    assert reward.status == Status.INVALID_TASK
    assert reward.reward == 0.0


@pytest.mark.parametrize(
    ("compare", "produced", "expected", "accepted"),
    [
        (Compare.EXACT, "7  \n\n", "7\n", True),
        (Compare.EXACT, "\n7\n", "7\n", True),
        (Compare.EXACT, "7 1\n", "7  1\n", False),
        (Compare.TOKENS, "7   1\n", "7 1\n", True),
        (Compare.TOKENS, "7 1\n", "7 2\n", False),
        (Compare.FLOAT, "0.3333333\n", "0.3333334\n", True),
        (Compare.FLOAT, "0.33\n", "0.34\n", False),
    ],
)
def test_stdio_compare_modes(tmp_path, compare, produced, expected, accepted):
    tests_dir = tmp_path / "tests"
    cases_dir = tests_dir / "cases"
    cases_dir.mkdir(parents=True)
    (cases_dir / "input_0.txt").write_text("x\n")
    (cases_dir / "output_0.txt").write_text(expected)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "solution.py").write_text(f"import sys\nsys.stdout.write({produced!r})\n")
    spec = StdioSpec(command=f"{sys.executable} solution.py", compare=compare, min_cases=1, float_tolerance=1e-6)
    assert grade_stdio.grade(spec, tests_dir, workspace).reward == (1.0 if accepted else 0.0)


JUDGE_ANY_ORDER = """
import sys

_, input_path, expected_path, got_path = sys.argv
expected = open(expected_path).read().split()
got = open(got_path).read().split()
print("1" if sorted(expected) == sorted(got) else "0")
"""

JUDGE_CRASH = "raise SystemExit('checker blew up')\n"


def _judge_task(tmp_path: Path, judge_source: str) -> tuple[Path, Path]:
    tests_dir = tmp_path / "tests"
    cases_dir = tests_dir / "cases"
    cases_dir.mkdir(parents=True)
    (cases_dir / "input_0.txt").write_text("x\n")
    (cases_dir / "output_0.txt").write_text("1 2 3\n")
    (tests_dir / "judge.py").write_text(judge_source)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "solution.py").write_text("print('3 1 2')\n")
    return tests_dir, workspace


def test_stdio_special_judge_accepts_reordered_answer(tmp_path):
    tests_dir, workspace = _judge_task(tmp_path, JUDGE_ANY_ORDER)
    spec = StdioSpec(command=f"{sys.executable} solution.py", special_judge="judge.py", min_cases=1)
    assert grade_stdio.grade(spec, tests_dir, workspace).reward == 1.0


def test_stdio_special_judge_rejection_overrides_token_match(tmp_path):
    tests_dir, workspace = _judge_task(tmp_path, "print('0')\n")
    (tests_dir / "cases" / "output_0.txt").write_text("3 1 2\n")
    spec = StdioSpec(command=f"{sys.executable} solution.py", special_judge="judge.py", min_cases=1)
    assert grade_stdio.grade(spec, tests_dir, workspace).reward == 0.0


def test_stdio_crashing_special_judge_rejects_instead_of_falling_back(tmp_path):
    tests_dir, workspace = _judge_task(tmp_path, JUDGE_CRASH)
    (tests_dir / "cases" / "output_0.txt").write_text("3 1 2\n")
    spec = StdioSpec(command=f"{sys.executable} solution.py", special_judge="judge.py", min_cases=1)
    assert grade_stdio.grade(spec, tests_dir, workspace).reward == 0.0


def test_stdio_binary_stdin_program_scores_correct_output(tmp_path):
    program = "import sys\nfor line in sys.stdin.buffer:\n    print(int(line.strip()) * 2)\n"
    tests_dir, workspace = _task(tmp_path, program)
    verdict = grade(_spec(), tests_dir, workspace)
    assert (verdict.reward, verdict.status) == (1.0, Status.SCORED)


@pytest.mark.parametrize("special_judge", [None, "judge.py"])
def test_stdio_correct_output_before_candidate_crash_scores_zero(tmp_path, special_judge):
    tests_dir, workspace = _task(tmp_path, DOUBLE + "raise SystemExit(7)\n")
    if special_judge:
        (tests_dir / special_judge).write_text("print('1')\n")
    verdict = grade(_spec(special_judge=special_judge), tests_dir, workspace)
    assert (verdict.reward, verdict.status) == (0.0, Status.SCORED)
    assert verdict.detail["passed"] == 0


@pytest.mark.parametrize(
    ("candidate", "expected", "score"),
    [
        (" 1.00  2e0\n  answer \n", "1 2\nanswer", 1.0),
        ("1 2", "1\n2", 0.0),
        ("1\n\n2", "1\n2", 0.0),
        ("value  1", "value 1", 0.0),
        ("1.00000000000000000001", "1", 0.0),
        ("1e4000", "10e3999", 1.0),
        ("-1e-4000", "1e-4000", 0.0),
        ("", "  ", 1.0),
        ("NaN", "1", 0.0),
        ("Infinity", "1", 0.0),
    ],
)
def test_decimal_lines_direct_and_executed_outputs_agree(tmp_path, candidate, expected, score):
    tests_dir, workspace = _task(tmp_path, f"print({candidate!r})", cases=1)
    (tests_dir / "cases" / "output_0.txt").write_text(expected)
    spec = _spec(compare=Compare.DECIMAL_LINES)
    direct = grade_stdio.grade_stdio_candidate(spec, candidate, expected)
    executed = grade(spec, tests_dir, workspace)
    assert (direct.reward, direct.status) == (score, Status.SCORED)
    assert (executed.reward, executed.status) == (score, Status.SCORED)


def test_nonfinite_decimal_reference_is_invalid_before_execution(tmp_path):
    tests_dir, workspace = _task(tmp_path, SLEEPER, cases=1)
    (tests_dir / "cases" / "output_0.txt").write_text("NaN")
    spec = _spec(compare=Compare.DECIMAL_LINES)
    with pytest.raises(InvalidTask, match="nonfinite"):
        grade_stdio.grade_stdio_candidate(spec, "NaN", "NaN")
    verdict = grade(spec, tests_dir, workspace)
    assert (verdict.reward, verdict.status) == (0.0, Status.INVALID_TASK)


@pytest.mark.parametrize(
    "options",
    [
        {"compare": "typo"},
        {"compare": Compare.FLOAT, "float_tolerance": float("nan")},
        {"compare": Compare.FLOAT, "float_tolerance": float("inf")},
        {"compare": Compare.FLOAT, "float_tolerance": -1.0},
    ],
)
def test_invalid_comparison_policy_cannot_award_stdout_credit(tmp_path, options):
    tests_dir, workspace = _task(tmp_path, "print(0)", cases=1)
    spec = _spec(**options)
    with pytest.raises(InvalidTask):
        grade_stdio.grade_stdio_candidate(spec, "0", "0")
    verdict = grade(spec, tests_dir, workspace)
    assert (verdict.reward, verdict.status) == (0.0, Status.INVALID_TASK)
