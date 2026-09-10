# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mode stdio: run the agent's program on ``input_<n>.txt`` and compare stdout to ``output_<n>.txt``.

The cases directory holds numbered pairs, either flat or split into ``inputs``/``outputs``
subdirectories. Every case must pass; the reward is all or nothing, as it is for the competitive
programming graders this mode replaces.
"""

import math
import re
import tempfile
from pathlib import Path
from typing import NamedTuple

from tasktrove_verify.modes.extract import last_line
from tasktrove_verify.modes.run import STDERR_TAIL, Completed, run_command, split_command, workdir
from tasktrove_verify.reward import InvalidTask, Reward, scored
from tasktrove_verify.spec import Compare, StdioSpec

INPUT_PATTERN = re.compile(r"^input_(.+)\.txt$")
JUDGE_ACCEPT = "1"


def grade(spec: StdioSpec, tests_dir: Path, workspace: Path) -> Reward:
    cases = _cases(tests_dir / spec.cases)
    if len(cases) < spec.min_cases:
        raise InvalidTask(f"stdio needs at least {spec.min_cases} cases, found {len(cases)}")
    directory = workdir(spec, workspace)
    argv = split_command(spec.command)
    judge = tests_dir / spec.special_judge if spec.special_judge else None
    if judge is not None and not judge.is_file():
        raise InvalidTask(f"special judge {spec.special_judge!r} is not in the tests directory")
    if spec.build:
        build = run_command(["bash", "-lc", spec.build], directory, spec.per_case_timeout * len(cases))
        if build.timed_out or build.returncode != 0:
            return scored(0.0, reason="build_failed", stderr=build.stderr[-STDERR_TAIL:], passed=0, total=len(cases))

    with tempfile.TemporaryDirectory(prefix="tasktrove-stdio-") as scratch:
        got_path = Path(scratch) / "got.txt"
        for index, (number, input_path, expected_path) in enumerate(cases):
            try:
                result = run_command(argv, directory, spec.per_case_timeout, input_path.read_text(errors="replace"))
            except (FileNotFoundError, NotADirectoryError, PermissionError) as error:
                return scored(0.0, reason="command_failed", error=str(error), passed=0, total=len(cases))
            if result.timed_out:
                return scored(0.0, reason="timeout", passed=index, total=len(cases), first_failure=number)
            expected = expected_path.read_text(errors="replace")
            if judge is None:
                accepted = _matches(result.stdout, expected, spec)
            else:
                got_path.write_text(result.stdout)
                accepted = _judge_accepts(judge, input_path, expected_path, got_path, spec.per_case_timeout)
            if not accepted:
                return scored(0.0, passed=index, total=len(cases), first_failure=number)
    return scored(1.0, passed=len(cases), total=len(cases))


class Case(NamedTuple):
    number: str
    stdin: Path
    expected: Path


def _cases(cases_dir: Path) -> list[Case]:
    """The cases under ``cases_dir``, in case-number order."""
    if not cases_dir.is_dir():
        raise InvalidTask(f"stdio cases directory {cases_dir} does not exist")
    pairs: list[Case] = []
    for input_path in cases_dir.rglob("input_*.txt"):
        match = INPUT_PATTERN.match(input_path.name)
        if match is None:
            continue
        number = match.group(1)
        expected = _expected(input_path, cases_dir, number)
        if expected is None:
            raise InvalidTask(f"stdio case {number} has no output_{number}.txt")
        pairs.append(Case(number, input_path, expected))
    pairs.sort(key=lambda case: (_sort_key(case.number), case.stdin))
    return pairs


def _expected(input_path: Path, cases_dir: Path, number: str) -> Path | None:
    name = f"output_{number}.txt"
    for candidate in (input_path.parent / name, cases_dir / "outputs" / name, cases_dir / name):
        if candidate.is_file():
            return candidate
    return None


def _sort_key(number: str) -> tuple[int, float, str]:
    if number.isdigit():
        return (0, int(number), "")
    return (1, 0.0, number)


def _matches(got: str, expected: str, spec: StdioSpec) -> bool:
    if spec.compare == Compare.EXACT:
        return _normalize_lines(got) == _normalize_lines(expected)
    got_tokens, expected_tokens = got.split(), expected.split()
    if spec.compare == Compare.TOKENS:
        return got_tokens == expected_tokens
    if len(got_tokens) != len(expected_tokens):
        return False
    return all(_float_equal(a, b, spec.float_tolerance) for a, b in zip(got_tokens, expected_tokens, strict=True))


def _normalize_lines(text: str) -> str:
    """Trailing whitespace on each line and blank lines at either end never change an answer."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    while lines and not lines[0]:
        lines.pop(0)
    return "\n".join(lines)


def _float_equal(got: str, expected: str, tolerance: float) -> bool:
    """Tokens match literally, or both are floats within absolute or relative ``tolerance``."""
    if got == expected:
        return True
    try:
        a, b = float(got), float(expected)
    except ValueError:
        return False
    if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
        return a == b
    difference = abs(a - b)
    return difference <= tolerance or difference <= tolerance * abs(b)


def _judge_accepts(judge: Path, input_path: Path, expected_path: Path, got_path: Path, timeout: float) -> bool:
    """The judge accepts iff its last non-empty stdout line is ``1``. A crashing judge rejects."""
    argv = ["python3", str(judge), str(input_path), str(expected_path), str(got_path)]
    result = run_command(argv, judge.parent, timeout)
    if result.timed_out or result.returncode != 0:
        return False
    return _verdict(result) == JUDGE_ACCEPT


def _verdict(result: Completed) -> str:
    return last_line(result.stdout) or ""
