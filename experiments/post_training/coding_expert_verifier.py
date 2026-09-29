# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure ShellSim verification for the coding-expert task packages."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from shellsim import python as shellsim_python

SOLUTION_PATH = "/workspace/solution.py"
CASE_CPU_LIMIT = 5_000_000
CASE_MEMORY_BYTES = 256 * 1024 * 1024
CASE_OUTPUT_BYTES = 1024 * 1024


@dataclass(frozen=True)
class CodeCase:
    stdin: bytes
    expected: str


@dataclass(frozen=True)
class VerificationSummary:
    reward: int
    message: str


def outputs_match(actual: str, expected: str) -> bool:
    """Compare standard output with the LiveCodeBench line and numeric rules."""
    actual_lines = [line.strip() for line in actual.strip().split("\n")]
    expected_lines = [line.strip() for line in expected.strip().split("\n")]
    if len(actual_lines) != len(expected_lines):
        return False

    for actual_line, expected_line in zip(actual_lines, expected_lines, strict=True):
        if actual_line == expected_line:
            continue
        try:
            actual_numbers = [Decimal(value) for value in actual_line.split()]
            expected_numbers = [Decimal(value) for value in expected_line.split()]
        except InvalidOperation:
            return False
        if actual_numbers != expected_numbers:
            return False
    return True


def parse_cases(payload: Any) -> tuple[CodeCase, ...]:
    """Validate the private standard-input cases at the verifier boundary."""
    if not isinstance(payload, list) or not payload:
        raise ValueError("cases.json must contain a nonempty list")

    cases = []
    for index, raw_case in enumerate(payload):
        if not isinstance(raw_case, dict):
            raise ValueError(f"case {index} must be an object")
        stdin = raw_case.get("input")
        expected = raw_case.get("output")
        if not isinstance(stdin, str) or not isinstance(expected, str):
            raise ValueError(f"case {index} must contain string input and output values")
        cases.append(CodeCase(stdin.encode(), expected))
    return tuple(cases)


def verify_source(source: str, cases: tuple[CodeCase, ...]) -> VerificationSummary:
    """Execute each case in an independent bounded ShellSim Python process."""
    for index, case in enumerate(cases):
        result = shellsim_python.run(
            source,
            argv=(SOLUTION_PATH,),
            stdin=case.stdin,
            cpu=CASE_CPU_LIMIT,
            memory=CASE_MEMORY_BYTES,
            output=CASE_OUTPUT_BYTES,
        )
        if result.returncode != 0:
            reason = result.stop_reason or f"exit {result.returncode}"
            return VerificationSummary(0, f"case {index + 1}/{len(cases)} failed: {reason}")
        actual = result.stdout.decode(errors="replace")
        if not outputs_match(actual, case.expected):
            return VerificationSummary(0, f"case {index + 1}/{len(cases)} failed: wrong output")
    return VerificationSummary(1, f"all {len(cases)} cases passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solution", type=str)
    parser.add_argument("cases", type=str)
    args = parser.parse_args()
    with open(args.solution) as solution_file:
        source = solution_file.read()
    with open(args.cases) as cases_file:
        cases = parse_cases(json.load(cases_file))
    print(json.dumps(verify_source(source, cases).__dict__, sort_keys=True))


if __name__ == "__main__":
    main()
