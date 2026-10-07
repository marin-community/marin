# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Enforce the reproduced recursive factorial and student-test contract."""

import hashlib
from dataclasses import dataclass

TEST_SHA = "0f00bf58569de0cb316f383aa81a7e8c9f92c85f500a3db1806955e0ee809d1d"
STUDENT_TEST_PATH = "/app/tests/test_factorial.py"
IMPORTS = """import inspect
import json
import os
import subprocess
import tempfile
import typing
from pathlib import Path

"""
CONTRACT_TESTS = """

def test_factorial_documentation_and_type_hints():
    assert calculate_factorial.__doc__ and calculate_factorial.__doc__.strip()
    signature = inspect.signature(calculate_factorial)
    assert len(signature.parameters) == 1
    hints = typing.get_type_hints(calculate_factorial)
    parameter = next(iter(signature.parameters))
    assert hints[parameter] is int
    assert hints['return'] is int


def test_factorial_uses_recursion():
    source = Path(inspect.getfile(calculate_factorial)).resolve()
    recursive_calls = []
    def observe(frame, event, _argument):
        if event != 'call' or Path(frame.f_code.co_filename).resolve() != source:
            return
        caller = frame.f_back
        while caller is not None:
            if caller.f_code is frame.f_code:
                recursive_calls.append(frame.f_code.co_name)
                return
            caller = caller.f_back
    previous = sys.getprofile()
    try:
        sys.setprofile(observe)
        assert calculate_factorial(8) == 40320
    finally:
        sys.setprofile(previous)
    assert recursive_calls, 'Use a recursive function to compute the factorial'


def test_student_tests_detect_requested_factorial_errors():
    path = Path('__STUDENT_TEST_PATH__')
    assert path.is_file(), f'Write your tests at {path}'
    with tempfile.TemporaryDirectory() as temporary:
        for mutant in ['correct', 'negative', '0', '1', '2', '3', '4', '5']:
            report = Path(temporary) / (mutant + '.json')
            environment = os.environ.copy()
            environment['PYTHONPATH'] = '/app:/tests'
            environment['FACTORIAL_MUTANT'] = mutant
            command = [sys.executable, '-m', 'pytest', '-q', str(path),
                       '--rootdir=/app', '-p', 'no:cacheprovider',
                       '--json-report', '--json-report-file=' + str(report)]
            if mutant != 'correct':
                command.extend(['-p', 'factorial_coverage'])
            process = subprocess.run(command, cwd='/app', env=environment,
                                     capture_output=True, text=True, timeout=20)
            assert report.is_file(), process.stdout + process.stderr
            summary = json.loads(report.read_text())['summary']
            if mutant == 'correct':
                assert process.returncode == 0 and summary.get('passed', 0) > 0, summary
            else:
                assert process.returncode == 1 and summary.get('failed', 0) > 0, (mutant, summary)
                assert summary.get('error', 0) == 0, summary
""".replace(
    "__STUDENT_TEST_PATH__", STUDENT_TEST_PATH
)
COVERAGE_PLUGIN = """import os

import factorial


def pytest_sessionstart(session):
    original = factorial.calculate_factorial
    mutant = os.environ['FACTORIAL_MUTANT']
    def changed(number):
        if mutant == 'negative' and number < 0:
            return 1
        if mutant != 'negative' and number == int(mutant):
            return -1
        factorial.calculate_factorial = original
        try:
            return original(number)
        finally:
            factorial.calculate_factorial = changed
    factorial.calculate_factorial = changed
"""
TEST_INTERFACE = """

Your tests must run with pytest and check negative-input validation, 0!, and factorials
of each integer from 1 through 5. The verifier also checks the function independently.
"""


@dataclass(frozen=True)
class FactorialContract:
    test: bytes
    instruction: str
    extra_files: dict[str, bytes]


def repair_contract(test: bytes, instruction: str) -> FactorialContract:
    """Repair the known factorial fixture without changing other Python tasks."""
    if hashlib.sha256(test).hexdigest() != TEST_SHA or "calculate_factorial" not in instruction:
        return FactorialContract(test, instruction, {})
    instruction = instruction.replace("/tests/test_factorial.py", STUDENT_TEST_PATH) + TEST_INTERFACE
    return FactorialContract(
        IMPORTS.encode() + test + CONTRACT_TESTS.encode(),
        instruction,
        {"tests/factorial_coverage.py": COVERAGE_PLUGIN.encode()},
    )
