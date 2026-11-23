# Taken from https://github.com/LiveCodeBench/LiveCodeBench/blob/998c52d394b836f15fff3b9a29866191108ff81b/lcb_runner/evaluation/testing_util.py
# Adapt from rllm: https://github.com/agentica-project/rllm/blob/main/rllm/rewards/reward_types.py#L24
import ast
import json
import logging
import os
import sys
import faulthandler
import platform
import re
import threading
from dataclasses import dataclass
from datetime import datetime
import signal
import struct

from io import BytesIO, StringIO
from unittest.mock import patch, mock_open
from types import ModuleType
from enum import Enum, StrEnum
from decimal import Decimal
from fractions import Fraction
import time

import multiprocessing
import resource
from collections.abc import Mapping
from typing import Any, NotRequired, TypedDict

# The sandbox prelude imports numpy and pandas after reliability_guard() has set
# os.putenv to None; on Linux their first import assigns os.environ entries, which
# calls os.putenv and fails every test with "'NoneType' object is not callable".
# A verifier child re-imports this module before the guard runs, so importing both
# here makes the prelude's imports sys.modules cache hits.
import numpy  # noqa: F401
import pandas  # noqa: F401


BASE_IMPORTS = """from itertools import accumulate, chain, combinations, count, permutations, product, groupby, islice, repeat
from copy import deepcopy
from string import ascii_lowercase, ascii_uppercase
from math import floor, log2, log10, sqrt, comb, gcd, ceil, inf, isqrt, factorial, atan2, pi
from collections import defaultdict, deque, Counter
from bisect import bisect, bisect_left, bisect_right, insort
from heapq import heappush, heappop, heapify, merge, nlargest, nsmallest, heapreplace
from functools import reduce, cache, lru_cache, cmp_to_key, reduce
from random import randrange, shuffle
from operator import itemgetter, sub, xor, or_
from re import search as re_search  # Assuming 're' refers to a regex search
from os.path import commonprefix
from typing import List, Tuple, Dict, Set, Optional, Union, Any, Callable, Iterable, Iterator, Generator, Deque
import copy
import string
import math
import collections
import bisect
import heapq
import functools
import random
import itertools
import operator
import re
import datetime
from time import time
import numpy as np
import pandas as pd
from math import log, prod  # 'log' and 'prod' are functions in the math module
from collections import deque, defaultdict, Counter, OrderedDict
from itertools import accumulate, permutations, combinations, product, groupby, islice, chain, repeat, zip_longest, cycle, pairwise
from functools import lru_cache, reduce, partial
from operator import iand
import sys
import io, os
"""

import_string = BASE_IMPORTS

# "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"


def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


class TestExecutionMode(Enum):
    collect_all = 0
    stop_on_failure = 1


STDIN_TEST_TYPE = "stdin"
FUNCTIONAL_TEST_TYPE = "functional"
FUNCTION_NAME_KEY = "func_name"
BINARY_REWARD_MODE = "binary"
FRACTIONAL_REWARD_MODE = "fractional"
LCB_REWARD_MODES = frozenset({BINARY_REWARD_MODE, FRACTIONAL_REWARD_MODE})
CodeGroundTruth = str | Mapping[str, Any] | list[Mapping[str, Any]]
ParsedCodeGroundTruth = Mapping[str, Any] | list[Any] | str | int | float | bool | None
_STDIN_SOURCE_TEST_TYPES = {STDIN_TEST_TYPE, "stdin_stdout"}
_FUNCTIONAL_SOURCE_TEST_TYPES = {FUNCTIONAL_TEST_TYPE, "call_based"}
_SOURCE_FUNCTION_NAME_KEY = "fn_name"


class CanonicalCodeCase(TypedDict):
    input: str
    output: str
    testtype: str
    metadata: NotRequired[dict[str, str]]


def _parse_code_ground_truth(ground_truth: CodeGroundTruth) -> ParsedCodeGroundTruth:
    if not isinstance(ground_truth, str):
        return ground_truth
    try:
        return json.loads(ground_truth)
    except json.JSONDecodeError as exc:
        raise ValueError("LCB ground_truth must be valid JSON.") from exc


def _canonical_code_case(case: Mapping[str, Any]) -> CanonicalCodeCase:
    test_input = case.get("input")
    expected_output = case.get("output")
    test_type = case.get("testtype", case.get("type"))
    if test_type in _STDIN_SOURCE_TEST_TYPES:
        if not isinstance(test_input, str) or not isinstance(expected_output, str):
            raise TypeError("Standard-input LCB test-case input and output must be strings.")
        return {"input": test_input, "output": expected_output, "testtype": STDIN_TEST_TYPE}
    if test_type in _FUNCTIONAL_SOURCE_TEST_TYPES:
        function_name = case.get(_SOURCE_FUNCTION_NAME_KEY)
        metadata = case.get("metadata")
        if function_name is None and isinstance(metadata, Mapping):
            function_name = metadata.get(FUNCTION_NAME_KEY)
        if not isinstance(function_name, str) or not function_name:
            raise ValueError("Functional LCB test cases require a function name.")
        if isinstance(test_input, list):
            test_input = "\n".join(json.dumps(argument) for argument in test_input)
        elif not isinstance(test_input, str):
            raise TypeError("Functional LCB test-case input must be a list of arguments or an encoded string.")
        if not isinstance(expected_output, str):
            expected_output = json.dumps(expected_output)
        return {
            "input": test_input,
            "output": expected_output,
            "testtype": FUNCTIONAL_TEST_TYPE,
            "metadata": {FUNCTION_NAME_KEY: function_name},
        }
    raise ValueError(f"Unsupported LCB test-case type: {test_type!r}.")


def _canonical_apps_cases(ground_truth: Mapping[str, Any]) -> list[CanonicalCodeCase]:
    inputs = ground_truth.get("inputs")
    outputs = ground_truth.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise TypeError("APPS ground_truth requires input and output lists.")
    if not inputs or len(inputs) != len(outputs):
        raise ValueError("APPS ground_truth requires equally sized, non-empty input and output lists.")
    function_name = ground_truth.get(_SOURCE_FUNCTION_NAME_KEY)
    test_type = FUNCTIONAL_TEST_TYPE if function_name is not None else STDIN_TEST_TYPE
    return [
        _canonical_code_case(
            {
                "input": test_input,
                "output": expected_output,
                "testtype": test_type,
                _SOURCE_FUNCTION_NAME_KEY: function_name,
            }
        )
        for test_input, expected_output in zip(inputs, outputs)
    ]


def normalize_lcb_ground_truth(ground_truth: CodeGroundTruth) -> str:
    """Return a canonical JSON-encoded LCB case list.

    Accepts canonical case lists, APPS ``inputs``/``outputs`` mappings, and open-r1
    ``test_cases`` mappings, either directly or as JSON text. All cases must use one
    execution mode and, for functional tests, one function name.
    """
    parsed = _parse_code_ground_truth(ground_truth)
    if isinstance(parsed, Mapping) and "test_cases" in parsed:
        if parsed.get("language") not in {None, "python"}:
            raise ValueError("LCB supports only Python verification specs.")
        cases = parsed["test_cases"]
    elif isinstance(parsed, Mapping):
        cases = _canonical_apps_cases(parsed)
    else:
        cases = parsed
    if not isinstance(cases, list) or not cases:
        raise ValueError("LCB ground_truth must contain at least one test case.")
    if not all(isinstance(case, Mapping) for case in cases):
        raise TypeError("LCB test cases must be mappings.")
    canonical_cases = [_canonical_code_case(case) for case in cases]
    test_types = {case["testtype"] for case in canonical_cases}
    if len(test_types) != 1:
        raise ValueError("LCB ground_truth cannot mix standard-input and functional test cases.")
    if FUNCTIONAL_TEST_TYPE in test_types:
        function_names = {case["metadata"][FUNCTION_NAME_KEY] for case in canonical_cases}
        if len(function_names) != 1:
            raise ValueError("All functional LCB test cases must use the same function name.")
    return json.dumps(canonical_cases, sort_keys=True)


# stuff for setting up signal timer
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    print("timeout occured: alarm went off")
    raise TimeoutException


# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)  # type: ignore
    except Exception:
        pass

    return code


def make_function(code: str) -> str:
    try:
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=all_other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        main_code = (
            import_string
            + "\n"
            + ast.unparse(import_stmts)  # type: ignore
            + "\n"
            + ast.unparse(function_ast)  # type: ignore
        )
        return main_code
    except Exception:
        return code


class _TextStdin(StringIO):
    """Text stdin patch that also exposes the underlying byte stream as ``buffer``."""

    def __init__(self, text: str):
        super().__init__(text)
        self.buffer = BytesIO(text.encode())


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", _TextStdin(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            pass
        finally:
            pass

    return _inner_call_method(method)


def get_function(compiled_sol, fn_name: str):  # type: ignore
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception:
        return


def compile_code(code: str, timeout: int):
    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        if "class Solution" in code:
            # leetcode wraps solutions in `Solution`
            # this is a hack to check if it is leetcode solution or not
            # currently livecodebench only supports LeetCode but
            # else condition allows future extensibility to other platforms
            compiled_sol = tmp_sol.Solution()
        else:
            # do nothing in the other case since function is accesible
            compiled_sol = tmp_sol

        assert compiled_sol is not None
    finally:
        signal.alarm(0)

    return compiled_sol


def convert_line_to_decimals(line: str) -> tuple[bool, list[Decimal]]:
    try:
        decimal_line = [Decimal(elem) for elem in line.split()]
    except Exception:
        return False, []
    return True, decimal_line


def get_stripped_lines(val: str):
    ## you don't want empty lines to add empty list after splitlines!
    val = val.strip()

    return [val_line.strip() for val_line in val.split("\n")]


class _ProgramError(Exception):
    """An exception the program raised in the verifier child, which reports it as text."""

    def __repr__(self) -> str:
        return self.args[0]


class _OversizedResult(Exception):
    """A result message longer than its bound, which the parent discarded without decoding it."""

    def __init__(self, size: int, bound: int):
        super().__init__(f"the program's result is {size} bytes, over the {bound}-byte bound for this test")


class _VerifierChildLost(BaseException):
    """The verifier child exited or missed its deadline before reporting a test.

    A ``BaseException``, so that the per-test ``except Exception`` handlers let it reach ``lcb_execution_result``.
    """


class _Unsupported:
    """A returned value the parent does not rebuild; it equals nothing.

    Either of a type the verifier child does not send back, a dict with a key that is not a string, or a NumPy
    scalar form whose value is not a bool, int or float.
    """

    def __init__(self, text: str):
        self.text = text

    def __eq__(self, other):
        return False

    def __str__(self) -> str:
        return self.text


_NON_STRING_KEY = "a dict with a key that is not a string"
"""How the parent shows a returned dict that it does not rebuild."""
_NON_SCALAR_NUMPY = "a NumPy scalar whose value is not a bool, int or float"
"""How the parent shows a NumPy scalar form that it does not rebuild."""

_NUMPY_SCALARS = frozenset(
    {"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64"}
)


def _value_form(value):
    """The JSON form of a value, from which ``_value_from_form`` rebuilds one equal to the same expected outputs.

    Subclasses of ``int``, ``float``, ``str``, ``list``, ``tuple`` and ``dict`` become their base type, so a class
    that overrides ``__eq__`` cannot decide a comparison. Raises ``TypeError`` for a value of any other type.
    """
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, numpy.generic):
        if value.dtype.name not in _NUMPY_SCALARS:
            raise TypeError(f"no value form for {value.dtype.name}")
        return ["numpy", value.dtype.name, value.item()]
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, Fraction):
        return ["fraction", value.numerator, value.denominator]
    if isinstance(value, Decimal):
        return ["decimal", str(value)]
    if isinstance(value, complex):
        return ["complex", value.real, value.imag]
    if isinstance(value, (list, tuple)):
        return ["tuple" if isinstance(value, tuple) else "list", [_value_form(item) for item in value]]
    if isinstance(value, dict):
        return ["dict", [[_value_form(key), _value_form(item)] for key, item in value.items()]]
    raise TypeError(f"no value form for {type(value).__name__}")


def _value_from_form(form):
    match form:
        case ["list", list(items)]:
            return [_value_from_form(item) for item in items]
        case ["tuple", list(items)]:
            return tuple(_value_from_form(item) for item in items)
        case ["dict", list(pairs)]:
            # A key that is not a string hashes by a public function of its value, so keys can be chosen to collide
            # and make building the dict take quadratic time. Expected outputs are JSON, whose keys are strings, so
            # such a dict equals none of them.
            if not all(isinstance(key, str) for key, _ in pairs):
                return _Unsupported(_NON_STRING_KEY)
            return {key: _value_from_form(item) for key, item in pairs}
        case ["numpy", str(name), bool() | int() | float() as value] if name in _NUMPY_SCALARS:
            return numpy.dtype(name).type(value)
        case ["numpy", str(name), _] if name in _NUMPY_SCALARS:
            # The NumPy constructors also build an array from a list, and comparing an array with an expected list
            # broadcasts: an (n, 1) array against n items allocates n * n results. ``_value_form`` writes only
            # ``item()``, a bool, int or float, so any other value equals nothing.
            return _Unsupported(_NON_SCALAR_NUMPY)
        case ["fraction", int(numerator), int(denominator)]:
            return Fraction(numerator, denominator)
        case ["decimal", str(text)]:
            return Decimal(text)
        case ["complex", float(real), float(imaginary)]:
            return complex(real, imaginary)
        case list():
            raise ValueError(f"malformed value form: {truncatefn(form)}")
    return form


class _ChildMessage(StrEnum):
    """The first field of each message a verifier child sends its parent."""

    READY = "ready"
    SETUP_ERROR = "setup_error"
    VALUE = "value"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


_MISSING_FUNCTION = "the program does not define the function the tests call"
"""The setup error a verifier child reports for a program that compiles but lacks the tested function."""
_FRAME_LENGTH = struct.Struct("!Q")
"""The length prefix of each message a verifier child writes to its parent's pipe."""
_PIPE_READ_BYTES = 1 << 20
"""The most one read takes from the pipe: a length prefix the program wrote can be arbitrarily large."""
_MESSAGE_FLOOR_BYTES = 1 << 20
"""The longest message the parent reads whatever the expected output: setup messages and small results."""
_MESSAGE_BYTES_PER_EXPECTED_BYTE = 16
"""How much longer than its expected output a result message may be. The value form of a list of one-digit numpy
integers is up to 12 times the compact JSON of the same list."""


def _result_bound(expected_output: str) -> int:
    """The longest result message the parent reads for a test whose expected output is ``expected_output``.

    The parent decodes and compares each result outside the child's memory cap, so the bound keeps what it
    allocates proportional to the expected outputs it already holds.
    """
    return _MESSAGE_FLOOR_BYTES + _MESSAGE_BYTES_PER_EXPECTED_BYTE * len(expected_output.encode())


def _message(*fields) -> bytes:
    return json.dumps(fields).encode()


def _send(connection, message: bytes) -> None:
    frame = memoryview(_FRAME_LENGTH.pack(len(message)) + message)
    while frame:
        frame = frame[os.write(connection.fileno(), frame) :]


class _ChildProgram:
    """The program running in a verifier child, as this process reads it: its result for each test, in order."""

    def __init__(self, connection, deadline: float):
        self._connection = connection
        self._deadline = deadline

    def _read(self, size: int) -> bytearray:
        # The program can reach the pipe and write part of a message, so every read waits only until the deadline.
        data = bytearray()
        while len(data) < size:
            if not self._connection.poll(max(0.0, self._deadline - time.monotonic())):
                raise _VerifierChildLost
            chunk = os.read(self._connection.fileno(), min(size - len(data), _PIPE_READ_BYTES))
            if not chunk:
                raise _VerifierChildLost
            data += chunk
        return data

    def _receive(self, bound: int) -> Any:
        """The next message, decoded; a message longer than ``bound`` bytes is read off the pipe and dropped."""
        (size,) = _FRAME_LENGTH.unpack(self._read(_FRAME_LENGTH.size))
        if size > bound:
            remaining = size
            while remaining:
                remaining -= len(self._read(min(remaining, _PIPE_READ_BYTES)))
            raise _OversizedResult(size, bound)
        return json.loads(self._read(size))

    def wait_until_ready(self) -> None:
        """Wait until the program has compiled and defines the tested function; raise why it cannot run otherwise."""
        match self._receive(_MESSAGE_FLOOR_BYTES):
            case [_ChildMessage.READY]:
                return
            case [_ChildMessage.SETUP_ERROR, str(text)]:
                raise _ProgramError(text)
            case message:
                raise ValueError(f"unexpected verifier child message: {truncatefn(message)}")

    def next_result(self, bound: int) -> tuple[Any, float]:
        """What the program returned or printed for the next test, and how long it ran.

        Raises ``_OversizedResult`` if the result's message is longer than ``bound`` bytes.
        """
        match self._receive(bound):
            case [_ChildMessage.VALUE, form, int() | float() as execution_time]:
                return _value_from_form(form), execution_time
            case [_ChildMessage.UNSUPPORTED, str(text), int() | float() as execution_time]:
                return _Unsupported(text), execution_time
            case [_ChildMessage.ERROR, str(text)]:
                raise _ProgramError(text)
            case message:
                raise ValueError(f"unexpected verifier child message: {truncatefn(message)}")


def _execution_failure(error: Exception, test_input, expected_output) -> tuple[int, dict[str, Any]]:
    if isinstance(error, _OversizedResult):
        error_code, error_message = -2, "Wrong Answer: result over the size bound"
    elif "timeoutexception" in repr(error).lower():
        error_code, error_message = -3, "Time Limit Exceeded"
    else:
        error_code, error_message = -4, "Runtime Error"
    return error_code, {
        "error": repr(error),
        "error_code": error_code,
        "error_message": error_message,
        "inputs": truncatefn(test_input),
        "expected": truncatefn(expected_output),
    }


def _call_arguments(all_inputs: list) -> list:
    return [[json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs]


def grade_call_based(
    program: _ChildProgram,
    all_inputs: list,
    all_outputs: list,
    execution_mode: TestExecutionMode = TestExecutionMode.collect_all,
):
    program.wait_until_ready()

    all_inputs = _call_arguments(all_inputs)

    bounds = [_result_bound(output) for output in all_outputs]
    all_outputs = [json.loads(output) for output in all_outputs]

    total_execution = 0
    all_results = []
    failure_metadata = None
    for idx, (gt_inp, gt_out, bound) in enumerate(zip(all_inputs, all_outputs, bounds)):
        try:
            prediction, execution_time = program.next_result(bound)
            total_execution += execution_time

            # don't penalize model if it produces tuples instead of lists
            # ground truth sequences are not tuples
            if isinstance(prediction, tuple):
                prediction = list(prediction)

            tmp_result = prediction == gt_out

            # handle floating point comparisons

            all_results.append(tmp_result)

            if not tmp_result and failure_metadata is None:
                failure_metadata = {
                    "output": truncatefn(prediction),
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                    "error_code": -2,
                    "error_message": "Wrong Answer",
                }
            if not tmp_result and execution_mode is TestExecutionMode.stop_on_failure:
                return all_results, failure_metadata
        except Exception as e:
            error_code, metadata = _execution_failure(e, gt_inp, gt_out)
            all_results.append(error_code)
            if failure_metadata is None:
                failure_metadata = metadata
            if execution_mode is TestExecutionMode.stop_on_failure:
                return all_results, failure_metadata

    return all_results, failure_metadata or {"execution time": total_execution}


def grade_stdio(
    program: _ChildProgram,
    all_inputs: list,
    all_outputs: list,
    execution_mode: TestExecutionMode = TestExecutionMode.collect_all,
):
    program.wait_until_ready()

    all_results = []
    total_execution_time = 0
    failure_metadata = None
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        try:
            prediction, execution_time = program.next_result(_result_bound(gt_out))
            total_execution_time += execution_time
        except Exception as e:
            error_code, metadata = _execution_failure(e, gt_inp, gt_out)
            all_results.append(error_code)
            if failure_metadata is None:
                failure_metadata = metadata
            if execution_mode is TestExecutionMode.stop_on_failure:
                return all_results, failure_metadata
            continue

        stripped_prediction_lines = get_stripped_lines(prediction)
        stripped_gt_out_lines = get_stripped_lines(gt_out)

        ## WA happens in multiple circumstances
        ## so cache the return to make it clean!
        WA_send_args = {
            "output": truncatefn(prediction),
            "inputs": truncatefn(gt_inp),
            "expected": truncatefn(gt_out),
            "error_code": -2,
        }

        if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
            all_results.append(-2)
            WA_send_args["error_message"] = "Wrong answer: mismatched output length"
            if failure_metadata is None:
                failure_metadata = WA_send_args
            if execution_mode is TestExecutionMode.stop_on_failure:
                return all_results, failure_metadata
            continue

        matches = True
        for output_line_idx, (
            stripped_prediction_line,
            stripped_gt_out_line,
        ) in enumerate(zip(stripped_prediction_lines, stripped_gt_out_lines)):
            WA_send_args["error_message"] = (
                f"Wrong answer at {output_line_idx=}: {truncatefn(stripped_prediction_line)} != {truncatefn(stripped_gt_out_line)}"
            )

            ## CASE 1: exact match
            if stripped_prediction_line == stripped_gt_out_line:
                continue

            ## CASE 2: element-wise comparision
            ## if there are floating elements
            ## use `decimal` library for good floating point comparision
            ## otherwise gotcha: np.isclose(50000000000000000, 50000000000000001) = True
            ## note that we should always be able to convert to decimals

            success, decimal_prediction_line = convert_line_to_decimals(stripped_prediction_line)
            if not success:
                matches = False
                break
            success, decimal_gtout_line = convert_line_to_decimals(stripped_gt_out_line)
            if not success:
                matches = False
                break

            if decimal_prediction_line == decimal_gtout_line:
                continue

            matches = False
            break
        all_results.append(True if matches else -2)
        if not matches and failure_metadata is None:
            failure_metadata = WA_send_args
        if not matches and execution_mode is TestExecutionMode.stop_on_failure:
            return all_results, failure_metadata

    return all_results, failure_metadata or {"execution time": total_execution_time}


def run_test(
    sample,
    program: _ChildProgram,
    debug=False,
    execution_mode=TestExecutionMode.collect_all,
):
    """Compare the results ``program`` reports for each test with the sample's expected outputs."""
    if debug:
        print(f"start = {datetime.now().time()}")

    try:
        in_outs = json.loads(sample["input_output"])
    except ValueError as e:
        raise e
        in_outs = None

    if in_outs:
        if in_outs.get("fn_name") is None:
            which_type = CODE_TYPE.standard_input  # Standard input
        else:
            which_type = CODE_TYPE.call_based  # Call-based

    if debug:
        print(f"loaded input_output = {datetime.now().time()}")

    grade = grade_call_based if which_type == CODE_TYPE.call_based else grade_stdio
    try:
        return grade(
            program=program,
            all_inputs=in_outs["inputs"],
            all_outputs=in_outs["outputs"],
            execution_mode=execution_mode,
        )
    except Exception as e:
        return [-4], {
            "error_code": -4,
            "error_message": f"Error during testing: {e}",
        }


def _program_method(code: str, fn_name: str | None, timeout: int):
    """The function each test calls: ``fn_name`` of a call-based program, else the whole program wrapped."""
    if fn_name is not None:
        return get_function(compile_code(import_string + "\n\n" + code, timeout), fn_name)
    ## runtime doesn't interact well with __name__ == '__main__'
    ## we wrap the given code inside another function
    code = make_function(clean_if_name(code))
    return get_function(compile_code(code, timeout), "wrapped_function")


def _program_output(method, test_input, code_type: CODE_TYPE, timeout: int) -> bytes:
    """The message reporting what the program returned or printed for one input, or what it raised."""
    signal.alarm(timeout)
    faulthandler.enable()
    try:
        start = time.time()
        if code_type is CODE_TYPE.call_based:
            prediction = method(*test_input)
        else:
            with Capturing() as captured_output:
                call_method(method, test_input)
            prediction = captured_output[0]
        execution_time = time.time() - start
    except Exception as e:
        return _message(_ChildMessage.ERROR, repr(e))
    finally:
        signal.alarm(0)
        faulthandler.disable()
    try:
        return _message(_ChildMessage.VALUE, _value_form(prediction), execution_time)
    except (TypeError, ValueError, RecursionError):
        # Any other value fails as a wrong answer, reported by its text.
        try:
            return _message(_ChildMessage.UNSUPPORTED, str(prediction), execution_time)
        except Exception as e:
            return _message(_ChildMessage.ERROR, repr(e))


def _tighten_memory_limit(limit_kind, maximum_memory_bytes):
    """Lower one soft rlimit to ``maximum_memory_bytes``, never raising it.

    Only the soft limit moves: lowering a hard limit below the current soft limit is
    refused on macOS. Platforms that refuse the change keep their existing limit after
    one warning.
    """
    soft, hard = resource.getrlimit(limit_kind)
    cap = maximum_memory_bytes if hard == resource.RLIM_INFINITY else min(maximum_memory_bytes, hard)
    if soft != resource.RLIM_INFINITY and soft <= cap:
        return
    try:
        resource.setrlimit(limit_kind, (cap, hard))
    except ValueError as error:
        _logger.warning("Could not tighten %s to %d bytes: %s", limit_kind, maximum_memory_bytes, error)


def reliability_guard(maximum_memory_bytes=None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        _tighten_memory_limit(resource.RLIMIT_AS, maximum_memory_bytes)
        _tighten_memory_limit(resource.RLIMIT_DATA, maximum_memory_bytes)
        if not platform.uname().system == "Darwin":
            _tighten_memory_limit(resource.RLIMIT_STACK, maximum_memory_bytes)

    faulthandler.disable()

    import builtins

    # builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    if isinstance(__builtins__, dict):
        __builtins__["help"] = None
    elif hasattr(__builtins__, "help"):
        setattr(__builtins__, "help", None)

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def postprocess_lcb_sample(sample):
    sample_inputs = [sample["input"] for sample in sample]
    sample_outputs = [sample["output"] for sample in sample]

    sample_dict = {
        "inputs": sample_inputs,
        "outputs": sample_outputs,
    }

    if sample[0].get("testtype") == FUNCTIONAL_TEST_TYPE:
        metadata = sample[0].get("metadata", {})
        fn_name = metadata.get(FUNCTION_NAME_KEY, None)
        assert fn_name is not None, (
            f"Function name is not found, check if your LCB data is preprocessed correctly: {metadata}"
        )
        # Fill in the blank
        sample_dict["fn_name"] = fn_name

    sample = {
        "input_output": json.dumps(sample_dict),
    }
    return sample


@dataclass(frozen=True)
class VerifierLimits:
    """Resource bounds for one verifier child process.

    ``max_memory_bytes`` caps the child's address space via soft RLIMIT_AS/DATA/STACK;
    ``total_timeout_seconds`` caps the wall-clock window one child can occupy regardless
    of test count. A ``None`` value disables that bound.
    """

    max_memory_bytes: int | None = 4 * 1024**3
    total_timeout_seconds: float | None = 300.0


DEFAULT_LIMITS = VerifierLimits()

_SPAWN = multiprocessing.get_context("spawn")
_verifier_slots_lock = threading.Lock()
_verifier_slots: threading.BoundedSemaphore | None = None
_logger = logging.getLogger(__name__)


def verifier_slots() -> threading.BoundedSemaphore:
    """Process-wide verifier-child slots, sized once from SKYRL_GYM_LCB_MAX_CONCURRENT.

    The trainer runs up to `max_env_workers` env threads, each able to hold one
    verifier child; a slot count keeps concurrent children (and their memory caps)
    summable. Default 8.
    """
    global _verifier_slots
    with _verifier_slots_lock:
        if _verifier_slots is None:
            try:
                count = int(os.environ.get("SKYRL_GYM_LCB_MAX_CONCURRENT", "8"))
            except ValueError:
                _logger.warning("Invalid SKYRL_GYM_LCB_MAX_CONCURRENT value; using 8 verifier slots")
                count = 8
            _verifier_slots = threading.BoundedSemaphore(max(1, count))
        return _verifier_slots


def _run_program_in_subprocess(generation, fn_name, all_inputs, connection, timeout, max_memory_bytes):
    """Run the program on each test input in order and report each result to the parent as JSON."""
    signal.signal(signal.SIGALRM, timeout_handler)

    # Disable functionalities that can make destructive changes to the test.
    # A non-null max_memory_bytes caps the child's address space via RLIMIT_AS/DATA/STACK.
    reliability_guard(maximum_memory_bytes=max_memory_bytes)

    try:
        method = _program_method(generation, fn_name, timeout)
    except Exception as e:
        _send(connection, _message(_ChildMessage.SETUP_ERROR, str(e)))
        return
    if method is None:
        _send(connection, _message(_ChildMessage.SETUP_ERROR, _MISSING_FUNCTION))
        return
    _send(connection, _message(_ChildMessage.READY))
    code_type = CODE_TYPE.standard_input if fn_name is None else CODE_TYPE.call_based
    for test_input in _call_arguments(all_inputs) if code_type is CODE_TYPE.call_based else all_inputs:
        _send(connection, _program_output(method, test_input, code_type, timeout))


def lcb_execution_result(
    sample,
    generation,
    timeout=6,
    debug=False,
    execution_mode=TestExecutionMode.collect_all,
    limits=None,
):
    """Return per-test pass/failure results and execution diagnostics.

    A child process runs the program on the test inputs and reports what it returned or
    printed for each; this process compares that with the expected outputs, which the
    child never receives, so the program cannot read an expected output or decide a
    comparison. A process-level timeout catches extreme cases not handled by the per-test
    alarms. Stop-on-failure mode preserves the binary scorer's original short circuit. Each
    child runs under ``limits`` (default :class:`VerifierLimits`): a per-child memory
    cap and a wall-clock deadline independent of test count, with deterministic
    reaping. See https://echo.oa.dev/wiki/471 for the failure mode these bounds exist
    to contain.
    """
    assert len(sample) >= 1, "Sample must contain at least one test case"
    limits = DEFAULT_LIMITS if limits is None else limits
    sample = postprocess_lcb_sample(sample)
    in_outs = json.loads(sample["input_output"])
    num_tests = len(in_outs["inputs"])
    deadline = (timeout + 1) * num_tests + 5
    if limits.total_timeout_seconds is not None:
        deadline = min(deadline, limits.total_timeout_seconds)

    # The child is spawned, not forked: a forked child would inherit this process's memory,
    # expected outputs included. The target must be picklable: the child re-imports it by
    # qualified name.
    receiver, sender = _SPAWN.Pipe(duplex=False)
    p = _SPAWN.Process(
        target=_run_program_in_subprocess,
        args=(generation, in_outs.get("fn_name"), in_outs["inputs"], sender, timeout, limits.max_memory_bytes),
    )
    with verifier_slots():
        p.start()
        sender.close()
        _logger.info(
            "LiveCodeBench verifier child started: pid=%s tests=%d per_test_timeout=%ss deadline=%.0fs",
            p.pid,
            num_tests,
            timeout,
            deadline,
        )
        started = time.monotonic()
        try:
            results = run_test(sample, _ChildProgram(receiver, started + deadline), debug, execution_mode)
        except _VerifierChildLost:
            results = None
        if results is not None:
            # After a first failure the child may already be running the next test.
            p.kill()
        p.join(max(0.0, deadline - (time.monotonic() - started)))
        timed_out = p.is_alive()
        if timed_out:
            p.kill()
            p.join()
    if timed_out:
        _logger.warning(
            "LiveCodeBench verifier child pid=%s exceeded its %.0fs deadline (%d tests); scoring all failed",
            p.pid,
            deadline,
            num_tests,
        )
    receiver.close()
    if results is not None:
        test_results, metadata = results
        return list(test_results), metadata
    if results is None:
        # No verdict was returned; retain the cause as well as failure sentinels.
        results = [-1 for _ in range(num_tests)]
        if debug:
            print("global timeout")
    return list(results), {
        "execution_error": "process_timeout" if timed_out else "child_crash",
        "exit_code": p.exitcode,
    }


def lcb_test_results(
    sample, generation, timeout=6, debug=False, execution_mode=TestExecutionMode.collect_all, limits=None
):
    results, _ = lcb_execution_result(sample, generation, timeout, debug, execution_mode, limits)
    return results


def lcb_check_correctness(sample, generation, timeout=6, debug=False, limits=None):
    return all(
        result is True
        for result in lcb_test_results(
            sample,
            generation,
            timeout,
            debug,
            execution_mode=TestExecutionMode.stop_on_failure,
            limits=limits,
        )
    )


def extract_code_from_model(model_response: str):
    """
    Extracts the code from a Markdown-style code block in an LLM output.

    Parameters:
        model_response (str): The text output from the LLM.

    Returns:
        str: The extracted code, or an empty string if no code block is found.
    """
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()


def compute_score(model_response, tests, reward_mode=BINARY_REWARD_MODE, *, timeout=6, limits=None):
    output_code = extract_code_from_model(model_response)
    if output_code is None:
        return output_code, 0.0
    if reward_mode == BINARY_REWARD_MODE:
        reward = 1.0 if lcb_check_correctness(tests, output_code, timeout=timeout, debug=False, limits=limits) else 0.0
    elif reward_mode == FRACTIONAL_REWARD_MODE:
        results = lcb_test_results(tests, output_code, timeout=timeout, debug=False, limits=limits)
        reward = sum(result is True for result in results) / len(results)
    else:
        raise ValueError(f"Unsupported LCB reward_mode: {reward_mode!r}.")
    return output_code, reward
