"""Candidate Python runtime. Hidden test outputs stay on the worker."""

# Taken from https://github.com/LiveCodeBench/LiveCodeBench/blob/998c52d394b836f15fff3b9a29866191108ff81b/lcb_runner/evaluation/testing_util.py
# Adapt from rllm: https://github.com/agentica-project/rllm/blob/main/rllm/rewards/reward_types.py#L24

import ast
import contextlib
import io
import json
import types
from collections.abc import Callable
from io import BytesIO, StringIO
from unittest.mock import mock_open, patch

CANDIDATE_RESULT_PREFIX = "VALUE:"

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


def clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)  # type: ignore
    except (SyntaxError, IndexError):
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
            BASE_IMPORTS
            + "\n"
            + ast.unparse(import_stmts)  # type: ignore
            + "\n"
            + ast.unparse(function_ast)  # type: ignore
        )
        return main_code
    except SyntaxError:
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

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", _TextStdin(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit as error:
            if error.code not in (None, 0):
                raise

    return _inner_call_method(method)


def _wire(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return [type(value).__name__, value]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [_wire(item) for item in value]]
    if isinstance(value, dict):
        return ["dict", [[_wire(key), _wire(item)] for key, item in value.items()]]
    raise ValueError("unsupported output")


def candidate_method(code: str, function: str | None) -> Callable:
    """Return a compiled candidate without reference answers."""
    namespace = {}
    program = BASE_IMPORTS + "\n\n" + code if function else make_function(clean_if_name(code))
    with contextlib.redirect_stdout(io.StringIO()):
        exec(program, namespace)
        if function:
            candidate = namespace["Solution"]() if "Solution" in namespace else types.SimpleNamespace(**namespace)
            return getattr(candidate, function)
        return namespace["wrapped_function"]


def evaluate(method: Callable, arguments, function: str | None) -> None:
    """Write candidate output or a runtime-failure marker for one test input."""
    try:
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            if function:
                prediction = method(*arguments)
                if isinstance(prediction, tuple):
                    prediction = list(prediction)
            else:
                call_method(method, arguments)
                prediction = captured.getvalue()
        print(CANDIDATE_RESULT_PREFIX + json.dumps(_wire(prediction), allow_nan=False))
    except (Exception, SystemExit):
        print("CANDIDATE_FAILURE")
