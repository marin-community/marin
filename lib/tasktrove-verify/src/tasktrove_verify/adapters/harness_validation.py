# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared validation for pinned harness task contracts."""

import ast
import functools
import hashlib
import inspect
import math
from collections.abc import Collection, Mapping, Sequence
from importlib import import_module
from pathlib import Path
from types import CodeType
from typing import Any

from tasktrove_verify.grade import InvalidTask


def pinned_source_bytes(path: Path, digests: Collection[str]) -> bytes | None:
    """Read one source snapshot for both digest verification and code comparison."""
    if not path.is_file():
        return None
    source = path.read_bytes()
    return source if hashlib.sha256(source).hexdigest() in digests else None


@functools.lru_cache(maxsize=4)
def compiled_source_functions(source: bytes, filename: str) -> tuple[CodeType, ...]:
    return tuple(code for code in compile(source, filename, "exec").co_consts if isinstance(code, CodeType))


def source_functions_match(
    namespace: dict[str, Any], codes: Sequence[CodeType], defaults: Mapping[str, tuple[object, ...]]
) -> bool:
    """Match function code, globals and defaults against the verified snapshot."""
    for code in codes:
        function = namespace.get(code.co_name)
        if (
            not inspect.isfunction(function)
            or function.__code__ != code
            or function.__globals__ is not namespace
            or function.__defaults__ != defaults.get(code.co_name)
            or function.__kwdefaults__ is not None
            or function.__closure__ is not None
        ):
            return False
    return True


def pinned_function_namespace(function, name: str, digests: Collection[str]) -> dict[str, Any]:
    """Validate the selected callable and top-level functions from one source snapshot."""
    if not inspect.isfunction(function) or function.__name__ != name:
        raise InvalidTask("source callable changed")
    path = Path(function.__code__.co_filename)
    source = pinned_source_bytes(path, digests)
    if source is None or function.__globals__.get(name) is not function:
        raise InvalidTask("source callable identity changed")
    names = {node.name for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    codes = [code for code in compiled_source_functions(source, str(path)) if code.co_name in names]
    if not source_functions_match(function.__globals__, codes, {}):
        raise InvalidTask("source function graph changed")
    return function.__globals__


def validate_default_filter(task, label: str) -> None:
    """Require the unmodified harness default ensemble and TakeFirst constructor."""
    ensemble_type = import_module("lm_eval.api.filter").FilterEnsemble
    take_first = import_module("lm_eval.filters.selection").TakeFirstFilter
    filters = task._filters
    if (
        task.config.filter_list is not None
        or len(filters) != 1
        or type(filters[0]) is not ensemble_type
        or filters[0].name != "none"
        or "apply" in vars(filters[0])
    ):
        raise InvalidTask(f"{label} requires the default response filter")
    constructors = filters[0].filters
    if (
        len(constructors) != 1
        or not isinstance(constructors[0], functools.partial)
        or constructors[0].func is not take_first
        or constructors[0].args
        or constructors[0].keywords
    ):
        raise InvalidTask(f"{label} requires the unmodified TakeFirstFilter")


def log_likelihoods(responses, label: str) -> list[float]:
    """Decode finite, nonpositive likelihoods with the source greedy flag intact."""
    values = []
    for response in responses:
        if not isinstance(response, (tuple, list)) or len(response) != 2 or type(response[1]) is not bool:
            raise InvalidTask(f"{label} likelihood responses must be (number, bool) pairs")
        value = response[0]
        try:
            finite = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or value > 0:
            raise InvalidTask(f"{label} log-likelihoods must be finite and nonpositive")
        values.append(value)
    return values
