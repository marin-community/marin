# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from functools import update_wrapper, wraps
from typing import Any, Generic, ParamSpec, TypeVar
import typing

from marin.execution.step_model import StepSpec, StepMeta
from pydantic import BaseModel

T = TypeVar("T")
P = ParamSpec("P")


@dataclass(frozen=True)
class StepCallDeferred(Generic[T], Any):
    """Represents a deferred function call"""

    _name: str
    _fn: Callable[..., T]
    _args: tuple
    _kwargs: dict

    def __getattr__(self, name: str) -> Any:
        raise TypeError(f"{self} is deferred, can't access the attribute {name!r}")

    def __repr__(self):
        return f"{StepCallDeferred.__name__}({self._name}, args={self._args}, kwargs={self._kwargs})"


@dataclass(init=False)
class Step(Generic[P, T]):
    """
    Represents a step function that can be invoked lazily via the `defer` method.
    """

    _fn: Callable[P, T]
    _name: str | None = None

    def __init__(self, fn: Callable[P, T], *, name: str | None = None):
        self._fn = fn
        self._name = name or getattr(fn, "__name__", None)
        update_wrapper(self, self._fn)  # type: ignore[invalid-argument-type]

    def defer(self, *args: P.args, **kwargs: P.kwargs) -> StepCallDeferred[T]:
        return StepCallDeferred(_fn=self._fn, _name=self._name, _args=args, _kwargs=kwargs)


# ---------------------------------------------------------------------------
# StepCallDeferred → list[ResolvedStep] resolution
# ---------------------------------------------------------------------------


def _version_value(obj: Any, resolved: dict[int, str]) -> Any:
    """Build a JSON-serializable version representation of obj.

    StepCallDeferred references become their output_path (already resolved).
    VersionedValue instances are unwrapped.  Dataclasses, lists, and dicts
    are recursed into.  Everything else is omitted (not part of the version).
    """
    from marin.execution.executor import VersionedValue

    if isinstance(obj, StepCallDeferred):
        return resolved[id(obj)]
    if isinstance(obj, VersionedValue):
        return _version_value(obj.value, resolved)
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _version_value(getattr(obj, f.name), resolved) for f in fields(obj)}
    if isinstance(obj, list | tuple):
        return [_version_value(x, resolved) for x in obj]
    if isinstance(obj, dict):
        return {k: _version_value(v, resolved) for k, v in obj.items()}
    # Primitives that are JSON-serializable contribute to the version.
    if isinstance(obj, str | int | float | bool | None):
        return obj
    return None


def _compute_output_path(deferred: StepCallDeferred, prefix: str, resolved: dict[int, str]) -> str:
    """Compute the output path for a deferred step: ``{prefix}/{fn_name}-{hash6}``."""
    fn_name = deferred._name
    version = {
        "name": fn_name,
        "args": _version_value(deferred._args, resolved),
        "kwargs": _version_value(deferred._kwargs, resolved),
    }
    version_str = json.dumps(version, sort_keys=True, default=str)
    hashed = hashlib.md5(version_str.encode()).hexdigest()[:6]
    return os.path.join(prefix, f"{fn_name}-{hashed}")


def _collect_deferred(obj: Any) -> list[StepCallDeferred]:
    """Find all StepCallDeferred instances reachable from obj (in args/kwargs, including nested structures)."""
    found: list[StepCallDeferred] = []
    if isinstance(obj, StepCallDeferred):
        found.append(obj)
    elif is_dataclass(obj) and not isinstance(obj, type):
        for f in fields(obj):
            found.extend(_collect_deferred(getattr(obj, f.name)))
    elif isinstance(obj, list | tuple | set):
        for x in obj:
            found.extend(_collect_deferred(x))
    elif isinstance(obj, dict):
        for v in obj.values():
            found.extend(_collect_deferred(v))
    return found


def _get_return_type(fn: Callable) -> type | None:
    """Extract the return type annotation from a callable, if it's a BaseModel subclass or dataclass."""
    hints = typing.get_type_hints(fn)
    ret = hints.get("return")
    if ret is not None and isinstance(ret, type) and (issubclass(ret, BaseModel) or is_dataclass(ret)):
        return ret
    return None


def _substitute_deferred(obj: Any, resolved: dict[int, str], return_types: dict[int, type | None]) -> Any:
    """Replace StepCallDeferred references with Artifact.load calls.

    Uses the upstream step's return type annotation for typed artifact loading.
    Must be called inside the step's fn closure so Artifact.load is deferred.
    """
    from marin.execution.artifact import Artifact

    if isinstance(obj, StepCallDeferred):
        artifact_type = return_types.get(id(obj))
        if artifact_type is not None:
            return Artifact.load(resolved[id(obj)], artifact_type)
        return Artifact.load(resolved[id(obj)])
    if is_dataclass(obj) and not isinstance(obj, type):
        replacements = {}
        for f in fields(obj):
            val = getattr(obj, f.name)
            new_val = _substitute_deferred(val, resolved, return_types)
            if new_val is not val:
                replacements[f.name] = new_val
        return dataclasses.replace(obj, **replacements) if replacements else obj
    if isinstance(obj, list):
        return [_substitute_deferred(x, resolved, return_types) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_substitute_deferred(x, resolved, return_types) for x in obj)
    if isinstance(obj, set):
        return {_substitute_deferred(x, resolved, return_types) for x in obj}
    if isinstance(obj, dict):
        return {k: _substitute_deferred(v, resolved, return_types) for k, v in obj.items()}
    return obj


def resolve_deferred(
    *deferred_steps: StepCallDeferred,
    prefix: str,
) -> list[StepSpec]:
    """Flatten a tree of ``StepCallDeferred`` into a topologically-sorted ``list[ResolvedStep]``.

    Each ``StepCallDeferred`` found in args/kwargs of another step is treated
    as a dependency.  At execution time, deferred references are replaced with
    ``Artifact.load(dep_output_path)`` so that the runner's auto-save on the
    upstream step feeds into the downstream step.
    """
    # id(deferred) → output_path, for already-resolved nodes
    resolved_paths: dict[int, str] = {}
    # id(deferred) → return type (BaseModel subclass or None)
    return_types: dict[int, type | None] = {}
    # Maintain insertion order for topological sort
    ordered: list[StepSpec] = []

    def resolve_one(deferred: StepCallDeferred) -> None:
        if id(deferred) in resolved_paths:
            return

        # Recurse into dependencies first (bottom-up)
        all_args = list(deferred._args) + list(deferred._kwargs.values())
        for dep in _collect_deferred(all_args):
            resolve_one(dep)

        # Compute output path (deps are resolved, so their paths are available)
        output_path = _compute_output_path(deferred, prefix, resolved_paths)
        resolved_paths[id(deferred)] = output_path
        return_types[id(deferred)] = _get_return_type(deferred._fn)

        # Collect direct dependency output_paths
        dep_paths = [resolved_paths[id(d)] for d in _collect_deferred(all_args)]

        # Snapshot mutable dicts so the closure sees the state at resolution time
        snapshot_paths = dict(resolved_paths)
        snapshot_types = dict(return_types)

        @wraps(deferred._fn)
        def fn(_: str) -> Any:
            args = _substitute_deferred(deferred._args, snapshot_paths, snapshot_types)
            kwargs = _substitute_deferred(deferred._kwargs, snapshot_paths, snapshot_types)
            return deferred._fn(*args, **kwargs)

        step = StepSpec(
            fn=fn,
            meta=StepMeta(
                name=deferred._name,
                output_path_prefix=None,
                override_output_path=output_path,
                deps=dep_paths,
            ),
        )
        ordered.append(step)

    for d in deferred_steps:
        resolve_one(d)

    return ordered
