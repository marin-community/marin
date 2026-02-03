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

from functools import update_wrapper
from dataclasses import dataclass
from typing import Any, Generic, ParamSpec, TypeVar
from collections.abc import Callable

T = TypeVar("T")
P = ParamSpec("P")


def safe_get_func_name(fn: Callable[..., T]) -> str:
    return getattr(fn, "__name__", "__unknown_step_fn__")


@dataclass(frozen=True)
class StepCallDeferred(Generic[T], Any):
    """Represents a deferred function call"""

    _fn: Callable[..., T]
    _args: tuple
    _kwargs: dict

    def __getattr__(self, name: str) -> Any:
        raise TypeError(f"{self} is deferred, can't access the attribute {name!r}")

    def __repr__(self):
        return f"{StepCallDeferred.__name__}({safe_get_func_name(self._fn)}, args={self._args}, kwargs={self._kwargs})"


@dataclass
class Step(Generic[P, T]):
    """
    Represents a step function that can be invoked lazily via the `defer` method.
    """

    _fn: Callable[P, T]

    def __post_init__(self):
        update_wrapper(self, self._fn)  # type: ignore[invalid-argument-type]

    def defer(self, *args: P.args, **kwargs: P.kwargs) -> StepCallDeferred[T]:
        return StepCallDeferred(_fn=self._fn, _args=args, _kwargs=kwargs)
