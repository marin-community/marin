# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared placeholder/value types used by both ``executor`` and ``step_spec``.

These were extracted from ``marin.execution.executor`` to break the import
cycle between ``executor`` and ``step_spec``. ``executor`` re-exports the
symbols defined here for backwards compatibility with existing call sites.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from fray.types import ResourceConfig

ConfigT_co = TypeVar("ConfigT_co", covariant=True)
T_co = TypeVar("T_co", covariant=True)

ExecutorFunction = Callable | None


@dataclass(frozen=True)
class ExecutorStep(Generic[ConfigT_co]):
    """A single step of an executor pipeline.

    See ``marin.execution.executor`` module docstring for the full description
    of how steps are versioned and how output paths are derived.
    """

    name: str
    fn: ExecutorFunction
    config: ConfigT_co
    description: str | None = None

    override_output_path: str | None = None
    """Specifies the `output_path` that should be used.  Print warning if it
    doesn't match the automatically computed one."""

    resources: ResourceConfig | None = None
    """If set, this step is submitted as its own Fray job using these
    resources. ``fn`` is invoked inside the submitted job.

    If ``None``, behavior is determined by ``fn``: a ``RemoteCallable``
    submits as a Fray job; a plain callable runs inline in-process.
    """

    def cd(self, name: str) -> InputName:
        """Refer to the `name` under `self`'s output_path."""
        return InputName(self, name=name)

    def __truediv__(self, other: str) -> InputName:
        """Alias for `cd`. That looks more Pythonic."""
        return InputName(self, name=other)

    def __hash__(self):
        """Hash based on the ID (every object is different)."""
        return hash(id(self))

    def with_output_path(self, output_path: str) -> ExecutorStep:
        """Return a copy of the step with the given output_path."""
        return dataclasses.replace(self, override_output_path=output_path)

    def as_input_name(self) -> InputName:
        return InputName(step=self, name=None)


@dataclass(frozen=True)
class InputName:
    """To be interpreted as a previous `step`'s output_path joined with `name`."""

    step: ExecutorStep | None
    name: str | None
    block_on_step: bool = True
    """
    If False, the step that uses this InputName
    will not block (or attempt to execute) `step`. We use this for
    documenting dependencies in the config, but where that step might not have technically finished...

    For instance, we sometimes use training checkpoints before the training step has finished.

    These "pseudo-dependencies" still impact the hash of the step, but they don't block execution.
    """

    def cd(self, name: str) -> InputName:
        return InputName(self.step, name=os.path.join(self.name, name) if self.name else name)

    def __truediv__(self, other: str) -> InputName:
        """Alias for `cd` that looks more Pythonic."""
        return self.cd(other)

    @staticmethod
    def hardcoded(path: str) -> InputName:
        """
        Sometimes we want to specify a path that is not part of the pipeline but is still relative to the prefix.
        Try to use this sparingly.
        """
        return InputName(None, name=path)

    def nonblocking(self) -> InputName:
        """
        the step will not block on (or attempt to execute) the parent step.

         (Note that if another step depends on the parent step, it will still block on it.)
        """
        return dataclasses.replace(self, block_on_step=False)


def get_executor_step(run: ExecutorStep | InputName) -> ExecutorStep:
    """Extract the ``ExecutorStep`` from an ``InputName`` or ``ExecutorStep``."""
    if isinstance(run, ExecutorStep):
        return run
    elif isinstance(run, InputName):
        step = run.step
        if step is None:
            raise ValueError(f"Hardcoded path {run.name} is not part of the pipeline")
        return step
    else:
        raise ValueError(f"Unexpected type {type(run)} for run: {run}")


def output_path_of(step: ExecutorStep, name: str | None = None) -> InputName:
    return InputName(step=step, name=name)


if TYPE_CHECKING:

    class OutputName(str):
        """Type-checking stub treated as a string so defaults like THIS_OUTPUT_PATH fit `str`."""

        name: str | None

else:

    @dataclass(frozen=True)
    class OutputName:
        """To be interpreted as part of this step's output_path joined with `name`."""

        name: str | None


def this_output_path(name: str | None = None):
    return OutputName(name=name)


# constant so we can use it in fields of dataclasses
THIS_OUTPUT_PATH = OutputName(None)


@dataclass(frozen=True)
class VersionedValue(Generic[T_co]):
    """Wraps a value, to signal that this value (part of a config) should be part of the version."""

    value: T_co


def versioned(value: T_co) -> VersionedValue[T_co]:
    if isinstance(value, VersionedValue):
        raise ValueError("Can't nest VersionedValue")
    elif isinstance(value, InputName):
        # TODO: We have also run into Versioned([InputName(...), ...])
        raise ValueError("Can't version an InputName")

    return VersionedValue(value)


def ensure_versioned(value: VersionedValue[T_co] | T_co) -> VersionedValue[T_co]:
    """Ensure that the value is wrapped in a VersionedValue. If it is already wrapped, return it as is."""
    return value if isinstance(value, VersionedValue) else VersionedValue(value)
