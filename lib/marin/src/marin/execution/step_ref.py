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

"""
StepRef and StepContext: The core types for the tracer-based executor design.

This module provides:
- StepRef: A tracer object representing a path that will be resolved at execution time
- StepContext: Context for defining step dependencies
- @step: Decorator for defining executor steps

Usage:
    from marin.execution import step, StepContext, StepRef

    @step(name="tokenize/fineweb")
    def tokenize_fineweb(ctx: StepContext):
        download = ctx.require(download_step)
        return TokenizeConfig(
            train_paths=[download / "train"],
            cache_path=ctx.output,
        )

    tokenize_step = tokenize_fineweb()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

if TYPE_CHECKING:
    from marin.execution.executor import ExecutorStep
    from marin.resources import ResourceConfig

ConfigT = TypeVar("ConfigT")


@dataclass(frozen=True)
class StepRef:
    """
    A reference to a path that will be resolved at execution time.

    This is the ONLY type used for paths during step construction.
    At execution time, all StepRefs are resolved to concrete strings.

    Attributes:
        _step: The ExecutorStep this references, or None for output/hardcoded paths
        _subpath: Subpath within the step's output
        _blocking: Whether this is a blocking dependency (affects execution order)
    """

    _step: ExecutorStep | None = None
    _subpath: str | None = None
    _blocking: bool = True

    def __truediv__(self, subpath: str) -> StepRef:
        """Navigate to subpath: ref / "data" / "train" """
        if self._subpath:
            new_subpath = os.path.join(self._subpath, subpath)
        else:
            new_subpath = subpath
        return replace(self, _subpath=new_subpath)

    def nonblocking(self) -> StepRef:
        """
        Mark as non-blocking dependency.

        Non-blocking dependencies affect version hashing but don't block execution.
        Use for referencing checkpoints from still-running training steps.
        """
        return replace(self, _blocking=False)

    @staticmethod
    def hardcoded(path: str) -> StepRef:
        """
        Create a reference to a hardcoded path (not part of pipeline).

        Use sparingly - prefer step references when possible.
        """
        return StepRef(_step=None, _subpath=path, _blocking=True)

    def resolve(self, output_path: str, output_paths: dict[ExecutorStep, str]) -> str:
        """
        Resolve to concrete path. Called by executor at execution time.

        Args:
            output_path: This step's output path (for self-references)
            output_paths: Map from ExecutorStep to resolved output paths
        """
        if self._step is None:
            # This step's output or hardcoded path
            if self._subpath is None:
                return output_path
            elif self._subpath.startswith("/") or "://" in self._subpath:
                # Absolute or URL path - use as-is
                return self._subpath
            else:
                return os.path.join(output_path, self._subpath)
        else:
            base = output_paths[self._step]
            if self._subpath:
                return os.path.join(base, self._subpath)
            return base


@dataclass
class StepContext:
    """
    Context object for defining step dependencies.

    Passed to step-defining functions. Tracks dependencies as they're declared.
    """

    _dependencies: list[ExecutorStep] = field(default_factory=list)
    _pseudo_dependencies: list[ExecutorStep] = field(default_factory=list)
    _step: ExecutorStep | None = None  # Set after step creation

    def require(self, dep: ExecutorStep | StepRef) -> StepRef:
        """
        Declare a blocking dependency.

        Returns a StepRef that will resolve to the dependency's output path.
        """
        if isinstance(dep, StepRef):
            if dep._step is not None and dep._step not in self._dependencies:
                if dep._blocking:
                    self._dependencies.append(dep._step)
                else:
                    if dep._step not in self._pseudo_dependencies:
                        self._pseudo_dependencies.append(dep._step)
            return dep
        else:
            if dep not in self._dependencies:
                self._dependencies.append(dep)
            return StepRef(_step=dep)

    def require_nonblocking(self, dep: ExecutorStep | StepRef) -> StepRef:
        """
        Declare a non-blocking dependency (for versioning only).

        The dependency affects the version hash but doesn't block execution.
        Use for checkpoints from still-running training.
        """
        if isinstance(dep, StepRef):
            if dep._step is not None:
                # Remove from blocking if present
                if dep._step in self._dependencies:
                    self._dependencies.remove(dep._step)
                if dep._step not in self._pseudo_dependencies:
                    self._pseudo_dependencies.append(dep._step)
            return dep.nonblocking()
        else:
            if dep in self._dependencies:
                self._dependencies.remove(dep)
            if dep not in self._pseudo_dependencies:
                self._pseudo_dependencies.append(dep)
            return StepRef(_step=dep, _blocking=False)

    @property
    def output(self) -> StepRef:
        """Reference to this step's output path."""
        return StepRef(_step=None)

    def output_subpath(self, subpath: str) -> StepRef:
        """Reference to subpath within this step's output."""
        return StepRef(_step=None, _subpath=subpath)


def step(
    name: str,
    *,
    fn: Callable | None = None,
    description: str | None = None,
    resources: ResourceConfig | None = None,
    pip_dependency_groups: list[str] | None = None,
):
    """
    Decorator to define an executor step.

    Usage:
        @step(name="tokenize/fineweb", fn=tokenize_fn)
        def tokenize_fineweb(ctx: StepContext):
            download = ctx.require(download_step)
            return TokenizeConfig(
                train_paths=[download / "train"],
                cache_path=ctx.output,
            )

        # Create the step
        my_step = tokenize_fineweb()

    Args:
        name: Step name, used for output path generation
        fn: The function to execute (receives resolved config)
        description: Human-readable description
        resources: GPU/TPU/CPU requirements
        pip_dependency_groups: Extra pip dependencies
    """

    def decorator(config_fn: Callable[[StepContext], ConfigT]) -> Callable[..., ExecutorStep[ConfigT]]:
        @wraps(config_fn)
        def make_step(*args: Any, **kwargs: Any) -> ExecutorStep[ConfigT]:
            from marin.execution.executor import ExecutorStep

            ctx = StepContext()
            config = config_fn(ctx, *args, **kwargs)

            step_obj = ExecutorStep(
                name=name,
                fn=fn,
                config=config,
                description=description,
                resources=resources,
                pip_dependency_groups=pip_dependency_groups,
                _context=ctx,
            )
            ctx._step = step_obj
            return step_obj

        return make_step

    return decorator
