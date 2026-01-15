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
JAX-style tracing for executor steps.

This module provides a tracing-based system where steps can call other steps
naturally, and the dependency graph is automatically constructed.

Usage:
    from marin.execution import step, output

    @step(name="download/fineweb", fn=download_fn)
    def download_fineweb():
        return DownloadConfig(output_path=output())

    @step(name="tokenize/fineweb", fn=tokenize_fn)
    def tokenize_fineweb():
        download = download_fineweb()  # Calls step, returns StepRef
        return TokenizeConfig(
            input_path=download / "train",
            output_path=output(),
        )

    # Entry point for executor - just a function that calls steps
    def my_pipeline():
        tokenize = tokenize_fineweb()
        train = train_model(data=tokenize)
        return train

    # Executor traces from entry point
    executor.run(my_pipeline)
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    from marin.execution.executor import ExecutorStep
    from marin.resources import ResourceConfig

ConfigT = TypeVar("ConfigT")

# Context variable tracking the current step being constructed
_tracing_context: ContextVar[StepContext | None] = ContextVar("tracing_context", default=None)


@dataclass(frozen=True)
class StepRef:
    """
    A reference to a step's output path, resolved at execution time.

    StepRef is returned when calling a @step decorated function during tracing.
    It can be used in configs and supports path navigation with `/`.

    Attributes:
        _step: The ExecutorStep this references, or None for current step's output
        _subpath: Subpath within the step's output
        _blocking: Whether this is a blocking dependency
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
    Internal context for tracking dependencies during step tracing.

    Users don't interact with this directly - use output() and call steps.
    """

    _dependencies: list[ExecutorStep] = field(default_factory=list)
    _pseudo_dependencies: list[ExecutorStep] = field(default_factory=list)
    _step: ExecutorStep | None = None

    def _add_dependency(self, step: ExecutorStep, blocking: bool = True) -> None:
        """Add a dependency to this step."""
        if blocking:
            if step not in self._dependencies:
                self._dependencies.append(step)
        else:
            if step not in self._pseudo_dependencies:
                self._pseudo_dependencies.append(step)

    @property
    def output(self) -> StepRef:
        """Reference to this step's output path."""
        return StepRef(_step=None)

    # Legacy API - still supported for backwards compatibility
    def require(self, dep: ExecutorStep | StepRef) -> StepRef:
        """Declare a blocking dependency (legacy API)."""
        if isinstance(dep, StepRef):
            if dep._step is not None:
                self._add_dependency(dep._step, dep._blocking)
            return dep
        else:
            self._add_dependency(dep, blocking=True)
            return StepRef(_step=dep)

    def require_nonblocking(self, dep: ExecutorStep | StepRef) -> StepRef:
        """Declare a non-blocking dependency (legacy API)."""
        if isinstance(dep, StepRef):
            if dep._step is not None:
                # Move from blocking to non-blocking if present
                if dep._step in self._dependencies:
                    self._dependencies.remove(dep._step)
                self._add_dependency(dep._step, blocking=False)
            return dep.nonblocking()
        else:
            if dep in self._dependencies:
                self._dependencies.remove(dep)
            self._add_dependency(dep, blocking=False)
            return StepRef(_step=dep, _blocking=False)


def output() -> StepRef:
    """
    Get a reference to the current step's output path.

    Must be called within a @step decorated function.

    Usage:
        @step(name="my_step", fn=my_fn)
        def my_step():
            return MyConfig(output_path=output())
    """
    ctx = _tracing_context.get()
    if ctx is None:
        raise RuntimeError("output() called outside of step context")
    return ctx.output


def output_subpath(subpath: str) -> StepRef:
    """
    Get a reference to a subpath within the current step's output.

    Must be called within a @step decorated function.

    Usage:
        @step(name="my_step", fn=my_fn)
        def my_step():
            return MyConfig(
                train_path=output_subpath("train"),
                val_path=output_subpath("val"),
            )
    """
    ctx = _tracing_context.get()
    if ctx is None:
        raise RuntimeError("output_subpath() called outside of step context")
    return StepRef(_step=None, _subpath=subpath)


def step(
    name: str,
    *,
    fn: Callable | None = None,
    description: str | None = None,
    resources: ResourceConfig | None = None,
    pip_dependency_groups: list[str] | None = None,
    override_output_path: str | None = None,
):
    """
    Decorator to define an executor step with JAX-style tracing.

    When called, a @step function:
    1. Creates a new tracing context
    2. Executes the function body (which may call other steps)
    3. Creates an ExecutorStep with the returned config
    4. Registers as a dependency of any enclosing step
    5. Returns a StepRef for use in configs

    Usage:
        @step(name="download/fineweb", fn=download_fn)
        def download_fineweb():
            return DownloadConfig(output_path=output())

        @step(name="tokenize/{dataset}", fn=tokenize_fn)
        def tokenize_dataset(dataset: str):
            download = download_fineweb()  # Automatically tracked as dependency
            return TokenizeConfig(
                input_path=download,
                output_path=output(),
            )

    Args:
        name: Step name, used for output path generation. Can contain {arg_name}
              placeholders that will be substituted with kwargs when called.
        fn: The function to execute at runtime (receives resolved config)
        description: Human-readable description. Can use {arg_name} placeholders.
        resources: GPU/TPU/CPU requirements
        pip_dependency_groups: Extra pip dependencies
        override_output_path: Explicit output path (overrides automatic generation).
    """

    def decorator(step_fn: Callable[..., ConfigT]) -> Callable[..., StepRef]:
        @wraps(step_fn)
        def traced_step(*args: Any, **kwargs: Any) -> StepRef:
            from marin.execution.executor import ExecutorStep

            # Format name with kwargs if it contains placeholders
            actual_name = name.format(**kwargs) if "{" in name else name

            # Format description if provided and contains placeholders
            actual_description = description
            if description and "{" in description:
                actual_description = description.format(**kwargs)

            # Format override_output_path if provided and contains placeholders
            actual_override_output_path = override_output_path
            if override_output_path and "{" in override_output_path:
                actual_override_output_path = override_output_path.format(**kwargs)

            # Create context for this step
            ctx = StepContext()

            # Set as current tracing context (save outer context)
            token = _tracing_context.set(ctx)
            try:
                # Execute the step function - may call other steps
                config = step_fn(*args, **kwargs)
            finally:
                # Restore outer context
                _tracing_context.reset(token)

            # Extract any StepRefs from config and register as dependencies
            # This handles StepRefs passed as arguments (not created by calling steps)
            _extract_step_refs(config, ctx)

            # Create the ExecutorStep
            step_obj = ExecutorStep(
                name=actual_name,
                fn=fn,
                config=config,
                description=actual_description,
                resources=resources,
                pip_dependency_groups=pip_dependency_groups,
                override_output_path=actual_override_output_path,
                _context=ctx,
            )
            ctx._step = step_obj

            # If there's an outer step context, register as dependency
            outer_ctx = _tracing_context.get()
            if outer_ctx is not None:
                outer_ctx._add_dependency(step_obj, blocking=True)

            # Return StepRef so caller can use it in configs
            return StepRef(_step=step_obj)

        return traced_step

    return decorator


def _extract_step_refs(obj: Any, ctx: StepContext, visited: set | None = None) -> None:
    """
    Walk an object (config) and extract all StepRefs, registering them as dependencies.

    This handles StepRefs that were passed as arguments rather than created by
    calling steps inside the function body.
    """
    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)

    if isinstance(obj, StepRef):
        if obj._step is not None:
            ctx._add_dependency(obj._step, obj._blocking)
    elif hasattr(obj, "__dataclass_fields__"):
        # Dataclass - walk fields
        for field_name in obj.__dataclass_fields__:
            _extract_step_refs(getattr(obj, field_name), ctx, visited)
    elif isinstance(obj, dict):
        for v in obj.values():
            _extract_step_refs(v, ctx, visited)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _extract_step_refs(item, ctx, visited)


def trace(entry_point: Callable[..., StepRef], *args: Any, **kwargs: Any) -> list[ExecutorStep]:
    """
    Trace an entry point function to discover all steps.

    Calls the entry point, which triggers step tracing, then collects
    all ExecutorSteps in dependency order.

    Args:
        entry_point: A function that calls @step decorated functions
        *args, **kwargs: Arguments to pass to entry_point

    Returns:
        List of ExecutorSteps in dependency order (dependencies first)

    Usage:
        def my_pipeline():
            tokenize = tokenize_dataset()
            train = train_model(data=tokenize)
            return train

        steps = trace(my_pipeline)
        executor.run_steps(steps)
    """
    # Call entry point to trigger tracing
    root_ref = entry_point(*args, **kwargs)

    # Handle multiple return values (tuple/list of StepRefs)
    if isinstance(root_ref, (list, tuple)):
        root_refs = list(root_ref)
    else:
        root_refs = [root_ref]

    # Collect all steps via DFS (use id() since ExecutorStep may not be hashable)
    visited: set[int] = set()
    ordered: list[ExecutorStep] = []

    def collect(ref: StepRef) -> None:
        if ref._step is None or id(ref._step) in visited:
            return
        visited.add(id(ref._step))

        # Visit dependencies first
        ctx = ref._step._context
        if ctx:
            for dep in ctx._dependencies:
                collect(StepRef(_step=dep))
            for dep in ctx._pseudo_dependencies:
                collect(StepRef(_step=dep))

        ordered.append(ref._step)

    for ref in root_refs:
        collect(ref)

    return ordered
