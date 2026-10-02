# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind DataKit steps to one shared Zephyr pool."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from marin.execution.remote import RemoteCallable
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from zephyr.context import ZephyrContext


def run_steps_in_pool(steps: Sequence[StepSpec], *, pool: ZephyrContext, max_concurrent: int) -> None:
    """Run all dependencies with the same pool, including remote step drivers."""
    bound: dict[int, StepSpec] = {}

    def bind_callable(fn: Callable[[str], Any]) -> Callable[[str], Any]:
        def run(output_path: str) -> Any:
            with pool.execution_scope():
                return fn(output_path)

        return run

    def bind(step: StepSpec) -> StepSpec:
        if id(step) in bound:
            return bound[id(step)]
        fn = step.fn
        if isinstance(fn, RemoteCallable):
            fn = replace(fn, fn=bind_callable(fn.fn))
        elif fn is not None:
            fn = bind_callable(fn)
        result = replace(step, deps=[bind(dep) for dep in step.deps], fn=fn)
        bound[id(step)] = result
        return result

    StepRunner().run([bind(step) for step in steps], max_concurrent=max_concurrent)
