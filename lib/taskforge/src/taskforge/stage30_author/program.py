# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build steps and the DAG they form.

An author writes each step as ``async def name(context, ...) -> Output`` and decorates it with
``@rl_env_build_step``. The decorator reads the step's inputs off its signature, taking every
parameter after the context in order. When the program runs, the step receives the build context
first, then one keyword argument per input:

* for an input named after another step, the output of that step;
* for ``PROPOSAL_INPUT``, the proposal the caller passed to the runner.

``rl_env_build_program`` checks the inputs, orders the steps so each one follows its inputs, and
picks the sink, the single step no other step takes as an input. The sink's output becomes the
program's output.
"""

import bisect
import inspect
import typing
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from taskforge.content_hash import digest, sha256_hex

PROPOSAL_INPUT = "proposal"
"""The parameter name a step takes to receive the proposal. A step may not use it as its own name."""


@dataclass(frozen=True)
class RLEnvBuildStep:
    """A decorated step: ``fn`` with its inputs, its return annotation and the sha256 of its source.

    ``inputs`` lists the names of ``fn``'s parameters after the context, in order.
    """

    name: str
    fn: Callable[..., Awaitable[Any]]
    inputs: tuple[str, ...]
    output_type: Any
    source_digest: str


def rl_env_build_step(fn: Callable[..., Awaitable[Any]]) -> RLEnvBuildStep:
    """Turn ``async def name(context, *inputs) -> Output`` into the step ``name`` taking ``inputs``.

    The step's inputs are the function's parameters after the context, in declaration order. The
    decorator hashes the function's own source text, decorator line included, into
    ``source_digest``.

    Raises:
        TypeError: the function is not ``async``, takes no context parameter, or lacks a return
            annotation.
    """
    name = fn.__name__
    if not inspect.iscoroutinefunction(fn):
        raise TypeError(f"Step {name} must be an async function")
    parameters = tuple(inspect.signature(fn).parameters)
    if not parameters:
        raise TypeError(f"Step {name} must take the build context as its first parameter")
    hints = typing.get_type_hints(fn)
    if "return" not in hints:
        raise TypeError(f"Step {name} needs a return annotation to decode its stored output")
    return RLEnvBuildStep(
        name=name,
        fn=fn,
        inputs=parameters[1:],
        output_type=hints["return"],
        source_digest=sha256_hex(inspect.getsource(fn).encode()),
    )


@dataclass(frozen=True)
class RLEnvBuildProgram:
    """A checked step DAG.

    Attributes:
        steps: Every step by name, in declaration order.
        order: The run order, with each step after all of its inputs.
        sink: The step whose output the program returns.
        digest: The sha256 over each step's name, inputs and source digest.
    """

    steps: Mapping[str, RLEnvBuildStep]
    order: tuple[str, ...]
    sink: str
    digest: str


def rl_env_build_program(*steps: RLEnvBuildStep) -> RLEnvBuildProgram:
    """Check ``steps`` and order each one after its inputs.

    The order comes from Kahn's algorithm. When several steps have all their inputs ready, the one
    declared first runs first, so the order depends only on the declarations.

    Raises:
        ValueError: two steps share a name, a step uses the name ``PROPOSAL_INPUT``, a step declares
            an input with no step of that name, the inputs form a cycle, or the program has zero
            sinks or several.
    """
    by_name: dict[str, RLEnvBuildStep] = {}
    for step in steps:
        if step.name == PROPOSAL_INPUT:
            raise ValueError(f"A step may not be named {PROPOSAL_INPUT!r}: steps use it to receive the proposal")
        if step.name in by_name:
            raise ValueError(f"Two steps are named {step.name!r}")
        by_name[step.name] = step
    consumers: dict[str, list[str]] = {name: [] for name in by_name}
    waiting: dict[str, set[str]] = {}
    for step in steps:
        unknown = [name for name in step.inputs if name not in by_name and name != PROPOSAL_INPUT]
        if unknown:
            raise ValueError(f"Step {step.name} declares inputs {unknown} with no step of that name")
        waiting[step.name] = {name for name in step.inputs if name != PROPOSAL_INPUT}
        for name in waiting[step.name]:
            consumers[name].append(step.name)
    position = {name: index for index, name in enumerate(by_name)}
    ready = [name for name in by_name if not waiting[name]]
    order: list[str] = []
    while ready:
        name = ready.pop(0)
        order.append(name)
        for consumer in consumers[name]:
            waiting[consumer].discard(name)
            if not waiting[consumer]:
                bisect.insort(ready, consumer, key=position.__getitem__)
    if len(order) != len(by_name):
        raise ValueError(f"Steps {sorted(set(by_name) - set(order))} form a cycle")
    sinks = [name for name in by_name if not consumers[name]]
    if len(sinks) != 1:
        raise ValueError(f"A program needs exactly one sink, a step no other step takes as an input; found {sinks}")
    declarations = sorted((step.name, step.inputs, step.source_digest) for step in steps)
    return RLEnvBuildProgram(steps=by_name, order=tuple(order), sink=sinks[0], digest=digest(declarations))
