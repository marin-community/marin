# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build steps, the sink, and the DAG they form.

An author writes each step as ``async def name(context, ...) -> Output`` and decorates it with
``@rl_env_build_step``. The decorator reads the step's inputs off its signature, taking every
parameter after the context in order. When the program runs, the step receives the build context
first, then one keyword argument per input:

* for a parameter whose name is another step's name, that step's output;
* for the parameter named ``proposal``, the proposal the caller passed to the runner.

The author writes the final step the same way, annotates it as returning a ``TaskSpec``, and
decorates it with ``@rl_env_build_sink``. ``rl_env_build_program`` receives the steps and, as
``sink``, this final step. It checks every input, orders the steps so each one follows its inputs,
and runs the sink last. The sink's ``TaskSpec`` becomes the program's output.
"""

import bisect
import inspect
import re
import types
import typing
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from taskcompendium.models import TaskSpec

from taskforge.content_hash import canonical_json, digest, sha256_hex

PROPOSAL_INPUT = "proposal"
"""The parameter name a step takes to receive the proposal. A step may not use it as its own name."""


MEMORY_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")
"""What a ``repr`` that changes from one process to the next contains."""


@dataclass(frozen=True)
class RLEnvBuildStep:
    """A decorated step: ``fn`` with its inputs, its return annotation and its source digest.

    ``inputs`` lists the names of ``fn``'s parameters after the context, in order.
    ``rl_env_build_step`` computes ``source_digest``.
    """

    name: str
    fn: Callable[..., Awaitable[Any]]
    inputs: tuple[str, ...]
    output_type: Any
    source_digest: str


@dataclass(frozen=True)
class RLEnvBuildSink:
    """The decorated final step of a program, returning the ``TaskSpec``.

    ``rl_env_build_sink`` fills the fields as ``rl_env_build_step`` does for a step, with
    ``output_type`` always ``TaskSpec``.
    """

    name: str
    fn: Callable[..., Awaitable[Any]]
    inputs: tuple[str, ...]
    output_type: Any
    source_digest: str


def rl_env_build_step(fn: Callable[..., Awaitable[Any]]) -> RLEnvBuildStep:
    """Turn ``async def name(context, *inputs) -> Output`` into the step ``name`` taking ``inputs``.

    The step's inputs are the function's parameters after the context, in declaration order. The
    decorator computes ``source_digest`` as the sha256 of the function's own source text, decorator
    line included. A program written by the language model defines the output types its steps
    return at module level, above the steps. When the return annotation is a class defined in the
    step's own module, the decorator appends that class's source to the step's source before
    hashing, so an edit to the class reruns the step. For a builtin, a generic alias, or a class
    imported from another module such as ``TaskSpec``, the decorator hashes the step's source
    alone, and a change to such a library type comes with an ``SDK_VERSION`` bump.

    Raises:
        TypeError: the function is not ``async``, takes no context parameter, or lacks a return
            annotation.
    """
    return _decorated(fn, RLEnvBuildStep)


def rl_env_build_sink(fn: Callable[..., Awaitable[TaskSpec]]) -> RLEnvBuildSink:
    """Turn ``async def name(context, *inputs) -> TaskSpec`` into the sink ``name``.

    The decorator reads the inputs and computes the source digest as ``rl_env_build_step`` does,
    then checks the return annotation.

    Raises:
        TypeError: ``rl_env_build_step`` would refuse the function, or its return annotation is not
            ``taskcompendium.models.TaskSpec``.
    """
    sink = _decorated(fn, RLEnvBuildSink)
    if sink.output_type is not TaskSpec:
        raise TypeError(f"Sink {sink.name} must return TaskSpec, not {sink.output_type!r}")
    return sink


def _decorated[S: (RLEnvBuildStep, RLEnvBuildSink)](fn: Callable[..., Awaitable[Any]], kind: type[S]) -> S:
    name = fn.__name__
    if not inspect.iscoroutinefunction(fn):
        raise TypeError(f"Step {name} must be an async function")
    parameters = tuple(inspect.signature(fn).parameters)
    if not parameters:
        raise TypeError(f"Step {name} must take the build context as its first parameter")
    hints = typing.get_type_hints(fn)
    if "return" not in hints:
        raise TypeError(f"Step {name} needs a return annotation to decode its stored output")
    return kind(
        name=name,
        fn=fn,
        inputs=parameters[1:],
        output_type=hints["return"],
        source_digest=_source_digest(fn, hints["return"]),
    )


def _source_digest(fn: Callable[..., Awaitable[Any]], output_type: Any) -> str:
    """The sha256 of the step's source together with everything it reaches in its own module.

    From the step's code and from its return annotation, the walk follows every name the module
    defines: a function or class adds its source and is walked in turn (a class through its methods,
    its field annotations and its bases), another step adds its own source digest, and any other value
    adds its canonical JSON, or its ``repr`` when it has none. A value whose ``repr`` names a memory
    address has no text that survives the process, so it is refused with ``TypeError``. Names bound by
    imports, modules and builtins stay out; ``SDK_VERSION`` covers
    the library. The step's source comes first, then each reached object under its name in sorted
    order, so a step that reaches nothing digests its source alone.
    """
    module, namespace = fn.__module__, fn.__globals__
    reached: dict[str, str] = {}
    pending = [*_annotated_classes(output_type, module), *_named(fn.__code__, namespace, module)]
    while pending:
        name, value = pending.pop()
        if name in reached:
            continue
        if inspect.isfunction(value):
            reached[name] = inspect.getsource(value)
            pending += _named(value.__code__, namespace, module)
        elif inspect.isclass(value):
            reached[name] = inspect.getsource(value)
            pending += _reached_by_class(value, namespace, module)
        elif isinstance(value, RLEnvBuildStep | RLEnvBuildSink):
            if value.fn is not fn:
                reached[name] = f"step {value.source_digest}"
        else:
            reached[name] = _value_text(name, value)
    parts = [inspect.getsource(fn), *(f"\n# {name}\n{text}" for name, text in sorted(reached.items()))]
    return sha256_hex("".join(parts).encode())


def _code_names(code: types.CodeType) -> set[str]:
    """The global names ``code`` and the code objects nested in it read."""
    names = set(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names |= _code_names(constant)
    return names


def _named(code: types.CodeType, namespace: Mapping[str, Any], module: str) -> list[tuple[str, Any]]:
    """The module-level objects ``code`` reads that ``module`` defines, by name."""
    found = []
    for name in sorted(_code_names(code)):
        if name.startswith("__") or name not in namespace:
            continue
        value = namespace[name]
        if isinstance(value, types.ModuleType):
            continue
        if (inspect.isfunction(value) or inspect.isclass(value)) and value.__module__ != module:
            continue
        found.append((name, value))
    return found


def _reached_by_class(cls: type, namespace: Mapping[str, Any], module: str) -> list[tuple[str, Any]]:
    """What a class defined in ``module`` reaches: its methods' names, its field annotations and its bases."""
    found = []
    for attribute in vars(cls).values():
        if inspect.isfunction(attribute):
            found += _named(attribute.__code__, namespace, module)
    try:
        hints = typing.get_type_hints(cls)
    except (NameError, TypeError):
        hints = {}
    for hint in hints.values():
        found += _annotated_classes(hint, module)
    for base in cls.__bases__:
        found += _annotated_classes(base, module)
    return found


def _annotated_classes(annotation: Any, module: str) -> list[tuple[str, Any]]:
    """The classes ``module`` defines that ``annotation`` names, at any depth of its type arguments."""
    if inspect.isclass(annotation):
        return [(annotation.__name__, annotation)] if annotation.__module__ == module else []
    found = []
    for argument in typing.get_args(annotation):
        found += _annotated_classes(argument, module)
    return found


def _value_text(name: str, value: Any) -> str:
    try:
        return canonical_json(value)
    except (TypeError, ValueError):
        text = repr(value)
    if MEMORY_ADDRESS.search(text):
        raise TypeError(
            f"{name} has no text a step's digest can keep across processes (its repr names a memory address): "
            "make it a function, a class, or a JSON value"
        )
    return text


@dataclass(frozen=True)
class RLEnvBuildProgram:
    """A checked step DAG.

    Attributes:
        steps: Every step except the sink, by name, in declaration order.
        sink: The final step. The program returns its ``TaskSpec``.
        order: The run order by name, with each step after all of its inputs and the sink last.
        digest: The sha256 over the name, inputs and source digest of each step and of the sink.
    """

    steps: Mapping[str, RLEnvBuildStep]
    sink: RLEnvBuildSink
    order: tuple[str, ...]
    digest: str


def rl_env_build_program(*steps: RLEnvBuildStep, sink: RLEnvBuildSink) -> RLEnvBuildProgram:
    """Check ``steps`` and ``sink``, and order each step after its inputs.

    The sink's inputs resolve like a step's. Every step has to feed another step or the sink. A
    step feeding neither would spend model calls on an output the build then discards, and in a
    program written by the language model it usually means a step was left out of the wiring, so
    the call refuses it.

    The order comes from Kahn's algorithm. When several steps have all their inputs ready, the one
    declared first runs first, so the order depends only on the declarations. Because every step
    leads to the sink, the sink comes last.

    Raises:
        ValueError: two steps share a name, a step uses the name ``PROPOSAL_INPUT``, a step declares
            an input with no step of that name, a step takes the sink as an input, a step feeds
            neither another step nor the sink, or the inputs form a cycle.
    """
    nodes = (*steps, sink)
    by_name: dict[str, RLEnvBuildStep | RLEnvBuildSink] = {}
    for node in nodes:
        if node.name == PROPOSAL_INPUT:
            raise ValueError(f"A step may not be named {PROPOSAL_INPUT!r}: steps use it to receive the proposal")
        if node.name in by_name:
            raise ValueError(f"Two steps are named {node.name!r}")
        by_name[node.name] = node
    consumers: dict[str, list[str]] = {name: [] for name in by_name}
    waiting: dict[str, set[str]] = {}
    for node in nodes:
        if sink.name in node.inputs:
            raise ValueError(f"Step {node.name} takes the sink {sink.name} as an input")
        unknown = [name for name in node.inputs if name not in by_name and name != PROPOSAL_INPUT]
        if unknown:
            raise ValueError(f"Step {node.name} declares inputs {unknown} with no step of that name")
        waiting[node.name] = {name for name in node.inputs if name != PROPOSAL_INPUT}
        for name in waiting[node.name]:
            consumers[name].append(node.name)
    dead = [step.name for step in steps if not consumers[step.name]]
    if dead:
        raise ValueError(f"Steps {dead} feed neither another step nor the sink")
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
    declarations = sorted((node.name, node.inputs, node.source_digest) for node in nodes)
    return RLEnvBuildProgram(
        steps={step.name: step for step in steps}, sink=sink, order=tuple(order), digest=digest(declarations)
    )
