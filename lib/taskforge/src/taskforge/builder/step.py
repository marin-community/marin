# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Memoized builder steps and the content-addressed store behind them.

A step is an ``async`` function whose first parameter is the build context. Decorating it with
``@step(role)`` makes every call go through ``StepCache.run``, which keys the call on

* the sha256 of the step's code: its source text (``inspect.getsource``) plus the source of every
  function, class, or step of the same module it references and the canonical JSON of every data
  global it references (strings, dicts, lists, dataclass instances), transitively, so editing a
  prompt, an answer-key constant, or a helper invalidates the step,
* the digest of every bound argument after the context (its canonical JSON),
* ``SDK_VERSION``, the proposal digest, and the digest of the build's ``LLMPolicy``.

A call whose key already has a record under ``<root>/items/<item>/steps/<name>/<key>/result.json``
is a cache hit: the stored output is decoded with the step's return annotation and the resources
the step emitted are replayed. Otherwise the step runs and its output is written to the blob store
(``<root>/blobs/<sha256>``) before ``result.json``, so a record never points at a missing blob. A
name passed to ``invalidate`` is recomputed (and its record replaced) on every call in this run.

Outputs must round-trip through a pydantic ``TypeAdapter`` of the return annotation: dataclasses,
pydantic models (``TaskSpec``, ``LoweredTaskSpec``), tuples, and primitives do. Large files belong in
the blob store as ``Blob`` references rather than inline in an output.
"""

import contextvars
import dis
import inspect
import types
import typing
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from pydantic import TypeAdapter
from pydantic_core import PydanticSerializationError

from taskforge.atomic_file import write_atomic
from taskforge.content_hash import canonical_json, digest, sha256_hex
from taskforge.ledger.records import EntryKind, Ledger, check_item_id, span

SDK_VERSION = "taskforge.builder/6"
"""Bump when a library change alters what an unchanged step would produce."""


@dataclass
class StepFrame:
    """The step whose body is running in this task, and the resources it has emitted so far."""

    name: str
    resources: list["Resource"] = field(default_factory=list)


CURRENT_STEP: contextvars.ContextVar[StepFrame | None] = contextvars.ContextVar("taskforge_current_step", default=None)


class StepRole(StrEnum):
    """What a step contributes; ``run_build`` checks the grader and controls roles."""

    SOURCES = "sources"
    FIXTURES = "fixtures"
    ENVIRONMENT = "environment"
    GRADER = "grader"
    CONTROLS = "controls"
    INSTRUCTIONS = "instructions"
    ASSEMBLE = "assemble"
    OTHER = "other"


class CacheStatus(StrEnum):
    HIT = "hit"
    MISS = "miss"
    INVALIDATED = "invalidated"


@dataclass(frozen=True)
class Blob:
    """A content-addressed file in the blob store."""

    digest: str
    size: int


@dataclass(frozen=True)
class Resource:
    """A named file a step emitted; the name is the path the task or a reviewer knows it by."""

    name: str
    blob: Blob


@dataclass(frozen=True)
class StepRecord:
    """One step call in a build: what ran, under which key, and whether it was reused."""

    name: str
    role: StepRole
    key: str
    status: CacheStatus
    output: Blob
    resources: tuple[Resource, ...]


class BlobStore:
    """Files stored once by their sha256 under ``root``."""

    def __init__(self, root: Path):
        self.root = root

    def path(self, blob: Blob) -> Path:
        return self.root / blob.digest

    def put(self, content: bytes) -> Blob:
        blob = Blob(digest=sha256_hex(content), size=len(content))
        path = self.path(blob)
        if not path.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            write_atomic(path, content)
        return blob

    def get(self, blob: Blob) -> bytes:
        content = self.path(blob).read_bytes()
        if sha256_hex(content) != blob.digest:
            raise ValueError(f"Blob {blob.digest} is corrupt on disk")
        return content


class StepContext(Protocol):
    """What ``Step`` needs from the build context; ``taskforge.builder.sdk.Build`` implements it."""

    async def run_step(self, step: "Step", arguments: Mapping[str, object]) -> Any: ...


@dataclass(frozen=True)
class Step:
    """A memoized builder step. Call it like the function it wraps: ``await my_step(b, ...)``."""

    fn: Callable[..., Awaitable[Any]]
    role: StepRole
    output_type: Any

    @property
    def name(self) -> str:
        return self.fn.__name__

    @property
    def source(self) -> str:
        return inspect.getsource(self.fn)

    @property
    def code(self) -> dict[str, str]:
        """The step's source and the same-module code and constants it references, by name."""
        return step_code(self.fn)

    def bind(self, args: Sequence[object], kwargs: Mapping[str, object]) -> dict[str, object]:
        """The call's arguments after the context, by parameter name, with defaults applied."""
        bound = inspect.signature(self.fn).bind(None, *args, **kwargs)
        bound.apply_defaults()
        context_name = next(iter(bound.arguments))
        return {name: value for name, value in bound.arguments.items() if name != context_name}

    async def __call__(self, b: StepContext, *args: object, **kwargs: object) -> Any:
        return await b.run_step(self, self.bind(args, kwargs))


GLOBAL_LOADS = frozenset({"LOAD_GLOBAL", "LOAD_NAME"})


def code_names(code: types.CodeType) -> set[str]:
    """Global names ``code`` and its nested functions load (not attribute names)."""
    names = {i.argval for i in dis.get_instructions(code) if i.opname in GLOBAL_LOADS}
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names |= code_names(constant)
    return names


def _is_data(value: object) -> bool:
    """A global that contributes its value to a memo key: anything but a module or a callable."""
    return not isinstance(value, types.ModuleType) and not callable(value)


def _data_json(name: str, value: object) -> str:
    try:
        return canonical_json(value)
    except PydanticSerializationError as error:
        raise TypeError(
            f"step code reads {name!r}, a {type(value).__name__} with no JSON form for its memo key"
        ) from error


def step_code(fn: Callable[..., object]) -> dict[str, str]:
    """``fn``'s source and, transitively, what it references from its own module.

    Functions, classes, and steps defined in ``fn``'s module contribute their source. Every other
    referenced global that is not a module or a callable is data (strings, numbers, dicts, lists,
    sets, ``Fraction``s, dataclass instances) and contributes its canonical JSON, plus the source
    of its class when that class is defined in ``fn``'s module. Imported modules, functions, and
    classes do not contribute: library changes are covered by ``SDK_VERSION``.

    Raises:
        TypeError: a referenced data global has no JSON form, so it cannot be part of a memo key.
    """
    module = fn.__module__
    namespace = getattr(fn, "__globals__", {})

    def local(value: object) -> bool:
        return getattr(value, "__module__", None) == module and (inspect.isfunction(value) or inspect.isclass(value))

    code: dict[str, str] = {}
    pending: list[tuple[str, object]] = [(fn.__name__, fn)]
    while pending:
        name, value = pending.pop()
        if name in code:
            continue
        if isinstance(value, Step):
            value = value.fn
        if _is_data(value):
            code[name] = _data_json(name, value)
            if local(type(value)):
                pending.append((type(value).__name__, type(value)))
            continue
        code[name] = inspect.getsource(value)  # pyrefly: ignore[bad-argument-type]
        body = getattr(value, "__code__", None)
        referenced = code_names(body) if body is not None else set(_class_names(value))
        for other in sorted(referenced):
            if other not in namespace or other in code:
                continue
            candidate = namespace[other]
            target = candidate.fn if isinstance(candidate, Step) else candidate
            if local(target) or _is_data(candidate):
                pending.append((other, candidate))
    return code


def _class_names(cls: object) -> set[str]:
    """Global names a class's methods load, plus the classes named in its field annotations."""
    if not inspect.isclass(cls):
        return set()
    names: set[str] = set()
    for member in vars(cls).values():
        function = getattr(member, "__func__", member)
        if inspect.isfunction(function):
            names |= code_names(function.__code__)
    pending = list(inspect.get_annotations(cls).values())
    while pending:
        annotation = pending.pop()
        if inspect.isclass(annotation):
            names.add(annotation.__name__)
        pending.extend(typing.get_args(annotation))
    return names


def step(role: StepRole) -> Callable[[Callable[..., Awaitable[Any]]], Step]:
    """Mark an ``async def name(b, ...) -> Output`` function as a memoized step with ``role``."""

    def wrap(fn: Callable[..., Awaitable[Any]]) -> Step:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"Step {fn.__name__} must be an async function")
        if not inspect.signature(fn).parameters:
            raise TypeError(f"Step {fn.__name__} must take the build context as its first parameter")
        hints = typing.get_type_hints(fn)
        if "return" not in hints:
            raise TypeError(f"Step {fn.__name__} needs a return annotation to store its output")
        return Step(fn=fn, role=role, output_type=hints["return"])

    return wrap


@dataclass(frozen=True)
class StepKey:
    """The inputs that identify a step call; ``digest`` is the memo key."""

    sdk_version: str
    source_digest: str
    argument_digests: dict[str, str]
    proposal_digest: str
    policy_digest: str

    @property
    def digest(self) -> str:
        return digest(self)


@dataclass
class StepCache:
    """Step records for one item under ``root/items/<item_id>/steps`` and blobs under ``root/blobs``."""

    root: Path
    item_id: str
    invalidated: frozenset[str] = frozenset()
    records: list[StepRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        check_item_id(self.item_id)

    @property
    def blobs(self) -> BlobStore:
        return BlobStore(self.root / "blobs")

    def step_dir(self, name: str, key: str) -> Path:
        return self.root / "items" / self.item_id / "steps" / name / key

    def key(self, step: Step, arguments: Mapping[str, object], proposal_digest: str, policy_digest: str) -> StepKey:
        return StepKey(
            sdk_version=SDK_VERSION,
            source_digest=digest(step.code),
            argument_digests={name: digest(value) for name, value in arguments.items()},
            proposal_digest=proposal_digest,
            policy_digest=policy_digest,
        )

    async def run(
        self,
        step: Step,
        arguments: Mapping[str, object],
        context: object,
        *,
        proposal_digest: str,
        policy_digest: str,
        ledger: Ledger,
        round: int,  # noqa: A002 - matches LedgerEntry.round
        emitted: list[Resource],
    ) -> Any:
        """Return the step's output, from its record when the key exists, else by running it.

        The step's emitted resources (collected in its ``StepFrame`` on a miss, read from its record
        on a hit) are appended to ``emitted``, the build's resource list.
        """
        key = self.key(step, arguments, proposal_digest, policy_digest)
        directory = self.step_dir(step.name, key.digest)
        result_path = directory / "result.json"
        adapter: TypeAdapter[Any] = TypeAdapter(step.output_type)
        with span(
            ledger,
            EntryKind.STEP,
            item_id=self.item_id,
            round=round,
            step=step.name,
            code_hash=key.source_digest,
            input_hash=key.digest,
        ) as fields:
            fields.attrs["role"] = step.role
            if result_path.exists() and step.name not in self.invalidated:
                record = _RECORD.validate_json(result_path.read_bytes())
                output = adapter.validate_json(self.blobs.get(record.output))
                status = CacheStatus.HIT
            else:
                status = CacheStatus.INVALIDATED if result_path.exists() else CacheStatus.MISS
                frame = StepFrame(step.name)
                token = CURRENT_STEP.set(frame)
                try:
                    output = await step.fn(context, **arguments)
                finally:
                    CURRENT_STEP.reset(token)
                output_blob = self.blobs.put(adapter.dump_json(output))
                record = StepRecord(step.name, step.role, key.digest, status, output_blob, tuple(frame.resources))
                directory.mkdir(parents=True, exist_ok=True)
                write_atomic(directory / "key.json", canonical_json(key).encode())
                write_atomic(directory / "code.json", canonical_json(step.code).encode())
                write_atomic(result_path, _RECORD.dump_json(record, indent=2))
            fields.attrs["cache"] = status
            fields.output_hash = record.output.digest
        emitted.extend(record.resources)
        self.records.append(StepRecord(step.name, step.role, key.digest, status, record.output, record.resources))
        return output


_RECORD: TypeAdapter[StepRecord] = TypeAdapter(StepRecord)
