# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Builder steps, the program DAG they form, and the per-item cache of their outputs.

A step is an ``async def name(b, *inputs) -> Output`` function decorated with ``@step(inputs=...)``:
``b`` is the build context, and each parameter after it is the output of the step of that name, or,
for ``"proposal"`` (``PROPOSAL_INPUT``), the ``TaskProposal`` being built. A program is a set of steps
with exactly one sink, the step no other step takes as an input; its output is the program's.

A step's cache key is ``StepKey``: ``SDK_VERSION``, the sha256 of the step's own source text, the
output digest of every declared input, and the digest of the model and sampling policy steps call.
So what a step depends on is an input, in its body, or covered by ``SDK_VERSION``: a helper or
constant the body reads from elsewhere is not part of the key, and changing one needs a new input
or a version bump.

``StepCache`` keeps one record per key under ``<root>/items/<item>/steps/<name>/<key>/record.json``,
with the key itself beside it as ``key.json`` for a reader, and each output once under
``<root>/blobs/<sha256>``. The blob is written before the record, so a
record never names a missing blob. Outputs must round-trip through a pydantic ``TypeAdapter`` of the
step's return annotation: dataclasses, pydantic models (``TaskSpec``), tuples and primitives do.
"""

import bisect
import inspect
import typing
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from taskforge.atomic_file import write_atomic
from taskforge.content_hash import canonical_json, digest, sha256_hex
from taskforge.ledger.records import check_item_id

SDK_VERSION = "taskforge.builder/8"
"""Bump when a library change alters what an unchanged step would produce."""

PROPOSAL_INPUT = "proposal"
"""The built-in input every step may declare: the proposal being built. No step may take this name."""

RECORD_FILE = "record.json"
KEY_FILE = "key.json"


class CacheStatus(StrEnum):
    HIT = "hit"
    MISS = "miss"


@dataclass(frozen=True)
class Blob:
    """A content-addressed file in the blob store."""

    digest: str
    size: int


@dataclass(frozen=True)
class StepRecord:
    """One step of a build: its key, whether it was reused, its inputs' digests by name, and its output."""

    name: str
    key: str
    status: CacheStatus
    inputs: dict[str, str]
    output: Blob


@dataclass(frozen=True)
class Step:
    """A builder step: ``fn`` called with the build context and its ``inputs`` by name."""

    name: str
    fn: Callable[..., Awaitable[Any]]
    inputs: tuple[str, ...]
    output_type: Any
    source_digest: str


def step(*, inputs: Sequence[str] = ()) -> Callable[[Callable[..., Awaitable[Any]]], Step]:
    """Make ``async def name(b, *inputs) -> Output`` a step named ``name`` taking ``inputs``.

    Raises:
        TypeError: the function is not ``async``, has no return annotation, or its parameters after
            the context are not ``inputs`` in order.
    """

    def wrap(fn: Callable[..., Awaitable[Any]]) -> Step:
        name = fn.__name__
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"Step {name} must be an async function")
        parameters = tuple(inspect.signature(fn).parameters)
        if not parameters:
            raise TypeError(f"Step {name} must take the build context as its first parameter")
        if parameters[1:] != tuple(inputs):
            raise TypeError(
                f"Step {name} takes {list(parameters[1:])} after the context but declares inputs {list(inputs)}"
            )
        hints = typing.get_type_hints(fn)
        if "return" not in hints:
            raise TypeError(f"Step {name} needs a return annotation to store its output")
        source_digest = sha256_hex(inspect.getsource(fn).encode())
        return Step(name=name, fn=fn, inputs=tuple(inputs), output_type=hints["return"], source_digest=source_digest)

    return wrap


@dataclass(frozen=True)
class Program:
    """A step DAG: ``order`` runs every step after its inputs, and ``sink`` is the one step no other
    step takes as an input. ``digest`` covers each step's name, inputs and source."""

    steps: Mapping[str, Step]
    order: tuple[str, ...]
    sink: str
    digest: str


def program(*steps: Step) -> Program:
    """The program of ``steps``, ordered so that among steps whose inputs are ready, the one declared
    first runs first.

    Raises:
        ValueError: two steps share a name, a step is named ``"proposal"``, a step takes an input no
            step produces, the inputs form a cycle, or the program has other than one sink.
    """
    by_name: dict[str, Step] = {}
    for each in steps:
        if each.name == PROPOSAL_INPUT:
            raise ValueError(f"No step may be named {PROPOSAL_INPUT!r}: it is the built-in proposal input")
        if each.name in by_name:
            raise ValueError(f"Two steps are named {each.name!r}")
        by_name[each.name] = each
    consumers: dict[str, list[str]] = {name: [] for name in by_name}
    waiting: dict[str, set[str]] = {}
    for each in steps:
        unknown = [name for name in each.inputs if name not in by_name and name != PROPOSAL_INPUT]
        if unknown:
            raise ValueError(f"Step {each.name} takes {unknown}, which no step produces")
        waiting[each.name] = {name for name in each.inputs if name != PROPOSAL_INPUT}
        for name in waiting[each.name]:
            consumers[name].append(each.name)
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
        raise ValueError(f"A program needs exactly one sink (a step no other step takes), found {sinks}")
    shape = sorted((each.name, each.inputs, each.source_digest) for each in steps)
    return Program(steps=by_name, order=tuple(order), sink=sinks[0], digest=digest(shape))


@dataclass(frozen=True)
class StepKey:
    """What identifies a step's output; ``digest`` is the cache key."""

    sdk_version: str
    source_digest: str
    inputs: dict[str, str]
    policy_digest: str

    @property
    def digest(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class StepCache:
    """Step records for one item under ``root/items/<item_id>/steps``, and output blobs under ``root/blobs``."""

    root: Path
    item_id: str

    def __post_init__(self) -> None:
        check_item_id(self.item_id)

    def step_dir(self, name: str, key: StepKey) -> Path:
        return self.root / "items" / self.item_id / "steps" / name / key.digest

    def blob_path(self, blob: Blob) -> Path:
        return self.root / "blobs" / blob.digest

    def load(self, step: Step, key: StepKey) -> tuple[StepRecord, Any] | None:
        """The ``HIT`` record and decoded output stored under ``key``, or ``None`` when there is none.

        Raises:
            ValueError: the record's blob does not match its digest.
        """
        path = self.step_dir(step.name, key) / RECORD_FILE
        if not path.exists():
            return None
        record = _RECORD.validate_json(path.read_bytes())
        content = self.blob_path(record.output).read_bytes()
        if sha256_hex(content) != record.output.digest:
            raise ValueError(f"Blob {record.output.digest} of step {step.name} is corrupt on disk")
        output = TypeAdapter(step.output_type).validate_json(content)
        return replace(record, status=CacheStatus.HIT), output

    def store(self, step: Step, key: StepKey, inputs: Mapping[str, str], output: object) -> StepRecord:
        """Store ``output`` under ``key``: its blob first, then the key and the ``MISS`` record."""
        content = TypeAdapter(step.output_type).dump_json(output)
        blob = Blob(digest=sha256_hex(content), size=len(content))
        blob_path = self.blob_path(blob)
        if not blob_path.exists():
            blob_path.parent.mkdir(parents=True, exist_ok=True)
            write_atomic(blob_path, content)
        record = StepRecord(name=step.name, key=key.digest, status=CacheStatus.MISS, inputs=dict(inputs), output=blob)
        directory = self.step_dir(step.name, key)
        directory.mkdir(parents=True, exist_ok=True)
        write_atomic(directory / KEY_FILE, canonical_json(key).encode())
        write_atomic(directory / RECORD_FILE, _RECORD.dump_json(record, indent=2))
        return record


_RECORD: TypeAdapter[StepRecord] = TypeAdapter(StepRecord)
