# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Memoized step outputs, keyed by the step's source, its inputs' content, and the model.

The runner calls ``memoized`` for each step with the step's ``RLEnvBuildStepKey``. When the store
holds a memo under that key, ``memoized`` decodes the stored output and the step does not run.
Otherwise ``memoized`` awaits the step and stores its output. The key covers the step's own
source, the source of an output type defined in the step's module, the content of each input, and
the model, so anything else a step relies on has to change with ``SDK_VERSION``. An output type
imported from a library, such as ``TaskSpec``, falls under ``SDK_VERSION`` too. When a step body
reads a helper or a constant defined elsewhere, editing that helper leaves the key as it was.
``stage30_author.loader`` therefore refuses an authored program with helpers or constants beside
its steps and output types, and a library change of that kind bumps ``SDK_VERSION``.

``RLEnvBuildStepStore`` reads and writes through rigging's ``StoragePath``, so ``root`` may be a
local path or an object-store URL. Under it the store keeps:

* ``blobs/<sha256>``: each output as canonical JSON, written once and shared by all workpieces;
* ``workpieces/<workpiece_id>/steps/<name>/<key digest>.json``: the memo of each key a workpiece stored.

A memo holds a step's stored output under one key. The ledger in a later PR calls its own rows
entries.

The store writes the blob before the memo, so every memo points at a complete blob, and it
writes each file to a temporary sibling before renaming it into place. A memo or blob that reads back
unreadable or altered raises ``StoreCorruption``, an ``OSError``, so a build reports it as the
storage fault it is rather than as the program's. On an object store the
rename is a copy and a delete, so a memo costs three requests and appears whole or not at all;
CI exercises a ``file://`` root, and ``tests/stage40_build_with_program/test_step_store_live.py`` (marker
``manual``) runs the same round trip against a bucket. A step's output has to come
back intact from ``canonical_json`` followed by a pydantic ``TypeAdapter`` of its output type.
Dataclasses, pydantic output types, tuples, strings and numbers come back intact. Raw ``bytes`` come back
garbled, because canonical JSON encodes them as base64 and pydantic decodes a JSON string as UTF-8.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

from pydantic import TypeAdapter, ValidationError
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.storage_path import StoragePath

from taskforge.content_hash import canonical_json, digest, sha256_hex
from taskforge.ledger.records import FILE_SAFE, check_workpiece_id

SDK_VERSION = "taskforge.builder/14"
"""Bump when a library change alters what an unchanged step would produce."""


class StoreCorruption(OSError):
    """A memo or blob in the store is unreadable or does not match its digest.

    The store is a host fault, not the program's.
    """


class CacheStatus(StrEnum):
    HIT = "hit"
    MISS = "miss"
    INVALIDATED = "invalidated"
    """The caller named the step in ``invalidate``, so the build reran it and replaced any stored memo."""


@dataclass(frozen=True)
class RLEnvBuildStepKey:
    """The memo key of a step's output.

    Attributes:
        sdk_version: ``SDK_VERSION`` when the output was computed.
        source_digest: The step's ``RLEnvBuildStep.source_digest``: the sha256 of its source text,
            followed by the source of its output type when the step's module defines that class.
        inputs: For each declared input, the sha256 of that input's canonical JSON: the upstream
            step's output, or the proposal. An upstream step that reruns and returns identical
            content leaves this map unchanged.
        model_digest: ``run.build_model_digest`` of the model, its sampling policy and the services
            settings that change what a step returns, or a caller's override.
    """

    sdk_version: str
    source_digest: str
    inputs: dict[str, str]
    model_digest: str

    @property
    def digest(self) -> str:
        """The sha256 of the key's canonical JSON, a mapping of each field name to its value."""
        return digest(self)


@dataclass(frozen=True)
class RLEnvBuildStepMemo:
    """The stored memo of a step's output under a key, written by ``RLEnvBuildStepStore.store``.

    ``RLEnvBuildStepStore.load`` reads it back. The memo repeats the key's fields except
    ``model_digest``. That digest enters every key, and ``WorkpieceBuildProvenance.model_digest`` keeps
    a copy for the build. A build pairs each memo with the ``CacheStatus`` it observed in an
    ``RLEnvBuildStepOutcome``, defined in ``run.py``.

    Attributes:
        name: The step name.
        key: The digest of the ``RLEnvBuildStepKey`` the output was stored under.
        inputs: The key's input digests by name.
        output_digest: The sha256 of the output's canonical JSON, and the blob's file name.
        output_size: The blob's length in bytes.
        sdk_version: The key's ``sdk_version``.
        source_digest: The key's ``source_digest``.
    """

    name: str
    key: str
    inputs: dict[str, str]
    output_digest: str
    output_size: int
    sdk_version: str
    source_digest: str


class RLEnvBuildStepStore:
    """Step outputs for the workpiece ``workpiece_id``, under ``root``.

    Memos are per workpiece: ``load`` looks only under this workpiece, so two workpieces that compute the same
    key each run the step once, while the blob they write is shared. A hit therefore says that this
    workpiece stored that key in an earlier build.

    A driver can also memoize a helper that a step body calls. It builds an ``RLEnvBuildStepKey``
    from the helper's source digest and its arguments' digests, then calls ``memoized`` with a name
    of the form ``<step>.<helper>``, the same call ``run_program`` makes for each step. The dot
    keeps helper memos apart from program steps, which appear in ``WorkpieceBuildProvenance.steps``,
    when a reader lists ``workpieces/<workpiece_id>/steps/``. A step receives no store, so a program written by the
    language model cannot memoize its own helpers.
    """

    def __init__(self, root: str, workpiece_id: str):
        check_workpiece_id(workpiece_id)
        self.root = StoragePath(root)
        self.workpiece_id = workpiece_id

    def load[T](self, key: RLEnvBuildStepKey, name: str, output_type: type[T]) -> tuple[RLEnvBuildStepMemo, T] | None:
        """Read the memo stored for ``name`` under ``key`` and decode its blob as ``output_type``.

        Returns:
            The memo and the decoded output, or ``None`` when no memo exists.

        Raises:
            StoreCorruption: the memo does not validate, the blob's content does not match the digest
                in the memo, or the blob does not decode as ``output_type``.
        """
        path = self._memo_path(key, name)
        if not path.exists():
            return None
        try:
            memo = _MEMO.validate_json(path.read_bytes())
        except ValidationError as error:
            raise StoreCorruption(f"Memo {path} of step {name} does not validate: {error}") from error
        return memo, self.read_output(memo, output_type)

    def read_output[T](self, memo: RLEnvBuildStepMemo, output_type: type[T]) -> T:
        """Decode the blob ``memo`` points at as ``output_type``.

        Raises:
            StoreCorruption: the blob's content does not match the digest in the memo, or the blob
                does not decode as ``output_type``.
        """
        content = (self.root / "blobs" / memo.output_digest).read_bytes()
        if sha256_hex(content) != memo.output_digest:
            raise StoreCorruption(f"Blob {memo.output_digest} of step {memo.name} does not match its digest")
        try:
            return TypeAdapter(output_type).validate_json(content)
        except ValidationError as error:
            raise StoreCorruption(
                f"Blob {memo.output_digest} of step {memo.name} does not decode as {output_type}: {error}"
            ) from error

    def store(self, key: RLEnvBuildStepKey, name: str, output: object) -> RLEnvBuildStepMemo:
        """Write ``output``'s blob unless it exists, then the memo for ``name`` under ``key``."""
        content = canonical_json(output).encode()
        output_digest = sha256_hex(content)
        blob = self.root / "blobs" / output_digest
        if not blob.exists():
            _write(blob, content)
        memo = RLEnvBuildStepMemo(
            name=name,
            key=key.digest,
            inputs=dict(key.inputs),
            output_digest=output_digest,
            output_size=len(content),
            sdk_version=key.sdk_version,
            source_digest=key.source_digest,
        )
        _write(self._memo_path(key, name), _MEMO.dump_json(memo, indent=2))
        return memo

    def _memo_path(self, key: RLEnvBuildStepKey, name: str) -> StoragePath:
        if not FILE_SAFE.fullmatch(name):
            raise ValueError(f"Step name {name!r} must match {FILE_SAFE.pattern}")
        return self.root / "workpieces" / self.workpiece_id / "steps" / name / f"{key.digest}.json"


async def memoized[T](
    store: RLEnvBuildStepStore,
    name: str,
    key: RLEnvBuildStepKey,
    output_type: type[T],
    compute: Callable[[], Awaitable[T]],
) -> tuple[CacheStatus, RLEnvBuildStepMemo, T]:
    """Load ``name``'s output under ``key`` from ``store``, or compute and store it on a miss.

    ``memoized`` calls ``store.load`` first. When a memo exists, it returns ``HIT`` with the
    memo and the decoded output, leaving ``compute`` uncalled. Otherwise it awaits ``compute()``,
    stores the result, and returns ``MISS`` with the new memo and the result decoded from its
    stored form, so a consumer receives the same value on a miss as on a later hit.
    """
    loaded = store.load(key, name, output_type)
    if loaded is not None:
        memo, output = loaded
        return CacheStatus.HIT, memo, output
    memo, output = await _computed(store, name, key, output_type, compute)
    return CacheStatus.MISS, memo, output


async def recomputed[T](
    store: RLEnvBuildStepStore,
    name: str,
    key: RLEnvBuildStepKey,
    output_type: type[T],
    compute: Callable[[], Awaitable[T]],
) -> tuple[CacheStatus, RLEnvBuildStepMemo, T]:
    """Compute ``name``'s output, store it under ``key`` over any memo there, and return ``INVALIDATED``.

    The output comes back decoded from its stored form, as ``memoized`` returns it on a miss.
    """
    memo, output = await _computed(store, name, key, output_type, compute)
    return CacheStatus.INVALIDATED, memo, output


async def _computed[T](
    store: RLEnvBuildStepStore,
    name: str,
    key: RLEnvBuildStepKey,
    output_type: type[T],
    compute: Callable[[], Awaitable[T]],
) -> tuple[RLEnvBuildStepMemo, T]:
    computed = await compute()
    memo = store.store(key, name, computed)
    return memo, TypeAdapter(output_type).validate_json(canonical_json(computed))


def _write(path: StoragePath, content: bytes) -> None:
    path.parent.mkdirs()
    with atomic_rename(str(path)) as temp:
        StoragePath(temp).write_bytes(content)


_MEMO: TypeAdapter[RLEnvBuildStepMemo] = TypeAdapter(RLEnvBuildStepMemo)
