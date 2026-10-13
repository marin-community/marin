# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Memoized step outputs, keyed by the step's source, its inputs' content, and the model.

The runner keys each step on an ``RLEnvBuildStepKey``. When it finds a record under that key, it
decodes the stored output and skips the step. Otherwise it runs the step and stores the output.
The key hashes the step's declared inputs and its own source text, so anything else a step relies
on has to change with ``SDK_VERSION``. When a step body reads a helper or a constant defined
elsewhere, editing that helper leaves the key as it was. Pass the value as a declared input, or bump
``SDK_VERSION`` for a library change.

``RLEnvBuildStepStore`` reads and writes through rigging's ``StoragePath``, so ``root`` may be a
local path or an object-store URL. Under it the store keeps:

* ``blobs/<sha256>``: each output as canonical JSON, written once and shared by all items;
* ``items/<item_id>/steps/<name>/<key digest>.json``: the record of each key an item stored.

The store writes the blob before the record, so every record points at a complete blob, and it
writes each file to a temporary sibling before renaming it into place. On an object store the
rename is a copy and a delete, so a record costs three requests and appears whole or not at all;
CI exercises a ``file://`` root, and ``tests/stage40_build/test_step_store_live.py`` (marker
``manual``) runs the same round trip against a bucket. A step's output has to come
back intact from ``canonical_json`` followed by a pydantic ``TypeAdapter`` of its output type.
Dataclasses, pydantic models, tuples, strings and numbers come back intact. Raw ``bytes`` come back
garbled, because canonical JSON encodes them as base64 and pydantic decodes a JSON string as UTF-8.
"""

import re
from dataclasses import dataclass, replace
from enum import StrEnum

from pydantic import TypeAdapter
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.storage_path import StoragePath

from taskforge.content_hash import canonical_json, digest, sha256_hex
from taskforge.llm.client import ModelClient
from taskforge.llm.policy import LLMPolicy

SDK_VERSION = "taskforge.builder/8"
"""Bump when a library change alters what an unchanged step would produce."""

FILE_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def check_item_id(item_id: str) -> None:
    """Raise ``ValueError`` unless ``item_id`` is safe to use as a directory name."""
    if not FILE_SAFE.fullmatch(item_id):
        raise ValueError(f"Item id {item_id!r} must match {FILE_SAFE.pattern}")


def model_digest(model: ModelClient, sampling: LLMPolicy) -> str:
    """Return ``content_hash.digest({"model": model.model, "sampling": sampling})``.

    The runner puts this digest in every step's key, so changing the model or any sampling field
    reruns every step. A client whose replies also depend on another setting, such as a replay
    switch or an agent-loop limit, needs a digest that folds that setting in; its caller computes
    one and passes it to ``run_build`` in place of this one.
    """
    return digest({"model": model.model, "sampling": sampling})


class CacheStatus(StrEnum):
    HIT = "hit"
    MISS = "miss"


@dataclass(frozen=True)
class RLEnvBuildStepKey:
    """The memo key of a step's output.

    Attributes:
        sdk_version: ``SDK_VERSION`` when the output was computed.
        source_digest: The sha256 of the step's source text.
        inputs: For each declared input, the sha256 of that input's canonical JSON: the upstream
            step's output, or the proposal. An upstream step that reruns and returns identical
            content leaves this map unchanged.
        model_digest: ``model_digest(model, sampling)``, or a caller's extension of it.
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
class RLEnvBuildStepRecord:
    """A step's entry in one build.

    The record repeats the key's fields except ``model_digest``. That digest, defined by
    ``model_digest``, enters every key, and ``Provenance.model_digest`` keeps a copy for the build.

    Attributes:
        name: The step name.
        key: The digest of the ``RLEnvBuildStepKey`` the output was stored under.
        status: ``HIT`` when this build decoded a stored output, ``MISS`` when it ran the step.
        inputs: The key's input digests by name.
        output_digest: The sha256 of the output's canonical JSON, and the blob's file name.
        output_size: The blob's length in bytes.
        sdk_version: The key's ``sdk_version``.
        source_digest: The key's ``source_digest``.
    """

    name: str
    key: str
    status: CacheStatus
    inputs: dict[str, str]
    output_digest: str
    output_size: int
    sdk_version: str
    source_digest: str


class RLEnvBuildStepStore:
    """Step outputs for the item ``item_id``, under ``root``.

    Records are per item: ``load`` looks only under this item, so two items that compute the same
    key each run the step once, while the blob they write is shared. A hit therefore says that this
    item stored that key in an earlier build.

    A driver can also memoize a helper that a step body calls: it builds an ``RLEnvBuildStepKey``
    from the helper's source digest and its arguments' digests and calls ``load`` and ``store`` with
    a name of the form ``<step>.<helper>``. The dot keeps helper records apart from program steps,
    which appear in ``Provenance.steps``, when a reader lists ``items/<item_id>/steps/``. A step
    receives no store, so a program written by a model cannot memoize its own helpers.
    """

    def __init__(self, root: str, item_id: str):
        check_item_id(item_id)
        self.root = StoragePath(root)
        self.item_id = item_id

    def load[T](self, key: RLEnvBuildStepKey, name: str, output_type: type[T]) -> tuple[RLEnvBuildStepRecord, T] | None:
        """Read the record stored for ``name`` under ``key`` and decode its blob as ``output_type``.

        Returns:
            The record with status ``HIT`` and the decoded output, or ``None`` when no record exists.

        Raises:
            ValueError: the blob's content does not match the digest in the record.
        """
        path = self._record_path(key, name)
        if not path.exists():
            return None
        record = _RECORD.validate_json(path.read_bytes())
        content = (self.root / "blobs" / record.output_digest).read_bytes()
        if sha256_hex(content) != record.output_digest:
            raise ValueError(f"Blob {record.output_digest} of step {name} does not match its digest")
        return replace(record, status=CacheStatus.HIT), TypeAdapter(output_type).validate_json(content)

    def store(self, key: RLEnvBuildStepKey, name: str, output: object) -> RLEnvBuildStepRecord:
        """Write ``output``'s blob unless it exists, then the ``MISS`` record for ``name`` under ``key``."""
        content = canonical_json(output).encode()
        output_digest = sha256_hex(content)
        blob = self.root / "blobs" / output_digest
        if not blob.exists():
            _write(blob, content)
        record = RLEnvBuildStepRecord(
            name=name,
            key=key.digest,
            status=CacheStatus.MISS,
            inputs=dict(key.inputs),
            output_digest=output_digest,
            output_size=len(content),
            sdk_version=key.sdk_version,
            source_digest=key.source_digest,
        )
        _write(self._record_path(key, name), _RECORD.dump_json(record, indent=2))
        return record

    def _record_path(self, key: RLEnvBuildStepKey, name: str) -> StoragePath:
        if not FILE_SAFE.fullmatch(name):
            raise ValueError(f"Step name {name!r} must match {FILE_SAFE.pattern}")
        return self.root / "items" / self.item_id / "steps" / name / f"{key.digest}.json"


def _write(path: StoragePath, content: bytes) -> None:
    path.parent.mkdirs()
    with atomic_rename(str(path)) as temp:
        StoragePath(temp).write_bytes(content)


_RECORD: TypeAdapter[RLEnvBuildStepRecord] = TypeAdapter(RLEnvBuildStepRecord)
