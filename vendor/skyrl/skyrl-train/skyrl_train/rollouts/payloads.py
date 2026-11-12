"""Rollout payload stores: where a written group waits between its worker's write and the trainer's read.

``MemoryPayloads`` keeps each trainable group in Ray's object store, owned by the process that wrote it: the
driver, or a rollout worker whose failure already fails training. A checkpoint copies the groups.
``ObjectStorePayloads`` writes each trainable group to its own object under a root directory, usually an S3
prefix, before its verdict reaches the buffer. The trainer reads each group by its URI, the root keeps every
trainable group the run generated, and a checkpoint records only the URIs.

Payload reads and writes run on their own thread pools. The default executor also runs rollout work, such as
GenRM grading, that can hold a thread for minutes, and a read queued behind it stalls training.
"""

from __future__ import annotations

import asyncio
import pickle
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import ray
from ray.actor import ActorHandle

from marinskyrl.resource_locator import join_resource_path
from skyrl_train.io import io
from skyrl_train.rollouts.buffer import (
    PayloadReference,
    RolloutContentPolicy,
    RolloutGroup,
    RolloutLease,
    RolloutWriter,
)

PAYLOAD_IO_THREADS = 16
ROLLOUT_OBJECT_SUFFIX = ".pkl"

_reads = ThreadPoolExecutor(max_workers=PAYLOAD_IO_THREADS, thread_name_prefix="rollout-payload-read")
_writes = ThreadPoolExecutor(max_workers=PAYLOAD_IO_THREADS, thread_name_prefix="rollout-payload-write")


class PayloadStore(Protocol):
    """How rollout payloads are written, read by the trainer, and carried through a checkpoint.

    A ``ReadyRollout.payload`` holds one reference, or None for a group whose verdict excludes it from training.
    """

    @property
    def object_store_root(self) -> str | None:
        """The directory that holds the payloads, or None when they live only in Ray's object store."""
        ...

    def writer(self, buffer: ActorHandle, content_policy: RolloutContentPolicy) -> RolloutWriter:
        """A picklable writer that rollout workers use to commit groups to ``buffer``."""
        ...

    async def fetch(self, payloads: Sequence) -> list[RolloutGroup]: ...

    async def checkpoint(self, payload: object | None) -> object | None:
        """The durable form of one payload for a checkpoint."""
        ...

    def restore(self, payload: object | None) -> object | None:
        """One reference for a new buffer from a checkpoint's durable payload."""
        ...


@dataclass(frozen=True)
class MemoryRolloutWriter:
    """Store payloads in Ray's object store, owned by the writing process, and commit their verdicts."""

    buffer: ActorHandle
    content_policy: RolloutContentPolicy

    async def write_rollout(self, lease: RolloutLease, group: RolloutGroup) -> None:
        verdict = self.content_policy.verdict(group)
        payload = None
        if verdict.trainable:
            # Not ``_owner=self.buffer``: Ray 2.51 can lose its local record of an object that a process put for
            # another owner when that process releases the object while receiving it back, as the synchronous
            # trainer's driver does, and reading it then never completes.
            payload = await asyncio.get_running_loop().run_in_executor(_writes, ray.put, group)
        await self.buffer.commit.remote(lease.lease_id, group.prompt, verdict, PayloadReference(payload))


class MemoryPayloads:
    """Payloads in Ray's object store; a checkpoint holds the groups themselves."""

    object_store_root = None

    def writer(self, buffer: ActorHandle, content_policy: RolloutContentPolicy) -> RolloutWriter:
        return MemoryRolloutWriter(buffer, content_policy)

    async def fetch(self, payloads: Sequence) -> list[RolloutGroup]:
        return list(await asyncio.gather(*payloads))

    async def checkpoint(self, payload: object | None) -> object | None:
        if payload is None:
            return None
        if not isinstance(payload, ray.ObjectRef):
            raise TypeError(f"memory payload must be a Ray ObjectRef, got {type(payload)}")
        return await payload

    def restore(self, payload: object | None) -> object | None:
        payload = _unwrap_legacy_checkpoint_payload(payload)
        if payload is None:
            return None
        if not isinstance(payload, RolloutGroup):
            raise TypeError(f"checkpointed memory payload must be a RolloutGroup, got {type(payload)}")
        return ray.put(payload)


@dataclass(frozen=True)
class ObjectStoreRolloutWriter:
    """Write each trainable group to its own object, then commit its verdict with the object's URI."""

    buffer: ActorHandle
    content_policy: RolloutContentPolicy
    object_store_root: str

    async def write_rollout(self, lease: RolloutLease, group: RolloutGroup) -> None:
        verdict = self.content_policy.verdict(group)
        payload = None
        if verdict.trainable:
            uri = join_resource_path(self.object_store_root, f"{lease.lease_id}{ROLLOUT_OBJECT_SUFFIX}")
            await asyncio.get_running_loop().run_in_executor(_writes, _write_group, uri, group)
            payload = uri
        await self.buffer.commit.remote(lease.lease_id, group.prompt, verdict, PayloadReference(payload))


@dataclass(frozen=True)
class ObjectStorePayloads:
    """Payloads as one object per group under ``object_store_root``; a checkpoint holds their URIs."""

    object_store_root: str

    def writer(self, buffer: ActorHandle, content_policy: RolloutContentPolicy) -> RolloutWriter:
        return ObjectStoreRolloutWriter(buffer, content_policy, self.object_store_root)

    async def fetch(self, payloads: Sequence) -> list[RolloutGroup]:
        loop = asyncio.get_running_loop()
        return list(await asyncio.gather(*(loop.run_in_executor(_reads, _read_group, uri) for uri in payloads)))

    async def checkpoint(self, payload: object | None) -> object | None:
        return payload

    def restore(self, payload: object | None) -> object | None:
        return _unwrap_legacy_checkpoint_payload(payload)


def _unwrap_legacy_checkpoint_payload(payload: object | None) -> object | None:
    """Older checkpoints stored each group's zero-or-one payloads in a list."""
    if isinstance(payload, list):
        if len(payload) > 1:
            raise ValueError("checkpointed rollout group has multiple payloads")
        return next(iter(payload), None)
    return payload


def _write_group(uri: str, group: RolloutGroup) -> None:
    io.write_bytes_atomic(uri, pickle.dumps(group, protocol=pickle.HIGHEST_PROTOCOL))


def _read_group(uri: str) -> RolloutGroup:
    return pickle.loads(io.read_bytes(uri))
