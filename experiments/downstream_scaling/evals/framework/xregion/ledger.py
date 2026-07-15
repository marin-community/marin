# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal chunk progress ledger for cross-region eval workers."""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import fsspec
from rigging.filesystem.distributed_lock import HEARTBEAT_INTERVAL, LeaseLostError, create_lock


class ChunkStatus(StrEnum):
    CLAIMED = "claimed"
    DONE = "done"


@dataclass(frozen=True)
class ChunkState:
    status: ChunkStatus
    owner: str | None = None


@dataclass(frozen=True)
class LedgerSummary:
    total: int
    claimed: int
    done: int


@dataclass
class ChunkClaim:
    ledger_path: str
    owner: str
    chunk: dict[str, Any]
    state_path: str
    lock: Any
    lease_lost_event: threading.Event
    refresh_lock: threading.Lock

    @property
    def chunk_id(self) -> int:
        return int(self.chunk["chunk_id"])

    def assert_owned(self) -> None:
        if self.lease_lost_event.is_set():
            raise LeaseLostError(f"Lease was lost for chunk {self.chunk_id}")
        with self.refresh_lock:
            self.lock.refresh()
        if self.lease_lost_event.is_set():
            raise LeaseLostError(f"Lease was lost for chunk {self.chunk_id}")


def convert_mirror_path(*, ledger_prefix: str, output_path: str) -> str:
    if not output_path.startswith("mirror://"):
        raise ValueError(f"Expected mirror:// output path, got {output_path!r}")

    relative_path = output_path.removeprefix("mirror://").lstrip("/")
    return os.path.join(ledger_prefix.rstrip("/"), relative_path, "ledger")


def _manifest_path(ledger_path: str) -> str:
    return f"{ledger_path.rstrip('/')}/manifest.jsonl"


def _state_path(ledger_path: str, chunk_id: int) -> str:
    return f"{ledger_path.rstrip('/')}/chunks/{chunk_id}.json"


def _ensure_parent(path: str) -> None:
    fs, fs_path = fsspec.core.url_to_fs(path)
    parent = os.path.dirname(fs_path)
    if parent:
        fs.makedirs(parent, exist_ok=True)


def _chunk_to_record(chunk: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(chunk):
        record = dataclasses.asdict(chunk)
    elif isinstance(chunk, dict):
        record = dict(chunk)
    else:
        raise TypeError(f"Unsupported chunk spec type: {type(chunk)!r}")
    if "chunk_id" not in record:
        raise ValueError(f"Chunk spec missing chunk_id: {record!r}")
    record["chunk_id"] = int(record["chunk_id"])
    return record


def _normalize_chunks(chunks: Sequence[Any]) -> list[dict[str, Any]]:
    records = [_chunk_to_record(chunk) for chunk in chunks]
    chunk_ids = [record["chunk_id"] for record in records]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError(f"Duplicate chunk ids in manifest: {chunk_ids}")
    return records


def ensure_manifest(ledger_path: str, chunks: Sequence[Any]) -> None:
    manifest_path = _manifest_path(ledger_path)
    lock = create_lock(f"{manifest_path}.lock")
    if not lock.try_acquire():
        raise RuntimeError(f"Could not acquire manifest lock for {manifest_path}")

    expected = _normalize_chunks(chunks)
    try:
        fs, fs_path = fsspec.core.url_to_fs(manifest_path)
        if fs.exists(fs_path):
            actual = read_manifest(ledger_path)
            if actual != expected:
                raise ValueError(f"Existing manifest at {manifest_path} does not match requested chunks")
            return

        _ensure_parent(manifest_path)
        with fsspec.open(manifest_path, "wt") as f:
            for record in expected:
                f.write(json.dumps(record, sort_keys=True) + "\n")
    finally:
        lock.release()


def read_manifest(ledger_path: str) -> list[dict[str, Any]]:
    path = _manifest_path(ledger_path)
    with fsspec.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_chunk_state(ledger_path: str, chunk_id: int) -> ChunkState | None:
    path = _state_path(ledger_path, chunk_id)
    fs, fs_path = fsspec.core.url_to_fs(path)
    if not fs.exists(fs_path):
        return None
    with fs.open(fs_path, "rt") as f:
        data = json.load(f)
    return ChunkState(status=ChunkStatus(data["status"]), owner=data.get("owner"))


def write_chunk_state(ledger_path: str, chunk_id: int, state: ChunkState) -> None:
    path = _state_path(ledger_path, chunk_id)
    _ensure_parent(path)
    with fsspec.open(path, "wt") as f:
        json.dump({"status": state.status.value, "owner": state.owner}, f, sort_keys=True)


def summarize(ledger_path: str) -> LedgerSummary:
    chunks = read_manifest(ledger_path)
    claimed = 0
    done = 0
    for chunk in chunks:
        state = read_chunk_state(ledger_path, int(chunk["chunk_id"]))
        if state is None:
            continue
        if state.status is ChunkStatus.CLAIMED:
            claimed += 1
        elif state.status is ChunkStatus.DONE:
            done += 1
    return LedgerSummary(total=len(chunks), claimed=claimed, done=done)


def done_chunk_ids(ledger_path: str) -> list[int]:
    ids: list[int] = []
    for chunk in read_manifest(ledger_path):
        chunk_id = int(chunk["chunk_id"])
        state = read_chunk_state(ledger_path, chunk_id)
        if state is not None and state.status is ChunkStatus.DONE:
            ids.append(chunk_id)
    return ids


@contextmanager
def claim_next_chunk(ledger_path: str, owner: str) -> Iterator[ChunkClaim | None]:
    for chunk in read_manifest(ledger_path):
        chunk_id = int(chunk["chunk_id"])
        state = read_chunk_state(ledger_path, chunk_id)
        if state is not None and state.status is ChunkStatus.DONE:
            continue

        state_path = _state_path(ledger_path, chunk_id)
        lock = create_lock(f"{state_path}.lock", owner)
        if not lock.try_acquire():
            continue

        claim: ChunkClaim | None = None
        stop_event = threading.Event()
        lease_lost_event = threading.Event()
        refresh_lock = threading.Lock()

        def heartbeat() -> None:
            while not stop_event.wait(HEARTBEAT_INTERVAL):
                try:
                    with refresh_lock:
                        lock.refresh()
                except LeaseLostError:
                    lease_lost_event.set()
                    return

        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        try:
            state = read_chunk_state(ledger_path, chunk_id)
            if state is not None and state.status is ChunkStatus.DONE:
                continue

            write_chunk_state(ledger_path, chunk_id, ChunkState(status=ChunkStatus.CLAIMED, owner=owner))
            heartbeat_thread.start()
            claim = ChunkClaim(
                ledger_path=ledger_path,
                owner=owner,
                chunk=chunk,
                state_path=state_path,
                lock=lock,
                lease_lost_event=lease_lost_event,
                refresh_lock=refresh_lock,
            )
            yield claim
            return
        finally:
            stop_event.set()
            if heartbeat_thread.is_alive():
                heartbeat_thread.join(timeout=5)
            lock.release()
            if claim is not None and lease_lost_event.is_set():
                raise LeaseLostError(f"Lease was lost during execution of chunk {chunk_id}")

    yield None


def mark_done(claim: ChunkClaim) -> None:
    claim.assert_owned()
    state = read_chunk_state(claim.ledger_path, claim.chunk_id)
    if state != ChunkState(status=ChunkStatus.CLAIMED, owner=claim.owner):
        raise LeaseLostError(f"Chunk {claim.chunk_id} is no longer claimed by {claim.owner}")
    write_chunk_state(
        claim.ledger_path,
        claim.chunk_id,
        ChunkState(status=ChunkStatus.DONE, owner=claim.owner),
    )
