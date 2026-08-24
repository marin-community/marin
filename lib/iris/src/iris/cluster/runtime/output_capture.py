# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded per-attempt output archiving shared by Iris execution runtimes."""

import hashlib
import logging
import os
import re
import stat
import tarfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import zstandard
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Deadline

from iris.cluster.config import TaskOutputPolicy
from iris.cluster.types import AttemptUid, JobName
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

_ARCHIVE_NAME = "outputs.tar.zst"
_SKIPPED_SAMPLE_LIMIT = 10
_SKIPPED_PATH_MAX_CHARS = 200
_ERROR_MAX_CHARS = 900
_CAPTURE_CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class TaskOutputLimits:
    max_bytes: int
    max_entries: int


@dataclass(frozen=True, slots=True)
class ResolvedOutputDestination:
    path: StoragePath
    retention: int
    ttl_days: int = 0


def resolve_task_output_destination(
    policy: TaskOutputPolicy,
    task_id: JobName,
    attempt_uid: AttemptUid,
    *,
    local_root: Path,
    source_prefix: str | None,
) -> ResolvedOutputDestination:
    """Resolve one attempt's archive path from execution-cluster policy."""
    relative = f"{task_id.to_wire().lstrip('/')}/{attempt_uid}/{_ARCHIVE_NAME}"
    if policy.destination == "file://":
        return ResolvedOutputDestination(
            path=StoragePath(f"file://{local_root / relative}"),
            retention=job_pb2.TaskOutputArchive.TASK_OUTPUT_ARCHIVE_RETENTION_LOCAL_CLUSTER,
        )
    prefix = policy.destination
    if prefix is None:
        prefix = marin_temp_bucket(
            policy.ttl_days,
            prefix="iris/task-outputs",
            source_prefix=source_prefix,
        )
    destination = StoragePath(prefix)
    if policy.destination is not None and destination.is_local:
        return ResolvedOutputDestination(
            path=destination / relative,
            retention=job_pb2.TaskOutputArchive.TASK_OUTPUT_ARCHIVE_RETENTION_LOCAL_CLUSTER,
        )
    ttl_match = re.search(r"/ttl=(\d+)d(?:/|$)", str(destination))
    if ttl_match is None:
        raise ValueError(f"Temporary output destination has no lifecycle TTL prefix: {destination}")
    return ResolvedOutputDestination(
        path=destination / relative,
        retention=job_pb2.TaskOutputArchive.TASK_OUTPUT_ARCHIVE_RETENTION_TTL,
        ttl_days=int(ttl_match.group(1)),
    )


@dataclass(frozen=True, slots=True)
class _Entry:
    path: Path
    relative: str
    mode: int


@dataclass(frozen=True, slots=True)
class _ArchiveWriteResult:
    size_bytes: int
    sha256: str
    skipped: tuple[job_pb2.TaskOutputSkippedEntry, ...]


class _CaptureError(RuntimeError):
    pass


class _ArchiveStopped(_CaptureError):
    pass


def _skipped_entry(entry: _Entry) -> job_pb2.TaskOutputSkippedEntry:
    return job_pb2.TaskOutputSkippedEntry(path=entry.relative[:_SKIPPED_PATH_MAX_CHARS], reason="special_file")


class _HashingWriter:
    def __init__(self, raw: BinaryIO, deadline: Deadline, stop: threading.Event):
        self.raw = raw
        self.deadline = deadline
        self.stop = stop
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, data: bytes) -> int:
        _check_running(self.deadline, self.stop)
        written = self.raw.write(data)
        chunk = data[:written]
        self.digest.update(chunk)
        self.size += written
        return written

    def flush(self) -> None:
        self.raw.flush()


def _check_running(deadline: Deadline, stop: threading.Event) -> None:
    if stop.is_set():
        raise _ArchiveStopped(_CAPTURE_CANCELLED)
    if deadline.expired():
        raise _ArchiveStopped("deadline_exceeded")


def _inventory(source: Path, limits: TaskOutputLimits, deadline: Deadline, stop: threading.Event) -> list[_Entry]:
    entries: list[_Entry] = []
    total_bytes = 0

    def walk(directory: Path, relative_dir: str) -> None:
        nonlocal total_bytes
        _check_running(deadline, stop)
        with os.scandir(directory) as iterator:
            children = sorted(iterator, key=lambda entry: entry.name)
        for child in children:
            _check_running(deadline, stop)
            relative = f"{relative_dir}/{child.name}" if relative_dir else child.name
            info = child.stat(follow_symlinks=False)
            mode = info.st_mode
            size = info.st_size if stat.S_ISREG(mode) else 0
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
                entries.append(_Entry(Path(child.path), relative, mode))
            else:
                total_bytes += size
                if total_bytes > limits.max_bytes:
                    raise _CaptureError(f"too_large: regular-file bytes exceed {limits.max_bytes}")
                entries.append(_Entry(Path(child.path), relative, mode))
            if len(entries) > limits.max_entries:
                raise _CaptureError(f"too_many_entries: entry count exceeds {limits.max_entries}")
            if stat.S_ISDIR(mode):
                walk(Path(child.path), relative)

    if source.exists():
        walk(source, "")
    return entries


def _normalized_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.pax_headers = {}
    return info


def _write_archive(
    entries: list[_Entry],
    destination: StoragePath,
    deadline: Deadline,
    stop: threading.Event,
) -> _ArchiveWriteResult:
    skipped: list[job_pb2.TaskOutputSkippedEntry] = []
    temporary = destination.parent / f".{_ARCHIVE_NAME}.partial-{uuid.uuid4().hex}"
    destination.parent.mkdirs(exist_ok=True)
    try:
        with temporary.open("wb") as raw:
            hashing = _HashingWriter(raw, deadline, stop)
            compressor = zstandard.ZstdCompressor(level=3)
            with compressor.stream_writer(hashing, closefd=False) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                    for entry in entries:
                        _check_running(deadline, stop)
                        if not (stat.S_ISDIR(entry.mode) or stat.S_ISREG(entry.mode) or stat.S_ISLNK(entry.mode)):
                            skipped.append(_skipped_entry(entry))
                            continue
                        info = _normalized_tarinfo(archive.gettarinfo(str(entry.path), arcname=entry.relative))
                        if stat.S_ISREG(entry.mode):
                            with entry.path.open("rb") as source:
                                archive.addfile(info, source)
                        else:
                            archive.addfile(info)
            hashing.flush()
        temporary.rename(destination)
        return _ArchiveWriteResult(hashing.size, hashing.digest.hexdigest(), tuple(skipped))
    except Exception:
        try:
            if temporary.exists():
                temporary.rm()
        except Exception:
            logger.warning("Failed to remove partial task output %s", temporary, exc_info=True)
        raise


def capture_task_outputs(
    source: Path,
    destination: ResolvedOutputDestination,
    limits: TaskOutputLimits,
    deadline: Deadline,
    stop: threading.Event,
) -> job_pb2.TaskOutputArchive:
    """Archive one stable output tree and return its terminal capture result."""
    try:
        entries = _inventory(source, limits, deadline, stop)
        eligible = [
            entry
            for entry in entries
            if stat.S_ISDIR(entry.mode) or stat.S_ISREG(entry.mode) or stat.S_ISLNK(entry.mode)
        ]
        skipped_entries = [
            entry
            for entry in entries
            if not (stat.S_ISDIR(entry.mode) or stat.S_ISREG(entry.mode) or stat.S_ISLNK(entry.mode))
        ]
        if not eligible:
            return job_pb2.TaskOutputArchive(
                state=job_pb2.TaskOutputArchive.TASK_OUTPUT_ARCHIVE_STATE_EMPTY,
                skipped_count=len(skipped_entries),
                skipped_sample=[_skipped_entry(entry) for entry in skipped_entries[:_SKIPPED_SAMPLE_LIMIT]],
            )
        archive = _write_archive(entries, destination.path, deadline, stop)
        return job_pb2.TaskOutputArchive(
            state=job_pb2.TaskOutputArchive.TASK_OUTPUT_ARCHIVE_STATE_UPLOADED,
            uri=str(destination.path),
            size_bytes=archive.size_bytes,
            sha256=archive.sha256,
            retention=destination.retention,
            ttl_days=destination.ttl_days,
            skipped_count=len(archive.skipped),
            skipped_sample=archive.skipped[:_SKIPPED_SAMPLE_LIMIT],
        )
    except _CaptureError as exc:
        state = (
            job_pb2.TaskOutputArchive.TASK_OUTPUT_ARCHIVE_STATE_UNAVAILABLE
            if str(exc) == _CAPTURE_CANCELLED
            else job_pb2.TaskOutputArchive.TASK_OUTPUT_ARCHIVE_STATE_FAILED
        )
        return job_pb2.TaskOutputArchive(state=state, error=str(exc))
    except Exception as exc:
        logger.exception("Task output capture failed")
        return task_output_storage_failure(exc)


def task_output_storage_failure(exc: Exception) -> job_pb2.TaskOutputArchive:
    """Return a bounded FAILED archive result for ``exc``."""
    return job_pb2.TaskOutputArchive(
        state=job_pb2.TaskOutputArchive.TASK_OUTPUT_ARCHIVE_STATE_FAILED,
        error=f"storage_error: {str(exc)[:_ERROR_MAX_CHARS]}",
    )


def capture_task_outputs_for_attempt(
    source: Path,
    policy: TaskOutputPolicy,
    task_id: JobName,
    attempt_uid: AttemptUid,
    *,
    local_root: Path,
    source_prefix: str | None,
    stop: threading.Event,
) -> job_pb2.TaskOutputArchive:
    """Resolve policy and capture one attempt's output tree."""
    try:
        destination = resolve_task_output_destination(
            policy,
            task_id,
            attempt_uid,
            local_root=local_root,
            source_prefix=source_prefix,
        )
        return capture_task_outputs(
            source,
            destination,
            TaskOutputLimits(max_bytes=policy.max_bytes, max_entries=policy.max_entries),
            Deadline.from_now(policy.finalization_timeout),
            stop,
        )
    except Exception as exc:
        logger.exception("Task output destination resolution failed")
        return task_output_storage_failure(exc)
