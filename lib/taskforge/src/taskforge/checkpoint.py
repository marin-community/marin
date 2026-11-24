# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish a workpiece's journal (format 15; its layout is in the README) under one fenced writer.

Immutable chunks and workpieces, named by their SHA-256, precede the head that names them. The head is claimed and
advanced only by compare-and-swap on the version this writer observed, so of two writers at most one proceeds.
"""

import asyncio
import errno
import hashlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from functools import partial
from pathlib import PurePosixPath
from typing import Annotated, override

from pydantic import Field, TypeAdapter
from rigging.filesystem.conditional_object import ConditionalObject, ConditionalWriteError, conditional_object
from rigging.filesystem.factory import open_url, url_to_fs
from rigging.filesystem.s3_errors import OBJECT_STORE_ERRORS, is_transient_s3_error_code, storage_error_status
from rigging.filesystem.storage_path import prefix_join
from rigging.timing import ExponentialBackoff, retry_with_backoff

from taskforge.effects.issues import CorruptJournal, LogFailure, PublicationFailed, Superseded, UnsupportedFormat
from taskforge.effects.log import BufferSink, Owners, Record, Sink, Status
from taskforge.model.workpiece import WORKPIECE, Workpiece
from taskforge.stages.classify import commitments

JOURNAL_FORMAT = 15
"""Format 15 checks judge settings when the author grades or submits a draft. A format-14 author could grade a
draft that now returns a tool error, so replay would change the next model request and could diverge from its
recorded sandbox operations. Formats 8 through 14 remain readable for reports.
Format 14 records a solver generation still running at ``solver.time_limit`` as stopped at the time limit (stop
reason and cause ``time_limit``), which a
format-13 reader cannot decode; at 14 it refuses the root as another format. Format 13 recorded such a generation as a
generation limit. A format-14 writer claimed only its own format because earlier trials used other clocks and
recording rules.
Format 13 commits each visit a continuing review decided on (``stage.completed``'s ``committed``) and names each
solver trial's row (``receipt``), so a later launch reuses that evidence. A format-13 writer claimed no format-12 root:
it would decide again on re-observed evidence.
Format 12 words two author requests otherwise: a final revision's request ends with its ordinal (``This is final
revision N of M.``), and a too-easy placement's line no longer calls the round the one hardening attempt. Its rows
decode as format 11's, but a format-12 writer claimed no format-11 root: a resumed format-11 author would send
requests its recorded calls did not, and buy answers its recorded sandbox history refuses.
Format 11 records a ``schema.author_reference`` row: the TaskSpec and Control schemas a root's author is shown,
generated once from the installed models and replayed by every later visit and launch. Format-10 readers cannot
decode that row; at 11 they refuse the root as another format. A format-11 writer claimed no earlier root: their
recorded author calls embedded a reference it cannot reproduce. Such a root stays with its own writer or is rerun at a
fresh output prefix.
Format 10 names the role that paid for each physical cached resolution (a row's ``billed``, outside the event), so
the usage report no longer guesses it from overlapping scopes; request digests and replay keys are unchanged. Every
format-8 and format-9 row means the same at 10, so a format-10 writer resumed both; the report
charges their rows, which name no payer, to one unattributed bucket. Format-9 readers refuse a format-10 root as
another format.
Format 9 records a ``rollouts.solver_generation`` row: a slot's finished generation that its decision model did not
score, which a resumed slot grades without generating again. Format-8 readers cannot decode that row, and would report
the root corrupt; at 9 they refuse it as another format, untouched. Every format-8 row means the same at 9.
Format 8 confirms an acceptance (``solver.confirmation_k`` more solver trials and a final review) before it is
terminal; a format-7 acceptance ended at its validation round and names neither as missing, and a resumed format-7
round would buy a confirmation its recorded history lacks. Its workpiece records each author response to a review
finding (``responses``, each marked ``discarded`` once a restore drops the revision it answered), a final
revision's optional notes, a confirmation record's shipped ``notes`` and a review finding's citation ``groups``: each
is omitted at its default (``OMIT_*`` in ``model/review.py``), and a fault revision omits its placement
(``OMIT_NONE``). This reader reads every
format-8 row, including a null placement; a format-8 reader that predates the omitted placement cannot read a fault
revision without one, but no such reader wrote or read a root outside tests, so the format stays 8.
Format 7 solve-reviews every reviewable round and names what an acceptance went without; a format-6 acceptance
names nothing although no review ran, and a resumed format-6 round would buy a review its recorded history lacks.
Format 6 records an `effect.invalidated` row naming the refused slot result an operator-authorised replacement
supersedes, and a `decision_model.read` row, which format-5 readers cannot decode.
Format 5 places a round over its scored solver slots, holds a slot without one, and falls back to an earlier
submission named by digest; format 4 hydrated each trial without a detail and restored only a too-easy baseline.
Format 4 records adversary findings as a defect direction with observed and deserved grades; format 3 recorded an
expected reward band, which cannot establish a deserved reward. Format 3 links sandbox rows to their environment
owners; format 2 replayed them by content, without owners."""
READABLE = frozenset({8, 9, 10, 11, 12, 13, 14, JOURNAL_FORMAT})
"""The head formats whose rows this code decodes, for the usage report; a writer claims only ``JOURNAL_FORMAT``."""
FORMAT_1 = frozenset({"events.jsonl", "events.jsonl.parts", "status.json", "workpiece.json"})
"""Format 1 named events by module path and wrote these objects beside its stream; any one marks its root."""
HEAD = "journal.json"
CHECKPOINT_EVENTS = 25
CHECKPOINT_SECONDS = 60
FINAL_ATTEMPTS = 6
FINAL_BACKOFF = ExponentialBackoff(initial=5.0, maximum=120.0, factor=2.0, jitter=0.25)
"""The final publication and every history read outlast a storage outage of several minutes beyond the adapters' own
retries."""
RETRIED_REFUSALS = frozenset({408, 429})
"""Request timeout and too many requests: 4xx answers that ask the client to try again."""
REFUSED_ERRNOS = frozenset(
    {
        errno.EACCES,
        errno.EPERM,
        errno.ENOENT,
        errno.ENOTDIR,
        errno.EISDIR,
        errno.EEXIST,
        errno.ENOSPC,
        errno.EDQUOT,
        errno.EROFS,
        errno.ENAMETOOLONG,
    }
)
REFUSED_PATHS = (PermissionError, FileNotFoundError, FileExistsError, IsADirectoryError, NotADirectoryError)


def is_transient_storage_error(error: BaseException) -> bool:
    """Whether a storage call that raised ``error`` may be retried: every storage error but an explicit refusal.

    In order: a retryable S3 code on the error or the SDK error it translates; a local refusal; any 4xx but 408/429.
    """
    if not isinstance(error, OBJECT_STORE_ERRORS):
        return False
    for cause in (error, error.__cause__):
        if isinstance(response := getattr(cause, "response", None), dict):
            if is_transient_s3_error_code(response.get("Error", {}).get("Code")):
                return True
    if isinstance(error, REFUSED_PATHS) or (isinstance(error, OSError) and error.errno in REFUSED_ERRNOS):
        return False
    status = storage_error_status(error)
    return status is None or status in RETRIED_REFUSALS or not 400 <= status < 500


def _write(
    target: ConditionalObject, data: bytes, expected: str | None, other: Callable[[bytes], LogFailure | None]
) -> None:
    """Write ``data`` over version ``expected`` (None: create); decide an inconclusive answer by reading the target.

    Its own bytes there mean the write applied; the fence still holding means it did not, and a retryable error says
    so; ``other`` judges anyone else's bytes, and bytes it does not judge are retried. A failed read raises its error.
    """
    try:
        target.write(data, expected_version=expected)
    except (ConditionalWriteError, *OBJECT_STORE_ERRORS) as error:
        if not isinstance(error, ConditionalWriteError) and not is_transient_storage_error(error):
            raise
        current = target.read()
        if current is not None and current.data == data:
            return
        if current is not None and current.version != expected and (failure := other(current.data)) is not None:
            raise failure from error
        # Not chained: the answer that left the fence holding is no refusal, whatever status it carries.
        raise OSError(errno.EAGAIN, f"{target.path} is still at version {expected} after {error!r}") from None


def _retried[T](call: Callable[[], T], operation: str, attempts: int) -> T:
    return retry_with_backoff(
        call, retryable=is_transient_storage_error, max_attempts=attempts, backoff=FINAL_BACKOFF, operation=operation
    )


@dataclass(frozen=True, slots=True)
class Head:
    format: int
    writer: str
    chunks: tuple[Annotated[str, Field(pattern=r"^[0-9a-f]{64}\.jsonl$")], ...]
    size: Annotated[int, Field(ge=0)]
    status: Status
    workpiece: Annotated[str, Field(pattern=r"^[0-9a-f]{64}\.json$")] | None
    published_at: float


HEAD_JSON = TypeAdapter(Head)


def _put(directory: str, suffix: str, data: bytes) -> str:
    """Create ``data`` under its content name in ``directory`` unless it is there already; return the name."""
    name = hashlib.sha256(data).hexdigest() + suffix
    path = prefix_join(directory, name)
    corrupt = CorruptJournal(f"{path} holds bytes its name does not digest")
    _write(conditional_object(path), data, None, lambda _: corrupt)
    return name


@dataclass(frozen=True, slots=True)
class Journal:
    """A committed head and the version it was read at; its history and workpiece are fetched on demand."""

    destination: str
    head: Head
    version: str

    def _fetch(self, directory: str, name: str) -> bytes:
        """Fetch an object the head names; a missing or altered one is corruption, other I/O errors propagate."""
        path = prefix_join(self.destination, f"{directory}/{name}")

        def fetch() -> bytes:
            with open_url(path, "rb") as source:
                return source.read()

        try:
            data = _retried(fetch, f"read of {path}", FINAL_ATTEMPTS)
        except FileNotFoundError as error:
            raise CorruptJournal(f"{path} is missing; the head names it") from error
        if hashlib.sha256(data).hexdigest() != name.split(".")[0]:
            raise CorruptJournal(f"{path} does not hold the bytes its name digests")
        return data

    def events(self) -> bytes:
        events = b"".join(self._fetch("chunks", name) for name in self.head.chunks)
        if len(events) != self.head.size:
            raise CorruptJournal(f"{self.destination} chunks hold {len(events)} bytes; its head covers {self.head.size}")
        if events and not events.endswith(b"\n"):
            raise CorruptJournal(f"{self.destination} ends in a torn line; committed history holds whole lines")
        return events

    def workpiece(self) -> Workpiece | None:
        if self.head.workpiece is None:
            return None
        try:
            return WORKPIECE.validate_json(self._fetch("workpieces", self.head.workpiece))
        except ValueError as error:
            raise CorruptJournal(f"{self.destination}/workpieces/{self.head.workpiece} is no workpiece") from error


def read_journal(destination: str) -> Journal | None:
    """Read the head alone; fail closed on a root in any other journal format."""
    head_object = conditional_object(prefix_join(destination, HEAD))
    current = _retried(head_object.read, f"read of {head_object.path}", FINAL_ATTEMPTS)
    if current is None:
        fs, root = url_to_fs(destination)
        if found := sorted(FORMAT_1 & {PurePosixPath(path).name for path in fs.glob(prefix_join(root, "*"))}):
            raise UnsupportedFormat(f"{destination} holds journal format 1 ({', '.join(found)}); migrate it first")
        return None
    try:
        found = json.loads(current.data).get("format")
        if type(found) is not int:
            raise ValueError(f"format {found!r} is no integer")
        if found not in READABLE:
            raise UnsupportedFormat(f"{destination}/{HEAD} has journal format {found}, not {JOURNAL_FORMAT}")
        head = HEAD_JSON.validate_json(current.data)
        if head.status.terminal is not None and head.workpiece is None:
            raise ValueError(f"terminal {head.status.terminal} names no workpiece")
    except (ValueError, AttributeError) as error:
        raise CorruptJournal(f"{destination}/{HEAD} is no format-{JOURNAL_FORMAT} head: {error}") from error
    return Journal(destination, head, current.version)


async def _drained(work: Coroutine[object, object, None]) -> None:
    """Run ``work`` to the end even when cancelled, then re-raise: a publishing thread cannot stop midway."""
    task = asyncio.create_task(work)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


class CheckpointSink(Sink):
    """Hold the stream's unpublished suffix in memory; publish it after 25 events or 60 seconds."""

    def __init__(self, destination: str) -> None:
        self.writer = uuid.uuid4().hex
        self.local, self.destination = BufferSink(), destination
        self.head = conditional_object(prefix_join(destination, HEAD))
        self.pending = 0
        self.deferred = False  # After a transient failure the timer retries, not every later event.
        self.lock = asyncio.Lock()
        self.owner: asyncio.Task[object] | None = None
        self.stopped: LogFailure | None = None
        """Why this writer publishes nothing more: supersession, corruption or a storage refusal."""

    def _swap(self, head: Head, expected: str | None) -> None:
        def successor(found: bytes) -> Superseded | None:
            if (writer := HEAD_JSON.validate_json(found).writer) != self.writer:
                return Superseded(f"Writer {writer} advanced {self.destination}/{HEAD}; writer {self.writer} stops")
            return None

        _write(self.head, HEAD_JSON.dump_json(head), expected, successor)

    def claim(self, journal: Journal | None) -> tuple[Record, ...]:
        """Own the head ``journal`` was read at and return its history, decoded before anything is claimed; a head of an
        earlier format is ``UnsupportedFormat``, untouched."""
        if journal is not None and journal.head.format != JOURNAL_FORMAT:
            raise UnsupportedFormat(
                f"{self.destination}/{HEAD} has journal format {journal.head.format}, not {JOURNAL_FORMAT}: resume it "
                "with the writer of its format or rerun at a fresh output prefix"
            )
        events = journal.events() if journal else b""
        try:
            records = self.local.start(events)
            Owners(records)  # owner references, pairing, outcomes and continuity, before anything is claimed
            commitments(records)  # each committed trial's receipt, as reconstruction checks it
        except LogFailure as error:
            raise CorruptJournal(f"{self.destination}: {error}") from error
        chunks, size = (journal.head.chunks, journal.head.size) if journal else ((), 0)
        head = Head(JOURNAL_FORMAT, self.writer, chunks, size, Status(), None, time.time())
        expected = journal.version if journal else None
        _retried(partial(self._swap, head, expected), f"claim of {self.destination}", FINAL_ATTEMPTS)
        return records

    def _publish(self, status: Status, workpiece: Workpiece | None) -> None:
        current = self.head.read()
        if current is None or (head := HEAD_JSON.validate_json(current.data)).writer != self.writer:
            raise Superseded(f"{self.destination}/{HEAD} is owned by another writer, not {self.writer}")
        self.local.drop(head.size)  # Named by this writer's head, even by a swap whose acknowledgment was lost.
        events = bytes(self.local.data)
        if events:
            chunk = _put(prefix_join(self.destination, "chunks"), ".jsonl", events)
            head = replace(head, chunks=(*head.chunks, chunk), size=head.size + len(events))
        elif workpiece is None:
            return  # An idle checkpoint only confirms ownership.
        if workpiece is not None:
            name = _put(prefix_join(self.destination, "workpieces"), ".json", WORKPIECE.dump_json(workpiece))
            if not events and (head.status, head.workpiece) == (status, name):
                return  # A retried final swap whose lost acknowledgment had in fact been applied.
            head = replace(head, workpiece=name)
        self._swap(replace(head, status=status, published_at=time.time()), current.version)

    def _stop(self, error: LogFailure) -> None:
        if self.stopped is None:
            self.stopped = error
            logging.error("%s", error)
            if self.owner is not None and self.owner is not asyncio.current_task():
                self.owner.cancel()

    @override
    async def write(self, records: Sequence[Record]) -> None:
        async with self.lock:
            if self.stopped is not None:
                raise self.stopped
            await self.local.write(records)
            self.pending += len(records)
            if self.pending >= CHECKPOINT_EVENTS and not self.deferred:
                await self._upload(Status(), None)

    async def _upload(self, status: Status, workpiece: Workpiece | None) -> None:
        """Publish the new lines (finally with status and workpiece) under the lock; after a transient error a
        checkpoint leaves them pending for the timer and the final publication retries, rereading the head."""
        if self.stopped is not None:
            raise self.stopped
        publish = asyncio.to_thread(
            _retried,
            partial(self._publish, status, workpiece),
            f"publication of {self.destination}",
            1 if workpiece is None else FINAL_ATTEMPTS,
        )
        try:
            await _drained(publish)
        except (Superseded, CorruptJournal) as error:
            self._stop(error)
            raise
        except OBJECT_STORE_ERRORS as error:
            if workpiece is None and is_transient_storage_error(error):
                logging.warning("Checkpoint upload to %s failed; the timer retries: %r", self.destination, error)
                self.deferred = True
                return
            failure = PublicationFailed(f"Cannot publish {self.destination}; its head stays as it was: {error!r}")
            self._stop(failure)
            raise failure from error
        self.pending, self.deferred = 0, False

    async def finish(self, status: Status, workpiece: Workpiece) -> None:
        """Publish the rest of the stream with the final status and workpiece."""
        async with self.lock:
            await self._upload(status, workpiece)

    async def _periodic(self, stop: asyncio.Event) -> None:
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=CHECKPOINT_SECONDS)
                return
            except TimeoutError:
                async with self.lock:
                    try:
                        await self._upload(Status(), None)
                    except LogFailure:  # Stopped; the owner is cancelled.
                        return

    @asynccontextmanager
    async def running(self) -> AsyncIterator[None]:
        """Run the timer, whose every tick reads the head; a writer that lost it is cancelled and raises why."""
        stop = asyncio.Event()
        self.owner = asyncio.current_task()
        timer = asyncio.create_task(self._periodic(stop))
        failure: Exception | None = None
        try:
            yield
        except Exception as error:  # Kept beside the stop unless it is a log failure, such as the stop itself.
            failure = error
            raise
        finally:
            owner, self.owner = self.owner, None
            stop.set()
            await timer
            if self.stopped is not None:
                if owner is not None and owner.cancelling():
                    owner.uncancel()
                if failure is not None and ExceptionGroup("block", [failure]).split(LogFailure)[1] is not None:
                    raise ExceptionGroup(f"{self.destination} stopped beside another failure", [self.stopped, failure])
                raise self.stopped
