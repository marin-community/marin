# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable log forwarder: tail one finelog, replicate it into another (the "log-proxy").

A :class:`LogForwarder` reads a source finelog's log store and re-appends each new batch
to a target finelog under the same keys. It is the mechanism behind Iris federation's
global-finelog relay (each cluster forwards its logs to one shared store), but it is
generic: source and target are injected and it holds no Iris concepts. Source and target
must be distinct stores — it does not dedupe its own writes, so forwarding a store into
itself would loop.

Durability leans on the source, which is itself durable (finelog persists to parquet
with a retention window). The forwarder therefore stores only a forward *watermark* —
the max source autoincrement id confirmed landed at the target — in a small JSON state
file, not a second copy of the log data. Forwarding uses :meth:`LogClient.push_batch`,
which returns only after the target durably persists the batch; the watermark advances
only after that ack, so a crash or transient failure re-forwards the in-flight batch on
the next tick rather than losing it (at-least-once; duplicate lines are bounded to the
failure boundary and tolerable for logs).

On first run the watermark is seeded at the source's current max cursor, so enabling a
forwarder ships new logs going forward without backfilling the whole retention window.
The state file also records the target it tracks; pointing a forwarder at a different
target reseeds rather than replaying stale cursors into a new store's id space.

The watermark is deliberately local best-effort state: it is fsync'd so it survives a
process restart on the same host, but it is not part of any external checkpoint. If the
state file is lost (e.g. the host is replaced), the forwarder reseeds at the source's
current max — accepting a bounded gap of un-forwarded logs rather than replaying the
retention window. That trade suits a log relay, where a gap degrades observability, not
correctness.
"""

import json
import logging
import os
import threading
from collections import defaultdict
from pathlib import Path

from finelog.client.log_client import LogClient
from finelog.rpc import logging_pb2

logger = logging.getLogger(__name__)

# All finelog log keys are absolute (`/user/...`, `/system/...`), so a PREFIX read on
# "/" tails the entire source store.
_ALL_KEYS_PREFIX = "/"

_DEFAULT_BATCH_LINES = 1000
_DEFAULT_POLL_INTERVAL_SECONDS = 5.0
_DEFAULT_STOP_TIMEOUT_SECONDS = 5.0


class CorruptForwarderStateError(RuntimeError):
    """The state file exists but is unreadable or malformed.

    Kept distinct from an absent file (a genuine first run): a corrupt watermark must
    not be treated as first-run, or the forwarder would seed at the source's current
    max and permanently skip the logs between the lost watermark and now. The operator
    repairs or removes the file; until then the forwarder refuses to advance.
    """


class LogForwarder:
    """Tails ``source`` and forwards new log batches to ``target`` under the same keys.

    The forwarder owns ``target`` (created for it) and closes it on :meth:`stop`; it
    never closes ``source`` (owned by the caller).
    """

    def __init__(
        self,
        *,
        source: LogClient,
        target: LogClient,
        target_label: str,
        state_path: Path,
        batch_lines: int = _DEFAULT_BATCH_LINES,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
    ):
        self._source = source
        self._target = target
        self._target_label = target_label
        self._state_path = Path(state_path)
        self._batch_lines = batch_lines
        self._poll_interval = poll_interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # In-memory observability counters (the state file only needs the cursor).
        self._forwarded = 0
        self._failed = 0
        # Rows fetched in the last `forward_once`; a full batch drives the drain loop.
        self._last_fetched = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="finelog-forwarder", daemon=False)
        self._thread.start()

    def stop(self, timeout_seconds: float = _DEFAULT_STOP_TIMEOUT_SECONDS) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_seconds)
            self._thread = None
        self._target.close()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._drain()
            except Exception:
                # A failed forward is already recorded and retried; an unexpected error
                # (e.g. a source read) must not kill the loop and silently stop shipping.
                logger.exception("finelog forwarder: unexpected error in tick")
            self._stop.wait(timeout=self._poll_interval)

    def _drain(self) -> int:
        """Forward batches until caught up, then return the total forwarded this pass.

        Shipping only one ``batch_lines`` batch per poll interval caps the forward rate; a
        cluster producing faster than that would fall behind until source retention evicts
        un-forwarded rows (silent loss). So we keep forwarding while each batch comes back
        full (more pending), and stop on a short batch (caught up), a failure, or stop.
        """
        total = 0
        while not self._stop.is_set():
            total += self.forward_once()
            if self._last_fetched < self._batch_lines:
                return total
        return total

    def forward_once(self) -> int:
        """Forward one batch of newly-ingested source logs to the target.

        Returns the number of entries forwarded this tick (0 = nothing new, or the
        initial seed). Exposed so a test can drive one tick deterministically.
        """
        self._last_fetched = 0
        cursor = self._read_watermark()
        if cursor is None:
            seed = self._source_max_cursor()
            self._write_watermark(seed)
            logger.info(
                "finelog forwarder: seeded watermark at source cursor %d for %s (new logs only)",
                seed,
                self._target_label,
            )
            return 0

        response = self._source.fetch_logs(
            logging_pb2.FetchLogsRequest(
                source=_ALL_KEYS_PREFIX,
                match_scope=logging_pb2.MATCH_SCOPE_PREFIX,
                cursor=cursor,
                max_lines=self._batch_lines,
            )
        )
        if not response.entries:
            return 0

        groups: dict[str, list[logging_pb2.LogEntry]] = defaultdict(list)
        for entry in response.entries:
            if not entry.key:
                # A keyless entry has no key to replicate under; drop it rather than
                # stall the watermark (the source always keys its writes).
                continue
            groups[entry.key].append(entry)

        try:
            for key, entries in groups.items():
                self._target.push_batch(key, entries)
        except Exception as exc:
            self._failed += 1
            logger.warning(
                "finelog forwarder: forward failed at cursor %d (%s: %s); retrying next tick",
                cursor,
                type(exc).__name__,
                exc,
            )
            return 0

        # A full fetch means the source has more pending past this batch — the drain loop
        # in `_run` reads this to keep forwarding within the tick instead of one batch per
        # poll interval. Reset to 0 on every non-forwarding path (seed, empty, failure).
        self._last_fetched = len(response.entries)

        # Count what was actually pushed, not what was fetched: keyless entries are
        # skipped above but still advance the watermark.
        forwarded = sum(len(entries) for entries in groups.values())
        self._forwarded += forwarded
        self._write_watermark(response.cursor)
        return forwarded

    def _source_max_cursor(self) -> int:
        """The source's current max autoincrement id (the seed watermark)."""
        response = self._source.fetch_logs(
            logging_pb2.FetchLogsRequest(
                source=_ALL_KEYS_PREFIX,
                match_scope=logging_pb2.MATCH_SCOPE_PREFIX,
                tail=True,
                max_lines=1,
            )
        )
        return response.cursor

    def _read_watermark(self) -> int | None:
        """The persisted watermark, or ``None`` on a genuine first run / intentional repoint.

        Returns ``None`` only when the state file is absent (first run) or tracks a
        different target (an intentional repoint — reseed rather than replay stale
        cursors into a new store's id space). A file that exists but is malformed raises
        :class:`CorruptForwarderStateError` so a damaged watermark surfaces loudly
        instead of silently reseeding past un-forwarded logs.
        """
        try:
            raw = self._state_path.read_text()
        except FileNotFoundError:
            return None
        try:
            state = json.loads(raw)
        except ValueError as exc:
            raise CorruptForwarderStateError(f"forwarder state file {self._state_path} is not valid JSON") from exc
        if not isinstance(state, dict):
            raise CorruptForwarderStateError(f"forwarder state file {self._state_path} is not a JSON object")
        if state.get("target") != self._target_label:
            logger.warning(
                "finelog forwarder: state file %s tracks target %r, not %r; reseeding for the new target",
                self._state_path,
                state.get("target"),
                self._target_label,
            )
            return None
        cursor = state.get("cursor")
        if not isinstance(cursor, int) or isinstance(cursor, bool):
            raise CorruptForwarderStateError(f"forwarder state file {self._state_path} has no integer cursor")
        return cursor

    def _write_watermark(self, cursor: int) -> None:
        """Atomically and durably persist the watermark (write-temp, fsync, rename).

        fsync of the temp file and the parent directory makes the watermark survive a
        crash: a lost state file reads as a first run and would reseed past un-forwarded
        logs, so its durability is what bounds forwarding to at-least-once.
        """
        payload = {
            "target": self._target_label,
            "cursor": cursor,
            "forwarded": self._forwarded,
            "failed": self._failed,
        }
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        with open(tmp, "w") as f:
            f.write(json.dumps(payload))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._state_path)
        dir_fd = os.open(self._state_path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
