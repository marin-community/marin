# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native sidecar that ships a task pod's container logs to finelog.

Runs as a native sidecar (initContainer with ``restartPolicy: Always``) in every
k8s task pod. It tails the task container's on-disk CRI log file from the node
(mounted read-only via hostPath at ``/var/log/pods``) and pushes the lines to the
finelog log server, so the controller does not pull logs per pod through the
apiserver.

Run as ``python -m iris.cluster.backends.k8s.logship``. Configuration comes from
the environment the pod manifest sets:

- ``IRIS_TASK_ID`` — the wire ``task:attempt`` id; the finelog key is derived
  from it with the attempt suffix always present (``/user/job/0:0``).
- ``IRIS_CONTROLLER_ADDRESS`` — resolves the log server endpoint.
- ``IRIS_POD_NAMESPACE`` / ``IRIS_POD_NAME`` — locate the CRI log directory.

The push is unauthenticated: the finelog log service performs no auth, the same
posture under which the controller writes its own logs.
"""

import glob
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

from finelog.client import LogClient
from finelog.rpc import logging_pb2
from rigging.timing import Timestamp

from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.cluster.log_keys import classify_log_level
from iris.rpc import controller_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync

logger = logging.getLogger(__name__)

# Name of the task container in the pod manifest. The CRI log directory holds one
# subdirectory per container; we tail only the task's.
_TASK_CONTAINER_NAME = "task"

# Max log entries per write_batch RPC. Keeps a burst of output from building one
# unbounded request while still amortizing the round-trip over many lines.
_MAX_BATCH_ENTRIES = 1000

# How long to wait for the CRI log file to appear before the first read, and the
# idle poll interval once it exists. The kubelet creates the file when the
# container starts, which is after this sidecar; tailing simply waits for it.
_FILE_WAIT_POLL_SECONDS = 0.5
_TAIL_IDLE_POLL_SECONDS = 0.25

# Flush deadline applied on shutdown so a finelog hiccup cannot hang pod teardown.
_SHUTDOWN_FLUSH_TIMEOUT_SECONDS = 5.0


@dataclass
class CriLogLine:
    """One parsed CRI log line.

    ``epoch_ms`` is the kubelet-stamped timestamp; ``stream`` is ``stdout`` or
    ``stderr``; ``message`` is the reassembled log text (partial ``P`` fragments
    joined onto the following ``F`` line).
    """

    epoch_ms: int
    stream: str
    message: str


@dataclass(frozen=True)
class ParsedCriLine:
    """One CRI log line after field splitting, before partial-line reassembly.

    ``is_full`` is True for an ``F`` tag (a complete line) and False for ``P`` (a
    continuation to be joined onto the next fragment).
    """

    epoch_ms: int
    stream: str
    is_full: bool
    message: str


def parse_cri_line(raw: str) -> ParsedCriLine | None:
    """Parse one CRI on-disk log line.

    The format is ``<rfc3339> <stdout|stderr> <F|P> <message>``: a timestamp, the
    stream, a full/partial tag, then the message (which may itself contain
    spaces). Returns None for a line that does not match the format.
    """
    parts = raw.split(" ", 3)
    if len(parts) < 4:
        return None
    timestamp_str, stream, tag, message = parts
    if stream not in ("stdout", "stderr") or tag not in ("F", "P"):
        return None
    epoch_ms = _rfc3339_to_epoch_ms(timestamp_str)
    if epoch_ms is None:
        return None
    return ParsedCriLine(epoch_ms=epoch_ms, stream=stream, is_full=tag == "F", message=message)


def _rfc3339_to_epoch_ms(timestamp_str: str) -> int | None:
    """Convert a CRI RFC3339 timestamp (nanosecond precision, ``Z`` suffix) to epoch ms."""
    text = timestamp_str.replace("Z", "+00:00")
    # datetime.fromisoformat accepts at most microsecond precision; CRI writes
    # nanoseconds, so truncate the fractional part to 6 digits before parsing.
    if "." in text:
        head, _, tail = text.partition(".")
        frac = tail
        offset = ""
        for sep in ("+", "-"):
            if sep in tail:
                frac, _, off = tail.partition(sep)
                offset = sep + off
                break
        text = f"{head}.{frac[:6]}{offset}"
    try:
        return Timestamp.from_seconds(datetime.fromisoformat(text).timestamp()).epoch_ms()
    except ValueError:
        return None


def _make_log_entry(line: CriLogLine, attempt_id: int) -> logging_pb2.LogEntry:
    """Build a finelog LogEntry from a parsed CRI line."""
    level = classify_log_level(line.stream, line.message)
    entry = logging_pb2.LogEntry(source=line.stream, data=line.message, attempt_id=attempt_id, level=level)
    entry.timestamp.epoch_ms = line.epoch_ms
    return entry


def log_key_and_attempt(task_id: str) -> tuple[str, int]:
    """Resolve ``IRIS_TASK_ID`` to the finelog log key and the attempt id.

    The finelog key must carry the attempt suffix — ``/user/job/0:0`` for the
    first attempt — so it matches the per-task log query, which is EXACT
    ``…:<attempt>`` for one attempt or PREFIX ``…:`` for all attempts. A bare key
    without the suffix only matches the job-level ``/user/job/`` prefix scan, so
    its logs would show under the job but never under the task.

    Only a trailing ``:<digits>`` is the attempt; a colon elsewhere is part of the
    job name (a legal name char) and stays in the key. A wire id that arrives
    without an attempt suffix is normalized to attempt 0.
    """
    _base, sep, suffix = task_id.rpartition(":")
    if sep and suffix.isdigit():
        return task_id, int(suffix)
    return f"{task_id}:0", 0


def _log_dir_glob(namespace: str, pod_name: str) -> str:
    """Glob pattern for the task container's CRI log directory.

    kubelet lays out logs at ``/var/log/pods/{namespace}_{name}_{uid}/{container}/``;
    the uid is unknown to the pod, so it is globbed.
    """
    return f"/var/log/pods/{namespace}_{pod_name}_*/{_TASK_CONTAINER_NAME}"


def _active_log_file(log_dir_glob: str) -> str | None:
    """Return the active CRI log file (``0.log``) under the globbed container dir.

    kubelet writes the live stream to ``0.log`` and rotates older content to
    ``0.log.<timestamp>`` / ``1.log`` etc. We only follow the active file.
    """
    for container_dir in glob.glob(log_dir_glob):
        candidate = os.path.join(container_dir, "0.log")
        if os.path.exists(candidate):
            return candidate
    return None


@dataclass
class _LineBuffer:
    """Reassembles CRI partial (``P``) fragments into full lines.

    A ``P``-tagged line is a fragment of a long line the kubelet split at its read
    buffer boundary; fragments accumulate until an ``F`` line closes them. The
    closing line's timestamp and stream are used for the reassembled entry.
    """

    pending: list[str] = field(default_factory=list)

    def feed(self, raw: str) -> CriLogLine | None:
        """Feed one raw CRI line; return a CriLogLine when a full line is complete."""
        parsed = parse_cri_line(raw)
        if parsed is None:
            return None
        if not parsed.is_full:
            self.pending.append(parsed.message)
            return None
        message = parsed.message
        if self.pending:
            message = "".join(self.pending) + message
            self.pending.clear()
        return CriLogLine(epoch_ms=parsed.epoch_ms, stream=parsed.stream, message=message)


class LogShipper:
    """Tails a task container's CRI log file and ships parsed lines to finelog.

    The shipper follows the active ``0.log``, reassembles partial lines, and
    writes batched LogEntries under the task's finelog key. It survives kubelet
    log rotation by reopening the active file when its inode changes or it
    shrinks. The ship loop never raises: a finelog outage must not crash the
    sidecar and wedge pod teardown.
    """

    def __init__(self, client: LogClient, log_dir_glob: str, key: str, attempt_id: int, stop: threading.Event):
        self._client = client
        self._log_dir_glob = log_dir_glob
        self._key = key
        self._attempt_id = attempt_id
        self._buffer = _LineBuffer()
        self._stop = stop

    def run(self) -> None:
        """Follow the active log file across rotations until stopped, then flush.

        The outer loop owns one open file handle per generation of ``0.log``;
        when kubelet rotates the file (new inode), the handle is closed and the
        loop reopens the fresh file from the start.
        """
        path = self._await_file()
        if path is None:
            return
        while path is not None:
            with open(path, encoding="utf-8", errors="replace") as handle:
                inode = os.fstat(handle.fileno()).st_ino
                path = self._tail(handle, inode)
        self._flush()

    def _tail(self, handle, inode: int) -> str | None:
        """Drain ``handle`` until stop or rotation.

        Returns the path of the rotated-in file to reopen, or None when the
        shipper was asked to stop (the caller then flushes and exits).
        """
        partial = ""
        while True:
            partial = self._drain(handle, partial)
            if self._stop.is_set():
                # Final drain to EOF after the task container exited.
                self._drain(handle, partial)
                return None
            rotated_path = self._rotated_path(handle, inode)
            if rotated_path is not None:
                return rotated_path
            time.sleep(_TAIL_IDLE_POLL_SECONDS)

    def _await_file(self) -> str | None:
        """Wait for the active CRI log file to appear; stop early if asked to."""
        while not self._stop.is_set():
            path = _active_log_file(self._log_dir_glob)
            if path is not None:
                return path
            time.sleep(_FILE_WAIT_POLL_SECONDS)
        return _active_log_file(self._log_dir_glob)

    def _drain(self, handle, partial: str) -> str:
        """Read all currently-available complete lines from ``handle`` and ship them.

        Returns the trailing partial read (an unterminated final chunk) to carry
        into the next drain so a line split across reads is not shipped twice.
        """
        chunk = handle.read()
        if not chunk:
            return partial
        data = partial + chunk
        lines = data.split("\n")
        trailing = lines.pop()
        batch: list[logging_pb2.LogEntry] = []
        for raw in lines:
            cri_line = self._buffer.feed(raw)
            if cri_line is not None:
                batch.append(_make_log_entry(cri_line, self._attempt_id))
            if len(batch) >= _MAX_BATCH_ENTRIES:
                self._write(batch)
                batch = []
        if batch:
            self._write(batch)
        return trailing

    def _rotated_path(self, handle, inode: int) -> str | None:
        """Return the active log file path when kubelet has rotated ``0.log``.

        Rotation replaces ``0.log`` with a fresh inode (and the file may briefly
        shrink below the current read offset). Returns the path to reopen on a
        detected rotation, or None when the current handle is still the live file.
        """
        path = _active_log_file(self._log_dir_glob)
        if path is None:
            return None
        try:
            stat = os.stat(path)
        except OSError:
            return None
        if stat.st_ino == inode and stat.st_size >= handle.tell():
            return None
        return path

    def _write(self, entries: list[logging_pb2.LogEntry]) -> None:
        try:
            self._client.write_batch(self._key, entries)
        except Exception:
            logger.warning("logship: write_batch of %d entries failed", len(entries), exc_info=True)

    def _flush(self) -> None:
        try:
            self._client.flush(timeout=_SHUTDOWN_FLUSH_TIMEOUT_SECONDS)
        except Exception:
            logger.warning("logship: final flush failed", exc_info=True)


def _resolve_log_service(controller_client: ControllerServiceClientSync, server_url: str) -> str:
    """Resolve the log server address via the controller's endpoint registry."""
    resp = controller_client.list_endpoints(
        controller_pb2.Controller.ListEndpointsRequest(prefix=server_url, exact=True),
    )
    if not resp.endpoints:
        raise ConnectionError(f"No {server_url!r} endpoint registered on controller")
    return resp.endpoints[0].address


def _connect_log_client(controller_address: str) -> LogClient:
    """Build a finelog write client that resolves the log server via the controller.

    Unauthenticated — the finelog log service performs no auth.
    """
    controller_client = ControllerServiceClientSync(
        address=controller_address,
        timeout_ms=10_000,
    )
    return LogClient.connect(
        LOG_SERVER_ENDPOINT_NAME,
        resolver=lambda server_url: _resolve_log_service(controller_client, server_url),
    )


def _ship(stop: threading.Event) -> None:
    """Connect to the log server and ship the task container's logs until stopped."""
    task_id = os.environ["IRIS_TASK_ID"]
    controller_address = os.environ["IRIS_CONTROLLER_ADDRESS"]
    namespace = os.environ["IRIS_POD_NAMESPACE"]
    pod_name = os.environ["IRIS_POD_NAME"]

    key, attempt_id = log_key_and_attempt(task_id)
    client = _connect_log_client(controller_address)
    shipper = LogShipper(client, _log_dir_glob(namespace, pod_name), key, attempt_id, stop)

    logger.info("logship: shipping logs for %s (attempt %d) from %s", key, attempt_id, pod_name)
    shipper.run()
    client.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s logship %(message)s")

    # As a native sidecar, this process must stay running for the task container
    # to start. Log shipping is best-effort: any failure is logged and the
    # process then idles until SIGTERM, so a broken shipper degrades to "no logs"
    # rather than wedging the task pod in PodInitializing.
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        _ship(stop)
    except Exception:
        logger.exception("logship: shipping failed; idling so the task pod is not blocked")
        stop.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
