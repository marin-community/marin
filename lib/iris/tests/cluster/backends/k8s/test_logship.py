# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the log-shipper sidecar: CRI parsing, tailing, rotation, key derivation."""

import threading
import time

from finelog.rpc import logging_pb2
from finelog.types import str_to_log_level
from iris.cluster.backends.k8s.logship import (
    LogShipper,
    _LineBuffer,
    _log_dir_glob,
    _make_log_entry,
    log_key_and_attempt,
    parse_cri_line,
)


class FakeLogWriter:
    """Records every write_batch call so tests can assert the shipped entries."""

    def __init__(self) -> None:
        self.batches: list[tuple[str, list[logging_pb2.LogEntry]]] = []

    def write_batch(self, key: str, messages: list[logging_pb2.LogEntry]) -> None:
        self.batches.append((key, list(messages)))

    def flush(self, timeout: float | None = None) -> None:
        pass

    def entries(self) -> list[logging_pb2.LogEntry]:
        return [entry for _key, batch in self.batches for entry in batch]


# ---------------------------------------------------------------------------
# CRI line parsing
# ---------------------------------------------------------------------------


def test_parse_full_line():
    parsed = parse_cri_line("2026-06-24T12:00:00.123456789Z stdout F hello world")
    assert parsed is not None
    assert parsed.stream == "stdout"
    assert parsed.is_full is True
    assert parsed.message == "hello world"
    # Nanosecond precision is truncated to ms.
    assert parsed.epoch_ms % 1000 == 123


def test_parse_stderr_stream():
    parsed = parse_cri_line("2026-06-24T12:00:00Z stderr F boom")
    assert parsed is not None
    assert parsed.stream == "stderr"
    assert parsed.message == "boom"


def test_parse_partial_line_flagged_not_full():
    parsed = parse_cri_line("2026-06-24T12:00:00Z stdout P frag")
    assert parsed is not None
    assert parsed.is_full is False
    assert parsed.message == "frag"


def test_parse_message_with_spaces_preserved():
    parsed = parse_cri_line("2026-06-24T12:00:00Z stdout F a b  c")
    assert parsed is not None
    assert parsed.message == "a b  c"


def test_parse_rejects_malformed_lines():
    assert parse_cri_line("garbage") is None
    assert parse_cri_line("2026-06-24T12:00:00Z notastream F x") is None
    assert parse_cri_line("2026-06-24T12:00:00Z stdout X x") is None
    assert parse_cri_line("not-a-timestamp stdout F x") is None


def test_full_line_carries_parsed_log_level():
    """A glog-style line yields a LogEntry with the parsed level."""
    buffer = _LineBuffer()
    line = buffer.feed("2026-06-24T12:00:00Z stderr F E0624 12:00:00 boom")
    assert line is not None
    entry = _make_log_entry(line, attempt_id=0)
    assert entry.level == str_to_log_level("ERROR")


def test_partial_lines_join_onto_following_full_line():
    buffer = _LineBuffer()
    assert buffer.feed("2026-06-24T12:00:00Z stdout P part1-") is None
    assert buffer.feed("2026-06-24T12:00:00Z stdout P part2-") is None
    line = buffer.feed("2026-06-24T12:00:00Z stdout F part3")
    assert line is not None
    assert line.message == "part1-part2-part3"
    assert line.stream == "stdout"


# ---------------------------------------------------------------------------
# IRIS_TASK_ID -> (key, attempt)
# ---------------------------------------------------------------------------


def test_log_key_and_attempt_with_retry_suffix():
    key, attempt = log_key_and_attempt("/u/job/0:2")
    assert key == "/u/job/0:2"
    assert attempt == 2


def test_log_key_and_attempt_first_attempt_gets_zero_suffix():
    """Attempt 0 arrives as the bare wire id but must be keyed with the ``:0``
    suffix so it matches the per-task log query, not just the job-level scan."""
    key, attempt = log_key_and_attempt("/u/job/0")
    assert key == "/u/job/0:0"
    assert attempt == 0


def test_log_key_and_attempt_tolerates_colon_in_job_name():
    """A colon is a legal job-name char (used by federated job names); only a
    trailing ``:<digits>`` is the attempt, so an interior colon stays in the key
    and the id still gains the ``:0`` suffix when no attempt is present."""
    key, attempt = log_key_and_attempt("/u/train:debug/0")
    assert key == "/u/train:debug/0:0"
    assert attempt == 0

    key, attempt = log_key_and_attempt("/u/train:debug/0:2")
    assert key == "/u/train:debug/0:2"
    assert attempt == 2


def test_log_dir_glob_targets_task_container():
    assert _log_dir_glob("iris", "iris-job-0-abc-0") == "/var/log/pods/iris_iris-job-0-abc-0_*/task"


# ---------------------------------------------------------------------------
# Tailing a CRI log file end-to-end
# ---------------------------------------------------------------------------


def _write_cri_lines(path, lines: list[str]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


def _run_shipper_until_drained(shipper: LogShipper) -> threading.Thread:
    thread = threading.Thread(target=shipper.run)
    thread.start()
    return thread


def test_tail_ships_full_lines(tmp_path):
    """Lines written to the active 0.log are shipped as LogEntries with the right
    key, stream, level, and message."""
    log_dir = tmp_path / "iris_pod_uid" / "task"
    log_dir.mkdir(parents=True)
    active = log_dir / "0.log"
    _write_cri_lines(
        active,
        [
            "2026-06-24T12:00:00.000000000Z stdout F starting up",
            "2026-06-24T12:00:01.000000000Z stderr F it broke",
        ],
    )

    writer = FakeLogWriter()
    stop = threading.Event()
    shipper = LogShipper(writer, str(tmp_path / "*" / "task"), key="/u/job/0:1", attempt_id=1, stop=stop)
    thread = _run_shipper_until_drained(shipper)
    try:
        _wait_until(lambda: len(writer.entries()) >= 2)
    finally:
        stop.set()
        thread.join(timeout=5)

    entries = writer.entries()
    by_data = {e.data: e for e in entries}
    assert by_data["starting up"].source == "stdout"
    assert by_data["starting up"].level == str_to_log_level("INFO")
    assert by_data["starting up"].attempt_id == 1
    assert by_data["it broke"].source == "stderr"
    assert by_data["it broke"].level == str_to_log_level("ERROR")
    assert all(key == "/u/job/0:1" for key, _batch in writer.batches)


def test_tail_continues_across_rotation_without_duplicates(tmp_path):
    """When kubelet rotates 0.log (replaces the inode), the shipper reopens the
    new file and continues; already-shipped lines are not re-sent."""
    log_dir = tmp_path / "iris_pod_uid" / "task"
    log_dir.mkdir(parents=True)
    active = log_dir / "0.log"
    _write_cri_lines(active, ["2026-06-24T12:00:00.000000000Z stdout F line-before-rotation"])

    writer = FakeLogWriter()
    stop = threading.Event()
    shipper = LogShipper(writer, str(tmp_path / "*" / "task"), key="/u/job/0", attempt_id=0, stop=stop)
    thread = _run_shipper_until_drained(shipper)
    try:
        _wait_until(lambda: any(e.data == "line-before-rotation" for e in writer.entries()))

        # Rotate: move the old file aside and create a fresh 0.log (new inode).
        active.rename(log_dir / "0.log.20260624")
        _write_cri_lines(active, ["2026-06-24T12:00:02.000000000Z stdout F line-after-rotation"])

        _wait_until(lambda: any(e.data == "line-after-rotation" for e in writer.entries()))
    finally:
        stop.set()
        thread.join(timeout=5)

    messages = [e.data for e in writer.entries()]
    assert messages.count("line-before-rotation") == 1
    assert messages.count("line-after-rotation") == 1


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("condition not met within timeout")
