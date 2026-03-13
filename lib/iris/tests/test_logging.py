# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from iris.logging import (
    BufferedLogRecord,
    LevelPrefixFormatter,
    LogRingBuffer,
    RingBufferHandler,
    parse_log_level,
    slow_log,
)


@pytest.fixture
def ring_buffer():
    return LogRingBuffer(maxlen=10)


def test_ring_buffer_fifo_eviction(ring_buffer):
    """Oldest records are evicted when buffer is full."""
    for i in range(15):
        ring_buffer.append(
            BufferedLogRecord(seq=i, timestamp=float(i), level="INFO", logger_name="test", message=f"msg-{i}")
        )
    results = ring_buffer.query()
    assert len(results) == 10
    assert results[0].message == "msg-5"
    assert results[-1].message == "msg-14"


def test_ring_buffer_query_prefix(ring_buffer):
    """Query filters records by logger name prefix."""
    ring_buffer.append(BufferedLogRecord(seq=1, timestamp=0.0, level="INFO", logger_name="iris.controller", message="a"))
    ring_buffer.append(BufferedLogRecord(seq=2, timestamp=1.0, level="INFO", logger_name="iris.worker", message="b"))
    ring_buffer.append(
        BufferedLogRecord(seq=3, timestamp=2.0, level="INFO", logger_name="iris.controller.scheduler", message="c")
    )

    results = ring_buffer.query(prefix="iris.controller")
    assert len(results) == 2
    assert results[0].logger_name == "iris.controller"
    assert results[1].logger_name == "iris.controller.scheduler"


def test_ring_buffer_query_limit(ring_buffer):
    """Query respects limit parameter, returning most recent records."""
    for i in range(10):
        ring_buffer.append(
            BufferedLogRecord(seq=i, timestamp=float(i), level="INFO", logger_name="test", message=f"msg-{i}")
        )
    results = ring_buffer.query(limit=3)
    assert len(results) == 3
    assert results[0].message == "msg-7"


def test_handler_captures_log_records():
    """RingBufferHandler captures formatted log records from Python logging."""
    buf = LogRingBuffer()
    handler = RingBufferHandler(buf)
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))

    logger = logging.getLogger("iris.test.handler_test")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        logger.info("hello world")
        results = buf.query(prefix="iris.test.handler_test")
        assert len(results) == 1
        assert "hello world" in results[0].message
        assert results[0].level == "INFO"
        assert results[0].logger_name == "iris.test.handler_test"
    finally:
        logger.removeHandler(handler)


def test_slow_log_emits_warning_when_slow(caplog):
    """slow_log emits a WARNING when the block exceeds the threshold."""
    log = logging.getLogger("iris.test.slow_log")
    with caplog.at_level(logging.WARNING, logger="iris.test.slow_log"):
        with slow_log(log, "test-op", threshold_ms=0):
            pass
    assert any("Slow test-op" in r.message for r in caplog.records)


def test_slow_log_silent_when_fast(caplog):
    """slow_log emits nothing when the block completes within budget."""
    log = logging.getLogger("iris.test.slow_log")
    with caplog.at_level(logging.DEBUG, logger="iris.test.slow_log"):
        with slow_log(log, "fast-op", threshold_ms=60_000):
            pass
    assert not any("Slow" in r.message for r in caplog.records)


def test_configure_logging_captures_records():
    """configure_logging installs a ring buffer handler that captures log records."""
    import iris.logging as iris_logging

    iris_logging._configured = False
    old_handlers = logging.getLogger().handlers[:]
    try:
        buf = iris_logging.configure_logging(level=logging.DEBUG)
        logger = logging.getLogger("iris.test.configure_test")
        logger.debug("cfg-test-msg")
        results = buf.query(prefix="iris.test.configure_test")
        assert len(results) >= 1
        assert "cfg-test-msg" in results[-1].message
    finally:
        iris_logging._configured = False
        root = logging.getLogger()
        root.handlers.clear()
        root.handlers.extend(old_handlers)


@pytest.mark.parametrize(
    "line,expected",
    [
        # Single-letter prefix format (the only supported format)
        ("I20260306 12:44:05 iris.worker starting up", "INFO"),
        ("E20260306 12:44:05 iris.worker failed", "ERROR"),
        ("W20260306 12:44:05 iris.worker slow", "WARNING"),
        ("D20260306 12:44:05 iris.worker debug", "DEBUG"),
        ("C20260306 12:44:05 iris.worker critical", "CRITICAL"),
        # Other formats are not recognized (we've normalized to single-letter prefix)
        ("[INFO] some message", None),
        ("2025-01-01 12:00:00 - INFO - message", None),
        # No level detected
        ("just some random output", None),
        ("", None),
        ("12345", None),
    ],
)
def test_parse_log_level(line, expected):
    """parse_log_level detects levels from the single-letter prefix format only."""
    assert parse_log_level(line) == expected


def test_level_prefix_formatter_produces_expected_format():
    """LevelPrefixFormatter prepends a single-letter level prefix."""
    formatter = LevelPrefixFormatter(
        fmt="%(levelprefix)s%(asctime)s %(name)s %(message)s",
        datefmt="%Y%m%d %H:%M:%S",
    )
    record = logging.LogRecord(
        name="iris.test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="hello",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    assert output.startswith("I"), f"Expected 'I' prefix, got: {output}"
    assert "iris.test" in output
    assert "hello" in output


def test_configure_logging_uses_level_prefix_format():
    """configure_logging produces log lines with single-letter level prefix."""

    import iris.logging as iris_logging

    iris_logging._configured = False
    old_handlers = logging.getLogger().handlers[:]
    try:
        iris_logging.configure_logging(level=logging.DEBUG)
        root = logging.getLogger()
        # Find the stderr handler and capture its output
        for h in root.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, RingBufferHandler):
                record = logging.LogRecord(
                    name="test.fmt",
                    level=logging.WARNING,
                    pathname="",
                    lineno=0,
                    msg="test-msg",
                    args=(),
                    exc_info=None,
                )
                output = h.format(record)
                assert output.startswith("W"), f"Expected 'W' prefix, got: {output}"
                assert "test.fmt" in output
                assert "test-msg" in output
                break
        else:
            pytest.fail("No StreamHandler found after configure_logging")
    finally:
        iris_logging._configured = False
        root = logging.getLogger()
        root.handlers.clear()
        root.handlers.extend(old_handlers)
