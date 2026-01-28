# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import pytest

from iris.logging import BufferedLogRecord, LogRingBuffer, RingBufferHandler


@pytest.fixture
def ring_buffer():
    return LogRingBuffer(maxlen=10)


def test_ring_buffer_fifo_eviction(ring_buffer):
    """Oldest records are evicted when buffer is full."""
    for i in range(15):
        ring_buffer.append(BufferedLogRecord(timestamp=float(i), level="INFO", logger_name="test", message=f"msg-{i}"))
    results = ring_buffer.query()
    assert len(results) == 10
    assert results[0].message == "msg-5"
    assert results[-1].message == "msg-14"


def test_ring_buffer_query_prefix(ring_buffer):
    """Query filters records by logger name prefix."""
    ring_buffer.append(BufferedLogRecord(0.0, "INFO", "iris.controller", "a"))
    ring_buffer.append(BufferedLogRecord(1.0, "INFO", "iris.worker", "b"))
    ring_buffer.append(BufferedLogRecord(2.0, "INFO", "iris.controller.scheduler", "c"))

    results = ring_buffer.query(prefix="iris.controller")
    assert len(results) == 2
    assert results[0].logger_name == "iris.controller"
    assert results[1].logger_name == "iris.controller.scheduler"


def test_ring_buffer_query_limit(ring_buffer):
    """Query respects limit parameter, returning most recent records."""
    for i in range(10):
        ring_buffer.append(BufferedLogRecord(float(i), "INFO", "test", f"msg-{i}"))
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
