# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E test for log level parsing and filtering.

Verifies that:
1. Task callables using the unified logging format produce logs with level tags
2. The controller can filter logs by minimum level
"""

import logging
import time

import pytest
from iris.rpc import cluster_pb2, logging_pb2

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(60)]

# Log propagation traverses two 5-second intervals (worker monitor poll +
# controller heartbeat), so worst-case delivery is 10s+ before any CI
# scheduling overhead.  30s gives comfortable margin.
_LOG_POLL_DEADLINE_S = 30
_LOG_POLL_INTERVAL_S = 0.25


def _emit_multi_level_logs():
    """Callable that emits log lines at multiple levels."""
    import sys

    # Set up logging with the unified format (matching CALLABLE_RUNNER)
    _LEVEL_PREFIX = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class _Fmt(logging.Formatter):
        def format(self, record):
            record.levelprefix = _LEVEL_PREFIX.get(record.levelname, "?")
            return super().format(record)

    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_Fmt(fmt="%(levelprefix)s%(asctime)s %(name)s %(message)s", datefmt="%Y%m%d %H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    log = logging.getLogger("test.levels")
    log.debug("debug-marker")
    log.info("info-marker")
    log.warning("warning-marker")
    log.error("error-marker")


def _poll_task_logs(cluster, job, marker: str, *, task_index: int = 0) -> list:
    """Poll controller until task logs containing *marker* appear.

    Returns the full list of log entries once the marker is found.
    Raises AssertionError if the deadline expires.
    """
    task_id = job.job_id.task(task_index).to_wire()
    deadline = time.monotonic() + _LOG_POLL_DEADLINE_S
    entries: list = []
    polls = 0
    while time.monotonic() < deadline:
        request = cluster_pb2.Controller.GetTaskLogsRequest(id=task_id)
        response = cluster.controller_client.get_task_logs(request)
        entries = [e for batch in response.task_logs for e in batch.logs]
        polls += 1
        if any(marker in e.data for e in entries):
            return entries
        time.sleep(_LOG_POLL_INTERVAL_S)

    raise AssertionError(
        f"{marker!r} not found after {polls} polls over {_LOG_POLL_DEADLINE_S}s. "
        f"Got {len(entries)} entries: {[e.data for e in entries]}"
    )


def test_task_logs_have_level_field(cluster):
    """Task emitting logs at different levels gets level fields populated."""
    job = cluster.submit(_emit_multi_level_logs, "log-levels")
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED

    entries = _poll_task_logs(cluster, job, "info-marker")

    markers_found = {}
    for entry in entries:
        for marker in ("info-marker", "warning-marker", "error-marker"):
            if marker in entry.data:
                markers_found[marker] = entry.level

    assert markers_found["info-marker"] == logging_pb2.LOG_LEVEL_INFO
    assert markers_found.get("warning-marker") == logging_pb2.LOG_LEVEL_WARNING
    assert markers_found.get("error-marker") == logging_pb2.LOG_LEVEL_ERROR


def test_log_level_filter(cluster):
    """Controller filters logs by minimum level."""
    job = cluster.submit(_emit_multi_level_logs, "log-level-filter")
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED

    task_id = job.job_id.task(0).to_wire()

    # Wait for logs to propagate before testing the filter
    _poll_task_logs(cluster, job, "info-marker")

    # Now fetch with min_level=WARNING - should exclude INFO and DEBUG
    request = cluster_pb2.Controller.GetTaskLogsRequest(id=task_id, min_level="WARNING")
    response = cluster.controller_client.get_task_logs(request)
    filtered = [e for batch in response.task_logs for e in batch.logs]

    filtered_data = [e.data for e in filtered]
    assert any(
        "warning-marker" in d for d in filtered_data
    ), f"warning-marker missing from filtered logs: {filtered_data}"
    assert any("error-marker" in d for d in filtered_data), f"error-marker missing from filtered logs: {filtered_data}"
    assert not any(
        "info-marker" in d for d in filtered_data if d
    ), f"info-marker should be filtered out at WARNING level: {filtered_data}"
