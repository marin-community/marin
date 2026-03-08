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

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(120)]


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


@pytest.mark.xfail(reason="Heartbeat-based log propagation is unreliable in CI")
def test_task_logs_have_level_field(cluster):
    """Task emitting logs at different levels gets level fields populated."""
    job = cluster.submit(_emit_multi_level_logs, "log-levels")
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED

    # Fetch logs via controller RPC.
    # Logs are forwarded from worker to controller via heartbeats and flushed
    # to the log store outside the state lock, so they may arrive shortly after
    # the job status transitions to SUCCEEDED. Use a generous deadline since
    # heartbeat intervals and CI load can delay log propagation.
    task_id = job.job_id.task(0).to_wire()
    deadline = time.monotonic() + 60
    entries = []
    while time.monotonic() < deadline:
        request = cluster_pb2.Controller.GetTaskLogsRequest(id=task_id)
        response = cluster.controller_client.get_task_logs(request)
        entries = []
        for batch in response.task_logs:
            entries.extend(batch.logs)
        # Wait until we see some entries with markers
        if any("info-marker" in e.data for e in entries):
            break
        time.sleep(0.5)

    # Verify that entries with our markers have the level field populated
    markers_found = {}
    for entry in entries:
        for marker in ("info-marker", "warning-marker", "error-marker"):
            if marker in entry.data:
                markers_found[marker] = entry.level

    assert "info-marker" in markers_found, (
        f"info-marker not found in logs after 60s polling. " f"Got {len(entries)} entries: {[e.data for e in entries]}"
    )
    assert markers_found["info-marker"] == logging_pb2.LOG_LEVEL_INFO
    assert markers_found.get("warning-marker") == logging_pb2.LOG_LEVEL_WARNING
    assert markers_found.get("error-marker") == logging_pb2.LOG_LEVEL_ERROR


@pytest.mark.xfail(reason="Heartbeat-based log propagation is unreliable in CI")
def test_log_level_filter(cluster):
    """Controller filters logs by minimum level."""
    job = cluster.submit(_emit_multi_level_logs, "log-level-filter")
    status = cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED

    task_id = job.job_id.task(0).to_wire()

    # Wait for logs to propagate via heartbeat. Use generous deadline since
    # heartbeat intervals and CI load can delay log propagation.
    deadline = time.monotonic() + 60
    all_entries = []
    while time.monotonic() < deadline:
        request = cluster_pb2.Controller.GetTaskLogsRequest(id=task_id)
        response = cluster.controller_client.get_task_logs(request)
        all_entries = []
        for batch in response.task_logs:
            all_entries.extend(batch.logs)
        if any("info-marker" in e.data for e in all_entries):
            break
        time.sleep(0.5)

    assert any(
        "info-marker" in e.data for e in all_entries
    ), f"info-marker not found in logs after 60s polling. Got {len(all_entries)} entries."

    # Now fetch with min_level=WARNING - should exclude INFO and DEBUG
    request = cluster_pb2.Controller.GetTaskLogsRequest(id=task_id, min_level="WARNING")
    response = cluster.controller_client.get_task_logs(request)
    filtered = []
    for batch in response.task_logs:
        filtered.extend(batch.logs)

    # Should have warning and error markers but not info
    filtered_data = [e.data for e in filtered]
    assert any(
        "warning-marker" in d for d in filtered_data
    ), f"warning-marker missing from filtered logs: {filtered_data}"
    assert any("error-marker" in d for d in filtered_data), f"error-marker missing from filtered logs: {filtered_data}"
    # Info marker should be filtered out (it has level=INFO which is below WARNING)
    assert not any(
        "info-marker" in d for d in filtered_data if d
    ), f"info-marker should be filtered out at WARNING level: {filtered_data}"
