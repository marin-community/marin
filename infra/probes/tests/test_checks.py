# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe behaviour: fakes at the RPC boundary, real ProbeResult assembly."""

from __future__ import annotations

from unittest.mock import MagicMock

from probes.checks.controller_ping import ControllerPing
from probes.probe import ErrorClass, ProbeOutcome


def test_controller_ping_success_records_worker_count():
    client = MagicMock()
    client.list_workers.return_value = [MagicMock(), MagicMock(), MagicMock()]
    result = ControllerPing(client).run(deadline_seconds=5.0)
    assert result.outcome is ProbeOutcome.SUCCESS
    assert result.extras == {"worker_count": 3}


def test_controller_ping_timeout_classified_as_timeout():
    client = MagicMock()
    client.list_workers.side_effect = TimeoutError("deadline")
    result = ControllerPing(client).run(deadline_seconds=5.0)
    assert result.outcome is ProbeOutcome.TIMEOUT
    assert result.error_class is ErrorClass.TIMEOUT


def test_controller_ping_generic_failure_is_remote_error():
    client = MagicMock()
    client.list_workers.side_effect = RuntimeError("boom")
    result = ControllerPing(client).run(deadline_seconds=5.0)
    assert result.outcome is ProbeOutcome.REMOTE_ERROR
    assert result.error_class is ErrorClass.RPC_ERROR
    assert "RuntimeError: boom" in (result.error_detail or "")


def test_controller_ping_connection_error_classified_as_connect():
    client = MagicMock()
    client.list_workers.side_effect = ConnectionError("no route")
    result = ControllerPing(client).run(deadline_seconds=5.0)
    assert result.outcome is ProbeOutcome.REMOTE_ERROR
    assert result.error_class is ErrorClass.CONNECT_ERROR
