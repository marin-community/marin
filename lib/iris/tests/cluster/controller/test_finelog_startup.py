# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller startup must tolerate Finelog being unreachable.

``LogClient.get_table`` issues a synchronous ``RegisterTable`` RPC. If Finelog
is down, the controller should still come up — leaving the affected Table
handle as ``None`` — and retry registration in the background.
"""

import time

import pytest
from finelog.client import LogClient

pytestmark = pytest.mark.timeout(15)


def test_controller_starts_when_finelog_table_registration_fails(make_controller, monkeypatch):
    real_get_table = LogClient.get_table
    call_count = {"n": 0}

    def flaky_get_table(self, namespace, schema):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ConnectionError("finelog unreachable (simulated)")
        return real_get_table(self, namespace, schema)

    monkeypatch.setattr(LogClient, "get_table", flaky_get_table)

    controller = make_controller()

    # Initial registration failed, so the service has no profile table yet.
    assert controller._service._profile_table is None

    # Background retry installs it on the next attempt (backoff starts ~1s).
    deadline = time.monotonic() + 5.0
    while controller._service._profile_table is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert controller._service._profile_table is not None
