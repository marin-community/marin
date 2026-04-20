# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the log-service-address advertisement rewrite."""

from __future__ import annotations

import pytest

from iris.cluster.controller.controller import _advertised_log_service_address


@pytest.mark.parametrize(
    "address, controller_url, expected",
    [
        # Loopback (sidecar) addresses: substitute the controller VM's host
        # but keep the sidecar's port. Workers connect directly to the
        # sidecar — it listens on host networking and its port is reachable
        # on the same firewall scope as the controller.
        ("http://localhost:10002", "http://10.0.0.5:10000", "http://10.0.0.5:10002"),
        ("http://127.0.0.1:10002", "http://10.0.0.5:10000", "http://10.0.0.5:10002"),
        ("http://0.0.0.0:10002", "http://10.0.0.5:10000", "http://10.0.0.5:10002"),
        ("http://localhost:10002", "http://ctrl.example.com:10000", "http://ctrl.example.com:10002"),
        # External log server addresses pass through unchanged — workers
        # connect directly to the configured URL.
        (
            "http://logs.example.com:10002",
            "http://10.0.0.5:10000",
            "http://logs.example.com:10002",
        ),
        ("http://10.0.0.9:10002", "http://10.0.0.5:10000", "http://10.0.0.9:10002"),
    ],
)
def test_advertised_log_service_address(address: str, controller_url: str, expected: str) -> None:
    assert _advertised_log_service_address(address, controller_url=controller_url) == expected
