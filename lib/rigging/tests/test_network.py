# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import socket
from types import SimpleNamespace

import psutil
import pytest
from rigging.network import interface_for_ipv4


@pytest.mark.parametrize("address", ["127.0.0.1", "localhost"])
def test_interface_for_ipv4_finds_loopback(address):
    interface = interface_for_ipv4(address)
    assert interface in dict(socket.if_nameindex()).values()
    assert "loopback" in psutil.net_if_stats()[interface].flags.split(",")


def test_interface_for_ipv4_finds_secondary_address(monkeypatch):
    monkeypatch.setattr(
        psutil,
        "net_if_addrs",
        lambda: {
            "lo0": [SimpleNamespace(family=socket.AF_INET, address="127.0.0.1")],
            "en0": [
                SimpleNamespace(family=socket.AF_INET6, address="fe80::1"),
                SimpleNamespace(family=socket.AF_INET, address="192.0.2.1"),
                SimpleNamespace(family=socket.AF_INET, address="192.0.2.2"),
            ],
        },
    )
    assert interface_for_ipv4("192.0.2.2") == "en0"
    with pytest.raises(RuntimeError, match="No local network interface owns IPv4 address"):
        interface_for_ipv4("192.0.2.3")
