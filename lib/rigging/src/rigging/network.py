# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small helpers for inspecting the local host network."""

import socket

import psutil


def interface_for_ipv4(address: str) -> str:
    """Return the local interface that owns ``address``."""
    resolved_address = socket.gethostbyname(address)
    for interface, addresses in psutil.net_if_addrs().items():
        for interface_address in addresses:
            if interface_address.family == socket.AF_INET and interface_address.address == resolved_address:
                return interface
    raise RuntimeError(f"No local network interface owns IPv4 address {address!r}")
