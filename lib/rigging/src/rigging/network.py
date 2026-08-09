# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small helpers for inspecting the local host network."""

import fcntl
import socket
import struct

_SIOCGIFADDR = 0x8915


def interface_for_ipv4(address: str) -> str:
    """Return the local interface that owns ``address``."""
    packed_address = socket.inet_aton(socket.gethostbyname(address))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for _, interface in socket.if_nameindex():
            request = struct.pack("256s", interface.encode()[:15])
            try:
                interface_address = fcntl.ioctl(sock.fileno(), _SIOCGIFADDR, request)[20:24]
            except OSError:
                continue
            if interface_address == packed_address:
                return interface
    raise RuntimeError(f"No local network interface owns IPv4 address {address!r}")
