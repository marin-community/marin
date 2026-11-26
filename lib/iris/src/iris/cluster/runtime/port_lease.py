# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Process-scoped TCP port claims shared by host-network tasks."""

import errno
import socket
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from random import SystemRandom

MAX_CANDIDATES_PER_PORT = 64


def host_tcp_port_candidates(excluded: Collection[int], count: int) -> list[int]:
    """Choose unprivileged ports outside the host's ephemeral and reserved ranges."""
    if count < 1:
        raise ValueError("count must be positive")

    # A native bind(0) must not take a claimed number between our TCP probe and
    # the listener's bind. Host-network pods share these kernel port settings.
    network = Path("/proc/sys/net/ipv4")
    low, high = map(int, (network / "ip_local_port_range").read_text().split())
    unprivileged = max(1024, int((network / "ip_unprivileged_port_start").read_text()))
    unavailable = set(excluded)
    for item in (network / "ip_local_reserved_ports").read_text().strip().split(","):
        if item:
            start, _, end = item.partition("-")
            unavailable.update(range(int(start), int(end or start) + 1))

    available = [port for port in range(unprivileged, 65536) if not low <= port <= high and port not in unavailable]
    return SystemRandom().sample(available, min(len(available), count * MAX_CANDIDATES_PER_PORT))


@contextmanager
def lease_tcp_port(candidates: Iterator[int]) -> Iterator[int]:
    """Claim an unbound TCP port until the context exits or its process dies.

    The claim uses a Linux abstract Unix socket, which is visible across pods
    sharing a network namespace even when their filesystems differ.
    """
    for value in islice(candidates, MAX_CANDIDATES_PER_PORT):
        lease = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            # Keep the existing SkyRL name so running launchers coordinate with
            # this shared Iris implementation during a rolling deployment.
            lease.bind(f"\0marin-skyrl-port-{value}")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("", value))
        except OSError as error:
            lease.close()
            if error.errno in (errno.EADDRINUSE, errno.EACCES):
                continue
            raise

        try:
            yield value
        finally:
            lease.close()
        return

    raise RuntimeError("Could not lease an unprivileged TCP port outside the kernel ephemeral range")
