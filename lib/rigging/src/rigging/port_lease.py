# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Process-scoped claims for TCP ports shared by host-network processes."""

import errno
import fcntl
import os
import socket
import sys
from collections.abc import Collection, Iterable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from random import Random

DYNAMIC_TCP_PORTS = range(20_000, 24_000)
PORT_ENV_PREFIX = "IRIS_PORT_"
_MAX_TCP_PORT = 65_535
_PROCESS_LEASES = ExitStack()


def port_env_var(name: str) -> str:
    """Convert a port name such as `dashboard` to `IRIS_PORT_DASHBOARD`."""
    return f"{PORT_ENV_PREFIX}{name.upper()}"


def configured_tcp_port(name: str) -> int:
    """Read a named port from the environment, defaulting to zero."""
    key = port_env_var(name)
    port = int(os.environ.get(key, "0"))
    if not 0 <= port <= _MAX_TCP_PORT:
        raise ValueError(f"{key} must be a TCP port in 0-{_MAX_TCP_PORT}")
    return port


def _probe_tcp_port(port: int) -> None:
    # Wildcard binds detect conflicts on individual interfaces. The probe
    # sockets close without listening.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("", port))
    if socket.has_ipv6:
        try:
            with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                probe.bind(("::", port))
        except OSError as error:
            if error.errno not in (errno.EAFNOSUPPORT, errno.EPROTONOSUPPORT, errno.EADDRNOTAVAIL):
                raise


@contextmanager
def claim_tcp_port(port: int) -> Iterator[None]:
    """Claim a specific port, including a port already bound by an adopted task."""
    if sys.platform == "linux":
        # Abstract names coordinate even when host-network pods have separate filesystems.
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as claim:
            claim.bind(f"\0rigging-tcp-port-{port}")
            yield
        return

    directory = Path("/tmp") / f"rigging-tcp-ports-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    # Keep the file so every contender locks the same inode.
    with (directory / str(port)).open("a") as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


@contextmanager
def lease_tcp_ports(
    count: int = 1,
    *,
    excluded: Collection[int] = (),
    candidates: Iterable[int] | None = None,
) -> Iterator[list[int]]:
    """Claim unbound host TCP ports until context exit or process death.

    Dynamic leases use the shared 20000-23999 range and skip ports in use.
    The TCP listener binds its own socket. Hosts whose ephemeral range
    overlaps the lease range can reserve it from kernel assignment.
    """
    if count < 1:
        raise ValueError("count must be positive")

    if candidates is None:
        shuffled = list(DYNAMIC_TCP_PORTS)
        Random().shuffle(shuffled)
        candidates = shuffled

    with ExitStack() as leases:
        selected: list[int] = []
        for port in candidates:
            if port in excluded:
                continue
            try:
                with ExitStack() as attempt:
                    attempt.enter_context(claim_tcp_port(port))
                    _probe_tcp_port(port)
                    leases.enter_context(attempt.pop_all())
            except OSError as error:
                if error.errno in (errno.EADDRINUSE, errno.EACCES, errno.EAGAIN):
                    continue
                raise
            selected.append(port)
            if len(selected) == count:
                yield selected
                return

        raise RuntimeError(f"Could not lease {count} unbound TCP ports")


@contextmanager
def lease_tcp_port(port: int = 0, *, name: str | None = None, excluded: Collection[int] = ()) -> Iterator[int]:
    """Use a named allocation, an explicit port, or a dynamic TCP lease.

    A nonzero ``IRIS_PORT_<NAME>`` takes precedence over ``port``. Assigned and
    explicitly specified ports are used as-is. Zero requests a lease.
    """
    if name is not None:
        port = configured_tcp_port(name) or port
    if port:
        yield port
        return
    with lease_tcp_ports(excluded=excluded) as ports:
        yield ports[0]


def allocate_tcp_port(port: int = 0, *, name: str | None = None, excluded: Collection[int] = ()) -> int:
    """Resolve a port like `lease_tcp_port` and hold any claim until process exit."""
    return _PROCESS_LEASES.enter_context(lease_tcp_port(port, name=name, excluded=excluded))


def allocate_tcp_ports(count: int, *, excluded: Collection[int] = ()) -> list[int]:
    """Allocate dynamic ports and hold their claims until process exit."""
    return _PROCESS_LEASES.enter_context(lease_tcp_ports(count, excluded=excluded))


@contextmanager
def lease_tcp_port_block(count: int) -> Iterator[int]:
    """Lease a contiguous block and yield its first port."""
    if count < 1:
        raise ValueError("count must be positive")
    bases = list(range(DYNAMIC_TCP_PORTS.start, DYNAMIC_TCP_PORTS.stop - count + 1))
    Random().shuffle(bases)
    with ExitStack() as leases:
        for base in bases:
            block = range(base, base + count)
            try:
                leases.enter_context(lease_tcp_ports(count, candidates=block))
            except RuntimeError:
                continue
            yield base
            return
    raise RuntimeError(f"Could not lease a contiguous block of {count} TCP ports")
