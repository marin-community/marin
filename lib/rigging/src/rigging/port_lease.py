# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Allocate TCP ports before a server can bind them."""

import errno
import os
import socket
import sys
import tempfile
from collections.abc import Callable, Collection, Iterator
from contextlib import ExitStack, contextmanager
from itertools import chain
from pathlib import Path
from random import randrange

from filelock import FileLock, Timeout

DYNAMIC_TCP_PORTS = range(20_000, 24_000)
PORT_ENV_PREFIX = "IRIS_PORT_"
_MAX_TCP_PORT = 65_535
_PROCESS_RELEASES: list[Callable[[], None]] = []


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
    # Detect conflicts on every interface. Neither socket calls listen().
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


def claim_tcp_port(port: int) -> Callable[[], None]:
    """Claim a port and return its release callback. The port may be bound."""
    if sys.platform == "linux":
        # Abstract names coordinate even when host-network pods have separate filesystems.
        claim = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            claim.bind(f"\0rigging-tcp-port-{port}")
        except BaseException:
            claim.close()
            raise
        return claim.close

    directory = Path(tempfile.gettempdir()) / f"rigging-tcp-ports-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    lock = FileLock(directory / str(port), thread_local=False)
    lock.acquire(timeout=0)
    return lock.release


def try_claim_free_tcp_port(port: int) -> Callable[[], None] | None:
    """Claim an unbound port, returning its release callback if available."""
    release = None
    try:
        release = claim_tcp_port(port)
        _probe_tcp_port(port)
    except Timeout:
        return None
    except BaseException as error:
        if release is not None:
            release()
        if isinstance(error, OSError) and error.errno == errno.EADDRINUSE:
            return None
        raise
    return release


def _port_candidates(ports: range) -> Iterator[int]:
    if ports:
        first = randrange(ports.start, ports.stop)
        yield from chain(range(first, ports.stop), range(ports.start, first))


def _claim_dynamic_tcp_port(excluded: Collection[int]) -> tuple[int, Callable[[], None]]:
    for port in _port_candidates(DYNAMIC_TCP_PORTS):
        if port not in excluded and (release := try_claim_free_tcp_port(port)) is not None:
            return port, release
    raise RuntimeError(f"No free TCP ports in {DYNAMIC_TCP_PORTS.start}-{DYNAMIC_TCP_PORTS.stop - 1}")


@contextmanager
def lease_tcp_port(port: int = 0, *, excluded: Collection[int] = ()) -> Iterator[int]:
    """Use an explicit port or hold a dynamic claim until context exit."""
    if port:
        yield port
        return
    port, release = _claim_dynamic_tcp_port(excluded)
    try:
        yield port
    finally:
        release()


def allocate_tcp_port(port: int = 0, *, name: str | None = None, excluded: Collection[int] = ()) -> int:
    """Use a named or explicit port, or claim a dynamic port until process exit."""
    if name is not None:
        port = configured_tcp_port(name) or port
    if port:
        return port
    port, release = _claim_dynamic_tcp_port(excluded)
    _PROCESS_RELEASES.append(release)
    return port


@contextmanager
def lease_tcp_ports(count: int, *, excluded: Collection[int] = ()) -> Iterator[list[int]]:
    """Hold multiple dynamic claims until context exit."""
    with ExitStack() as leases:
        yield [leases.enter_context(lease_tcp_port(excluded=excluded)) for _ in range(count)]


@contextmanager
def lease_tcp_port_block(count: int) -> Iterator[int]:
    """Lease a contiguous block and yield its first port."""
    if count < 1:
        raise ValueError("count must be positive")
    bases = range(DYNAMIC_TCP_PORTS.start, DYNAMIC_TCP_PORTS.stop - count + 1)
    for base in _port_candidates(bases):
        with ExitStack() as leases:
            for port in range(base, base + count):
                release = try_claim_free_tcp_port(port)
                if release is None:
                    break
                leases.callback(release)
            else:
                yield base
                return
    raise RuntimeError(f"Could not lease a contiguous block of {count} TCP ports")
