# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Allowlist HTTP CONNECT proxy used to give sandboxed pytest runs selective egress.

The SWE-rebench images frequently need to fetch packages or test fixtures
from a small set of public registries (PyPI, crates.io, npm, GitHub, the
Debian package mirrors). Giving the sandbox arbitrary outbound network
access is unsafe, and stripping network access entirely makes ~half the
images unusable.

This module starts a tiny stdlib-only HTTP CONNECT proxy that allows
``CONNECT host:port HTTP/1.1`` requests where ``host`` matches an
allowlisted DNS name (exact match or ``*.suffix`` wildcard). Anything
else gets a 403. The proxy is hostname-only (it doesn't peek into TLS),
which is the right granularity for ``pip install`` / ``cargo fetch`` /
``git clone`` over HTTPS — those tools take ``HTTPS_PROXY`` and use
CONNECT.

The proxy runs in a thread inside the Zephyr worker process. The map
function points the sandboxed container at it via ``HTTPS_PROXY`` and
``HTTP_PROXY`` env vars. The runsc network stack reaches the proxy over
the worker's loopback because the runsc rootless network bridge has
host loopback access via the worker's host gateway.

Default allowlist covers the registries pip/cargo/npm/git use most.
Override via ``MARIN_PROXY_ALLOWLIST`` (colon-separated host patterns).
"""

from __future__ import annotations

import logging
import os
import re
import socket
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Default allowlist. Each entry is either an exact hostname or a
# ``*.suffix`` glob. Match is case-insensitive.
DEFAULT_ALLOWLIST: tuple[str, ...] = (
    # Python
    "pypi.org",
    "*.pypi.org",
    "files.pythonhosted.org",
    # Rust
    "crates.io",
    "*.crates.io",
    "static.crates.io",
    "index.crates.io",
    # Node
    "registry.npmjs.org",
    # Java / Maven
    "repo.maven.apache.org",
    "repo1.maven.org",
    # Debian / Ubuntu apt
    "deb.debian.org",
    "security.debian.org",
    "archive.ubuntu.com",
    "security.ubuntu.com",
    # GitHub (release/raw + git over HTTPS)
    "github.com",
    "*.github.com",
    "objects.githubusercontent.com",
    "raw.githubusercontent.com",
    "codeload.github.com",
    # Misc
    "ftp.debian.org",
)


def _compile_pattern(pattern: str) -> re.Pattern[str]:
    """Compile a glob-ish hostname pattern to a regex.

    ``*.example.com`` requires at least one subdomain label and does NOT
    match the bare apex ``example.com``. Add ``example.com`` separately if
    you also want the apex.
    """
    pattern = pattern.strip().lower()
    if pattern.startswith("*."):
        suffix = re.escape(pattern[2:])
        return re.compile(rf"^([a-z0-9-]+\.)+{suffix}$", re.IGNORECASE)
    return re.compile(rf"^{re.escape(pattern)}$", re.IGNORECASE)


@dataclass
class ProxyConfig:
    allowlist: tuple[str, ...] = DEFAULT_ALLOWLIST
    bind_host: str = "0.0.0.0"
    bind_port: int = 0  # 0 = pick a free port
    max_connections: int = 256
    _patterns: list[re.Pattern[str]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._patterns = [_compile_pattern(p) for p in self.allowlist]

    def host_allowed(self, host: str) -> bool:
        host = host.lower().rstrip(".")
        return any(p.match(host) for p in self._patterns)


@dataclass
class ProxyHandle:
    """Handle to a running proxy server. Stop with .shutdown()."""

    host: str
    port: int
    url: str
    _server_socket: socket.socket
    _thread: threading.Thread
    _stop: threading.Event
    config: ProxyConfig

    def shutdown(self) -> None:
        self._stop.set()
        try:
            self._server_socket.close()
        except OSError:
            pass
        self._thread.join(timeout=5.0)


def _parse_connect_target(line: str) -> tuple[str, int] | None:
    """Parse the request line of an HTTP CONNECT request.

    Returns (host, port) on success, None on parse error.
    """
    # Expected: "CONNECT host:port HTTP/1.1"
    parts = line.split()
    if len(parts) != 3 or parts[0].upper() != "CONNECT":
        return None
    target = parts[1]
    if ":" not in target:
        return None
    host, _, port_str = target.rpartition(":")
    try:
        port = int(port_str)
    except ValueError:
        return None
    if not host or not (1 <= port <= 65535):
        return None
    # Strip surrounding brackets for IPv6 literals.
    host = host.strip("[]")
    return host, port


def _read_request_headers(client: socket.socket) -> bytes:
    """Read until we hit \\r\\n\\r\\n. Returns the raw header block."""
    buf = bytearray()
    while b"\r\n\r\n" not in buf:
        chunk = client.recv(4096)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > 16384:
            break
    return bytes(buf)


def _shovel(src: socket.socket, dst: socket.socket) -> None:
    """Copy bytes one direction until either side closes."""
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _handle_client(client: socket.socket, addr: tuple[str, int], config: ProxyConfig) -> None:
    try:
        client.settimeout(30.0)
        headers = _read_request_headers(client)
        if not headers:
            return
        first_line, _, _ = headers.partition(b"\r\n")
        target = _parse_connect_target(first_line.decode("latin-1", errors="replace"))
        if target is None:
            client.sendall(b"HTTP/1.1 400 Bad Request\r\nProxy-Agent: marin-swetrace\r\n\r\n")
            logger.warning("proxy: rejected non-CONNECT or malformed request from %s", addr)
            return
        host, port = target
        if not config.host_allowed(host):
            client.sendall(b"HTTP/1.1 403 Forbidden\r\nProxy-Agent: marin-swetrace\r\n\r\n")
            logger.info("proxy: DENY %s:%d (not in allowlist)", host, port)
            return
        try:
            upstream = socket.create_connection((host, port), timeout=15.0)
        except OSError as e:
            client.sendall(b"HTTP/1.1 502 Bad Gateway\r\nProxy-Agent: marin-swetrace\r\n\r\n")
            logger.warning("proxy: upstream connect failed for %s:%d: %s", host, port, e)
            return

        client.sendall(b"HTTP/1.1 200 Connection Established\r\nProxy-Agent: marin-swetrace\r\n\r\n")
        logger.info("proxy: ALLOW %s:%d", host, port)

        # Bidirectional copy via two threads. The simpler "select on both"
        # approach also works but threads make the shutdown path obvious.
        t1 = threading.Thread(target=_shovel, args=(client, upstream), daemon=True)
        t2 = threading.Thread(target=_shovel, args=(upstream, client), daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        try:
            upstream.close()
        except OSError:
            pass
    finally:
        try:
            client.close()
        except OSError:
            pass


def _serve_loop(server_sock: socket.socket, stop: threading.Event, config: ProxyConfig) -> None:
    server_sock.settimeout(0.5)
    while not stop.is_set():
        try:
            client, addr = server_sock.accept()
        except TimeoutError:
            continue
        except OSError:
            break
        threading.Thread(
            target=_handle_client,
            args=(client, addr, config),
            daemon=True,
        ).start()


def start_proxy(config: ProxyConfig | None = None) -> ProxyHandle:
    """Start the proxy in a background thread and return a handle."""
    if config is None:
        config = ProxyConfig(allowlist=_load_allowlist_from_env())

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((config.bind_host, config.bind_port))
    server_sock.listen(config.max_connections)

    actual_host, actual_port = server_sock.getsockname()[:2]

    stop = threading.Event()
    thread = threading.Thread(
        target=_serve_loop,
        args=(server_sock, stop, config),
        name="marin-swetrace-proxy",
        daemon=True,
    )
    thread.start()

    return ProxyHandle(
        host=actual_host,
        port=actual_port,
        url=f"http://{actual_host}:{actual_port}",
        _server_socket=server_sock,
        _thread=thread,
        _stop=stop,
        config=config,
    )


def _load_allowlist_from_env() -> tuple[str, ...]:
    raw = os.environ.get("MARIN_PROXY_ALLOWLIST")
    if not raw:
        return DEFAULT_ALLOWLIST
    parts: Iterable[str] = (p.strip() for p in raw.split(":") if p.strip())
    return tuple(parts)
