# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the swe_rebench_trace allowlist proxy."""

from __future__ import annotations

import socket
import threading

import pytest

from experiments.swe_rebench_trace.proxy import (
    ProxyConfig,
    _compile_pattern,
    _parse_connect_target,
    start_proxy,
)

# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern,host,expected",
    [
        ("pypi.org", "pypi.org", True),
        ("pypi.org", "evil.pypi.org", False),
        ("*.pypi.org", "files.pypi.org", True),
        ("*.pypi.org", "pypi.org", False),
        ("*.crates.io", "static.crates.io", True),
        ("*.crates.io", "index.crates.io", True),
        ("*.crates.io", "evil.com", False),
        ("github.com", "github.com", True),
        ("github.com", "GITHUB.com", True),  # case-insensitive
    ],
)
def test_pattern_match(pattern: str, host: str, expected: bool):
    pat = _compile_pattern(pattern)
    assert (pat.match(host) is not None) == expected


def test_proxy_config_default_allowlist_includes_pypi():
    cfg = ProxyConfig()
    assert cfg.host_allowed("pypi.org")
    assert cfg.host_allowed("files.pythonhosted.org")
    assert cfg.host_allowed("static.crates.io")
    assert not cfg.host_allowed("evil.example.com")


def test_proxy_config_default_bind_is_loopback():
    """The default bind must be loopback so the proxy isn't exposed beyond the worker."""
    cfg = ProxyConfig()
    assert cfg.bind_host == "127.0.0.1"


# ---------------------------------------------------------------------------
# CONNECT request parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line,expected",
    [
        ("CONNECT pypi.org:443 HTTP/1.1", ("pypi.org", 443)),
        ("connect example.com:8080 HTTP/1.0", ("example.com", 8080)),
        ("GET / HTTP/1.1", None),  # not CONNECT
        ("CONNECT bare HTTP/1.1", None),  # missing port
        ("CONNECT host:notaport HTTP/1.1", None),
        ("CONNECT [::1]:443 HTTP/1.1", ("::1", 443)),
    ],
)
def test_parse_connect_target(line: str, expected):
    assert _parse_connect_target(line) == expected


# ---------------------------------------------------------------------------
# End-to-end: start proxy, send request, observe behavior
# ---------------------------------------------------------------------------


def _connect_through_proxy(proxy_host: str, proxy_port: int, target: str) -> tuple[int, str]:
    """Send a CONNECT request through the proxy and return (status_code, status_line)."""
    s = socket.create_connection((proxy_host, proxy_port), timeout=5.0)
    try:
        s.sendall(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
        s.settimeout(5.0)
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
        first_line = buf.split(b"\r\n", 1)[0].decode("latin-1", errors="replace")
        parts = first_line.split(" ", 2)
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1]), first_line
        return -1, first_line
    finally:
        s.close()


@pytest.fixture
def loopback_proxy():
    """Start a proxy on 127.0.0.1 with a tight allowlist for testing.

    The allowlist includes ``localhost`` so we can test the allow path
    against a real upstream we control without paying for DNS.
    """
    cfg = ProxyConfig(
        allowlist=("localhost", "allowed.example.com", "*.allowed.example.com"),
        bind_host="127.0.0.1",
    )
    handle = start_proxy(cfg)
    yield handle
    handle.shutdown()


def test_proxy_rejects_non_allowlisted_host(loopback_proxy):
    status, _ = _connect_through_proxy(
        loopback_proxy.host,
        loopback_proxy.port,
        "evil.example.com:443",
    )
    assert status == 403


def test_proxy_rejects_malformed_request(loopback_proxy):
    s = socket.create_connection((loopback_proxy.host, loopback_proxy.port), timeout=5.0)
    try:
        s.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        s.settimeout(5.0)
        first = s.recv(4096).split(b"\r\n", 1)[0]
    finally:
        s.close()
    assert b"400" in first


def test_proxy_allows_localhost_and_proxies_bytes():
    """Allowlisted host: start a tiny TCP echo upstream, tunnel through proxy, verify bytes flow."""
    # Upstream echo server.
    upstream = socket.socket()
    upstream.bind(("127.0.0.1", 0))
    upstream.listen(1)
    upstream_port = upstream.getsockname()[1]

    def echo_once():
        conn, _ = upstream.accept()
        try:
            data = conn.recv(64)
            if data:
                conn.sendall(b"echo:" + data)
        finally:
            conn.close()

    upstream_thread = threading.Thread(target=echo_once, daemon=True)
    upstream_thread.start()

    cfg = ProxyConfig(allowlist=("localhost",), bind_host="127.0.0.1")
    proxy = start_proxy(cfg)
    try:
        client = socket.create_connection((proxy.host, proxy.port), timeout=5.0)
        try:
            client.sendall(f"CONNECT localhost:{upstream_port} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode())
            client.settimeout(5.0)
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = client.recv(4096)
                if not chunk:
                    break
                buf += chunk
            assert b"200 Connection Established" in buf, buf

            client.sendall(b"hello")
            reply = b""
            client.settimeout(5.0)
            while len(reply) < len(b"echo:hello"):
                chunk = client.recv(4096)
                if not chunk:
                    break
                reply += chunk
            assert reply == b"echo:hello"
        finally:
            client.close()
    finally:
        proxy.shutdown()
        upstream.close()
        upstream_thread.join(timeout=2.0)


def test_proxy_handle_shutdown_is_idempotent(loopback_proxy):
    loopback_proxy.shutdown()
    loopback_proxy.shutdown()  # second call should not raise


def test_proxy_header_deadline_disconnects_slow_client(loopback_proxy):
    """A client that connects but never sends headers must be disconnected by the deadline."""
    import time as _time

    s = socket.create_connection((loopback_proxy.host, loopback_proxy.port), timeout=10.0)
    s.settimeout(10.0)
    started = _time.monotonic()
    try:
        # Don't send anything; wait for the proxy to give up.
        # The proxy's header read deadline is 5s; recv() will return when the
        # proxy closes its end.
        try:
            data = s.recv(4096)
        except (TimeoutError, OSError):
            data = b""
    finally:
        s.close()
    elapsed = _time.monotonic() - started
    # Must be bounded by the header deadline (5s) plus a small slack.
    assert elapsed < 8.0, f"slow client held the connection for {elapsed:.1f}s"
    # The proxy may have sent a 400, or may have closed without writing.
    if data:
        assert b"400" in data
