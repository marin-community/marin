# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for rigging.proxy.

ssh_proxy itself is not exercised here — it shells out to ssh and is
covered by the live off-cluster test in iris (skipped by default).
"""

import socket
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from rigging import proxy as proxy_module
from rigging.proxy import active_stack, is_reachable, proxy_stack

# ---------------------------------------------------------------------------
# is_reachable
# ---------------------------------------------------------------------------


def test_is_reachable_localhost_listener():
    """Real loopback connect — verifies the positive path."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    try:
        port = sock.getsockname()[1]
        assert is_reachable("127.0.0.1", port, timeout=1.0)
    finally:
        sock.close()


def test_is_reachable_unbound_port_returns_false():
    # Port 1 (tcpmux) is reserved and ~never bound; connect refused → False.
    assert is_reachable("127.0.0.1", 1, timeout=0.2) is False


# ---------------------------------------------------------------------------
# ProxyStack + proxy_stack scope
# ---------------------------------------------------------------------------


class _TunnelLog:
    """Records each opener invocation + close."""

    def __init__(self):
        self.opened: list[tuple[str, int]] = []
        self.closed: list[tuple[str, int]] = []
        self._next_port = 50000

    def opener_for(self, addr: tuple[str, int]):
        @contextmanager
        def _cm() -> Iterator[tuple[str, int]]:
            self.opened.append(addr)
            local = ("127.0.0.1", self._next_port)
            self._next_port += 1
            try:
                yield local
            finally:
                self.closed.append(addr)

        return _cm


def test_active_stack_outside_block_is_none():
    assert active_stack() is None


def test_proxy_stack_sets_active_inside_block():
    with proxy_stack() as s:
        assert active_stack() is s
    assert active_stack() is None


def test_proxy_stack_caches_per_remote():
    log = _TunnelLog()
    with proxy_stack() as stack:
        first = stack.proxy(("10.0.0.1", 1), log.opener_for(("10.0.0.1", 1)))
        second = stack.proxy(("10.0.0.1", 1), log.opener_for(("10.0.0.1", 1)))
        third = stack.proxy(("10.0.0.2", 2), log.opener_for(("10.0.0.2", 2)))
    assert first == second
    assert first != third
    # Opener invoked once per distinct remote.
    assert log.opened == [("10.0.0.1", 1), ("10.0.0.2", 2)]


def test_proxy_stack_closes_tunnels_on_exit():
    log = _TunnelLog()
    with proxy_stack() as stack:
        stack.proxy(("10.0.0.1", 1), log.opener_for(("10.0.0.1", 1)))
        stack.proxy(("10.0.0.2", 2), log.opener_for(("10.0.0.2", 2)))
        assert log.closed == []
    assert sorted(log.closed) == [("10.0.0.1", 1), ("10.0.0.2", 2)]


def test_proxy_stack_nests():
    """Inner scope shadows the outer; outer restored on exit."""
    with proxy_stack() as outer:
        assert active_stack() is outer
        with proxy_stack() as inner:
            assert active_stack() is inner
            assert outer is not inner
        assert active_stack() is outer
    assert active_stack() is None


def test_proxy_stack_propagates_exceptions():
    """Exiting via exception still tears down opened tunnels."""
    log = _TunnelLog()
    with pytest.raises(RuntimeError, match="boom"):
        with proxy_stack() as stack:
            stack.proxy(("10.0.0.1", 1), log.opener_for(("10.0.0.1", 1)))
            raise RuntimeError("boom")
    assert log.closed == [("10.0.0.1", 1)]


def test_proxy_stack_independent_caches():
    """Two sibling scopes don't share the cache."""
    log = _TunnelLog()
    with proxy_stack() as a:
        a.proxy(("10.0.0.1", 1), log.opener_for(("10.0.0.1", 1)))
    with proxy_stack() as b:
        b.proxy(("10.0.0.1", 1), log.opener_for(("10.0.0.1", 1)))
    assert log.opened == [("10.0.0.1", 1), ("10.0.0.1", 1)]


# ---------------------------------------------------------------------------
# Module-level helpers exposed by ProxyStack
# ---------------------------------------------------------------------------


def test_probe_timeout_default_is_short(monkeypatch):
    # Sanity: the constant exists and is small enough for an interactive
    # CLI. We don't pin it tightly because hardware varies.
    assert 0 < proxy_module.PROBE_TIMEOUT_SECONDS <= 2.0
