# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dispatch tests for :func:`rigging.resolver.resolve`.

The resolver is a small registry: bare ``host:port`` short-circuits; URL
schemes dispatch to handlers installed via :func:`register_scheme`. Built-in
handlers cover ``gcp://``; package plugins (e.g. ``iris.client``) install
others. These tests exercise the registry mechanism directly without
spinning up any RPC server.
"""

import pytest

from rigging.resolver import register_scheme, resolve
from rigging.resolver import resolver as resolver_module


@pytest.fixture(autouse=True)
def _isolate_handler_registry():
    """Snapshot the registry before each test and restore it after.

    Tests register their own schemes (or override built-ins like ``gcp``)
    without leaking changes to other tests.
    """
    snapshot = dict(resolver_module._HANDLERS)
    yield
    resolver_module._HANDLERS.clear()
    resolver_module._HANDLERS.update(snapshot)


# ---------------------------------------------------------------------------
# Bare host:port: short-circuits before any registry lookup.
# ---------------------------------------------------------------------------


def test_resolve_bare_host_port_short_circuits():
    # If the registry were consulted at all, this would raise (no handler
    # named ""). The short-circuit means we never look it up.
    assert resolve("10.0.0.5:10002") == ("10.0.0.5", 10002)


def test_resolve_bare_hostname_port():
    assert resolve("log-server.internal:10002") == ("log-server.internal", 10002)


# ---------------------------------------------------------------------------
# gcp:// scheme: built-in handler delegates to vm_address.
# ---------------------------------------------------------------------------


def test_resolve_gcp_scheme_calls_vm_address(monkeypatch):
    calls: list[tuple] = []

    def _fake_vm_address(name: str, provider: str):
        calls.append((name, provider))
        return ("10.0.0.7", 10002)

    monkeypatch.setattr(resolver_module, "vm_address", _fake_vm_address)
    assert resolve("gcp://log-server") == ("10.0.0.7", 10002)
    assert calls == [("log-server", "gcp")]


# ---------------------------------------------------------------------------
# Registry mechanism: plugins install handlers, dispatch hits them.
# ---------------------------------------------------------------------------


def test_register_scheme_dispatches_to_handler():
    captured: list = []

    def _handler(url):
        captured.append(url)
        return ("plugin-host", 4242)

    register_scheme("plugin", _handler)

    assert resolve("plugin://my-service?endpoint=/foo") == ("plugin-host", 4242)
    assert len(captured) == 1
    assert captured[0].scheme == "plugin"
    assert captured[0].host == "my-service"
    assert captured[0].query == {"endpoint": "/foo"}


def test_register_scheme_overwrites_previous():
    register_scheme("dup", lambda url: ("first", 1))
    register_scheme("dup", lambda url: ("second", 2))
    # Last registration wins; matches dict semantics, documented in
    # register_scheme's docstring.
    assert resolve("dup://anything") == ("second", 2)


# ---------------------------------------------------------------------------
# Unsupported scheme.
# ---------------------------------------------------------------------------


def test_resolve_unsupported_scheme_raises():
    with pytest.raises(ValueError, match="unsupported scheme"):
        resolve("coreweave://log-server")


def test_resolve_unregistered_scheme_raises():
    # iris:// is not built-in; it requires importing iris.client to
    # activate the plugin. Without that import it should fail clearly.
    with pytest.raises(ValueError, match="unsupported scheme"):
        resolve("iris://marin?endpoint=/system/log-server")
