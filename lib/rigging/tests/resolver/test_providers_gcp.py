# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`rigging.resolver.vm_address` and the GCP backend.

GCP is an I/O boundary — we mock the credentials helper and the REST call,
which is the seam between rigging and ``google.auth`` / ``httpx``.
"""

import sys

import pytest

from rigging.resolver import DEFAULT_GCP_PORT, vm_address
from rigging.resolver import _gcp_lookup as gcp_lookup


def test_unsupported_provider_raises():
    with pytest.raises(ValueError, match="unsupported provider: coreweave"):
        vm_address("vm", provider="coreweave")


def test_unsupported_provider_k8s_raises():
    with pytest.raises(ValueError, match="unsupported provider: k8s"):
        vm_address("vm", provider="k8s")


def test_vm_address_gcp_returns_internal_ip_and_default_port(monkeypatch):
    monkeypatch.setattr(gcp_lookup, "_gcp_credentials", lambda: ("test-project", "tok"))

    captured: list[tuple] = []

    def _fake_fetch(project_id: str, token: str, name: str) -> dict | None:
        captured.append((project_id, token, name))
        return {
            "name": name,
            "networkInterfaces": [{"networkIP": "10.0.0.42"}],
        }

    monkeypatch.setattr(gcp_lookup, "_fetch_vm_aggregated", _fake_fetch)

    host, port = vm_address("log-server", provider="gcp")
    assert host == "10.0.0.42"
    assert port == DEFAULT_GCP_PORT
    assert captured == [("test-project", "tok", "log-server")]


def test_vm_address_gcp_explicit_port(monkeypatch):
    monkeypatch.setattr(gcp_lookup, "_gcp_credentials", lambda: ("p", "t"))
    monkeypatch.setattr(
        gcp_lookup,
        "_fetch_vm_aggregated",
        lambda project_id, token, name: {"networkInterfaces": [{"networkIP": "10.0.0.5"}]},
    )
    assert vm_address("log-server", provider="gcp", port=12345) == ("10.0.0.5", 12345)


def test_vm_address_gcp_missing_vm_raises_lookup_error(monkeypatch):
    monkeypatch.setattr(gcp_lookup, "_gcp_credentials", lambda: ("p", "t"))
    monkeypatch.setattr(gcp_lookup, "_fetch_vm_aggregated", lambda *_args: None)

    with pytest.raises(LookupError, match="no GCP VM named"):
        vm_address("missing-vm", provider="gcp")


def test_vm_address_gcp_no_network_interfaces(monkeypatch):
    monkeypatch.setattr(gcp_lookup, "_gcp_credentials", lambda: ("p", "t"))
    monkeypatch.setattr(
        gcp_lookup,
        "_fetch_vm_aggregated",
        lambda *_args: {"networkInterfaces": []},
    )
    with pytest.raises(LookupError, match="no network interfaces"):
        vm_address("vm", provider="gcp")


def test_vm_address_gcp_no_internal_ip(monkeypatch):
    monkeypatch.setattr(gcp_lookup, "_gcp_credentials", lambda: ("p", "t"))
    monkeypatch.setattr(
        gcp_lookup,
        "_fetch_vm_aggregated",
        lambda *_args: {"networkInterfaces": [{}]},
    )
    with pytest.raises(LookupError, match="no networkIP"):
        vm_address("vm", provider="gcp")


# ---------------------------------------------------------------------------
# Missing-deps path: credentials helper raises NotImplementedError when
# google-auth isn't importable.
# ---------------------------------------------------------------------------


def test_gcp_credentials_missing_google_auth_raises_not_implemented(monkeypatch):
    # Simulate `import google.auth` failing inside the function.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _faulty_import(name, *args, **kwargs):
        if name.startswith("google.auth"):
            raise ImportError(f"mocked: no module named {name!r}")
        return real_import(name, *args, **kwargs)

    # Wipe any cached google.auth so the import statement is re-run.
    monkeypatch.delitem(sys.modules, "google.auth", raising=False)
    monkeypatch.delitem(sys.modules, "google.auth.transport.requests", raising=False)
    monkeypatch.setitem(
        __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__, "__import__", _faulty_import
    )

    with pytest.raises(NotImplementedError, match="google-auth"):
        gcp_lookup._gcp_credentials()


def test_fetch_vm_aggregated_missing_httpx_raises_not_implemented(monkeypatch):
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _faulty_import(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("mocked: no module named 'httpx'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "httpx", raising=False)
    monkeypatch.setitem(
        __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__, "__import__", _faulty_import
    )

    with pytest.raises(NotImplementedError, match="httpx"):
        gcp_lookup._fetch_vm_aggregated("p", "t", "vm")
