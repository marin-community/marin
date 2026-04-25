# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from rigging import resolver as resolver_module
from rigging.resolver import ServiceURL, resolve, vm_address

# ---------------------------------------------------------------------------
# ServiceURL parsing
# ---------------------------------------------------------------------------


def test_parse_iris_with_endpoint_query():
    url = ServiceURL.parse("iris://marin?endpoint=/system/log-server")
    assert url.scheme == "iris"
    assert url.host == "marin"
    assert url.query == {"endpoint": "/system/log-server"}


def test_parse_gcp_no_query():
    url = ServiceURL.parse("gcp://log-server")
    assert url.scheme == "gcp"
    assert url.host == "log-server"
    assert url.query == {}


def test_parse_missing_scheme_raises():
    with pytest.raises(ValueError, match="missing scheme"):
        ServiceURL.parse("marin?endpoint=/x")


def test_parse_missing_host_raises():
    with pytest.raises(ValueError, match="missing host"):
        ServiceURL.parse("iris://")


def test_parse_rejects_userinfo():
    with pytest.raises(ValueError, match="userinfo not supported"):
        ServiceURL.parse("iris://user@marin?endpoint=/x")


def test_parse_accepts_port():
    url = ServiceURL.parse("gcp://log-server:1234")
    assert url.host == "log-server"
    assert url.port == 1234


def test_parse_port_absent_when_omitted():
    assert ServiceURL.parse("gcp://log-server").port is None


def test_query_missing_key_absent():
    assert "endpoint" not in ServiceURL.parse("iris://marin").query


def test_query_duplicate_key_first_wins():
    assert ServiceURL.parse("iris://marin?endpoint=/a&endpoint=/b").query == {"endpoint": "/a"}


def test_query_multiple_distinct_keys():
    url = ServiceURL.parse("gcp://vm?zone=us-central1-a&port=10002")
    assert url.query == {"zone": "us-central1-a", "port": "10002"}


# ---------------------------------------------------------------------------
# resolve() dispatch
# ---------------------------------------------------------------------------


def test_resolve_bare_host_port():
    assert resolve("10.0.0.5:10002") == ("10.0.0.5", 10002)
    assert resolve("log-server.internal:10002") == ("log-server.internal", 10002)


def test_resolve_gcp_scheme_calls_vm_address(monkeypatch):
    calls: list[tuple] = []

    def _fake_vm_address(name: str, provider: str, *, port: int):
        calls.append((name, provider, port))
        return (name, port)

    monkeypatch.setattr(resolver_module, "vm_address", _fake_vm_address)
    assert resolve("gcp://log-server") == ("log-server", 10002)
    assert resolve("gcp://log-server:8080") == ("log-server", 8080)
    assert calls == [("log-server", "gcp", 10002), ("log-server", "gcp", 8080)]


def test_register_scheme_dispatches_to_handler(monkeypatch):
    captured: list = []

    def _handler(url):
        captured.append(url)
        return ("plugin-host", 4242)

    # Restore the registry after the test so the temporary scheme doesn't leak.
    monkeypatch.setitem(resolver_module._HANDLERS, "plugin", _handler)

    assert resolve("plugin://my-service?endpoint=/foo") == ("plugin-host", 4242)
    assert len(captured) == 1
    assert captured[0].scheme == "plugin"
    assert captured[0].host == "my-service"
    assert captured[0].query == {"endpoint": "/foo"}


@pytest.mark.parametrize("ref", ["coreweave://log-server", "iris://marin?endpoint=/system/log-server"])
def test_resolve_unsupported_scheme_raises(ref: str):
    with pytest.raises(ValueError, match="unsupported scheme"):
        resolve(ref)


# ---------------------------------------------------------------------------
# vm_address: GCP backend (mocked at the I/O boundary)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["coreweave", "k8s"])
def test_unsupported_provider_raises(provider: str):
    with pytest.raises(ValueError, match=f"unsupported provider: {provider}"):
        vm_address("vm", provider=provider)


def test_vm_address_gcp_returns_internal_ip_and_default_port(monkeypatch):
    monkeypatch.setattr(resolver_module, "_gcp_credentials", lambda: ("test-project", "tok"))

    captured: list[tuple] = []

    def _fake_fetch(project_id: str, token: str, name: str) -> dict | None:
        captured.append((project_id, token, name))
        return {"name": name, "networkInterfaces": [{"networkIP": "10.0.0.42"}]}

    monkeypatch.setattr(resolver_module, "_fetch_vm_aggregated", _fake_fetch)

    host, port = vm_address("log-server", provider="gcp")
    assert host == "10.0.0.42"
    assert port == 10002
    assert captured == [("test-project", "tok", "log-server")]


def test_vm_address_gcp_explicit_port(monkeypatch):
    monkeypatch.setattr(resolver_module, "_gcp_credentials", lambda: ("p", "t"))
    monkeypatch.setattr(
        resolver_module,
        "_fetch_vm_aggregated",
        lambda project_id, token, name: {"networkInterfaces": [{"networkIP": "10.0.0.5"}]},
    )
    assert vm_address("log-server", provider="gcp", port=12345) == ("10.0.0.5", 12345)


def test_vm_address_gcp_missing_vm_raises_lookup_error(monkeypatch):
    monkeypatch.setattr(resolver_module, "_gcp_credentials", lambda: ("p", "t"))
    monkeypatch.setattr(resolver_module, "_fetch_vm_aggregated", lambda *_args: None)

    with pytest.raises(LookupError, match="no GCP VM named"):
        vm_address("missing-vm", provider="gcp")


def test_vm_address_gcp_no_network_interfaces(monkeypatch):
    monkeypatch.setattr(resolver_module, "_gcp_credentials", lambda: ("p", "t"))
    monkeypatch.setattr(resolver_module, "_fetch_vm_aggregated", lambda *_args: {"networkInterfaces": []})
    with pytest.raises(LookupError, match="no network interfaces"):
        vm_address("vm", provider="gcp")


def test_vm_address_gcp_no_internal_ip(monkeypatch):
    monkeypatch.setattr(resolver_module, "_gcp_credentials", lambda: ("p", "t"))
    monkeypatch.setattr(resolver_module, "_fetch_vm_aggregated", lambda *_args: {"networkInterfaces": [{}]})
    with pytest.raises(LookupError, match="no networkIP"):
        vm_address("vm", provider="gcp")
