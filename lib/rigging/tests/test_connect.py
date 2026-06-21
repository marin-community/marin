# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import gc
from contextlib import ExitStack
from dataclasses import dataclass

import pytest
from rigging.auth import StaticTokenProvider
from rigging.connect import (
    ChainedAuth,
    DirectTransport,
    Endpoint,
    IapAuth,
    JwtAuth,
    NoAuth,
    Transport,
    TunnelTransport,
    connect,
    disconnect,
    parse_transport,
    proxy_path,
)
from rigging.tunnel import GcpSshForwardTarget, K8sPortForwardTarget


def record_factory(endpoint: Endpoint) -> Endpoint:
    """A trivial factory whose 'client' is the endpoint it was handed."""
    return endpoint


# --- parse_transport -------------------------------------------------------


def test_parse_https_direct():
    parsed = parse_transport("https://finelog-1.example:7000/some/path")
    assert isinstance(parsed.transport, DirectTransport)
    assert isinstance(parsed.auth, NoAuth)
    assert parsed.path == "/some/path"
    assert parsed.transport.open(ExitStack(), 1.0).url == "https://finelog-1.example:7000"


def test_parse_http_direct():
    parsed = parse_transport("http://localhost:8080")
    assert isinstance(parsed.transport, DirectTransport)
    assert isinstance(parsed.auth, NoAuth)
    assert parsed.path == ""
    assert parsed.transport.open(ExitStack(), 1.0).url == "http://localhost:8080"


def test_parse_iap_https_attaches_iap_auth():
    parsed = parse_transport("iap+https://iris-dev.oa.dev/proxy/system.log-server?audience=AUD")
    assert isinstance(parsed.transport, DirectTransport)
    assert isinstance(parsed.auth, IapAuth)
    assert parsed.path == "/proxy/system.log-server"
    assert parsed.transport.open(ExitStack(), 1.0).url == "https://iris-dev.oa.dev"
    interceptors = parsed.auth.interceptors()
    assert len(interceptors) == 1
    assert interceptors[0].header == "proxy-authorization"


def test_parse_iap_https_without_audience_defers_edge_auth():
    # No audience: HTTPS transport, but no auto IAP token — the caller supplies
    # the provider via connect(..., auth=IapAuth(...)).
    parsed = parse_transport("iap+https://iris-dev.oa.dev/proxy/system.log-server")
    assert isinstance(parsed.transport, DirectTransport)
    assert isinstance(parsed.auth, NoAuth)
    assert parsed.path == "/proxy/system.log-server"
    assert parsed.transport.open(ExitStack(), 1.0).url == "https://iris-dev.oa.dev"


def test_parse_ssh_gcp():
    parsed = parse_transport(
        "ssh+gcp://iris-controller:10000/proxy/system.log-server?project=marin&zone=us-central1-a&sa=svc@x&iap=true"
    )
    assert isinstance(parsed.transport, TunnelTransport)
    assert isinstance(parsed.auth, NoAuth)
    assert parsed.path == "/proxy/system.log-server"
    target = parsed.transport._target
    assert isinstance(target, GcpSshForwardTarget)
    assert target.project == "marin"
    assert target.zone == "us-central1-a"
    assert target.instance == "iris-controller"
    assert target.port == 10000
    assert target.impersonate_service_account == "svc@x"
    assert target.tunnel_through_iap is True


@pytest.mark.parametrize(
    "url",
    [
        "ssh+gcp://iris-controller:10000/path?zone=us-central1-a",  # no project
        "ssh+gcp://iris-controller:10000/path?project=marin",  # no zone
    ],
)
def test_parse_ssh_gcp_missing_required_param_raises(url):
    with pytest.raises(ValueError, match="ssh\\+gcp"):
        parse_transport(url)


def test_parse_k8s():
    parsed = parse_transport("k8s://iris-controller:10000/proxy/system.log-server?namespace=iris&context=ctx")
    assert isinstance(parsed.transport, TunnelTransport)
    assert isinstance(parsed.auth, NoAuth)
    assert parsed.path == "/proxy/system.log-server"
    target = parsed.transport._target
    assert isinstance(target, K8sPortForwardTarget)
    assert target.namespace == "iris"
    assert target.service == "iris-controller"
    assert target.port == 10000
    assert target.context == "ctx"


def test_parse_k8s_without_namespace_raises():
    with pytest.raises(ValueError, match="k8s"):
        parse_transport("k8s://iris-controller:10000/path")


def test_parse_unknown_scheme_raises():
    with pytest.raises(ValueError, match="unsupported transport scheme"):
        parse_transport("ftp://host:21/path")


# --- proxy_path ------------------------------------------------------------


def test_proxy_path():
    assert proxy_path("/system/log-server") == "/proxy/system.log-server"
    assert proxy_path("system/log-server") == "/proxy/system.log-server"


# --- motivating example 1: SSH + proxy to finelog --------------------------


def test_connect_ssh_proxy_to_finelog(monkeypatch):
    captured = {}

    @contextlib.contextmanager
    def fake_open_tunnel(target, *, timeout=60.0, **kwargs):
        captured["target"] = target
        yield "http://127.0.0.1:54321"

    monkeypatch.setattr("rigging.connect.open_tunnel", fake_open_tunnel)

    endpoint = connect(
        "ssh+gcp://iris-controller:10000/proxy/system.log-server?project=marin&zone=us-central1-a",
        record_factory,
    )

    assert endpoint == Endpoint(url="http://127.0.0.1:54321/proxy/system.log-server")

    target = captured["target"]
    assert target.project == "marin"
    assert target.zone == "us-central1-a"
    assert target.instance == "iris-controller"
    assert target.port == 10000


# --- motivating example 2: IAP + proxy to finelog --------------------------


def test_connect_iap_proxy_to_finelog():
    endpoint = connect(
        "iap+https://iris-dev.oa.dev/proxy/system.log-server?audience=AUD",
        record_factory,
        auth=JwtAuth(StaticTokenProvider("jwt")),
    )

    assert endpoint.url == "https://iris-dev.oa.dev/proxy/system.log-server"
    # ChainedAuth(scheme_auth, auth): IAP edge header (from the scheme) first,
    # then the JWT app header.
    assert [i.header for i in endpoint.interceptors] == ["proxy-authorization", "authorization"]


def test_connect_iap_with_caller_supplied_provider():
    # The human/desktop path: iap+https with no audience defers the IAP token to
    # the caller, who supplies it as proxy-authorization via auth=IapAuth(...).
    endpoint = connect(
        "iap+https://iris-marin.oa.dev/proxy/system.log-server",
        record_factory,
        auth=IapAuth(StaticTokenProvider("desktop-id-token")),
    )

    assert endpoint.url == "https://iris-marin.oa.dev/proxy/system.log-server"
    assert [i.header for i in endpoint.interceptors] == ["proxy-authorization"]


# --- lifetime --------------------------------------------------------------


class RecordingTransport:
    """A fake transport that registers a recording cleanup on the stack."""

    def __init__(self, cleanups: list[str]):
        self._cleanups = cleanups

    def open(self, stack: ExitStack, timeout: float) -> Endpoint:
        stack.callback(self._cleanups.append, "torn-down")
        return Endpoint("http://127.0.0.1:9")


class Client:
    """A plain weak-referenceable object to stand in for a real RPC client."""


def make_client(endpoint: Endpoint) -> Client:
    return Client()


def test_disconnect_runs_cleanup_once_and_is_idempotent():
    cleanups: list[str] = []
    client = connect(RecordingTransport(cleanups), make_client)

    assert cleanups == []
    disconnect(client)
    assert cleanups == ["torn-down"]

    # Second disconnect is a no-op, no double-run, no error.
    disconnect(client)
    assert cleanups == ["torn-down"]


def test_dropping_client_reaps_transport_via_finalizer():
    cleanups: list[str] = []
    client = connect(RecordingTransport(cleanups), make_client)
    assert cleanups == []

    del client
    gc.collect()
    assert cleanups == ["torn-down"]


def test_derived_object_keeps_transport_alive():
    """The client is the lifetime anchor: a derived object that strong-refs it
    keeps the transport alive until both are dropped."""
    cleanups: list[str] = []
    client = connect(RecordingTransport(cleanups), make_client)

    class Derived:
        def __init__(self, owner):
            self.owner = owner

    derived = Derived(client)

    # Drop only the direct client variable; derived still strong-refs it.
    del client
    gc.collect()
    assert cleanups == []

    del derived
    gc.collect()
    assert cleanups == ["torn-down"]


def test_connect_error_path_tears_down_transport():
    cleanups: list[str] = []

    def boom_factory(endpoint: Endpoint):
        raise RuntimeError("factory failed")

    with pytest.raises(RuntimeError, match="factory failed"):
        connect(RecordingTransport(cleanups), boom_factory)

    assert cleanups == ["torn-down"]


# --- auth composition ------------------------------------------------------


def test_chained_auth_orders_injectors():
    p1 = StaticTokenProvider("iap")
    p2 = StaticTokenProvider("jwt")
    interceptors = ChainedAuth(IapAuth(p1), JwtAuth(p2)).interceptors()

    assert [i.header for i in interceptors] == ["proxy-authorization", "authorization"]


def test_chained_auth_flattens_noauth():
    interceptors = ChainedAuth(NoAuth(), JwtAuth(StaticTokenProvider("jwt")), NoAuth()).interceptors()

    assert len(interceptors) == 1
    assert interceptors[0].header == "authorization"


def test_connect_with_explicit_transport_object_no_scheme_auth():
    cleanups: list[str] = []
    endpoint = connect(
        RecordingTransport(cleanups),
        record_factory,
        path="/proxy/x",
        auth=JwtAuth(StaticTokenProvider("jwt")),
    )

    assert endpoint.url == "http://127.0.0.1:9/proxy/x"
    assert len(endpoint.interceptors) == 1
    assert endpoint.interceptors[0].header == "authorization"


def test_connect_conflicting_paths_raises():
    with pytest.raises(ValueError, match="conflicts"):
        connect(
            "https://host:7000/a",
            record_factory,
            path="/b",
        )


def test_connect_threads_connect_timeout_to_transport():
    seen = {}

    class TimeoutTransport(Transport):
        def open(self, stack: ExitStack, timeout: float) -> Endpoint:
            seen["timeout"] = timeout
            return Endpoint("http://x")

    connect(TimeoutTransport(), record_factory, connect_timeout=12.5)
    assert seen["timeout"] == 12.5


# --- malformed input rejection ---------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https:///proxy",
        "iap+https:///proxy?audience=AUD",
        "ssh+gcp://:10000/p?project=marin&zone=z",
        "k8s://:10000/p?namespace=iris",
    ],
)
def test_parse_missing_host_raises(url):
    with pytest.raises(ValueError):
        parse_transport(url)


def test_connect_rejects_relative_path():
    with pytest.raises(ValueError, match="start with '/'"):
        connect(DirectTransport("http://x"), record_factory, path="proxy/a")


def test_connect_rejects_double_slash_path():
    with pytest.raises(ValueError, match="'//'"):
        connect(DirectTransport("http://x"), record_factory, path="//proxy/a")


# --- client-keying invariants (id-keyed, not hash/eq-keyed) ----------------


def test_connect_rejects_non_weakreferenceable_client():
    """A client that cannot be weak-referenced fails loudly, and the transport
    it would have anchored is torn down rather than leaked."""
    cleanups: list[str] = []

    with pytest.raises(TypeError, match="weak-referenceable"):
        connect(RecordingTransport(cleanups), lambda endpoint: [])  # list: not weak-referenceable

    assert cleanups == ["torn-down"]


@dataclass(frozen=True)
class EqClient:
    """Frozen dataclass clients compare equal across instances (hash/eq collide)."""

    tag: str = "same"


def test_value_equal_clients_have_isolated_lifetimes():
    """Two distinct-but-equal clients must not share a finalizer slot: tearing
    down one must not reach into the other. (Identity keying, not hash/eq.)"""
    cleanups_a: list[str] = []
    cleanups_b: list[str] = []

    client_a = connect(RecordingTransport(cleanups_a), lambda endpoint: EqClient())
    client_b = connect(RecordingTransport(cleanups_b), lambda endpoint: EqClient())
    assert client_a == client_b and client_a is not client_b

    disconnect(client_a)
    assert cleanups_a == ["torn-down"]
    assert cleanups_b == []  # b's transport untouched

    disconnect(client_b)
    assert cleanups_b == ["torn-down"]


def test_reusing_one_client_object_closes_the_prior_transport():
    """If a factory hands back the same live client for two connects, the first
    transport is closed when the second takes the slot — never silently leaked
    (the client's single GC can only reclaim one)."""
    cleanups_first: list[str] = []
    cleanups_second: list[str] = []
    shared = Client()

    connect(RecordingTransport(cleanups_first), lambda endpoint: shared)
    assert cleanups_first == []

    connect(RecordingTransport(cleanups_second), lambda endpoint: shared)
    assert cleanups_first == ["torn-down"]  # first transport reclaimed on overwrite
    assert cleanups_second == []

    disconnect(shared)
    assert cleanups_second == ["torn-down"]
