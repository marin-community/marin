# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for actor RPC through the native endpoint proxy.

Tests the full round-trip: ActorClient → ProxyResolver → native listener →
actor server → response. There is no special actor-routing header; the encoded
actor name lives in the URL path.
"""

import json
from dataclasses import asdict

import pytest
from connectrpc.errors import ConnectError
from iris.actor.client import ActorClient
from iris.actor.resolver import ProxyResolver
from iris.actor.server import ActorServer
from iris.cluster.controller.endpoint_service import ProxyEndpointMapping, ProxyRegistrySnapshot
from iris.cluster.controller.native_proxy import NativeProxy
from iris.managed_thread import ThreadContainer


class StatusActor:
    """Minimal actor that returns status counters (mimics Zephyr Coordinator)."""

    def __init__(self):
        self.documents_processed = 0

    def get_status(self) -> dict:
        self.documents_processed += 1
        return {"documents_processed": self.documents_processed, "healthy": True}

    def echo(self, message: str) -> str:
        return f"echo: {message}"


def _start_proxy(
    auth_config_json: str,
    *,
    endpoints: dict[str, str] | None = None,
) -> tuple[str, NativeProxy]:
    endpoints = endpoints if endpoints is not None else {}
    proxy = NativeProxy(
        "127.0.0.1",
        0,
        "http://127.0.0.1:9",
        "actor-native-proxy-test",
        auth_config_json,
    )
    snapshot = ProxyRegistrySnapshot(
        generation=1,
        endpoints=tuple(
            ProxyEndpointMapping(
                endpoint_id=f"actor-{index}",
                name=name,
                address=address,
                link_access=False,
                peer_id=None,
                task_id=None,
                timeout_seconds=None,
                lease_deadline_epoch_ms=None,
            )
            for index, (name, address) in enumerate(endpoints.items())
        ),
    )
    proxy.replace_registry(json.dumps(asdict(snapshot)))
    return proxy.address, proxy


def test_proxy_round_trip(permissive_native_proxy_auth_json: str):
    """Full round-trip: ActorClient → ProxyResolver → native proxy → actor server."""
    threads = ThreadContainer()

    actor_name = "test-ns/status"
    actor_server = ActorServer(host="127.0.0.1", threads=threads)
    actor_server.register(actor_name, StatusActor())
    actor_port = actor_server.serve_background()

    try:
        proxy_url, proxy = _start_proxy(
            permissive_native_proxy_auth_json,
            endpoints={actor_name: f"127.0.0.1:{actor_port}"},
        )

        resolver = ProxyResolver(proxy_url)
        client = ActorClient(resolver, actor_name, max_call_attempts=1)

        result = client.get_status()
        assert result["documents_processed"] == 1
        assert result["healthy"] is True

        result = client.echo("hello")
        assert result == "echo: hello"

        # Second call increments the counter.
        result = client.get_status()
        assert result["documents_processed"] == 2
    finally:
        proxy.stop()
        threads.stop()


def test_proxy_namespaced_actor(permissive_native_proxy_auth_json: str):
    """ProxyResolver encodes slash-prefixed namespaced names with dot substitution.

    Mirrors real Iris backend behavior where actors are registered under paths
    like /user/job/coordinator/actor-0 and the address includes the http:// scheme.
    """
    threads = ThreadContainer()

    actor_name = "/user/my-job/coordinator/status-0"
    actor_server = ActorServer(host="127.0.0.1", threads=threads)
    actor_server.register(actor_name, StatusActor())
    actor_port = actor_server.serve_background()

    try:
        proxy_url, proxy = _start_proxy(
            permissive_native_proxy_auth_json,
            endpoints={actor_name: f"http://127.0.0.1:{actor_port}"},
        )

        resolver = ProxyResolver(proxy_url)
        client = ActorClient(resolver, actor_name, max_call_attempts=1)

        result = client.get_status()
        assert result["documents_processed"] == 1
    finally:
        proxy.stop()
        threads.stop()


def test_proxy_unknown_endpoint(permissive_native_proxy_auth_json: str):
    """The native proxy returns an error when the actor endpoint is not registered.

    The proxy returns 404; ConnectClientSync translates this to a ConnectError
    that propagates to the caller.
    """
    threads = ThreadContainer()

    try:
        proxy_url, proxy = _start_proxy(permissive_native_proxy_auth_json, endpoints={})

        resolver = ProxyResolver(proxy_url)
        client = ActorClient(resolver, "no-such-ns/no-such-actor", max_call_attempts=1)

        with pytest.raises(ConnectError):
            client.get_status()
    finally:
        proxy.stop()
        threads.stop()
