# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import asdict
from urllib.parse import unquote

import httpx
import uvicorn
from iris.cluster.config import AuthConfig
from iris.cluster.controller.auth import (
    VERIFIED_IDENTITY_HEADER,
    NativeProxyAuthConfig,
    NativeProxyAuthMode,
    create_controller_auth,
)
from iris.cluster.controller.endpoint_service import ProxyEndpointMapping, ProxyRegistrySnapshot
from iris.cluster.controller.native_proxy import PROXY_DECISION_PATH, NativeProxy
from iris.managed_thread import ThreadContainer
from rigging import telltale
from rigging.timing import Duration, ExponentialBackoff
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse
from starlette.routing import Route

_ENDPOINT_NAME = "/system/native-test"
_ENCODED_NAME = "system.native-test"


def _standalone_proxy(auth_config_json: str) -> NativeProxy:
    proxy = NativeProxy(
        "127.0.0.1",
        0,
        "http://127.0.0.1:9",
        "native-proxy-test",
        auth_config_json,
    )
    proxy.replace_registry(
        json.dumps(
            asdict(
                ProxyRegistrySnapshot(
                    generation=1,
                    endpoints=(
                        ProxyEndpointMapping(
                            endpoint_id="native-test",
                            name=_ENDPOINT_NAME,
                            address="http://127.0.0.1:9",
                            link_access=False,
                            peer_id=None,
                            task_id=None,
                            timeout_seconds=None,
                            lease_deadline_epoch_ms=None,
                        ),
                    ),
                )
            )
        )
    )
    return proxy


def _start_upstream(threads: ThreadContainer) -> tuple[str, list[bytes]]:
    received_bodies: list[bytes] = []

    async def echo(request: Request) -> JSONResponse:
        body = await request.body()
        received_bodies.append(body)
        return JSONResponse(
            {
                "body_bytes": len(body),
                "authorization": request.headers.get("authorization"),
                "cookie": request.headers.get("cookie"),
                "proxy_prefix": request.headers.get("x-forwarded-prefix"),
                "verified_identity": request.headers.get(VERIFIED_IDENTITY_HEADER),
                "decision_secret": request.headers.get("x-iris-decision-secret"),
                "iap_assertion": request.headers.get("x-goog-iap-jwt-assertion"),
            },
            headers={"x-native-upstream": "reached"},
        )

    async def redirect(_request: Request) -> RedirectResponse:
        return RedirectResponse("/login", status_code=307)

    async def reject(_request: Request) -> JSONResponse:
        return JSONResponse({"detail": "upstream auth"}, status_code=401)

    server = uvicorn.Server(
        uvicorn.Config(
            Starlette(
                routes=[
                    Route("/echo", echo, methods=["GET", "POST"]),
                    Route("/redirect", redirect),
                    Route("/reject", reject),
                ]
            ),
            host="127.0.0.1",
            port=0,
            log_level="error",
            log_config=None,
        )
    )
    threads.spawn_server(server, name="native-proxy-upstream")
    ExponentialBackoff(initial=0.01, maximum=0.1).wait_until(
        lambda: server.started,
        timeout=Duration.from_seconds(5),
    )
    assert server.servers
    port = server.servers[0].sockets[0].getsockname()[1]
    return f"http://127.0.0.1:{port}", received_bodies


def test_native_listener_preserves_public_routes_and_streams_to_endpoint(
    make_controller,
) -> None:
    threads = ThreadContainer()
    try:
        upstream, received_bodies = _start_upstream(threads)
        controller = make_controller(
            host="127.0.0.1",
            port=0,
            endpoints={_ENDPOINT_NAME: upstream},
        )
        controller.start()

        payload = b"native request body" * 4096
        with httpx.Client(base_url=controller.url, follow_redirects=False) as client:
            assert client.get("/health").status_code == 200
            assert client.get(PROXY_DECISION_PATH).status_code == 404
            redirect = client.get(f"/proxy/{_ENCODED_NAME}")
            response = client.post(
                f"/proxy/{_ENCODED_NAME}/echo",
                content=payload,
                headers={
                    "authorization": "Bearer browser-secret",
                    "cookie": "session=browser-secret",
                },
            )
            controller_rpc = client.post("/iris.cluster.ControllerService/ListJobs", json={})

        assert redirect.status_code == 307
        assert redirect.headers["location"] == f"/proxy/{_ENCODED_NAME}/"
        assert response.status_code == 200
        assert controller_rpc.status_code == 200
        assert response.headers["x-native-upstream"] == "reached"
        assert response.json() == {
            "body_bytes": len(payload),
            "authorization": None,
            "cookie": None,
            "proxy_prefix": f"/proxy/{_ENCODED_NAME}",
            "verified_identity": None,
            "decision_secret": None,
            "iap_assertion": None,
        }
        assert received_bodies == [payload]
    finally:
        threads.stop()


def test_native_rpc_metrics_aggregate_controllers_in_one_process(make_controller, tmp_path) -> None:
    controllers = [
        make_controller(
            host="127.0.0.1",
            port=0,
            local_state_dir=tmp_path / name,
            remote_state_dir=f"file://{tmp_path}/{name}-remote",
        )
        for name in ("first", "second")
    ]
    for controller in controllers:
        controller.start()
        response = httpx.post(
            f"{controller.url}/iris.cluster.ControllerService/ListJobs",
            json={},
        )
        assert response.status_code == 200

    labels = {
        "service": "iris.cluster.ControllerService",
        "method": "ListJobs",
        "upstream": "controller",
    }

    def sample_value(name: str, **extra_labels: str) -> float:
        matching = [
            family_sample.sample.value
            for family_sample in telltale.samples()
            if family_sample.sample.name == name and family_sample.sample.labels == {**labels, **extra_labels}
        ]
        assert len(matching) == 1
        return matching[0]

    assert sample_value("iris_rpc_requests_total") == 2
    assert sample_value("iris_rpc_responses_total", status="200") == 2
    assert sample_value("iris_rpc_in_flight") == 0
    assert sample_value("iris_rpc_duration_seconds_count") == 2


def test_native_proxy_transport_metrics_count_forwarded_bytes(make_controller) -> None:
    """A forwarded endpoint request lands in `iris_proxy_*`, byte-exact; a controller
    RPC does not. This also pins the Rust snapshot JSON to the collector's contract:
    a shape drift would raise when the collector unpacks the snapshot."""
    threads = ThreadContainer()
    try:
        upstream, _ = _start_upstream(threads)
        controller = make_controller(
            host="127.0.0.1",
            port=0,
            endpoints={_ENDPOINT_NAME: upstream},
        )
        controller.start()

        payload = b"native request body" * 4096
        with httpx.Client(base_url=controller.url, follow_redirects=False) as client:
            response = client.post(f"/proxy/{_ENCODED_NAME}/echo", content=payload)
            # A controller RPC terminates at the controller; it is RPC load, not
            # proxied transport, so it must not appear in the proxy family.
            assert client.post("/iris.cluster.ControllerService/ListJobs", json={}).status_code == 200
        assert response.status_code == 200

        def sample(name: str, labels: dict[str, str]) -> float:
            matching = [
                family_sample.sample.value
                for family_sample in telltale.samples()
                if family_sample.sample.name == name and family_sample.sample.labels == labels
            ]
            assert len(matching) == 1, f"{name}{labels}: {matching}"
            return matching[0]

        total = {"scope": "total", "endpoint": "", "method": "", "route_kind": ""}
        assert sample("iris_proxy_requests_total", total) == 1
        assert sample("iris_proxy_request_bytes_total", total) == len(payload)
        assert sample("iris_proxy_response_bytes_total", total) > 0
        assert sample("iris_proxy_responses_total", {**total, "status": "200"}) == 1

        # The bounded per-endpoint breakdown attributes the same request to its
        # endpoint, and no ControllerService route leaks into the proxy family.
        endpoint_requests = [
            family_sample.sample
            for family_sample in telltale.samples()
            if family_sample.sample.name == "iris_proxy_requests_total"
            and family_sample.sample.labels.get("scope") == "endpoint"
        ]
        assert len(endpoint_requests) == 1
        assert endpoint_requests[0].labels["endpoint"] == _ENCODED_NAME
        assert endpoint_requests[0].value == 1
        assert sample(
            "iris_proxy_request_bytes_total",
            {"scope": "endpoint", "endpoint": _ENCODED_NAME, "method": "POST", "route_kind": "endpoint"},
        ) == len(payload)
    finally:
        threads.stop()


def test_native_listener_caches_verified_jwt(make_controller) -> None:
    threads = ThreadContainer()
    try:
        upstream, _ = _start_upstream(threads)
        auth = create_controller_auth(None, cluster_name="native-test")
        assert auth.jwt_manager is not None
        token = auth.jwt_manager.create_token("test-user", "user", "native-test", ttl_seconds=300)
        endpoint_token = auth.jwt_manager.create_endpoint_token(_ENDPOINT_NAME, "native-endpoint-test")
        controller = make_controller(
            host="127.0.0.1",
            port=0,
            endpoints={_ENDPOINT_NAME: upstream},
            auth=auth,
        )
        controller.start()

        with httpx.Client(base_url=controller.url, headers={"authorization": f"Bearer {token}"}) as client:
            assert client.get(f"/proxy/{_ENCODED_NAME}/echo").status_code == 200
            assert client.get(f"/proxy/{_ENCODED_NAME}/echo").status_code == 200
        with httpx.Client(
            base_url=controller.url,
            headers={"authorization": f"Bearer {endpoint_token}"},
        ) as client:
            assert client.get(f"/proxy/{_ENCODED_NAME}/echo").status_code == 403

        stats = controller.native_proxy_stats
        assert stats is not None
        assert stats.jwt_cache_misses == 2
        assert stats.jwt_cache_hits == 1
    finally:
        threads.stop()


def test_native_listener_preserves_direct_controller_auth_without_trusting_forwarded_request(
    make_controller,
) -> None:
    auth = create_controller_auth(
        AuthConfig(trusted_cidrs=["10.0.0.0/8"]),
        cluster_name="native-controller-auth-test",
    )
    controller = make_controller(
        host="127.0.0.1",
        port=0,
        auth=auth,
    )
    controller.start()
    assert auth.jwt_manager is not None
    token = auth.jwt_manager.create_token("alice", "user", "controller-handoff", ttl_seconds=60)

    auth_config = httpx.get(
        f"{controller.url}/auth/config",
        headers={"x-forwarded-for": "203.0.113.10"},
    )
    authenticated_auth_config = httpx.get(
        f"{controller.url}/auth/config",
        headers={
            "authorization": f"Bearer {token}",
            "x-forwarded-for": "203.0.113.10",
        },
    )
    direct = httpx.post(
        f"{controller.url}/iris.cluster.ControllerService/ListJobs",
        json={},
        headers={"content-type": "application/json"},
    )
    forwarded = httpx.post(
        f"{controller.url}/iris.cluster.ControllerService/ListJobs",
        json={},
        headers={
            "content-type": "application/json",
            "x-forwarded-for": "203.0.113.10",
        },
    )
    authenticated = httpx.post(
        f"{controller.url}/iris.cluster.ControllerService/ListJobs",
        json={},
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {token}",
            "x-forwarded-for": "203.0.113.10",
        },
    )
    spoofed = httpx.post(
        f"{controller.url}/iris.cluster.ControllerService/ListJobs",
        json={},
        headers={
            "content-type": "application/json",
            "x-forwarded-for": "203.0.113.10",
            VERIFIED_IDENTITY_HEADER: "%7B%22user_id%22%3A%22admin%22%2C%22role%22%3A%22admin%22%7D",
        },
    )

    assert auth_config.json()["authenticated"] is False
    assert authenticated_auth_config.json()["authenticated"] is True
    assert direct.status_code == 200
    assert forwarded.status_code == 401
    assert authenticated.status_code == 200
    assert spoofed.status_code == 401


def test_native_listener_stamps_controller_identity_and_strips_caller_credentials() -> None:
    threads = ThreadContainer()
    try:
        upstream, _ = _start_upstream(threads)
        auth = create_controller_auth(None, cluster_name="native-controller-handoff")
        assert auth.jwt_manager is not None
        token = auth.jwt_manager.create_token("alice", "user", "controller-handoff", ttl_seconds=60)
        issuers, jwks = auth.jwt_manager.native_proxy_verification_material()
        proxy = NativeProxy(
            "127.0.0.1",
            0,
            upstream,
            "controller-handoff-secret",
            json.dumps(
                asdict(
                    NativeProxyAuthConfig(
                        mode=NativeProxyAuthMode.ENFORCING,
                        issuers=issuers,
                        jwks=jwks,
                        leeway_seconds=0,
                        cache_capacity=16,
                        cache_ttl_seconds=60,
                        trusted_cidrs=(),
                    )
                )
            ),
        )

        response = httpx.get(
            f"{proxy.address}/echo",
            headers={
                "authorization": f"Bearer {token}",
                "cookie": "iris_session=caller-cookie",
                "x-goog-iap-jwt-assertion": "caller-assertion",
                VERIFIED_IDENTITY_HEADER: "caller-spoof",
                "x-iris-decision-secret": "caller-spoof",
                "x-forwarded-for": "203.0.113.10",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["authorization"] is None
        assert payload["cookie"] is None
        assert payload["iap_assertion"] is None
        assert payload["decision_secret"] == "controller-handoff-secret"
        assert json.loads(unquote(payload["verified_identity"])) == {
            "user_id": "alice",
            "role": "user",
            "audience": None,
        }
        proxy.stop()
    finally:
        threads.stop()


def test_native_listener_owns_endpoint_access_policy() -> None:
    threads = ThreadContainer()
    try:
        upstream, _ = _start_upstream(threads)
        auth = create_controller_auth(None, cluster_name="native-access-test")
        assert auth.jwt_manager is not None
        issuers, jwks = auth.jwt_manager.native_proxy_verification_material()
        proxy = NativeProxy(
            "127.0.0.1",
            0,
            "http://127.0.0.1:9",
            "native-access-test",
            json.dumps(
                asdict(
                    NativeProxyAuthConfig(
                        mode=NativeProxyAuthMode.ENFORCING,
                        issuers=issuers,
                        jwks=jwks,
                        leeway_seconds=0,
                        cache_capacity=16,
                        cache_ttl_seconds=60,
                        trusted_cidrs=(),
                    )
                )
            ),
        )
        proxy.replace_registry(
            json.dumps(
                asdict(
                    ProxyRegistrySnapshot(
                        generation=1,
                        endpoints=(
                            ProxyEndpointMapping(
                                endpoint_id="native-access",
                                name=_ENDPOINT_NAME,
                                address=upstream,
                                link_access=True,
                                peer_id=None,
                                task_id=None,
                                timeout_seconds=None,
                                lease_deadline_epoch_ms=None,
                            ),
                        ),
                    )
                )
            )
        )
        matching = auth.jwt_manager.create_endpoint_token(_ENDPOINT_NAME, "matching")
        wrong = auth.jwt_manager.create_endpoint_token("/system/other", "wrong")
        full = auth.jwt_manager.create_token("alice", "user", "full", ttl_seconds=60)
        forwarded = {"x-forwarded-for": "203.0.113.1"}

        with httpx.Client(base_url=proxy.address) as client:
            assert client.get(f"/proxy/t/{matching}/{_ENCODED_NAME}/echo", headers=forwarded).status_code == 200
            assert (
                client.get(
                    f"/proxy/{_ENCODED_NAME}/echo",
                    headers={**forwarded, "authorization": f"Bearer {wrong}"},
                ).status_code
                == 403
            )
            assert (
                client.get(
                    f"/proxy/{_ENCODED_NAME}/echo",
                    headers={**forwarded, "authorization": f"Bearer {full}"},
                ).status_code
                == 200
            )
            assert client.get(f"/proxy/{_ENCODED_NAME}/echo", headers=forwarded).status_code == 401

        proxy.replace_registry(
            json.dumps(
                asdict(
                    ProxyRegistrySnapshot(
                        generation=2,
                        endpoints=(
                            ProxyEndpointMapping(
                                endpoint_id="native-access",
                                name=_ENDPOINT_NAME,
                                address=upstream,
                                link_access=False,
                                peer_id=None,
                                task_id=None,
                                timeout_seconds=None,
                                lease_deadline_epoch_ms=None,
                            ),
                        ),
                    )
                )
            )
        )
        response = httpx.get(
            f"{proxy.address}/proxy/{_ENCODED_NAME}/echo",
            headers={**forwarded, "authorization": f"Bearer {matching}"},
        )
        assert response.status_code == 403
        proxy.stop()
    finally:
        threads.stop()


def test_native_listener_handles_subdomains_and_response_safety(make_controller) -> None:
    threads = ThreadContainer()
    try:
        upstream, _ = _start_upstream(threads)
        auth = create_controller_auth(
            AuthConfig(trusted_cidrs=["10.0.0.0/8"]),
            cluster_name="native-loopback-test",
        )
        controller = make_controller(
            host="127.0.0.1",
            port=0,
            endpoints={_ENDPOINT_NAME: upstream},
            auth=auth,
        )
        controller.start()

        with httpx.Client(base_url=controller.url, follow_redirects=False) as client:
            subdomain = client.get(
                "/echo",
                headers={"host": f"{_ENCODED_NAME}.proxy.example.test"},
            )
            redirect = client.get(f"/proxy/{_ENCODED_NAME}/redirect")
            rejected = client.get(f"/proxy/{_ENCODED_NAME}/reject")

        assert subdomain.status_code == 200
        assert subdomain.json()["proxy_prefix"] is None
        assert redirect.status_code == 307
        assert redirect.headers["location"] == f"/proxy/{_ENCODED_NAME}/login"
        assert rejected.status_code == 502
    finally:
        threads.stop()


def test_native_listener_fails_closed_while_registry_is_paused(permissive_native_proxy_auth_json: str) -> None:
    with _standalone_proxy(permissive_native_proxy_auth_json) as proxy:
        proxy.pause_registry()

        response = httpx.get(f"{proxy.address}/proxy/{_ENCODED_NAME}/echo")

    assert response.status_code == 503
