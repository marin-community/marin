# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import asdict

import httpx
import uvicorn
from iris.cluster.config import AuthConfig
from iris.cluster.controller.auth import NativeProxyAuthConfig, NativeProxyAuthMode, create_controller_auth
from iris.cluster.controller.endpoint_service import ProxyEndpointMapping, ProxyRegistrySnapshot
from iris.cluster.controller.native_proxy import PROXY_DECISION_PATH, NativeProxy
from iris.managed_thread import ThreadContainer
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

        assert redirect.status_code == 307
        assert redirect.headers["location"] == f"/proxy/{_ENCODED_NAME}/"
        assert response.status_code == 200
        assert response.headers["x-native-upstream"] == "reached"
        assert response.json() == {
            "body_bytes": len(payload),
            "authorization": None,
            "cookie": None,
            "proxy_prefix": f"/proxy/{_ENCODED_NAME}",
        }
        assert received_bodies == [payload]
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
