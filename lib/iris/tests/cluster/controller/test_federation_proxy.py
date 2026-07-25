# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-cluster endpoint federation, end to end over real HTTP.

Exercises the whole workflow in one pass against two real controller dashboards
and a real upstream:

    parent hands a cluster-pinned job to the peer
      -> peer materializes it and registers an endpoint on the received task
      -> parent mirrors the endpoint via federation_sync
      -> a request to the parent's /proxy/<name>/ is authorized, forwarded to the
         peer under a freshly minted federation bearer, resolved locally on the
         peer, and served by the upstream
      -> unregistering the endpoint on the peer drops it from the parent's proxy.

The federation bearer is a real EdDSA JWT: the parent signs with its key, the
peer's Rust listener verifies it, then the private control-plane decision checks
the RECEIVED handle the handoff created.
"""

import json
import socket
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field

import httpx
import pytest
import uvicorn
from iris.cluster.controller import reads
from iris.cluster.controller.auth import (
    CONTROL_PLANE_AUDIENCES,
    FederationTokenProvider,
    FederationTokenVerifier,
    JwtTokenManager,
    NativeProxyAuthConfig,
    NativeProxyAuthMode,
    create_controller_auth,
)
from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.endpoint_service import ProxyEndpointMapping, ProxyRegistrySnapshot
from iris.cluster.controller.federation_proxy import FederatedEndpointHandoff
from iris.cluster.controller.native_proxy import (
    DECISION_SECRET_HEADER,
    PROXY_DECISION_PATH,
    UPSTREAM_URL_HEADER,
    NativeProxy,
)
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.types import EndpointAccess, JobName
from iris.managed_thread import ThreadContainer
from iris.rpc import controller_pb2
from rigging.server_auth import RequestAuthPolicy
from rigging.timing import Duration, ExponentialBackoff
from rigging.token_authority import JwksVerifier, JwtSigner, generate_ed25519_keypair, signing_key_from_private_pem
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.testclient import TestClient

from ._test_support import ControllerTestState
from .conftest import promote_queued_federation, query_task
from .test_federation_handoff import (
    _attach_federation,
    _cluster_pinned_request,
    _InProcessPeerConnection,
    _make_service,
)

# The parent cluster's id: the requester stamped on the handoff, the federation
# token's issuer, and the peer_id the peer's RECEIVED handle is keyed by.
PARENT_ID = "parent"
# The peer's id inside the parent's federation manager (the resolved endpoint's peer_id).
PEER_ID = "cw"

ENDPOINT_NAME = "/serve/foo"
# Proxy-encoded form of ENDPOINT_NAME (``.`` for ``/``), how a /proxy request names it.
ENDPOINT_PROXY_NAME = "serve.foo"
UPSTREAM_MARKER = "served-by-the-federated-endpoint"


@dataclass
class UpstreamObservation:
    """What the upstream serving process saw on each forwarded request."""

    paths: list[str] = field(default_factory=list)
    headers: list[dict[str, str]] = field(default_factory=list)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(server: uvicorn.Server) -> None:
    ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
        lambda: server.started,
        timeout=Duration.from_seconds(5.0),
    )


def _serve(threads: ThreadContainer, app, *, name: str) -> str:
    """Serve ``app`` on a free 127.0.0.1 port and return its base URL once started."""
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", log_config=None)
    server = uvicorn.Server(config)
    threads.spawn_server(server, name=name)
    _wait_for_server(server)
    return f"http://127.0.0.1:{port}"


def _build_upstream_app(seen: UpstreamObservation) -> Starlette:
    """The real serving process the endpoint address points at; echoes a marker and
    records the forwarded path and headers so the test can assert what crossed."""

    async def serve(request: Request) -> Response:
        seen.paths.append(request.url.path + (f"?{request.url.query}" if request.url.query else ""))
        seen.headers.append({k.lower(): v for k, v in request.headers.items()})
        return JSONResponse({"marker": UPSTREAM_MARKER, "path": request.url.path, "method": request.method})

    app = Starlette(routes=[Route("/{path:path}", serve, methods=["GET", "POST"])])
    app.router.redirect_slashes = False
    return app


def _federation_auth(requester: str = PARENT_ID):
    """A parent federation-token minter plus the peer verifier that trusts its key.

    Returns ``(mint_token, peer_verifier)``: ``mint_token`` yields an ``aud="federation"``
    EdDSA bearer issued by ``requester``; ``peer_verifier`` resolves that bearer to a
    federation-peer identity whose ``user_id`` is ``requester``.
    """
    key = signing_key_from_private_pem(generate_ed25519_keypair().private_pem)
    signer = JwtSigner(key, issuer=requester)
    control_plane_verifier = JwksVerifier(
        issuers={requester: [key.public_pem]}, expected_audiences=CONTROL_PLANE_AUDIENCES
    )
    parent_manager = JwtTokenManager(signer, control_plane_verifier)
    mint_token = FederationTokenProvider(requester, parent_manager).get_token
    return mint_token, FederationTokenVerifier({requester: key.public_pem}), key.public_pem


def _register_endpoint(
    peer_service: ControllerServiceImpl,
    peer_state: ControllerTestState,
    job_id: JobName,
    address: str,
) -> str:
    """Register a LINK endpoint on the peer's received-job task; return its id."""
    task = job_id.task(0)
    response = peer_service.endpoint_service.register_endpoint(
        controller_pb2.Controller.RegisterEndpointRequest(
            name=ENDPOINT_NAME,
            address=address,
            task_id=task.to_wire(),
            attempt_id=query_task(peer_state, task).current_attempt_id,
            access=EndpointAccess.ENDPOINT_ACCESS_LINK,
        ),
        None,
    )
    return response.endpoint_id


def _native_proxy(
    stack: ExitStack,
    decision_url: str,
    decision_secret: str,
    *,
    mode: NativeProxyAuthMode,
    issuers: tuple[str, ...] = (),
    jwks: dict | None = None,
    federation_keys: dict[str, str] | None = None,
) -> NativeProxy:
    """A native proxy on an ephemeral port with the standard cache/leeway/CIDR policy,
    registered for teardown. Call sites vary only the decision endpoint and the auth
    material; the caller installs whatever registry it wants to serve.
    """
    proxy = NativeProxy(
        "127.0.0.1",
        0,
        decision_url,
        decision_secret,
        json.dumps(
            asdict(
                NativeProxyAuthConfig(
                    mode=mode,
                    issuers=issuers,
                    jwks=jwks if jwks is not None else {"keys": []},
                    leeway_seconds=60,
                    cache_capacity=16,
                    cache_ttl_seconds=60,
                    trusted_cidrs=(),
                    federation_keys=federation_keys or {},
                )
            )
        ),
    )
    stack.callback(proxy.stop)
    return proxy


@pytest.fixture
def threads() -> Iterator[ThreadContainer]:
    container = ThreadContainer()
    try:
        yield container
    finally:
        container.stop()


def test_federated_endpoint_serves_through_the_parent_proxy_end_to_end(tmp_path, log_client, threads):
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, peer_state = _make_service(stack, "peer", tmp_path, log_client)
        manager = _attach_federation(parent_service, _InProcessPeerConnection(peer_service))

        # The serving process the endpoint points at; the peer dials it directly.
        upstream = UpstreamObservation()
        upstream_url = _serve(threads, _build_upstream_app(upstream), name="upstream")

        # Data plane: hand the job off, let the peer materialize its task, register
        # the endpoint on it.
        job_id = JobName.from_wire(parent_service.launch_job(_cluster_pinned_request("fed-serve"), None).job_id)
        promote_queued_federation(manager, parent_state)
        endpoint_id = _register_endpoint(peer_service, peer_state, job_id, upstream_url)

        # Peer dashboard: enforces real federation auth on the inbound /proxy and
        # authorizes it by the RECEIVED handle the handoff created on the peer.
        def peer_owns(root_job: JobName, peer_id: str) -> bool:
            with peer_state._db.read_snapshot() as q:
                return reads.has_received_job_from_peer(q, peer_id, root_job)

        mint_token, peer_verifier, parent_public_key = _federation_auth(PARENT_ID)
        decision_secret = "federation-native-proxy-test"
        peer_dashboard = ControllerDashboard(
            peer_service,
            auth_policy=RequestAuthPolicy.enforcing(verifier=peer_verifier),
            federation_owner_check=peer_owns,
            proxy_decision_secret=decision_secret,
        )
        peer_private_url = _serve(threads, peer_dashboard.app, name="peer-dashboard")
        peer_proxy = _native_proxy(
            stack,
            peer_private_url,
            decision_secret,
            mode=NativeProxyAuthMode.ENFORCING,
            federation_keys={PARENT_ID: parent_public_key},
        )
        peer_proxy.replace_registry(json.dumps(asdict(peer_service.endpoint_service.proxy_registry_snapshot())))
        peer_url = peer_proxy.address

        # Parent dashboard: resolves the mirrored remote endpoint and forwards to the
        # peer under a freshly minted federation bearer.
        parent_decision_secret = "parent-federation-native-proxy-test"
        parent_dashboard = ControllerDashboard(
            parent_service,
            auth_policy=RequestAuthPolicy.permissive(),
            federated_handoff=FederatedEndpointHandoff(
                lambda pid: peer_url if pid == PEER_ID else None,
                mint_token,
            ),
            proxy_decision_secret=parent_decision_secret,
        )
        parent_private_url = _serve(threads, parent_dashboard.app, name="parent-dashboard")

        # Sync mirrors the peer's endpoint onto the parent as a remote (peer_id) row.
        manager.sync_once()
        parent_proxy = _native_proxy(
            stack,
            parent_private_url,
            parent_decision_secret,
            mode=NativeProxyAuthMode.PERMISSIVE,
        )
        parent_proxy.replace_registry(json.dumps(asdict(parent_service.endpoint_service.proxy_registry_snapshot())))
        parent_url = parent_proxy.address

        # The whole forward: parent /proxy -> federation bearer -> peer /proxy -> upstream.
        with httpx.Client() as client:
            resp = client.get(
                f"{parent_url}/proxy/{ENDPOINT_PROXY_NAME}/greet",
                params={"q": "1"},
                headers={"cookie": "session=secret", "authorization": "Bearer browser-user-token"},
            )
            subdomain_resp = client.get(
                f"{parent_url}/greet",
                headers={"host": f"{ENDPOINT_PROXY_NAME}.proxy.example.test"},
            )

        assert resp.status_code == 200, resp.text
        assert resp.json()["marker"] == UPSTREAM_MARKER
        assert subdomain_resp.status_code == 502
        # The upstream served the forwarded sub-path and query verbatim.
        assert upstream.paths[-1] == "/greet?q=1"
        # The browser's own session credentials never reached the serving process:
        # only the federation bearer crosses the boundary.
        served_headers = upstream.headers[-1]
        assert "cookie" not in served_headers
        assert served_headers.get("authorization") != "Bearer browser-user-token"

        # Lifecycle: unregister on the peer, re-sync, and the parent no longer serves it.
        peer_service.endpoint_service.unregister_endpoint(
            controller_pb2.Controller.UnregisterEndpointRequest(endpoint_id=endpoint_id), None
        )
        manager.sync_once()
        parent_proxy.replace_registry(json.dumps(asdict(parent_service.endpoint_service.proxy_registry_snapshot())))

        with httpx.Client() as client:
            gone = client.get(f"{parent_url}/proxy/{ENDPOINT_PROXY_NAME}/greet")
        assert gone.status_code == 404


# The child cluster's id: the tag a minted URL carries and the peer_id the parent
# resolves it against.
CHILD_CLUSTER = "cw"


def _child_capability_proxy(stack: ExitStack, upstream_url: str) -> tuple[str, str]:
    """A child native proxy that owns ``/serve/foo`` as a link endpoint and validates
    its own capability token. Returns ``(proxy_url, capability_token)``.

    This is the child exactly as it serves a direct capability URL — the relay adds
    nothing to it; the parent forwards the same ``/proxy/t/<token>/<name>`` request.
    """
    auth = create_controller_auth(None, cluster_name=CHILD_CLUSTER)
    assert auth.jwt_manager is not None
    issuers, jwks = auth.jwt_manager.native_proxy_verification_material()
    # A local capability request posts no decision, so the decision URL is unused.
    proxy = _native_proxy(
        stack,
        "http://127.0.0.1:9",
        "child-relay-decision",
        mode=NativeProxyAuthMode.ENFORCING,
        issuers=issuers,
        jwks=jwks,
    )
    proxy.replace_registry(
        json.dumps(
            asdict(
                ProxyRegistrySnapshot(
                    generation=1,
                    endpoints=(
                        ProxyEndpointMapping(
                            endpoint_id="child-serve",
                            name=ENDPOINT_NAME,
                            address=upstream_url,
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
    return proxy.address, auth.jwt_manager.create_endpoint_token(ENDPOINT_NAME, "iris_ket_relay")


def test_child_capability_url_relays_through_the_parent_end_to_end(tmp_path, log_client, threads):
    with ExitStack() as stack:
        upstream = UpstreamObservation()
        upstream_url = _serve(threads, _build_upstream_app(upstream), name="upstream")

        # The child owns the endpoint and mints its own capability token.
        child_url, token = _child_capability_proxy(stack, upstream_url)
        # The token works against the child's own origin — the relay changes nothing here.
        with httpx.Client() as client:
            direct = client.get(f"{child_url}/proxy/t/{token}/{ENDPOINT_PROXY_NAME}/greet")
        assert direct.status_code == 200, direct.text

        # The parent knows the child as a peer but mirrors none of its endpoints; a
        # relay mints no bearer, so the token minter is never called.
        parent_service, _ = _make_service(stack, "parent", tmp_path, log_client)
        parent_decision_secret = "parent-relay-decision"
        parent_dashboard = ControllerDashboard(
            parent_service,
            auth_policy=RequestAuthPolicy.permissive(),
            federated_handoff=FederatedEndpointHandoff(
                lambda pid: child_url if pid == CHILD_CLUSTER else None,
                lambda: "relay-mints-no-bearer",
            ),
            proxy_decision_secret=parent_decision_secret,
        )
        parent_private_url = _serve(threads, parent_dashboard.app, name="parent-dashboard")
        # The parent enforces its own auth material, which cannot validate the
        # child's token — reproducing the production iris.oa.dev 403. A 200 below
        # therefore proves the relay short-circuits before the parent verifies.
        parent_auth = create_controller_auth(None, cluster_name="parent")
        assert parent_auth.jwt_manager is not None
        parent_issuers, parent_jwks = parent_auth.jwt_manager.native_proxy_verification_material()
        parent_proxy = _native_proxy(
            stack,
            parent_private_url,
            parent_decision_secret,
            mode=NativeProxyAuthMode.ENFORCING,
            issuers=parent_issuers,
            jwks=parent_jwks,
        )
        # A relay consults no local registry, but the proxy still gates proxy traffic
        # on a ready registry, so install an empty one.
        parent_proxy.replace_registry(json.dumps(asdict(ProxyRegistrySnapshot(generation=1, endpoints=()))))

        with httpx.Client() as client:
            relayed = client.get(
                f"{parent_proxy.address}/proxy/{CHILD_CLUSTER}/t/{token}/{ENDPOINT_PROXY_NAME}/greet",
                params={"q": "1"},
                headers={"cookie": "session=secret", "authorization": "Bearer browser-user-token"},
            )
            unknown = client.get(f"{parent_proxy.address}/proxy/nope/t/{token}/{ENDPOINT_PROXY_NAME}/greet")

        assert relayed.status_code == 200, relayed.text
        assert relayed.json()["marker"] == UPSTREAM_MARKER
        # The child served the sub-path and query verbatim on its own /proxy/t path.
        assert upstream.paths[-1] == "/greet?q=1"
        # The browser's own credentials never crossed to the serving process.
        served = upstream.headers[-1]
        assert "cookie" not in served
        assert served.get("authorization") != "Bearer browser-user-token"
        # A cluster tag the parent has no peer for is refused, not forwarded.
        assert unknown.status_code == 404


def _relay_decision_payload(**overrides) -> dict:
    payload = {
        "direction": "relay",
        "encoded_name": ENDPOINT_PROXY_NAME,
        "sub_path": "v1/models",
        "query": "q=1",
        "proxy_prefix": f"/proxy/{CHILD_CLUSTER}/t/tok/{ENDPOINT_PROXY_NAME}",
        "peer_id": CHILD_CLUSTER,
        "task_id": None,
        "token": "tok",
        "local_upstream": None,
        "timeout_seconds": None,
    }
    return {**payload, **overrides}


def test_relay_decision_builds_the_child_upstream_and_rejects_an_unknown_peer(tmp_path, log_client):
    """The relay branch of the decision handler, without the native proxy: a
    configured peer yields the child /proxy/t upstream and no credential; an
    unconfigured tag is a 404 so the parent never relays it."""
    with ExitStack() as stack:
        parent_service, _ = _make_service(stack, "parent", tmp_path, log_client)
        secret = "relay-decision-secret"
        dashboard = ControllerDashboard(
            parent_service,
            auth_policy=RequestAuthPolicy.permissive(),
            federated_handoff=FederatedEndpointHandoff(
                lambda pid: "https://iris-cw.oa.dev" if pid == CHILD_CLUSTER else None,
                lambda: "relay-mints-no-bearer",
            ),
            proxy_decision_secret=secret,
        )
        client = TestClient(dashboard.app)

        ok = client.post(PROXY_DECISION_PATH, json=_relay_decision_payload(), headers={DECISION_SECRET_HEADER: secret})
        assert ok.status_code == 204
        assert (
            ok.headers[UPSTREAM_URL_HEADER] == f"https://iris-cw.oa.dev/proxy/t/tok/{ENDPOINT_PROXY_NAME}/v1/models?q=1"
        )
        assert "x-iris-upstream-authorization" not in ok.headers

        unknown = client.post(
            PROXY_DECISION_PATH, json=_relay_decision_payload(peer_id="nope"), headers={DECISION_SECRET_HEADER: secret}
        )
        assert unknown.status_code == 404
