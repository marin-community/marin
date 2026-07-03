# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Auth tests for Iris controller with static token authentication."""

import pytest
from connectrpc.errors import ConnectError
from iris.cluster.local_cluster import LocalCluster
from iris.cluster.types import Entrypoint, ResourceSpec
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.version import client_revision_date
from rigging.auth import BearerTokenInjector, StaticTokenProvider
from rigging.server_auth import (
    AuthRequest,
    RequestAuthPolicy,
    StaticTokenVerifier,
    is_trusted_loopback,
    resolve_auth,
)

from tests.conftest import _make_controller_only_config

_AUTH_TOKEN = "e2e-test-token"
_AUTH_USER = "test-user"


def _login_for_jwt(url: str, identity_token: str) -> str:
    """Exchange a raw identity token for a JWT via the Login RPC."""
    client = ControllerServiceClientSync(address=url, timeout_ms=10000)
    try:
        resp = client.login(job_pb2.LoginRequest(identity_token=identity_token))
        return resp.token
    finally:
        client.close()


def _quick():
    return 1


def test_static_auth_rpc_access():
    """Static auth over loopback: a tokenless request is trusted as admin
    (loopback trust), a wrong token is rejected, a valid JWT is accepted."""

    config = _make_controller_only_config()
    config.auth.static.tokens[_AUTH_TOKEN] = _AUTH_USER
    controller = LocalCluster(config)
    url = controller.start()

    try:
        list_req = controller_pb2.Controller.ListWorkersRequest()

        # The test client connects over loopback, so a tokenless request is
        # trusted as the anonymous admin rather than rejected.
        unauth_client = ControllerServiceClientSync(address=url, timeout_ms=5000)
        assert unauth_client.list_workers(list_req) is not None
        unauth_client.close()

        wrong_injector = BearerTokenInjector(StaticTokenProvider("wrong-token"), "authorization")
        wrong_client = ControllerServiceClientSync(address=url, timeout_ms=5000, interceptors=[wrong_injector])
        with pytest.raises(ConnectError, match=r"(?i)authenticat"):
            wrong_client.list_workers(list_req)
        wrong_client.close()

        jwt_token = _login_for_jwt(url, _AUTH_TOKEN)
        valid_injector = BearerTokenInjector(StaticTokenProvider(jwt_token), "authorization")
        valid_client = ControllerServiceClientSync(address=url, timeout_ms=5000, interceptors=[valid_injector])
        response = valid_client.list_workers(list_req)
        assert response is not None
        valid_client.close()
    finally:
        controller.close()


def test_static_auth_job_ownership():
    """Job ownership: user A cannot terminate user B's job."""

    _TOKEN_A = "token-user-a"
    _TOKEN_B = "token-user-b"

    config = _make_controller_only_config()
    config.auth.static.tokens[_TOKEN_A] = "user-a"
    config.auth.static.tokens[_TOKEN_B] = "user-b"
    controller = LocalCluster(config)
    url = controller.start()

    try:
        jwt_a = _login_for_jwt(url, _TOKEN_A)
        jwt_b = _login_for_jwt(url, _TOKEN_B)

        injector_a = BearerTokenInjector(StaticTokenProvider(jwt_a), "authorization")
        client_a = ControllerServiceClientSync(address=url, timeout_ms=10000, interceptors=[injector_a])

        entrypoint = Entrypoint.from_callable(_quick)
        launch_req = controller_pb2.Controller.LaunchJobRequest(
            name="/user-a/auth-owned-job",
            entrypoint=entrypoint.to_proto(),
            resources=ResourceSpec(cpu=1, memory="1g").to_proto(),
            client_revision_date=client_revision_date(),
        )
        resp = client_a.launch_job(launch_req)
        job_id = resp.job_id

        injector_b = BearerTokenInjector(StaticTokenProvider(jwt_b), "authorization")
        client_b = ControllerServiceClientSync(address=url, timeout_ms=10000, interceptors=[injector_b])
        with pytest.raises(ConnectError, match="cannot access resources owned by"):
            client_b.terminate_job(controller_pb2.Controller.TerminateJobRequest(job_id=job_id))

        client_a.terminate_job(controller_pb2.Controller.TerminateJobRequest(job_id=job_id))

        client_a.close()
        client_b.close()
    finally:
        controller.close()


# ---------------------------------------------------------------------------
# Loopback trust — the SSH-tunnel transition path (docs/auth-loopback-transition.md)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "client_address,headers,expected",
    [
        # Genuine direct loopback peer (SSH tunnel / on-host): nonzero port, no XFF.
        ("127.0.0.1:54321", {}, True),
        ("::1:54321", {}, True),
        # Forwarded value: uvicorn zeroes the port when it derives the client
        # from X-Forwarded-For, so a public client spoofing 127.0.0.1 lands here.
        ("127.0.0.1:0", {"x-forwarded-for": "127.0.0.1"}, False),
        # Port 0 alone (forwarded) is disqualifying even without the header check.
        ("127.0.0.1:0", {}, False),
        # A genuine loopback peer that nonetheless carries XFF (a proxy in front)
        # is not trusted: trust means "no proxy between caller and controller".
        ("127.0.0.1:54321", {"x-forwarded-for": "203.0.113.7"}, False),
        # Non-loopback peers are never trusted.
        ("203.0.113.7:443", {}, False),
        ("10.0.0.5:8080", {}, False),
        # Missing / malformed addresses.
        (None, {}, False),
        ("garbage", {}, False),
    ],
)
def test_is_trusted_loopback(client_address, headers, expected):
    assert is_trusted_loopback(client_address, headers) is expected


def test_resolve_auth_token_wins_over_loopback():
    """A present token is verified even on a loopback connection."""
    verifier = StaticTokenVerifier({"tok": "alice"})
    identity = resolve_auth(
        AuthRequest(token="tok", headers={}, client_address="127.0.0.1:54321"),
        RequestAuthPolicy.enforcing(verifier=verifier).authenticators,
    )
    assert identity is not None
    assert identity.user_id == "alice"


def test_resolve_auth_loopback_is_admin():
    """A tokenless loopback caller is always trusted as the anonymous admin."""
    identity = resolve_auth(
        AuthRequest(token=None, headers={}, client_address="127.0.0.1:54321"),
        RequestAuthPolicy.enforcing(verifier=StaticTokenVerifier({})).authenticators,
    )
    assert identity is not None
    assert identity.user_id == "anonymous"
    assert identity.role == "admin"


def test_resolve_auth_public_tokenless_rejected():
    """A tokenless non-loopback (public) request is rejected when auth is required."""
    with pytest.raises(ValueError, match="Missing authentication"):
        resolve_auth(
            AuthRequest(token=None, headers={}, client_address="203.0.113.7:443"),
            RequestAuthPolicy.enforcing(verifier=StaticTokenVerifier({})).authenticators,
        )


def test_resolve_auth_spoofed_loopback_rejected():
    """A public caller spoofing X-Forwarded-For: 127.0.0.1 must not be trusted.

    uvicorn rewrites scope["client"] to ("127.0.0.1", 0) and the X-Forwarded-For
    header is present — both disqualify the connection from loopback trust.
    """
    with pytest.raises(ValueError, match="Missing authentication"):
        resolve_auth(
            AuthRequest(token=None, headers={"x-forwarded-for": "127.0.0.1"}, client_address="127.0.0.1:0"),
            RequestAuthPolicy.enforcing(verifier=StaticTokenVerifier({})).authenticators,
        )


def test_loopback_resolves_as_admin():
    """End-to-end: a tokenless loopback client is resolved as the anonymous admin."""

    config = _make_controller_only_config()
    config.auth.static.tokens[_AUTH_TOKEN] = _AUTH_USER
    controller = LocalCluster(config)
    url = controller.start()

    try:
        client = ControllerServiceClientSync(address=url, timeout_ms=10000)
        resp = client.get_current_user(job_pb2.GetCurrentUserRequest())
        assert resp.user_id == "anonymous"
        assert resp.role == "admin"
        client.close()
    finally:
        controller.close()


def test_loopback_admin_submits_as_named_user():
    """A tokenless loopback caller is admin, so the job name's owner segment is
    authoritative — jobs are attributed to the user the client names (the CLI
    fills this with $USER), matching the null-auth behaviour SSH users rely on."""

    config = _make_controller_only_config()
    config.auth.static.tokens[_AUTH_TOKEN] = _AUTH_USER
    controller = LocalCluster(config)
    url = controller.start()

    try:
        client = ControllerServiceClientSync(address=url, timeout_ms=10000)
        launch_req = controller_pb2.Controller.LaunchJobRequest(
            name="/bob/acted-job",
            entrypoint=Entrypoint.from_callable(_quick).to_proto(),
            resources=ResourceSpec(cpu=1, memory="1g").to_proto(),
            client_revision_date=client_revision_date(),
        )
        resp = client.launch_job(launch_req)
        assert resp.job_id.startswith("/bob/"), resp.job_id
        client.close()
    finally:
        controller.close()


def test_token_user_owner_pinned_to_principal():
    """A non-admin token caller cannot impersonate: the job owner is forced to
    the verified principal regardless of the name segment they request."""

    config = _make_controller_only_config()
    config.auth.static.tokens["carol-token"] = "carol"  # role defaults to "user"
    controller = LocalCluster(config)
    url = controller.start()

    try:
        jwt = _login_for_jwt(url, "carol-token")
        client = ControllerServiceClientSync(
            address=url, timeout_ms=10000, interceptors=[BearerTokenInjector(StaticTokenProvider(jwt), "authorization")]
        )
        launch_req = controller_pb2.Controller.LaunchJobRequest(
            name="/bob/impersonated-job",
            entrypoint=Entrypoint.from_callable(_quick).to_proto(),
            resources=ResourceSpec(cpu=1, memory="1g").to_proto(),
            client_revision_date=client_revision_date(),
        )
        resp = client.launch_job(launch_req)
        assert resp.job_id.startswith("/carol/"), resp.job_id
        client.close()
    finally:
        controller.close()
