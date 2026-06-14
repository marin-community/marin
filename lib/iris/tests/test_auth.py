# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Auth tests for Iris controller with static token authentication."""

import pytest
from connectrpc.errors import ConnectError
from iris.cluster.backends.local.cluster import LocalCluster
from iris.cluster.types import Entrypoint, ResourceSpec
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.auth import (
    IRIS_USER_HEADER,
    AuthTokenInjector,
    LoopbackUserInjector,
    StaticTokenProvider,
    StaticTokenVerifier,
    is_trusted_loopback,
    resolve_auth,
)
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.version import client_revision_date

from .conftest import _make_controller_only_config

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
    """Static auth rejects unauthenticated and wrong-token RPCs, accepts valid JWT."""

    config = _make_controller_only_config()
    config.auth.static.tokens[_AUTH_TOKEN] = _AUTH_USER
    controller = LocalCluster(config)
    url = controller.start()

    try:
        list_req = controller_pb2.Controller.ListWorkersRequest()

        unauth_client = ControllerServiceClientSync(address=url, timeout_ms=5000)
        with pytest.raises(ConnectError, match=r"(?i)(authorization|authenticat)"):
            unauth_client.list_workers(list_req)
        unauth_client.close()

        wrong_injector = AuthTokenInjector(StaticTokenProvider("wrong-token"))
        wrong_client = ControllerServiceClientSync(address=url, timeout_ms=5000, interceptors=[wrong_injector])
        with pytest.raises(ConnectError, match=r"(?i)authenticat"):
            wrong_client.list_workers(list_req)
        wrong_client.close()

        jwt_token = _login_for_jwt(url, _AUTH_TOKEN)
        valid_injector = AuthTokenInjector(StaticTokenProvider(jwt_token))
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

        injector_a = AuthTokenInjector(StaticTokenProvider(jwt_a))
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

        injector_b = AuthTokenInjector(StaticTokenProvider(jwt_b))
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
    """A present token is verified even on a trusted-loopback connection."""
    verifier = StaticTokenVerifier({"tok": "alice"})
    identity = resolve_auth(
        "tok",
        verifier,
        optional=False,
        client_address="127.0.0.1:54321",
        headers={IRIS_USER_HEADER: "mallory"},
        trust_loopback=True,
    )
    assert identity is not None
    assert identity.user_id == "alice"


def test_resolve_auth_loopback_uses_declared_user():
    """Tokenless loopback caller is trusted as the X-Iris-User it declares."""
    verifier = StaticTokenVerifier({})
    identity = resolve_auth(
        None,
        verifier,
        optional=False,
        client_address="127.0.0.1:54321",
        headers={IRIS_USER_HEADER: "alice"},
        trust_loopback=True,
    )
    assert identity is not None
    assert identity.user_id == "alice"
    assert identity.role == "admin"


def test_resolve_auth_loopback_defaults_to_anonymous():
    """Tokenless loopback caller without a header falls back to anonymous admin."""
    identity = resolve_auth(
        None,
        StaticTokenVerifier({}),
        optional=False,
        client_address="127.0.0.1:54321",
        headers={},
        trust_loopback=True,
    )
    assert identity is not None
    assert identity.user_id == "anonymous"


def test_resolve_auth_public_tokenless_rejected_with_trust_loopback():
    """trust_loopback must NOT trust tokenless non-loopback (public) requests."""
    with pytest.raises(ValueError, match="Missing authentication"):
        resolve_auth(
            None,
            StaticTokenVerifier({}),
            optional=False,
            client_address="203.0.113.7:443",
            headers={IRIS_USER_HEADER: "mallory"},
            trust_loopback=True,
        )


def test_resolve_auth_spoofed_loopback_rejected():
    """A public caller spoofing X-Forwarded-For: 127.0.0.1 must not be trusted.

    uvicorn rewrites scope["client"] to ("127.0.0.1", 0) and the X-Forwarded-For
    header is present — both disqualify the connection from loopback trust.
    """
    with pytest.raises(ValueError, match="Missing authentication"):
        resolve_auth(
            None,
            StaticTokenVerifier({}),
            optional=False,
            client_address="127.0.0.1:0",
            headers={"x-forwarded-for": "127.0.0.1", IRIS_USER_HEADER: "mallory"},
            trust_loopback=True,
        )


def test_loopback_trust_resolves_declared_identity():
    """End-to-end: with trust_loopback on, a tokenless loopback client is
    resolved as the X-Iris-User it declares, with the admin role."""

    config = _make_controller_only_config()
    config.auth.static.tokens[_AUTH_TOKEN] = _AUTH_USER
    config.auth.trust_loopback = True
    controller = LocalCluster(config)
    url = controller.start()

    try:
        client = ControllerServiceClientSync(address=url, timeout_ms=10000, interceptors=[LoopbackUserInjector("alice")])
        resp = client.get_current_user(job_pb2.GetCurrentUserRequest())
        assert resp.user_id == "alice"
        assert resp.role == "admin"

        # Jobs launched over the trusted loopback connection are attributed to
        # the declared user.
        launch_req = controller_pb2.Controller.LaunchJobRequest(
            name="/alice/loopback-job",
            entrypoint=Entrypoint.from_callable(_quick).to_proto(),
            resources=ResourceSpec(cpu=1, memory="1g").to_proto(),
            client_revision_date=client_revision_date(),
        )
        client.launch_job(launch_req)
        client.close()
    finally:
        controller.close()


def test_loopback_trust_disabled_rejects_tokenless():
    """Without trust_loopback, tokenless loopback requests are rejected when auth is on."""

    config = _make_controller_only_config()
    config.auth.static.tokens[_AUTH_TOKEN] = _AUTH_USER
    # trust_loopback defaults to False.
    controller = LocalCluster(config)
    url = controller.start()

    try:
        client = ControllerServiceClientSync(address=url, timeout_ms=5000, interceptors=[LoopbackUserInjector("alice")])
        with pytest.raises(ConnectError, match=r"(?i)(authorization|authenticat)"):
            client.list_workers(controller_pb2.Controller.ListWorkersRequest())
        client.close()
    finally:
        controller.close()
