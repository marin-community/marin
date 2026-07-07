# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Auth tests for the Iris controller: loopback trust and identity attribution.

A local controller boots without a persistent signing key, so it runs in
null-auth (permissive) mode; these integration tests exercise loopback trust and
job-owner attribution against it. The lower-level token-verification and
authenticator-chain logic is unit-tested directly against ``resolve_auth``.
"""

import pytest
from iris.cluster.local_cluster import LocalCluster
from iris.cluster.types import Entrypoint, ResourceSpec
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.version import client_revision_date
from rigging.server_auth import (
    AuthRequest,
    RequestAuthPolicy,
    is_trusted_loopback,
    resolve_auth,
)
from rigging.testing import MockVerifier

from tests.conftest import _make_controller_only_config


def _quick():
    return 1


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
    verifier = MockVerifier({"tok": "alice"})
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
        RequestAuthPolicy.enforcing(verifier=MockVerifier({})).authenticators,
    )
    assert identity is not None
    assert identity.user_id == "anonymous"
    assert identity.role == "admin"


def test_resolve_auth_public_tokenless_rejected():
    """A tokenless non-loopback (public) request is rejected when auth is required."""
    with pytest.raises(ValueError, match="Missing authentication"):
        resolve_auth(
            AuthRequest(token=None, headers={}, client_address="203.0.113.7:443"),
            RequestAuthPolicy.enforcing(verifier=MockVerifier({})).authenticators,
        )


def test_resolve_auth_spoofed_loopback_rejected():
    """A public caller spoofing X-Forwarded-For: 127.0.0.1 must not be trusted.

    uvicorn rewrites scope["client"] to ("127.0.0.1", 0) and the X-Forwarded-For
    header is present — both disqualify the connection from loopback trust.
    """
    with pytest.raises(ValueError, match="Missing authentication"):
        resolve_auth(
            AuthRequest(token=None, headers={"x-forwarded-for": "127.0.0.1"}, client_address="127.0.0.1:0"),
            RequestAuthPolicy.enforcing(verifier=MockVerifier({})).authenticators,
        )


def test_loopback_resolves_as_admin():
    """End-to-end: a tokenless loopback client is resolved as the anonymous admin."""

    config = _make_controller_only_config()
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
