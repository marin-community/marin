# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scoped endpoint tokens: minting/verification and the per-endpoint proxy gate.

Covers the auth mechanism a served endpoint relies on off-cluster:

- ``JwtTokenManager.create_endpoint_token`` mints a ``scope=proxy`` token bound
  to the fixed proxy-plane ``aud="iris-proxy"`` with the endpoint in an
  ``endpoint`` claim; ``verify()`` surfaces that endpoint as the identity's
  audience and accepts both it and an ordinary control-plane token (the
  control-plane verifier's ``expected_audiences`` spans both planes).
- ``authorize_method`` denies any audience-bearing identity every RPC.
- ``_authorize_proxy`` enforces the PRIVATE / PUBLIC / BEARER access modes and is
  the only place a scoped token is accepted.
"""

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.controller.auth import CONTROL_PLANE_AUDIENCES, ENDPOINT_TOKEN_ROLE, JwtTokenManager
from iris.cluster.controller.dashboard import _authorize_proxy
from iris.cluster.controller.endpoint_service import ResolvedEndpoint
from iris.cluster.types import EndpointAccess
from iris.rpc.auth import authorize_method
from rigging.server_auth import RequestAuthPolicy, VerifiedIdentity
from rigging.token_authority import JwksVerifier, JwtSigner, generate_ed25519_keypair, signing_key_from_private_pem
from starlette.requests import Request

_ISSUER = "test-cluster"
_ENDPOINT = "/serve/foo"


@pytest.fixture
def jwt() -> JwtTokenManager:
    key = signing_key_from_private_pem(generate_ed25519_keypair().private_pem)
    signer = JwtSigner(key, issuer=_ISSUER)
    verifier = JwksVerifier(issuers={_ISSUER: [key.public_pem]}, expected_audiences=CONTROL_PLANE_AUDIENCES)
    return JwtTokenManager(signer, verifier)


def _resolved(access: int, name: str = _ENDPOINT) -> ResolvedEndpoint:
    return ResolvedEndpoint(name=name, address="up:8000", access=access)


def _request(*, token: str | None = None, host: str = "1.2.3.4", port: int = 5000) -> Request:
    """Build a minimal ASGI Request, optionally carrying a bearer token."""
    headers = [(b"authorization", f"Bearer {token}".encode())] if token else []
    return Request({"type": "http", "headers": headers, "client": (host, port), "method": "GET"})


# --- minting + verification ----------------------------------------------------


def test_endpoint_token_carries_audience(jwt):
    token = jwt.create_endpoint_token(_ENDPOINT, "iris_ket_1", ttl_seconds=60)
    identity = jwt.verify(token)
    assert identity.audience == _ENDPOINT
    assert identity.role == ENDPOINT_TOKEN_ROLE


def test_full_token_has_no_audience_through_same_verify(jwt):
    """Both a control-plane and an endpoint token verify through one ``verify()``.

    The control-plane verifier's ``expected_audiences`` spans both the ``iris``
    and ``iris-proxy`` planes, so a full-identity token (``aud="iris"``, no scope)
    surfaces no audience while an endpoint token (``aud="iris-proxy"``,
    ``scope="proxy"``) surfaces its bound endpoint.
    """
    assert jwt.verify(jwt.create_token("alice", "admin", "k1", ttl_seconds=60)).audience is None
    assert jwt.verify(jwt.create_endpoint_token(_ENDPOINT, "k2", ttl_seconds=60)).audience == _ENDPOINT


# --- RPC over-grant is closed --------------------------------------------------


def test_authorize_method_denies_scoped_identity():
    scoped = VerifiedIdentity(user_id="endpoint:/serve/foo", role=ENDPOINT_TOKEN_ROLE, audience=_ENDPOINT)
    # Even a read RPC that a normal role could call is denied to a scoped token.
    with pytest.raises(ConnectError) as exc:
        authorize_method(scoped, "ListEndpoints")
    assert exc.value.code is Code.PERMISSION_DENIED
    # A full identity is unaffected.
    authorize_method(VerifiedIdentity(user_id="alice", role="user"), "ListEndpoints")


# --- the per-endpoint proxy gate ----------------------------------------------


@pytest.fixture
def policy(jwt) -> RequestAuthPolicy:
    return RequestAuthPolicy.enforcing(verifier=jwt)


def test_public_allows_without_token(policy):
    assert _authorize_proxy(_request(), _resolved(EndpointAccess.ENDPOINT_ACCESS_PUBLIC), policy) is None


def test_bearer_accepts_matching_scoped_token(jwt, policy):
    token = jwt.create_endpoint_token(_ENDPOINT, "k", ttl_seconds=60)
    assert _authorize_proxy(_request(token=token), _resolved(EndpointAccess.ENDPOINT_ACCESS_BEARER), policy) is None


def test_bearer_rejects_scoped_token_for_other_endpoint(jwt, policy):
    token = jwt.create_endpoint_token("/serve/other", "k", ttl_seconds=60)
    deny = _authorize_proxy(_request(token=token), _resolved(EndpointAccess.ENDPOINT_ACCESS_BEARER), policy)
    assert deny is not None and deny.status_code == 403


def test_bearer_accepts_full_identity(jwt, policy):
    token = jwt.create_token("alice", "admin", "k1", ttl_seconds=60)
    assert _authorize_proxy(_request(token=token), _resolved(EndpointAccess.ENDPOINT_ACCESS_BEARER), policy) is None


def test_private_rejects_scoped_token(jwt, policy):
    token = jwt.create_endpoint_token(_ENDPOINT, "k", ttl_seconds=60)
    deny = _authorize_proxy(_request(token=token), _resolved(EndpointAccess.ENDPOINT_ACCESS_PRIVATE), policy)
    assert deny is not None and deny.status_code == 403


def test_private_rejects_missing_token(policy):
    deny = _authorize_proxy(_request(), _resolved(EndpointAccess.ENDPOINT_ACCESS_PRIVATE), policy)
    assert deny is not None and deny.status_code == 401


def test_private_accepts_full_identity(jwt, policy):
    token = jwt.create_token("alice", "admin", "k1", ttl_seconds=60)
    assert _authorize_proxy(_request(token=token), _resolved(EndpointAccess.ENDPOINT_ACCESS_PRIVATE), policy) is None


def test_unknown_endpoint_treated_as_private(policy):
    deny = _authorize_proxy(_request(), None, policy)
    assert deny is not None and deny.status_code == 401


def test_permissive_policy_allows_private_endpoint(jwt):
    """Null-auth (permissive chain, worker verifier present) leaves the proxy open."""
    null = RequestAuthPolicy.permissive(verifier=jwt)
    assert _authorize_proxy(_request(), _resolved(EndpointAccess.ENDPOINT_ACCESS_PRIVATE), null) is None


def test_token_in_url_override(jwt, policy):
    """The URL-token fallback reuses the same check via the token override."""
    token = jwt.create_endpoint_token(_ENDPOINT, "k", ttl_seconds=60)
    assert _authorize_proxy(_request(), _resolved(EndpointAccess.ENDPOINT_ACCESS_BEARER), policy, token=token) is None
    wrong = jwt.create_endpoint_token("/serve/other", "k", ttl_seconds=60)
    deny = _authorize_proxy(_request(), _resolved(EndpointAccess.ENDPOINT_ACCESS_BEARER), policy, token=wrong)
    assert deny is not None and deny.status_code == 403
