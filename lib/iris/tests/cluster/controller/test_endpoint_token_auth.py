# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scoped endpoint token minting and control-plane isolation.

Covers the auth mechanism a served endpoint relies on off-cluster:

- ``JwtTokenManager.create_endpoint_token`` mints a ``scope=proxy`` token bound
  to the fixed proxy-plane ``aud="iris-proxy"`` with the endpoint in an
  ``endpoint`` claim; ``verify()`` surfaces that endpoint as the identity's
  audience and accepts both it and an ordinary control-plane token (the
  control-plane verifier's ``expected_audiences`` spans both planes).
- ``authorize_method`` denies any audience-bearing identity every RPC.
The Rust listener's access-mode checks are covered in ``test_native_proxy.py``.
"""

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.controller.auth import CONTROL_PLANE_AUDIENCES, ENDPOINT_TOKEN_ROLE, JwtTokenManager
from iris.rpc.auth import authorize_method
from rigging.server_auth import VerifiedIdentity
from rigging.token_authority import JwksVerifier, JwtSigner, generate_ed25519_keypair, signing_key_from_private_pem

_ISSUER = "test-cluster"
_ENDPOINT = "/serve/foo"


@pytest.fixture
def jwt() -> JwtTokenManager:
    key = signing_key_from_private_pem(generate_ed25519_keypair().private_pem)
    signer = JwtSigner(key, issuer=_ISSUER)
    verifier = JwksVerifier(issuers={_ISSUER: [key.public_pem]}, expected_audiences=CONTROL_PLANE_AUDIENCES)
    return JwtTokenManager(signer, verifier)


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
