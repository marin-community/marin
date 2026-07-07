# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior gate for the declarative auth stack (spec §1.3).

Every ControllerAuth state must compile — via ``request_auth_policy`` →
``AuthStackConfig`` → ``RequestAuthPolicy.from_config`` — to a specific
authenticator chain. This asserts both the compiled chain SHAPE (the authenticator
sequence) and the admit/deny OUTCOME over a representative request matrix, per state.
"""

import pytest
from iris.cluster.config import AuthConfig, IapAuthConfig
from iris.cluster.controller.auth import WORKER_USER, create_controller_auth, request_auth_policy
from rigging.server_auth import (
    ANONYMOUS_ADMIN,
    AnonymousAuthenticator,
    BestEffortJwtAuthenticator,
    CidrAuthenticator,
    IapAssertionAuthenticator,
    JwtAuthenticator,
    LoopbackAuthenticator,
)
from rigging.token_authority import generate_ed25519_keypair

_CLUSTER = "test-cluster"
_SIGNING_KEY = generate_ed25519_keypair().private_pem
_ASSERTION_AUD = "/projects/1/global/backendServices/2"

_LOOPBACK = "127.0.0.1:54321"
_IN_CIDR = "10.1.2.3:5555"
_EXTERNAL = "203.0.113.9:5555"


def _iap(**auth_kw) -> AuthConfig:
    """An IAP AuthConfig (signed-header assertion) with AuthConfig-level overrides."""
    return AuthConfig(iap=IapAuthConfig(signed_header_audience=_ASSERTION_AUD), **auth_kw)


def _chain_types(policy):
    return [type(a) for a in policy.authenticators]


# ---------------------------------------------------------------------------
# Chain SHAPE: each ControllerAuth state compiles to the exact authenticator
# sequence in the expected list.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config, signing_key, expected",
    [
        # null-auth: permissive [BestEffortJwt, Anonymous].
        (AuthConfig(), None, [BestEffortJwtAuthenticator, AnonymousAuthenticator]),
        # iap signed-header assertion: [Jwt, IapAssertion, Loopback].
        (_iap(), _SIGNING_KEY, [JwtAuthenticator, IapAssertionAuthenticator, LoopbackAuthenticator]),
        # iap + optional: anonymous tail appended.
        (
            _iap(optional=True),
            _SIGNING_KEY,
            [JwtAuthenticator, IapAssertionAuthenticator, LoopbackAuthenticator, AnonymousAuthenticator],
        ),
        # iap + trusted_cidrs: cidr layer between assertion and loopback.
        (
            _iap(trusted_cidrs=["10.0.0.0/8"]),
            _SIGNING_KEY,
            [JwtAuthenticator, IapAssertionAuthenticator, CidrAuthenticator, LoopbackAuthenticator],
        ),
        # cidr-only trust: [Jwt, Cidr, Loopback]. The jwt layer is retained (the
        # request verifier is always the JWT manager) so the cluster worker token
        # still verifies — see the §1.3 cidr-only NOTE in auth.py.
        (
            AuthConfig(trusted_cidrs=["10.0.0.0/8"]),
            _SIGNING_KEY,
            [JwtAuthenticator, CidrAuthenticator, LoopbackAuthenticator],
        ),
        # cidr-only + optional: anonymous tail appended.
        (
            AuthConfig(trusted_cidrs=["10.0.0.0/8"], optional=True),
            _SIGNING_KEY,
            [JwtAuthenticator, CidrAuthenticator, LoopbackAuthenticator, AnonymousAuthenticator],
        ),
    ],
    ids=["null-auth", "iap-assertion", "iap-optional", "iap-cidr", "cidr-only", "cidr-only-optional"],
)
def test_compiled_chain_shape(config, signing_key, expected):
    auth = create_controller_auth(config, cluster_name=_CLUSTER, signing_key_pem=signing_key)
    assert _chain_types(request_auth_policy(auth)) == expected


# ---------------------------------------------------------------------------
# Admit/deny OUTCOME matrix per state.
# ---------------------------------------------------------------------------


def test_null_auth_outcomes():
    auth = create_controller_auth(AuthConfig(), cluster_name=_CLUSTER)
    policy = request_auth_policy(auth)
    assert policy.allows_anonymous
    # Tokenless and invalid-token requests are admitted as anonymous admin;
    # a valid worker JWT still attributes the caller.
    assert policy.resolve(None, headers={}) == ANONYMOUS_ADMIN
    assert policy.resolve("bogus", headers={}) == ANONYMOUS_ADMIN
    assert policy.resolve(auth.worker_token, headers={}).user_id == WORKER_USER


def test_iap_optional_outcomes():
    auth = create_controller_auth(_iap(optional=True), cluster_name=_CLUSTER, signing_key_pem=_SIGNING_KEY)
    policy = request_auth_policy(auth)
    assert policy.allows_anonymous
    # The anonymous tail admits tokenless external; an invalid token still rejects.
    assert policy.resolve(None, client_address=_EXTERNAL, headers={}) == ANONYMOUS_ADMIN
    with pytest.raises(ValueError):
        policy.resolve("bogus", client_address=_EXTERNAL, headers={})


def test_cidr_only_outcomes():
    auth = create_controller_auth(
        AuthConfig(trusted_cidrs=["10.0.0.0/8"]), cluster_name=_CLUSTER, signing_key_pem=_SIGNING_KEY
    )
    policy = request_auth_policy(auth)
    assert not policy.allows_anonymous
    # In-CIDR direct peer => admin; the cluster worker JWT verifies from anywhere.
    assert policy.resolve(None, client_address=_IN_CIDR, headers={}).role == "admin"
    assert policy.resolve(auth.worker_token, client_address=_EXTERNAL, headers={}).user_id == WORKER_USER
    # External tokenless is rejected; a forwarded in-CIDR hop never borrows the cidr.
    with pytest.raises(ValueError, match="Missing authentication"):
        policy.resolve(None, client_address=_EXTERNAL, headers={})
    with pytest.raises(ValueError, match="Missing authentication"):
        policy.resolve(None, client_address=_IN_CIDR, headers={"x-forwarded-for": "203.0.113.9"})


def test_iap_assertion_outcomes():
    auth = create_controller_auth(_iap(), cluster_name=_CLUSTER, signing_key_pem=_SIGNING_KEY)
    policy = request_auth_policy(auth)
    assert not policy.allows_anonymous
    # No assertion header + tokenless external => rejected; loopback still admin;
    # the worker JWT still verifies. (The assertion-present path is covered with a
    # mocked IAP verifier in test_auth.py.)
    assert policy.resolve(None, client_address=_LOOPBACK, headers={}) == ANONYMOUS_ADMIN
    assert policy.resolve(auth.worker_token, client_address=_EXTERNAL, headers={}).user_id == WORKER_USER
    with pytest.raises(ValueError, match="Missing authentication"):
        policy.resolve(None, client_address=_EXTERNAL, headers={})
