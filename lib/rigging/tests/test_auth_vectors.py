# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Runs the shared cross-impl auth-stack conformance vectors (rigging/auth_vectors.json).

The SAME vector file is the single source of truth for evaluator behaviour across
the Python (rigging) and Rust (finelog) engines. Here the jwt layer's verifier is
mocked: the vector's ``token`` is ``"valid"`` (accepted), ``"invalid"`` (present but
rejected), or ``null`` (absent).
"""

import json
from importlib.resources import files

import pytest
from rigging.auth_config import AuthStackConfig
from rigging.server_auth import (
    AnonymousAuthenticator,
    AuthDecision,
    AuthRequest,
    BestEffortJwtAuthenticator,
    CidrAuthenticator,
    IapAssertionAuthenticator,
    JwtAuthenticator,
    LoopbackAuthenticator,
    RequestAuthPolicy,
)
from rigging.testing import MockVerifier

# The mock jwt verifier accepts exactly the token "valid".
_MOCK_JWT_VERIFIER = MockVerifier({"valid": "vector-user"})

# Map a compiled authenticator back to the wire layer ``type`` it came from, so a
# vector can assert WHICH layer admitted (the shared walk contract).
_AUTHENTICATOR_LAYER_TYPE = {
    JwtAuthenticator: "jwt",
    BestEffortJwtAuthenticator: "jwt",
    IapAssertionAuthenticator: "iap_assertion",
    CidrAuthenticator: "cidr",
    LoopbackAuthenticator: "loopback",
    AnonymousAuthenticator: "anonymous",
}


def _load_vectors() -> dict:
    raw = (files("rigging") / "auth_vectors.json").read_text(encoding="utf-8")
    return json.loads(raw)


_VECTORS = _load_vectors()


def _auth_request(request: dict) -> AuthRequest:
    return AuthRequest(
        token=request["token"],
        headers=request.get("headers", {}),
        client_address=request.get("peer"),
    )


def _walk(policy: RequestAuthPolicy, request: AuthRequest) -> tuple[str, str | None]:
    """First-match walk: return (verdict, matched-layer-type)."""
    for authenticator in policy.authenticators:
        outcome = authenticator.authenticate(request)
        if outcome.decision is AuthDecision.AUTHENTICATED:
            return "allow", _AUTHENTICATOR_LAYER_TYPE[type(authenticator)]
        if outcome.decision is AuthDecision.REJECTED:
            return "deny", None
    return "deny", None


@pytest.mark.parametrize("vector", _VECTORS["vectors"], ids=[v["name"] for v in _VECTORS["vectors"]])
def test_auth_vector(vector):
    policy = RequestAuthPolicy.from_config(AuthStackConfig.from_json(vector["stack"]), jwt_verifier=_MOCK_JWT_VERIFIER)
    request = _auth_request(vector["request"])

    verdict, matched = _walk(policy, request)
    assert verdict == vector["expect"]["verdict"], vector["description"]
    assert matched == vector["expect"]["matched"], vector["description"]

    # The public resolve() path must agree with the walk: allow => an identity,
    # deny => a raised ValueError.
    if vector["expect"]["verdict"] == "allow":
        assert policy.resolve(request.token, client_address=request.client_address, headers=request.headers)
    else:
        with pytest.raises(ValueError):
            policy.resolve(request.token, client_address=request.client_address, headers=request.headers)


@pytest.mark.parametrize("case", _VECTORS["parse_error_stacks"], ids=[c["name"] for c in _VECTORS["parse_error_stacks"]])
def test_parse_error_stack_raises(case):
    with pytest.raises(ValueError):
        AuthStackConfig.from_json(case["stack"])
