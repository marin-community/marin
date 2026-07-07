# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the declarative auth-stack schema (rigging.auth_config) and its
compilation into an authenticator chain (RequestAuthPolicy.from_config)."""

import pytest
from rigging.auth_config import (
    AnonymousLayer,
    AuthStackConfig,
    CidrLayer,
    IapAssertionLayer,
    JwtLayer,
    LoopbackLayer,
)
from rigging.server_auth import (
    ANONYMOUS_ADMIN,
    AnonymousAuthenticator,
    BestEffortJwtAuthenticator,
    CidrAuthenticator,
    IapAssertionAuthenticator,
    JwtAuthenticator,
    LoopbackAuthenticator,
    RequestAuthPolicy,
    VerifiedIdentity,
)
from rigging.testing import MockVerifier


class _FakeAssertionVerifier:
    """IapAssertionVerifier stand-in: a present signed-header => a fixed identity."""

    def identity_from_headers(self, headers):
        if headers.get("x-goog-iap-jwt-assertion"):
            return VerifiedIdentity(user_id="asserted@example.com", role="dashboard")
        return None


# ---------------------------------------------------------------------------
# AuthStackConfig (de)serialization
# ---------------------------------------------------------------------------


def test_from_json_parses_all_layer_types():
    stack = AuthStackConfig.from_json(
        [
            {"type": "jwt"},
            {"type": "iap_assertion"},
            {"type": "cidr", "cidrs": ["10.0.0.0/8", "127.0.0.0/8"]},
            {"type": "loopback"},
            {"type": "anonymous"},
        ]
    )
    assert stack.layers == (
        JwtLayer(optional=False),
        IapAssertionLayer(),
        CidrLayer(cidrs=("10.0.0.0/8", "127.0.0.0/8")),
        LoopbackLayer(),
        AnonymousLayer(),
    )


def test_from_json_accepts_a_json_string():
    stack = AuthStackConfig.from_json('[{"type": "jwt", "optional": true}, {"type": "anonymous"}]')
    assert stack.layers == (JwtLayer(optional=True), AnonymousLayer())


@pytest.mark.parametrize(
    "stack",
    [
        AuthStackConfig(layers=(JwtLayer(optional=True), AnonymousLayer())),
        AuthStackConfig(
            layers=(
                JwtLayer(),
                IapAssertionLayer(),
                CidrLayer(cidrs=("10.0.0.0/8",)),
                LoopbackLayer(),
                AnonymousLayer(),
            )
        ),
        AuthStackConfig(layers=(CidrLayer(cidrs=("127.0.0.0/8", "::1/128")),)),
    ],
)
def test_round_trip_from_json_of_to_json(stack):
    # from_json ∘ to_json is the identity on a well-formed stack.
    assert AuthStackConfig.from_json(stack.to_json()) == stack


def test_to_json_matches_the_wire_shape():
    stack = AuthStackConfig(
        layers=(
            JwtLayer(optional=True),
            CidrLayer(cidrs=("10.0.0.0/8",)),
            LoopbackLayer(),
        )
    )
    assert stack.to_json() == [
        {"type": "jwt", "optional": True},
        {"type": "cidr", "cidrs": ["10.0.0.0/8"]},
        {"type": "loopback"},
    ]


def test_jwt_layer_omits_optional_when_false():
    # A non-optional jwt layer serializes to the minimal `{"type": "jwt"}`, matching
    # finelog's wire shape; optional=false round-trips through the missing field.
    assert JwtLayer(optional=False).to_dict() == {"type": "jwt"}
    assert AuthStackConfig.from_json([{"type": "jwt"}]).layers == (JwtLayer(optional=False),)


def test_from_json_raises_on_empty_list():
    with pytest.raises(ValueError, match="empty list"):
        AuthStackConfig.from_json([])
    with pytest.raises(ValueError, match="empty list"):
        AuthStackConfig.from_json("[]")


def test_from_json_raises_on_unknown_type():
    # `static`/`gcp`/`iap_id_token` are login-exchange verifiers, not request-chain
    # layers — an unknown type is a parse error, not a silent skip.
    with pytest.raises(ValueError, match="unknown auth layer type"):
        AuthStackConfig.from_json([{"type": "static"}])
    with pytest.raises(ValueError, match="unknown auth layer type"):
        AuthStackConfig.from_json([{"type": "not-a-layer"}])


def test_from_json_raises_on_missing_type():
    with pytest.raises(ValueError, match="unknown auth layer type"):
        AuthStackConfig.from_json([{"cidrs": ["10.0.0.0/8"]}])


def test_cidr_layer_requires_non_empty_cidrs():
    with pytest.raises(ValueError, match="non-empty 'cidrs'"):
        AuthStackConfig.from_json([{"type": "cidr", "cidrs": []}])
    with pytest.raises(ValueError, match="non-empty 'cidrs'"):
        AuthStackConfig.from_json([{"type": "cidr"}])


# ---------------------------------------------------------------------------
# RequestAuthPolicy.from_config — compiling a stack into a chain
# ---------------------------------------------------------------------------


def test_from_config_compiles_layers_to_authenticators():
    verifier = MockVerifier({"tok": "alice"})
    iap_verifier = _FakeAssertionVerifier()
    stack = AuthStackConfig(
        layers=(
            JwtLayer(),
            IapAssertionLayer(),
            CidrLayer(cidrs=("10.0.0.0/8",)),
            LoopbackLayer(),
            AnonymousLayer(),
        )
    )
    policy = RequestAuthPolicy.from_config(stack, jwt_verifier=verifier, iap_assertion_verifier=iap_verifier)
    assert [type(a) for a in policy.authenticators] == [
        JwtAuthenticator,
        IapAssertionAuthenticator,
        CidrAuthenticator,
        LoopbackAuthenticator,
        AnonymousAuthenticator,
    ]
    # The jwt_verifier is exposed for out-of-band checks (session-cookie exchange).
    assert policy.verifier is verifier


def test_from_config_optional_jwt_is_best_effort():
    policy = RequestAuthPolicy.from_config(
        AuthStackConfig(layers=(JwtLayer(optional=True), AnonymousLayer())),
        jwt_verifier=MockVerifier({"worker": "system:worker"}),
    )
    assert [type(a) for a in policy.authenticators] == [BestEffortJwtAuthenticator, AnonymousAuthenticator]
    # Valid token attributes; invalid falls through to anonymous admin.
    assert policy.resolve("worker", headers={}).user_id == "system:worker"
    assert policy.resolve("stale", headers={}) == ANONYMOUS_ADMIN


def test_from_config_raises_on_empty_stack():
    with pytest.raises(ValueError, match="empty auth stack"):
        RequestAuthPolicy.from_config(AuthStackConfig(layers=()))


def test_from_config_raises_when_jwt_verifier_missing():
    with pytest.raises(ValueError, match="'jwt' auth layer requires a jwt_verifier"):
        RequestAuthPolicy.from_config(AuthStackConfig(layers=(JwtLayer(), AnonymousLayer())))


def test_from_config_raises_when_iap_assertion_verifier_missing():
    with pytest.raises(ValueError, match="'iap_assertion' auth layer requires"):
        RequestAuthPolicy.from_config(
            AuthStackConfig(layers=(IapAssertionLayer(), LoopbackLayer())),
        )


# ---------------------------------------------------------------------------
# enforcing()/permissive() are thin wrappers over from_config — same chains
# ---------------------------------------------------------------------------


def test_enforcing_wrapper_builds_the_same_chain_as_from_config():
    verifier = MockVerifier({"tok": "alice"})
    iap_verifier = _FakeAssertionVerifier()
    wrapped = RequestAuthPolicy.enforcing(
        verifier=verifier, iap_assertion_verifier=iap_verifier, trusted_cidrs=["10.0.0.0/8"], optional=True
    )
    explicit = RequestAuthPolicy.from_config(
        AuthStackConfig(
            layers=(
                JwtLayer(),
                IapAssertionLayer(),
                CidrLayer(cidrs=("10.0.0.0/8",)),
                LoopbackLayer(),
                AnonymousLayer(),
            )
        ),
        jwt_verifier=verifier,
        iap_assertion_verifier=iap_verifier,
    )
    assert [type(a) for a in wrapped.authenticators] == [type(a) for a in explicit.authenticators]
    assert wrapped.verifier is explicit.verifier is verifier


def test_permissive_wrapper_builds_best_effort_then_anonymous():
    verifier = MockVerifier({"worker": "system:worker"})
    policy = RequestAuthPolicy.permissive(verifier=verifier)
    assert [type(a) for a in policy.authenticators] == [BestEffortJwtAuthenticator, AnonymousAuthenticator]


def test_permissive_without_verifier_is_anonymous_only():
    policy = RequestAuthPolicy.permissive()
    assert [type(a) for a in policy.authenticators] == [AnonymousAuthenticator]
    assert policy.verifier is None


def test_enforcing_cidrs_alone_has_no_jwt_layer():
    # No verifier => no jwt layer; the chain is [Cidr, Loopback].
    policy = RequestAuthPolicy.enforcing(trusted_cidrs=["10.0.0.0/8"])
    assert [type(a) for a in policy.authenticators] == [CidrAuthenticator, LoopbackAuthenticator]
    assert policy.verifier is None
