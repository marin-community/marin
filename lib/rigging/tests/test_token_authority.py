# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the EdDSA token-signing/verification primitive.

These use real Ed25519 crypto end to end — no mocking of the signing or
verifying path — because the security contract is the crypto behavior: the
cross-plane audience guard, alg-confusion rejection, and the stable RFC 7638
kid are exactly what finelog's Rust verifier must agree with byte for byte.
"""

import base64
import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from rigging.token_authority import (
    JwksVerifier,
    JwtSigner,
    generate_ed25519_keypair,
    public_jwk_from_pem,
    signing_key_from_private_pem,
)

ISSUER = "cluster-a"


def _signer(issuer: str = ISSUER) -> JwtSigner:
    keypair = generate_ed25519_keypair()
    return JwtSigner(signing_key_from_private_pem(keypair.private_pem), issuer=issuer)


def _verifier(signer: JwtSigner, *, audiences=("iris",), issuer: str = ISSUER, extra_pems=()) -> JwksVerifier:
    pems = [signer._key.public_pem, *extra_pems]
    return JwksVerifier(issuers={issuer: pems}, expected_audiences=frozenset(audiences))


def test_mint_verify_roundtrip_preserves_caller_claims():
    signer = _signer()
    token = signer.mint({"sub": "alice", "role": "admin", "jti": "k1"}, audience="iris", ttl_seconds=3600)

    verified = _verifier(signer).verify(token)

    assert verified.sub == "alice"
    assert verified.iss == ISSUER
    assert verified.aud == "iris"
    # Caller-supplied policy claims survive verification unmodified.
    assert verified.claims["role"] == "admin"
    assert verified.claims["jti"] == "k1"


def test_scope_claim_surfaced_on_verified_claims():
    signer = _signer()
    token = signer.mint({"sub": "endpoint:foo", "scope": "proxy"}, audience="foo", ttl_seconds=3600)
    verified = JwksVerifier(issuers={ISSUER: [signer._key.public_pem]}, expected_audiences=frozenset({"foo"})).verify(
        token
    )
    assert verified.scope == "proxy"


def test_header_matches_frozen_contract():
    # The header is part of the frozen cross-impl contract: EdDSA + typ JWT + the
    # public key's RFC 7638 thumbprint as kid.
    signer = _signer()
    token = signer.mint({"sub": "alice"}, audience="iris", ttl_seconds=60)
    header = jwt.get_unverified_header(token)
    assert header == {"alg": "EdDSA", "typ": "JWT", "kid": signer._key.kid}


def test_token_for_other_plane_is_rejected():
    # The load-bearing invariant: a token minted for aud="iris" must not verify at
    # a surface that only expects aud="finelog" (cross-plane replay guard).
    signer = _signer()
    iris_token = signer.mint({"sub": "alice"}, audience="iris", ttl_seconds=3600)

    finelog_verifier = _verifier(signer, audiences=("finelog",))
    with pytest.raises(ValueError):
        finelog_verifier.verify(iris_token)

    # Same key, same signer — only the audience differs, and it decides admission.
    delegation_token = signer.mint({"sub": "relay"}, audience="finelog", ttl_seconds=3600)
    assert finelog_verifier.verify(delegation_token).aud == "finelog"


def test_expired_token_is_rejected():
    signer = _signer()
    # Mint with a positive ttl, then verify with negative leeway so "now" is past
    # exp — exercises the exp check without sleeping.
    token = signer.mint({"sub": "alice"}, audience="iris", ttl_seconds=1)
    verifier = JwksVerifier(
        issuers={ISSUER: [signer._key.public_pem]},
        expected_audiences=frozenset({"iris"}),
        leeway_seconds=-5,
    )
    with pytest.raises(ValueError):
        verifier.verify(token)


def test_tampered_payload_is_rejected():
    signer = _signer()
    token = signer.mint({"sub": "alice", "role": "user"}, audience="iris", ttl_seconds=3600)
    header_b64, _payload_b64, sig_b64 = token.split(".")

    # Re-encode the payload with role escalated to admin, keeping the original
    # signature — a forgery the EdDSA signature must reject.
    forged_payload = jwt.decode(token, options={"verify_signature": False}) | {"role": "admin"}
    forged_b64 = base64.urlsafe_b64encode(json.dumps(forged_payload).encode()).rstrip(b"=").decode()
    forged_token = f"{header_b64}.{forged_b64}.{sig_b64}"

    with pytest.raises(ValueError):
        _verifier(signer).verify(forged_token)


def test_token_from_unknown_issuer_is_rejected():
    signer = _signer(issuer="rogue-cluster")
    token = signer.mint({"sub": "alice"}, audience="iris", ttl_seconds=3600)
    # The verifier only trusts ISSUER, so a token from a different iss is rejected
    # even though it is validly self-signed.
    verifier = JwksVerifier(issuers={ISSUER: [signer._key.public_pem]}, expected_audiences=frozenset({"iris"}))
    with pytest.raises(ValueError, match="issuer"):
        verifier.verify(token)


def test_wrong_signing_key_for_known_issuer_is_rejected():
    signer = _signer()
    other = _signer()
    token = signer.mint({"sub": "alice"}, audience="iris", ttl_seconds=3600)
    # The issuer is trusted, but the configured public key is a different key —
    # the signature must not verify.
    verifier = JwksVerifier(issuers={ISSUER: [other._key.public_pem]}, expected_audiences=frozenset({"iris"}))
    with pytest.raises(ValueError):
        verifier.verify(token)


def test_alg_confusion_hs256_token_is_rejected():
    # Classic downgrade: a token whose header claims HS256 must never verify under
    # the EdDSA-only verifier, regardless of the HMAC secret used to forge it.
    signer = _signer()
    now = int(time.time())
    hs_token = jwt.encode(
        {"iss": ISSUER, "aud": "iris", "sub": "attacker", "iat": now, "exp": now + 3600},
        "a-shared-secret-that-is-at-least-32-bytes-long",
        algorithm="HS256",
    )
    with pytest.raises(ValueError):
        _verifier(signer).verify(hs_token)


def test_expected_audiences_is_required():
    signer = _signer()
    with pytest.raises(ValueError, match="expected_audiences"):
        JwksVerifier(issuers={ISSUER: [signer._key.public_pem]}, expected_audiences=frozenset())


def test_rotation_verifier_accepts_old_and_new_keys():
    old_signer = _signer()
    new_signer = _signer()
    # A verifier configured with BOTH the retired and the current public key
    # accepts tokens signed by either — the rotation overlap window.
    verifier = JwksVerifier(
        issuers={ISSUER: [old_signer._key.public_pem, new_signer._key.public_pem]},
        expected_audiences=frozenset({"iris"}),
    )
    old_token = old_signer.mint({"sub": "alice"}, audience="iris", ttl_seconds=3600)
    new_token = new_signer.mint({"sub": "bob"}, audience="iris", ttl_seconds=3600)
    assert verifier.verify(old_token).sub == "alice"
    assert verifier.verify(new_token).sub == "bob"


def test_public_jwks_serves_current_and_previous_keys():
    signer = _signer()
    previous = _signer()
    jwks = signer.public_jwks(also=[previous._key.public_pem])
    kids = [key["kid"] for key in jwks["keys"]]
    assert kids == [signer._key.kid, previous._key.kid]
    for entry in jwks["keys"]:
        assert entry["kty"] == "OKP"
        assert entry["crv"] == "Ed25519"
        assert entry["use"] == "sig"
        assert entry["alg"] == "EdDSA"
        # x is the 32-byte raw public key, base64url without padding.
        assert len(base64.urlsafe_b64decode(entry["x"] + "==")) == 32


def test_public_jwk_from_pem_roundtrips_generated_key():
    keypair = generate_ed25519_keypair()
    jwk = public_jwk_from_pem(keypair.public_pem)
    # The PEM view and the JWK view of one key agree, including the kid.
    assert jwk == keypair.public_jwk
    assert jwk["kid"] == keypair.kid


def test_kid_is_stable_thumbprint_for_a_given_key():
    keypair = generate_ed25519_keypair()
    # The kid is a pure function of the public key: independent loads agree.
    assert signing_key_from_private_pem(keypair.private_pem).kid == keypair.kid
    assert public_jwk_from_pem(keypair.public_pem)["kid"] == keypair.kid


def test_kid_matches_rfc8037_thumbprint_vector():
    # RFC 8037 Appendix A.2 Ed25519 public key -> its RFC 7638 thumbprint. Pins the
    # canonical-JWK form (member order / no whitespace) the Rust verifier shares.
    x = "11qYAYKxCrfVS_7TyWQHOg7hcvPapiMlrwIaaPcHURo"
    raw = base64.urlsafe_b64decode(x + "=")
    public_pem = (
        Ed25519PublicKey.from_public_bytes(raw).public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode()
    )
    assert public_jwk_from_pem(public_pem)["kid"] == "kPrK_qmxVWaYVA9wwBF6Iuo3vVzz7TxHCTwXBygrS4k"


def test_mint_rejects_signer_owned_claims():
    signer = _signer()
    # A caller cannot smuggle iss/aud/iat/exp through the claims dict — the signer
    # owns those, and a silent override would be a confusing security footgun.
    with pytest.raises(ValueError, match="signer-owned"):
        signer.mint({"sub": "alice", "aud": "sneaky"}, audience="iris", ttl_seconds=60)
