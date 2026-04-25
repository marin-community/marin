# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import time

import jwt
import pytest

from rigging.auth import JWT_ALGORITHM, JwtVerifier, VerifiedIdentity


def _mint(signing_key: str, *, sub: str = "alice", role: str = "user", jti: str = "k1", ttl: int = 60) -> str:
    now = int(time.time())
    payload = {"sub": sub, "role": role, "jti": jti, "iat": now, "exp": now + ttl}
    return jwt.encode(payload, signing_key, algorithm=JWT_ALGORITHM)


def test_verify_roundtrip():
    verifier = JwtVerifier("test-signing-key-abcdef1234567890")
    token = _mint(verifier.signing_key, sub="alice", role="user", jti="k1")
    assert verifier.verify(token) == VerifiedIdentity(user_id="alice", role="user")


def test_verify_default_role_when_claim_missing():
    signing_key = "test-signing-key-defaults"
    now = int(time.time())
    token = jwt.encode({"sub": "bob", "jti": "k2", "iat": now, "exp": now + 60}, signing_key, algorithm=JWT_ALGORITHM)
    assert JwtVerifier(signing_key).verify(token) == VerifiedIdentity(user_id="bob", role="user")


def test_verify_rejects_expired():
    verifier = JwtVerifier("test-key-expired-abc1234567890")
    token = _mint(verifier.signing_key, ttl=-1)
    with pytest.raises(ValueError, match="expired"):
        verifier.verify(token)


def test_verify_rejects_wrong_signature():
    minter = JwtVerifier("key-a-abcdef1234567890abcdef")
    other = JwtVerifier("key-b-abcdef1234567890abcdef")
    with pytest.raises(ValueError, match="Invalid token"):
        other.verify(_mint(minter.signing_key))


def test_verify_rejects_tampered_token():
    verifier = JwtVerifier("test-key-tampered-abc123")
    token = _mint(verifier.signing_key)
    flipped = "0" if token[-1] != "0" else "1"
    with pytest.raises(ValueError, match="Invalid token"):
        verifier.verify(token[:-1] + flipped)


def test_verify_rejects_garbage_string():
    verifier = JwtVerifier("test-key-garbage-abc123")
    with pytest.raises(ValueError, match="Invalid token"):
        verifier.verify("not-a-jwt")


def test_revoke_rejects_revoked_jti():
    verifier = JwtVerifier("test-key-revoke-abc1234567890")
    token = _mint(verifier.signing_key, jti="revoke-me")
    verifier.revoke("revoke-me")
    with pytest.raises(ValueError, match="revoked"):
        verifier.verify(token)


def test_revoke_only_blocks_named_jti():
    verifier = JwtVerifier("test-key-revoke-narrow")
    revoked = _mint(verifier.signing_key, jti="bad")
    other = _mint(verifier.signing_key, jti="good")
    verifier.revoke("bad")
    with pytest.raises(ValueError, match="revoked"):
        verifier.verify(revoked)
    assert verifier.verify(other).user_id == "alice"


def test_set_revocations_replaces_set():
    verifier = JwtVerifier("test-key-set-revocations")
    verifier.revoke("first")
    first_token = _mint(verifier.signing_key, jti="first")
    second_token = _mint(verifier.signing_key, jti="second")

    verifier.set_revocations({"second", "third"})
    assert verifier.verify(first_token).user_id == "alice"
    with pytest.raises(ValueError, match="revoked"):
        verifier.verify(second_token)


def test_verify_full_returns_payload():
    verifier = JwtVerifier("test-key-verify-full")
    token = _mint(verifier.signing_key, sub="carol", role="admin", jti="kx")
    identity, payload = verifier.verify_full(token)
    assert identity == VerifiedIdentity(user_id="carol", role="admin")
    assert payload["sub"] == "carol"
    assert payload["role"] == "admin"
    assert payload["jti"] == "kx"
