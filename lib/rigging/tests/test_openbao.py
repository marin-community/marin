# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic HTTP OpenBao boundary tests; real OpenBao proof lives in infra/auth."""

import base64
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jwt
import pytest
import requests
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, load_pem_private_key
from rigging.grants import Grant, GrantDenied, Permission
from rigging.openbao import (
    OpenBaoClient,
    OpenBaoIdentityVerifier,
    OpenBaoStorageIssuer,
    OpenBaoTransitSigner,
    OpenBaoTransitVerifier,
)
from rigging.token_authority import Ed25519Keypair, JwksVerifier, generate_ed25519_keypair


@dataclass
class BaoBoundary:
    token_file: Path
    address: str = ""
    keys: dict[int, Ed25519Keypair] = field(default_factory=lambda: {1: generate_ed25519_keypair()})
    latest_version: int = 1
    accepted_token: str = "broker-one"
    returned_version: int | None = None
    storage_subject: str = "entity-storage"
    storage_expiry: int = field(default_factory=lambda: int(time.time()) + 60)
    key_metadata_gate: tuple[threading.Event, threading.Event] | None = None

    @property
    def issuer(self):
        return self.address + "/v1/identity/oidc"

    def identity_token(self, *, subject="entity-one", audience="marin-auth", issuer=None, expires_at=None):
        key = self.keys[self.latest_version]
        now = int(time.time())
        return jwt.encode(
            {
                "iss": issuer or self.issuer,
                "sub": subject,
                "aud": audience,
                "iat": now,
                "exp": now + 300 if expires_at is None else expires_at,
                # An entity cannot choose an owner by asserting a JWT claim.
                "owner": "administrator",
            },
            key.private_pem,
            algorithm="EdDSA",
            headers={"kid": key.kid},
        )


@pytest.fixture
def bao(tmp_path):
    state = BaoBoundary(tmp_path / "broker-token")
    state.token_file.write_text(state.accepted_token)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format, *args):
            pass

        def respond(self, body, status=200):
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path == "/v1/identity/oidc/.well-known/keys":
                self.respond({"keys": [key.public_jwk for key in state.keys.values()]})
                return
            if self.headers.get("X-Vault-Token") != state.accepted_token:
                self.respond({"errors": ["permission denied"]}, 403)
                return
            if self.path == "/v1/transit/keys/marin-grants":
                if state.key_metadata_gate is not None:
                    started, release = state.key_metadata_gate
                    started.set()
                    assert release.wait(timeout=5)
                self.respond(
                    {
                        "data": {
                            "latest_version": state.latest_version,
                            "keys": {
                                str(version): {
                                    "public_key": (
                                        base64.b64encode(
                                            load_pem_private_key(key.private_pem.encode(), password=None)
                                            .public_key()
                                            .public_bytes(Encoding.Raw, PublicFormat.Raw)
                                        ).decode()
                                    )
                                }
                                for version, key in state.keys.items()
                            },
                        }
                    }
                )
                return
            if self.path == "/v1/identity/oidc/token/cw-read":
                self.respond(
                    {
                        "data": {
                            "token": state.identity_token(
                                subject=state.storage_subject,
                                audience="https://coreweave.com/iam",
                                expires_at=state.storage_expiry,
                            )
                        }
                    }
                )
                return
            self.respond({}, 404)

        def do_POST(self):
            if self.headers.get("X-Vault-Token") != state.accepted_token:
                self.respond({"errors": ["permission denied"]}, 403)
                return
            if self.path != "/v1/transit/sign/marin-grants":
                self.respond({}, 404)
                return
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            version = request["key_version"]
            key = load_pem_private_key(state.keys[version].private_pem.encode(), password=None)
            signature = base64.b64encode(key.sign(base64.b64decode(request["input"]))).decode()
            self.respond({"data": {"signature": f"vault:v{state.returned_version or version}:{signature}"}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.address = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def signer(bao):
    return OpenBaoTransitSigner(OpenBaoClient(bao.address, bao.token_file), issuer="https://auth.example", key_version=1)


def test_remote_transit_signature_matches_rigging_contract_and_rotating_broker_credential(bao, signer):
    verifier = JwksVerifier(
        issuers={signer.issuer: signer.public_keys()}, expected_audiences=frozenset({"loom"}), leeway_seconds=0
    )
    token = signer.mint({"sub": "loom:one", "grant_id": "one"}, audience="loom", ttl_seconds=60)
    assert verifier.verify(token).claims["grant_id"] == "one"
    assert jwt.get_unverified_header(token)["typ"] == "at+jwt"
    bao.accepted_token = "broker-two"
    with pytest.raises(requests.HTTPError):
        signer.mint({"sub": "loom:one"}, audience="loom", ttl_seconds=60)
    bao.token_file.write_text("broker-two")
    renewed = signer.mint({"sub": "loom:one", "grant_id": "two"}, audience="loom", ttl_seconds=60)
    assert verifier.verify(renewed).claims["grant_id"] == "two"


def test_remote_transit_rotation_accepts_retained_and_new_keys_after_refresh_bound(bao, signer):
    clock = [0.0]
    verifier = OpenBaoTransitVerifier(signer, audiences=frozenset({"loom"}), now=lambda: clock[0])
    old = signer.mint({"sub": "loom:one"}, audience="loom", ttl_seconds=60)
    bao.keys[2] = generate_ed25519_keypair()
    bao.latest_version = 2
    assert jwt.get_unverified_header(signer.mint({"sub": "pinned"}, audience="loom", ttl_seconds=60))["kid"] == (
        jwt.get_unverified_header(old)["kid"]
    )
    next_signer = OpenBaoTransitSigner(signer.client, issuer=signer.issuer, key_version=2)
    new = next_signer.mint({"sub": "loom:two"}, audience="loom", ttl_seconds=60)
    with pytest.raises(ValueError):
        verifier.verify(new)
    clock[0] = 1.0
    assert verifier.verify(new).sub == "loom:two"
    assert verifier.verify(old).sub == "loom:one"


def test_transit_wrong_signing_version_cannot_return_a_grant_token(bao, signer):
    bao.returned_version = 2
    with pytest.raises(ValueError):
        signer.mint({"sub": "loom:one"}, audience="loom", ttl_seconds=60)


def test_transit_key_retirement_invalidates_cached_key_after_five_minutes(bao, signer):
    old = signer.mint({"sub": "loom:one"}, audience="loom", ttl_seconds=600)
    bao.keys[2] = generate_ed25519_keypair()
    next_signer = OpenBaoTransitSigner(signer.client, issuer=signer.issuer, key_version=2)
    new = next_signer.mint({"sub": "loom:two"}, audience="loom", ttl_seconds=600)
    clock = [0.0]
    verifier = OpenBaoTransitVerifier(next_signer, audiences=frozenset({"loom"}), now=lambda: clock[0])
    del bao.keys[1]
    clock[0] = 299.0
    assert verifier.verify(old).sub == "loom:one"
    clock[0] = 300.0
    with pytest.raises(ValueError):
        verifier.verify(old)
    assert verifier.verify(new).sub == "loom:two"


def test_transit_expired_metadata_cannot_authorize_during_refresh_failure(bao, signer):
    clock = [0.0]
    verifier = OpenBaoTransitVerifier(signer, audiences=frozenset({"loom"}), now=lambda: clock[0])
    token = signer.mint({"sub": "loom:one"}, audience="loom", ttl_seconds=600)
    bao.accepted_token = "broker-two"
    assert verifier.verify(token).sub == "loom:one"
    clock[0] = 300.0
    with pytest.raises(requests.HTTPError):
        verifier.verify(token)
    with pytest.raises(ValueError):
        verifier.verify(token)
    bao.token_file.write_text("broker-two")
    clock[0] = 301.0
    assert verifier.verify(token).sub == "loom:one"


@pytest.mark.parametrize("refresh_time,known_authorized", [(1.0, True), (300.0, False)])
def test_transit_concurrent_refresh_preserves_only_fresh_cached_authority(bao, signer, refresh_time, known_authorized):
    clock = [0.0]
    verifier = OpenBaoTransitVerifier(signer, audiences=frozenset({"loom"}), now=lambda: clock[0])
    old = signer.mint({"sub": "loom:one"}, audience="loom", ttl_seconds=600)
    bao.keys[2] = generate_ed25519_keypair()
    next_signer = OpenBaoTransitSigner(signer.client, issuer=signer.issuer, key_version=2)
    new = next_signer.mint({"sub": "loom:two"}, audience="loom", ttl_seconds=600)
    started, release = threading.Event(), threading.Event()
    bao.key_metadata_gate = started, release
    clock[0] = refresh_time
    with ThreadPoolExecutor(max_workers=3) as pool:
        refreshing = pool.submit(verifier.verify, new)
        try:
            assert started.wait(timeout=1)
            known = pool.submit(verifier.verify, old)
            if known_authorized:
                assert known.result(timeout=1).sub == "loom:one"
            else:
                with pytest.raises(ValueError):
                    known.result(timeout=1)
            concurrent_unknown = pool.submit(verifier.verify, new)
            with pytest.raises(ValueError):
                concurrent_unknown.result(timeout=1)
        finally:
            release.set()
        assert refreshing.result(timeout=1).sub == "loom:two"


def test_identity_is_mapped_from_verified_entity_and_exact_issuer_and_audience(bao):
    verifier = OpenBaoIdentityVerifier(issuer=bao.issuer, audience="marin-auth", subjects={"entity-one": "loom:one"})
    assert verifier.verify(bao.identity_token()) == "loom:one"
    bad_tokens = (
        bao.identity_token(subject="entity-two"),
        bao.identity_token(audience="loom"),
        bao.identity_token(audience=["marin-auth", "loom"]),
        bao.identity_token(issuer="https://imposter.example/v1/identity/oidc"),
        bao.identity_token(expires_at=int(time.time()) - 1),
    )
    for token in bad_tokens:
        with pytest.raises(ValueError):
            verifier.verify(token)


def test_storage_issuer_checks_native_policy_identity_and_lifetime(bao):
    permission = Permission("cw_s3", frozenset({"s3:GetObject"}), "s3://logs/alerts/")
    issuer = OpenBaoStorageIssuer(
        OpenBaoClient(bao.address, bao.token_file),
        role="cw-read",
        issuer=bao.issuer,
        subject="entity-storage",
        permissions=(permission,),
    )
    now = time.time()
    grant = Grant(
        "grant-one", "user:alice", "loom:one", (permission,), None, "user:alice", now, now + 300, 0, None, "v1"
    )
    issued = issuer.issue(grant)
    assert jwt.decode(issued["token"], options={"verify_signature": False})["sub"] == "entity-storage"
    narrow = replace(grant, permissions=(replace(permission, resource="s3://logs/alerts/one/"),))
    with pytest.raises(GrantDenied):
        issuer.issue(narrow)
    bao.storage_expiry = int(grant.expires_at) + 60
    with pytest.raises(GrantDenied):
        issuer.issue(grant)
    bao.storage_expiry = int(now) + 60
    bao.storage_subject = "different-identity"
    with pytest.raises(ValueError):
        issuer.issue(grant)
