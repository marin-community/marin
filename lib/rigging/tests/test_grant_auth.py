# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import time
from dataclasses import dataclass
from importlib.resources import files

import httpx
import jwt
import pytest
from rigging.auth import StaticTokenProvider
from rigging.grant_auth import (
    ACCESS_TOKEN_TYPE,
    JWT_TOKEN_TYPE,
    TOKEN_EXCHANGE_GRANT,
    GrantAuthService,
    GrantTokenProvider,
    create_app,
    verified_grant_token,
)
from rigging.grant_database import SqliteGrantDatabase
from rigging.grants import GrantDenied, GrantStore, Permission, PrincipalAuthority
from rigging.shared_grants import SHARED_GRANT_ROLE, SharedGrantClient, SharedGrantConfig, SharedGrantUnavailable
from rigging.token_authority import JwksVerifier, generate_ed25519_keypair
from starlette.testclient import TestClient


class IdentityDirectory:
    def verify(self, token: str) -> str:
        identities = {
            "alice-identity": "user:alice",
            "session-one-identity": "loom:one",
            "session-two-identity": "loom:two",
        }
        if token not in identities:
            raise ValueError("Unknown identity")
        return identities[token]


class AccessSigner:
    """Local authority standing in for the remote OpenBao signing boundary."""

    def __init__(self):
        self.key = generate_ed25519_keypair()
        self.after_sign = lambda: None

    def mint(self, claims, *, audience, ttl_seconds):
        now = int(time.time())
        token = jwt.encode(
            {**claims, "iss": "https://auth.example", "aud": audience, "iat": now, "exp": now + ttl_seconds},
            self.key.private_pem,
            algorithm="EdDSA",
            headers={"kid": self.key.kid, "typ": "at+jwt"},
        )
        self.after_sign()
        return token


@dataclass
class AuthEnvironment:
    store: GrantStore
    service: GrantAuthService
    signer: AccessSigner
    permissions: tuple[Permission, ...]
    clock: list[float]


@pytest.fixture
def refresh_audiences():
    return frozenset({"cw_s3"})


@pytest.fixture
def auth(tmp_path, grant_permissions, grant_policy, refresh_audiences):
    clock = [time.time()]
    store = GrantStore(
        SqliteGrantDatabase(tmp_path / "grants.db"),
        [PrincipalAuthority("user:alice", "user:alice", 600, 2, "policy-v1")],
        grant_policy,
        now=lambda: clock[0],
    )
    signer = AccessSigner()
    verifier = JwksVerifier(
        issuers={"https://auth.example": [signer.key.public_pem]},
        expected_audiences=frozenset({"loom", "iris", "cw_s3"}),
        leeway_seconds=0,
    )
    service = GrantAuthService(
        store,
        IdentityDirectory(),
        signer,
        verifier,
        max_token_ttl=60,
        refresh_audiences=refresh_audiences,
        now=lambda: clock[0],
    )
    yield AuthEnvironment(store, service, signer, grant_permissions, clock)
    store.close()


def _session_grant(auth):
    return auth.store.create(
        "user:alice", actor="loom:one", permissions=auth.permissions, expires_at=auth.clock[0] + 300
    )


def _token_request(token, grant_id, *, audience="iris", token_type=JWT_TOKEN_TYPE):
    return {
        "grant_type": TOKEN_EXCHANGE_GRANT,
        "subject_token_type": token_type,
        "subject_token": token,
        "audience": audience,
        "grant_id": grant_id,
    }


def test_http_exchange_authorizes_multiple_services_with_distinct_audiences(auth):
    grant = _session_grant(auth)
    with TestClient(create_app(auth.service)) as client:
        provider = GrantTokenProvider(
            StaticTokenProvider("session-one-identity"),
            lambda form: client.post("/oauth/token", data=form).json(),
            grant_id=grant.id,
            audience="iris",
        )
        token = provider.get_token()
        assert jwt.decode(token, options={"verify_signature": False})["grant_version"] == 1
        authorized = auth.service.authorize(token, audience="iris", action="job.submit", resource="/alice/train")
        assert authorized.owner == "user:alice"
        with pytest.raises(GrantDenied):
            auth.service.authorize(token, audience="loom", action="session.create", resource="/alice/session")
        storage = auth.service.storage_token(authorized)
        assert (
            auth.service.authorize(
                storage, audience="cw_s3", action="s3:GetObject", resource="s3://logs/alerts/current.json"
            ).id
            == grant.id
        )
        with pytest.raises(GrantDenied):
            auth.service.authorize(storage, audience="cw_s3", action="s3:PutObject", resource="s3://logs/alerts/write")
        with pytest.raises(GrantDenied):
            auth.service.authorize(storage, audience="cw_s3", action="s3:GetObject", resource="s3://logs/private/read")


def test_receiving_service_checks_current_authority_and_recovers_after_outage(auth, monkeypatch):
    grant = _session_grant(auth)
    token = auth.service.exchange("session-one-identity", grant_id=grant.id, audience="iris")["access_token"]
    remote = SharedGrantClient(
        SharedGrantConfig(
            issuer="https://auth.example",
            public_keys=(auth.signer.key.public_pem,),
            authority_url="https://auth.example",
        ),
        audience="iris",
    )
    unavailable = [False]
    with TestClient(create_app(auth.service)) as authority:

        def post(_url, *, headers, json, **_kwargs):
            if unavailable[0]:
                return httpx.Response(503)
            return authority.post("/v1/authorize", headers=headers, json=json)

        monkeypatch.setattr("rigging.shared_grants.requests.post", post)
        assert remote.verify(token).role == SHARED_GRANT_ROLE
        assert remote.authorize(token, action="job.submit", resource="/alice/train").grant_id == grant.id
        with pytest.raises(GrantDenied):
            remote.authorize(token, action="job.submit", resource="/other/train")
        unavailable[0] = True
        with pytest.raises(SharedGrantUnavailable):
            remote.authorize(token, action="job.submit", resource="/alice/train")
        unavailable[0] = False
        assert remote.authorize(token, action="job.submit", resource="/alice/train").grant_id == grant.id
        auth.store.revoke(grant.id, principal=grant.owner)
        with pytest.raises(GrantDenied):
            remote.authorize(token, action="job.submit", resource="/alice/train")


def test_retrying_grant_creation_preserves_identity_and_expiry(auth):
    request = dict(
        actor="loom:one", permissions=auth.permissions, expires_at=auth.clock[0] + 100, idempotency_key="alert-one"
    )
    original = auth.store.create("user:alice", **request)
    retry = auth.store.create("user:alice", **{**request, "expires_at": auth.clock[0] + 200})
    assert retry.id == original.id
    assert retry.expires_at == original.expires_at
    with pytest.raises(GrantDenied):
        auth.store.create("user:alice", **{**request, "actor": "loom:two"})
    auth.store.revoke(original.id, principal=original.owner)
    with pytest.raises(GrantDenied):
        auth.store.create("user:alice", **request)


def test_http_owner_cannot_be_self_asserted_and_actor_must_authenticate(auth):
    with TestClient(create_app(auth.service)) as client:
        body = {
            "actor": "loom:one",
            "permissions": [{"service": "iris", "actions": ["job.submit"], "resource": "/alice/"}],
            "expires_at": auth.clock[0] + 100,
        }
        forged = client.post(
            "/v1/grants", headers={"Authorization": "Bearer alice-identity"}, json={**body, "owner": "admin"}
        )
        assert forged.status_code == 400
        created = client.post("/v1/grants", headers={"Authorization": "Bearer alice-identity"}, json=body)
        assert created.status_code == 201
        assert created.json()["owner"] == "user:alice"
        grant_id = created.json()["id"]
        denied = client.post("/oauth/token", data=_token_request("alice-identity", grant_id))
        assert denied.status_code == 403
        other = client.post("/oauth/token", data=_token_request("session-two-identity", grant_id))
        assert other.status_code == 403
        allowed = client.post("/oauth/token", data=_token_request("session-one-identity", grant_id))
        assert allowed.status_code == 200
        assert allowed.headers["cache-control"] == "no-store"


def test_storage_refresh_preserves_audience_grant_and_narrowed_token_bounds(auth, grant_policy):
    permissions = (Permission("cw_s3", frozenset({"s3:GetObject", "s3:PutObject"}), "s3://logs/alerts/"),)
    grant_policy.permissions["user:alice"] += permissions
    grant = auth.store.create("user:alice", actor="loom:one", permissions=permissions, expires_at=auth.clock[0] + 300)
    result = auth.service.mint_for(
        grant,
        audience="cw_s3",
        permissions=(Permission("cw_s3", frozenset({"s3:GetObject"}), "s3://logs/alerts/train/"),),
    )
    with TestClient(create_app(auth.service)) as client:
        refreshed = client.post(
            "/oauth/token",
            data=_token_request(result["access_token"], grant.id, audience="cw_s3", token_type=ACCESS_TOKEN_TYPE),
        )
        assert refreshed.status_code == 200
        assert refreshed.json()["access_token"] != result["access_token"]
        token = refreshed.json()["access_token"]
        assert (
            auth.service.authorize(
                token, audience="cw_s3", action="s3:GetObject", resource="s3://logs/alerts/train/current.json"
            ).id
            == grant.id
        )
        with pytest.raises(GrantDenied):
            auth.service.authorize(
                token, audience="cw_s3", action="s3:PutObject", resource="s3://logs/alerts/train/current.json"
            )
        with pytest.raises(GrantDenied):
            auth.service.authorize(
                token, audience="cw_s3", action="s3:GetObject", resource="s3://logs/alerts/elsewhere/current.json"
            )
        different_audience = client.post(
            "/oauth/token", data=_token_request(token, grant.id, audience="loom", token_type=ACCESS_TOKEN_TYPE)
        )
        assert different_audience.status_code == 403
        other_grant = _session_grant(auth)
        different_grant = client.post(
            "/oauth/token", data=_token_request(token, other_grant.id, audience="cw_s3", token_type=ACCESS_TOKEN_TYPE)
        )
        assert different_grant.status_code == 403


@pytest.mark.parametrize("audience", ["loom", "iris"])
def test_http_service_tokens_require_identity_for_renewal(auth, audience):
    grant = _session_grant(auth)
    with TestClient(create_app(auth.service)) as client:
        issued = client.post("/oauth/token", data=_token_request("session-one-identity", grant.id, audience=audience))
        assert issued.status_code == 200
        refreshed = client.post(
            "/oauth/token",
            data=_token_request(
                issued.json()["access_token"], grant.id, audience=audience, token_type=ACCESS_TOKEN_TYPE
            ),
        )
        assert refreshed.status_code == 403
        reissued = client.post("/oauth/token", data=_token_request("session-one-identity", grant.id, audience=audience))
        assert reissued.status_code == 200


@pytest.mark.parametrize("refresh_audiences,expected", [(frozenset(), 403), (frozenset({"iris"}), 200)])
def test_http_refresh_uses_deployment_policy(auth, refresh_audiences, expected):
    grant = _session_grant(auth)
    with TestClient(create_app(auth.service)) as client:
        issued = client.post("/oauth/token", data=_token_request("session-one-identity", grant.id))
        assert issued.status_code == 200
        refreshed = client.post(
            "/oauth/token",
            data=_token_request(issued.json()["access_token"], grant.id, token_type=ACCESS_TOKEN_TYPE),
        )
        assert refreshed.status_code == expected


def test_revocation_stops_existing_tokens_exchange_and_refresh(auth):
    grant = _session_grant(auth)
    token = auth.service.exchange("session-one-identity", grant_id=grant.id, audience="cw_s3")["access_token"]
    auth.service.revoke(grant.id, "user:alice")
    assert not auth.service.active(grant.id, "loom:one")
    with pytest.raises(GrantDenied):
        auth.service.authorize(token, audience="cw_s3", action="s3:GetObject", resource="s3://logs/alerts/train")
    with pytest.raises(GrantDenied):
        auth.service.refresh(token, audience="cw_s3")
    with pytest.raises(GrantDenied):
        auth.service.exchange("session-one-identity", grant_id=grant.id, audience="cw_s3")


def test_revocation_during_signing_does_not_return_a_new_token(auth):
    grant = _session_grant(auth)
    auth.signer.after_sign = lambda: auth.store.revoke(grant.id, principal="user:alice")
    with pytest.raises(GrantDenied):
        auth.service.exchange("session-one-identity", grant_id=grant.id, audience="iris")


def test_policy_withdrawal_stops_existing_tokens_exchange_and_refresh(auth, grant_policy):
    grant = _session_grant(auth)
    token = auth.service.exchange("session-one-identity", grant_id=grant.id, audience="cw_s3")["access_token"]
    grant_policy.permissions.clear()
    with pytest.raises(GrantDenied):
        auth.service.authorize(token, audience="cw_s3", action="s3:GetObject", resource="s3://logs/alerts/train")
    with pytest.raises(GrantDenied):
        auth.service.exchange("session-one-identity", grant_id=grant.id, audience="cw_s3")
    with pytest.raises(GrantDenied):
        auth.service.refresh(token, audience="cw_s3")


def test_child_policy_withdrawal_stops_token_issuance_without_disabling_parent(auth, grant_policy):
    parent = _session_grant(auth)
    child = auth.store.create(
        "loom:one",
        actor="loom:two",
        permissions=(Permission("cw_s3", frozenset({"s3:GetObject"}), "s3://logs/alerts/restricted/"),),
        expires_at=auth.clock[0] + 100,
        parent_id=parent.id,
    )
    token = auth.service.exchange("session-two-identity", grant_id=child.id, audience="cw_s3")["access_token"]
    grant_policy.denied_grant_resources.add("s3://logs/alerts/restricted/")
    assert auth.store.active(parent.id).id == parent.id
    with pytest.raises(GrantDenied):
        auth.service.exchange("session-two-identity", grant_id=child.id, audience="cw_s3")
    with pytest.raises(GrantDenied):
        auth.service.refresh(token, audience="cw_s3")
    with pytest.raises(GrantDenied):
        auth.service.authorize(
            token, audience="cw_s3", action="s3:GetObject", resource="s3://logs/alerts/restricted/train"
        )


def test_live_grant_expiry_rejects_cryptographically_valid_token(auth):
    grant = _session_grant(auth)
    token = auth.service.exchange("session-one-identity", grant_id=grant.id, audience="cw_s3")["access_token"]
    auth.clock[0] = grant.expires_at
    with pytest.raises(GrantDenied):
        auth.service.authorize(token, audience="cw_s3", action="s3:GetObject", resource="s3://logs/alerts/train")
    with pytest.raises(GrantDenied):
        auth.service.refresh(token, audience="cw_s3")


def test_jwt_expiry_and_header_type_are_checked_before_authorization(auth):
    grant = _session_grant(auth)
    token = auth.service.exchange("session-one-identity", grant_id=grant.id, audience="iris")["access_token"]
    claims = jwt.decode(token, options={"verify_signature": False})
    expired = jwt.encode(
        {**claims, "exp": int(time.time()) - 1},
        auth.signer.key.private_pem,
        algorithm="EdDSA",
        headers={"typ": "at+jwt"},
    )
    with pytest.raises(ValueError):
        auth.service.authorize(expired, audience="iris", action="job.submit", resource="/alice/train")
    identity_type = jwt.encode(claims, auth.signer.key.private_pem, algorithm="EdDSA", headers={"typ": "JWT"})
    with pytest.raises(GrantDenied):
        auth.service.authorize(identity_type, audience="iris", action="job.submit", resource="/alice/train")


def test_signed_token_cannot_borrow_another_session_grant(auth):
    grant = _session_grant(auth)
    original = auth.service.exchange("session-one-identity", grant_id=grant.id, audience="iris")["access_token"]
    claims = jwt.decode(original, options={"verify_signature": False})
    for key in ("iss", "aud", "iat", "exp"):
        del claims[key]
    forged = auth.signer.mint({**claims, "sub": "loom:two", "act": {"sub": "loom:two"}}, audience="iris", ttl_seconds=60)
    with pytest.raises(GrantDenied):
        auth.service.authorize(forged, audience="iris", action="job.submit", resource="/alice/train")


def test_signed_shared_contract_vectors_match_rust_verifier():
    vectors = json.loads((files("rigging") / "grant_vectors.json").read_text())
    verifier = JwksVerifier(
        issuers={vectors["issuer"]: vectors["public_keys"]},
        expected_audiences=frozenset({vectors["audience"]}),
        leeway_seconds=0,
    )
    for vector in vectors["vectors"]:
        if not vector["valid"]:
            with pytest.raises(ValueError):
                verified_grant_token(verifier, vector["token"], audience=vectors["audience"])
            continue
        access = verified_grant_token(verifier, vector["token"], audience=vectors["audience"])
        assert access.actor == "loom:run-one", vector["name"]
        assert access.owner == "user:power", vector["name"]
        for operation in vectors["operations"]:
            assert access.allows(operation["action"], operation["resource"]) == operation["allowed"], vector["name"]
