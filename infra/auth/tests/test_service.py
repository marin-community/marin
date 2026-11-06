# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json as json_module
import sqlite3
import time
from dataclasses import dataclass

import httpx
import jwt
import pytest
import requests
from rigging.grant_auth import ACCESS_TOKEN_TYPE, JWT_TOKEN_TYPE, TOKEN_EXCHANGE_GRANT, GrantAuthService
from rigging.token_authority import generate_ed25519_keypair, public_jwk_from_pem
from starlette.applications import Starlette
from starlette.testclient import TestClient

from infra.auth import service
from infra.auth.loom import LoomLauncher


class TransitSigner:
    def __init__(self, client, *, issuer, key_version):
        self.issuer = issuer
        self.key = generate_ed25519_keypair()

    def public_keys(self):
        return (self.key.public_pem,)

    def public_jwks(self):
        return {"keys": [public_jwk_from_pem(self.key.public_pem)]}

    def mint(self, claims, *, audience, ttl_seconds):
        now = int(time.time())
        return jwt.encode(
            {**claims, "iss": self.issuer, "aud": audience, "iat": now, "exp": now + ttl_seconds},
            self.key.private_pem,
            algorithm="EdDSA",
            headers={"typ": "at+jwt", "kid": self.key.kid},
        )


class IdentityVerifier:
    def __init__(self, **kwargs):
        pass

    def verify(self, token):
        if token != "grafana-identity":
            raise ValueError("Unknown test identity")
        return "service:grafana"


class StorageIssuer:
    def __init__(self, client, *, permissions, **kwargs):
        self.permissions = permissions

    def issue(self, grant):
        return {"token": "synthetic-native-token", "expires_at": int(time.time()) + 30}


class LoomHTTP:
    def __init__(self):
        self.grant_id = ""
        self.requests = []

    def post(self, url, *, json, headers, **kwargs):
        self.requests.append((url, json, headers))
        if url.endswith("/api/runs/list"):
            body = []
        elif url.endswith("/api/runs/create"):
            body = {
                "id": "run-one",
                "session_id": "session-one",
                "status": "running",
                "shared_grant_id": self.grant_id,
                "actor_subject": "service:grafana",
                "profile": "diagnostics",
            }
        else:
            raise AssertionError(f"Unexpected Loom API request: {url}")
        response = requests.Response()
        response.status_code = 200
        response._content = json_module.dumps(body).encode()
        return response

    def close(self):
        pass


@dataclass
class Environment:
    app: Starlette
    authority: GrantAuthService
    config: service.ServiceConfig
    loom: LoomHTTP


@pytest.fixture
def environment(tmp_path, monkeypatch, openfga_http):
    config = service.ServiceConfig.model_validate(
        {
            "bao_address": "http://127.0.0.1:18200",
            "bao_issuer": "https://bao.example/v1/identity/oidc",
            "issuer": "https://auth.example",
            "broker_token_file": tmp_path / "broker.token",
            "signing_key_version": 1,
            "database": tmp_path / "grants.db",
            "subjects": {"grafana": "service:grafana"},
            "openfga": {
                **{key: value for key, value in openfga_http.items() if key != "state"},
                "principals": [
                    {"principal": "service:grafana", "owner": "service:grafana", "max_ttl": 3600, "delegation_depth": 1}
                ],
            },
            "storage": [
                {
                    "name": name,
                    "role": "marin-coreweave",
                    "subject": name,
                    "token_file": tmp_path / f"{name}.token",
                    "permissions": [{"service": "cw_s3", "actions": ["s3:GetObject"], "resource": resource}],
                }
                for name, resource in (("diagnostics", "s3://logs/incidents/"), ("all-logs", "s3://logs/"))
            ],
            "max_token_ttl": 60,
            "loom": {"base_url": "https://loom.example", "observer_token_file": tmp_path / "observer.token"},
        }
    )
    config.loom.observer_token_file.write_text("loom_synthetic-operator")
    loom = LoomHTTP()
    monkeypatch.setattr(service, "OpenBaoTransitSigner", TransitSigner)
    monkeypatch.setattr(service, "OpenBaoIdentityVerifier", IdentityVerifier)
    monkeypatch.setattr(service, "OpenBaoStorageIssuer", StorageIssuer)
    monkeypatch.setattr(service.requests, "Session", lambda: loom)
    app, authority = service.create_service(config)
    yield Environment(app, authority, config, loom)
    authority.store.close()


def create_grant(client, *, parent_id=None, ttl=1800):
    response = client.post(
        "/v1/grants",
        headers={"Authorization": "Bearer grafana-identity"},
        json={
            "actor": "service:grafana",
            "permissions": [
                {"service": "loom", "actions": ["session.create"], "resource": "/diagnostics/"},
                {"service": "cw_s3", "actions": ["s3:GetObject"], "resource": "s3://logs/incidents/"},
            ],
            "expires_at": time.time() + ttl,
            "parent_id": parent_id,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def exchange(client, grant_id, audience):
    response = client.post(
        "/oauth/token",
        data={
            "grant_type": TOKEN_EXCHANGE_GRANT,
            "subject_token_type": JWT_TOKEN_TYPE,
            "subject_token": "grafana-identity",
            "grant_id": grant_id,
            "audience": audience,
        },
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.mark.parametrize("audience,expected", [("cw_s3", 200), ("loom", 403)])
def test_service_refreshes_only_storage_tokens(environment, audience, expected):
    with TestClient(environment.app) as client:
        grant_id = create_grant(client)
        headers = exchange(client, grant_id, audience)
        response = client.post(
            "/oauth/token",
            data={
                "grant_type": TOKEN_EXCHANGE_GRANT,
                "subject_token_type": ACCESS_TOKEN_TYPE,
                "subject_token": headers["Authorization"].removeprefix("Bearer "),
                "grant_id": grant_id,
                "audience": audience,
            },
        )
        assert response.status_code == expected


def test_service_denies_broader_provider_and_revoked_parent(environment):
    with TestClient(environment.app) as client:
        health = client.get("/healthz")
        assert health.status_code == 200 and health.json()["ready"] is True
        parent = create_grant(client)
        child = create_grant(client, parent_id=parent, ttl=1200)
        storage = exchange(client, child, "cw_s3")
        loom = exchange(client, child, "loom")
        allowed = client.post("/v1/storage/diagnostics/token", headers=storage)
        assert allowed.status_code == 200 and allowed.json()["token"] == "synthetic-native-token"
        assert allowed.headers["cache-control"] == "no-store"
        assert client.post("/v1/storage/all-logs/token", headers=storage).status_code == 403
        check = {"audience": "loom", "action": "session.create", "resource": "/diagnostics/"}
        assert client.post("/v1/authorize", headers=loom, json=check).status_code == 200
        revoked = client.post(f"/v1/grants/{parent}/revoke", headers={"Authorization": "Bearer grafana-identity"})
        assert revoked.status_code == 200
        assert client.post("/v1/authorize", headers=loom, json=check).status_code == 403
        assert client.post("/v1/storage/diagnostics/token", headers=storage).status_code == 403


def test_service_registers_credential_intent_and_forwards_only_the_shared_launch_token(environment):
    with TestClient(environment.app) as client:
        grant_id = create_grant(client)
        headers = exchange(client, grant_id, "loom")
        credentials = client.post("/v1/loom/credentials", headers=headers, json={"resource": "/diagnostics/"})
        assert credentials.status_code == 200
        result = credentials.json()
        assert result["grant_id"] == grant_id and result["actor"] == "service:grafana"
        assert jwt.decode(result["storage_token"], options={"verify_signature": False})["aud"] == "cw_s3"
        with sqlite3.connect(environment.config.database.parent / "loom-runs.db") as db:
            assert db.execute("SELECT grant_id,actor,template FROM loom_grant_runs").fetchall() == [
                (grant_id, "service:grafana", "diagnostics")
            ]
        launch_grant = create_grant(client)
        launch_headers = exchange(client, launch_grant, "loom")
        environment.loom.grant_id = launch_grant
        launched = client.post(
            "/v1/loom/launch",
            headers=launch_headers,
            json={"template": "diagnostics", "title": "Inspect", "goal": "Read the approved incident"},
        )
        assert launched.status_code == 201 and launched.json()["run_id"] == "run-one"
        delivery = [request for request in environment.loom.requests if request[0].endswith("/api/runs/create")]
        assert len(delivery) == 1 and delivery[0][2] == launch_headers


def test_openfga_outage_denies_requests_without_permanently_revoking_live_grants(environment, openfga_http):
    with TestClient(environment.app) as client:
        grant_id = create_grant(client)
        headers = exchange(client, grant_id, "loom")
        check = {"audience": "loom", "action": "session.create", "resource": "/diagnostics/"}
        assert client.post("/v1/authorize", headers=headers, json=check).status_code == 200
        grant = environment.authority.store.active(grant_id)
        launcher = LoomLauncher(
            environment.config.database.parent / "outage-observer.db",
            base_url="https://loom.example",
            observer_tokens=service.LoomObserverTokenProvider(environment.config.loom.observer_token_file),
            grants=environment.authority,
            session=environment.loom,
        )
        launcher.observe(grant, "diagnostics")
        openfga_http["state"]["status"] = 503
        assert client.post("/v1/authorize", headers=headers, json=check).status_code == 503
        launcher.reconcile()
        openfga_http["state"]["status"] = 200
        assert client.post("/v1/authorize", headers=headers, json=check).status_code == 200
        assert environment.authority.store.active(grant_id).id == grant_id


@pytest.mark.asyncio
async def test_fatal_reconciler_failure_makes_health_unavailable(environment):
    # Losing the observer ledger exercises an actual fatal storage error, beyond per-row recovery.
    (environment.config.database.parent / "loom-runs.db").unlink()
    with pytest.raises(ExceptionGroup) as error:
        async with environment.app.router.lifespan_context(environment.app):
            await asyncio.Future()
    assert any(isinstance(exception, sqlite3.Error) for exception in error.value.exceptions)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(environment.app), base_url="http://test") as client:
        response = await client.get("/healthz")
    assert response.status_code == 503 and response.json()["ready"] is False


def test_observer_does_not_send_an_openbao_token_from_a_distinct_file(environment):
    with TestClient(environment.app) as client:
        grant_id = create_grant(client)
        headers = exchange(client, grant_id, "loom")
        assert (
            client.post("/v1/loom/credentials", headers=headers, json={"resource": "/diagnostics/"}).status_code == 200
        )
        environment.config.loom.observer_token_file.write_text("hvs.synthetic-bao-secret")
        observer = LoomLauncher(
            environment.config.database.parent / "loom-runs.db",
            base_url=environment.config.loom.base_url,
            observer_tokens=service.LoomObserverTokenProvider(environment.config.loom.observer_token_file),
            grants=environment.authority,
            session=environment.loom,
        )
        observer.reconcile()
        assert not any(
            request_headers.get("Authorization") == "Bearer hvs.synthetic-bao-secret"
            for _, _, request_headers in environment.loom.requests
        )


@pytest.mark.parametrize("source", ["broker", "storage", "storage-alias"])
def test_observer_cannot_forward_an_openbao_credential(environment, source, tmp_path):
    config = environment.config.model_dump()
    if source == "broker":
        path = config["broker_token_file"]
    elif source == "storage":
        path = config["storage"][0]["token_file"]
    else:
        path = tmp_path / "observer-alias"
        path.symlink_to(config["storage"][0]["token_file"])
    config["loom"]["observer_token_file"] = path
    with pytest.raises(ValueError):
        service.ServiceConfig.model_validate(config)
