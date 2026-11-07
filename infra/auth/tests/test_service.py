# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json as json_module
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import jwt
import pytest
import requests
from rigging.grant_auth import ACCESS_TOKEN_TYPE, JWT_TOKEN_TYPE, TOKEN_EXCHANGE_GRANT, GrantAuthService
from rigging.token_authority import generate_ed25519_keypair, public_jwk_from_pem
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
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
        self.runs = {}

    def post(self, url, *, json, headers, **kwargs):
        self.requests.append((url, json, headers))
        if url.endswith("/api/runs/list"):
            body = []
        elif url.endswith("/api/runs/get"):
            body = self.runs.get(json["id"])
            if body is None:
                response = requests.Response()
                response.status_code = 404
                raise requests.HTTPError(response=response)
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


@dataclass
class SecretBoundary:
    token_file: Path
    address: str = ""
    requests: list = field(default_factory=list)
    status: int = 200
    on_read: Callable[[], None] | None = None


@pytest.fixture
def secret_http(tmp_path):
    state = SecretBoundary(tmp_path / "canary.token")
    state.token_file.write_text("native-canary-read-only")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state.requests.append((self.path, self.headers.get("X-Vault-Token")))
            status = state.status
            if (
                self.path != "/v1/auth-proof/data/task-canary"
                or self.headers.get("X-Vault-Token") != "native-canary-read-only"
            ):
                status = 403
            if state.on_read is not None:
                state.on_read()
            data = (
                {"data": {"data": {"canary": "synthetic-private-canary"}}}
                if status == 200
                else {"errors": ["synthetic-private-canary must never be relayed"]}
            )
            body = json_module.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        state.address = f"http://127.0.0.1:{server.server_port}"
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield state
        finally:
            server.shutdown()
            thread.join()


@pytest.fixture
def environment(tmp_path, monkeypatch, openfga_http, secret_http):
    config = service.ServiceConfig.model_validate(
        {
            "bao_address": secret_http.address,
            "bao_issuer": "https://bao.example/v1/identity/oidc",
            "issuer": "https://auth.example",
            "broker_token_file": tmp_path / "broker.token",
            "signing_key_version": 1,
            "database": {"backend": "sqlite", "path": tmp_path / "grants.db"},
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
            "secrets": [
                {
                    "name": "task-canary",
                    "mount": "auth-proof",
                    "path": "task-canary",
                    "token_file": secret_http.token_file,
                }
            ],
            "max_token_ttl": 60,
            "loom": {
                "base_url": "https://loom.example",
                "observer_token_file": tmp_path / "observer.token",
                "callback_token_file": tmp_path / "callback.token",
            },
        }
    )
    config.loom.observer_token_file.write_text("loom_synthetic-operator")
    config.loom.callback_token_file.write_text("independent-loom-callback")
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


def reserve_loom_run(environment, headers):
    token = headers["Authorization"].removeprefix("Bearer ")
    claims = jwt.decode(token, options={"verify_signature": False})
    run_id = f"reserved-{claims['grant_id']}"
    environment.loom.runs[run_id] = {
        "id": run_id,
        "shared_grant_id": claims["grant_id"],
        "actor_subject": claims["sub"],
        "profile": "diagnostics",
        "status": "creating",
        "session_id": f"session-{run_id}",
    }
    return run_id


def runtime_credentials(client, environment, headers):
    run_id = reserve_loom_run(environment, headers)
    return client.post(
        "/v1/loom/credentials",
        headers={**headers, "X-Marin-Loom-Callback": environment.config.loom.callback_token_file.read_text()},
        json={"resource": "/diagnostics/", "run_id": run_id, "session_id": environment.loom.runs[run_id]["session_id"]},
    )


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
        credentials = runtime_credentials(client, environment, headers)
        assert credentials.status_code == 200
        result = credentials.json()
        assert result["grant_id"] == grant_id and result["actor"] == "service:grafana"
        assert jwt.decode(result["storage_token"], options={"verify_signature": False})["aud"] == "cw_s3"
        with sqlite3.connect(environment.config.database.path) as db:
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
            environment.authority.store.database,
            base_url="https://loom.example",
            observer_tokens=service.LoomObserverTokenProvider(environment.config.loom.observer_token_file),
            grants=environment.authority,
            session=environment.loom,
        )
        run_id = reserve_loom_run(environment, headers)
        launcher.observe(grant, "diagnostics", run_id, environment.loom.runs[run_id]["session_id"])
        openfga_http["state"]["status"] = 503
        assert client.post("/v1/authorize", headers=headers, json=check).status_code == 503
        launcher.reconcile()
        openfga_http["state"]["status"] = 200
        assert client.post("/v1/authorize", headers=headers, json=check).status_code == 200
        assert environment.authority.store.active(grant_id).id == grant_id


@pytest.mark.asyncio
async def test_fatal_reconciler_failure_makes_health_unavailable(environment):
    # Losing the observer ledger exercises an actual fatal storage error, beyond per-row recovery.
    with environment.authority.store.database.transaction() as db:
        db.execute(text("DROP TABLE loom_grant_runs"))
    with pytest.raises(ExceptionGroup) as error:
        async with environment.app.router.lifespan_context(environment.app):
            await asyncio.Future()
    assert any(isinstance(exception, SQLAlchemyError) for exception in error.value.exceptions)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(environment.app), base_url="http://test") as client:
        response = await client.get("/healthz")
    assert response.status_code == 503 and response.json()["ready"] is False


def test_observer_does_not_send_an_openbao_token_from_a_distinct_file(environment):
    with TestClient(environment.app) as client:
        grant_id = create_grant(client)
        headers = exchange(client, grant_id, "loom")
        assert runtime_credentials(client, environment, headers).status_code == 200
        environment.config.loom.observer_token_file.write_text("hvs.synthetic-bao-secret")
        observer = LoomLauncher(
            environment.authority.store.database,
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


def test_runtime_tokens_share_one_narrowed_session_child_and_revoke_with_launch_grant(environment):
    with TestClient(environment.app) as client:
        response = client.post(
            "/v1/grants",
            headers={"Authorization": "Bearer grafana-identity"},
            json={
                "actor": "service:grafana",
                "permissions": [
                    {"service": "loom", "actions": ["session.create"], "resource": "/diagnostics/"},
                    {"service": "iris", "actions": ["job.submit"], "resource": "/diagnostics/"},
                    {"service": "cw_s3", "actions": ["s3:GetObject"], "resource": "s3://logs/incidents/"},
                ],
                "expires_at": time.time() + 300,
            },
        )
        assert response.status_code == 201, response.text
        parent = response.json()
        headers = exchange(client, parent["id"], "loom")
        first = runtime_credentials(client, environment, headers)
        retry = runtime_credentials(client, environment, headers)
        assert first.status_code == retry.status_code == 200
        result = first.json()
        iris = jwt.decode(result["iris_token"], options={"verify_signature": False})
        storage = jwt.decode(result["storage_token"], options={"verify_signature": False})
        retried = jwt.decode(retry.json()["iris_token"], options={"verify_signature": False})
        assert iris["grant_id"] == storage["grant_id"] == retried["grant_id"]
        assert iris["grant_id"] != parent["id"]
        assert iris["sub"] == storage["sub"] == f"loom:{parent['id']}"
        assert result["grant_id"] == parent["id"] and result["actor"] == parent["actor"]
        child = environment.authority.store.active(iris["grant_id"])
        assert child.parent_id == parent["id"] and child.expires_at == parent["expires_at"]
        assert {item.service for item in child.permissions} == {"iris", "cw_s3"}
        check = {"audience": "iris", "action": "job.submit", "resource": "/diagnostics/job"}
        runtime_headers = {"Authorization": f"Bearer {result['iris_token']}"}
        assert client.post("/v1/authorize", headers=runtime_headers, json=check).status_code == 200
        assert (
            client.post(
                f"/v1/grants/{parent['id']}/revoke", headers={"Authorization": "Bearer grafana-identity"}
            ).status_code
            == 200
        )
        assert client.post("/v1/authorize", headers=runtime_headers, json=check).status_code == 403


def test_database_outage_returns_sanitized_503_and_recovers(environment):
    assert isinstance(environment.config.database, service.SqliteDatabaseConfig)
    path = environment.config.database.path
    backup = path.with_suffix(".saved")
    client = TestClient(environment.app)
    try:
        create_grant(client)
        path.rename(backup)
        path.mkdir()
        try:
            response = client.post(
                "/v1/grants",
                headers={"Authorization": "Bearer grafana-identity"},
                json={
                    "actor": "service:grafana",
                    "permissions": [{"service": "loom", "actions": ["session.create"], "resource": "/diagnostics/"}],
                    "expires_at": time.time() + 300,
                },
            )
            assert response.status_code == 503
            assert response.json() == {"error": "Authorization is unavailable"}
        finally:
            path.rmdir()
            backup.rename(path)
        create_grant(client)
    finally:
        client.close()


def secret_grant(client, *, resource="/secrets/task-canary/"):
    response = client.post(
        "/v1/grants",
        headers={"Authorization": "Bearer grafana-identity"},
        json={
            "actor": "service:grafana",
            "expires_at": time.time() + 300,
            "permissions": [{"service": "openbao", "actions": ["secret.read"], "resource": resource}],
        },
    )
    assert response.status_code == 201, response.text
    grant_id = response.json()["id"]
    return grant_id, exchange(client, grant_id, "openbao")


def test_secret_reads_use_only_the_fixed_binding_and_separate_native_identity(environment, secret_http):
    with TestClient(environment.app) as client:
        grant_id, headers = secret_grant(client)
        response = client.post("/v1/secrets/task-canary", headers=headers, json={"path": "/transit/keys/marin-grants"})
        assert response.status_code == 200
        assert response.json() == {"data": {"canary": "synthetic-private-canary"}}
        assert response.headers["cache-control"] == "no-store"
        assert secret_http.requests == [("/v1/auth-proof/data/task-canary", "native-canary-read-only")]
        assert client.post("/v1/secrets/other", headers=headers).status_code == 403
        _, other_headers = secret_grant(client, resource="/secrets/other/")
        assert client.post("/v1/secrets/task-canary", headers=other_headers).status_code == 403
        assert len(secret_http.requests) == 1
        client.post(f"/v1/grants/{grant_id}/revoke", headers={"Authorization": "Bearer grafana-identity"})
        assert client.post("/v1/secrets/task-canary", headers=headers).status_code == 403
        assert len(secret_http.requests) == 1


def test_secret_provider_failure_never_relays_sensitive_error_data(environment, secret_http, caplog):
    with TestClient(environment.app) as client:
        _, headers = secret_grant(client)
        secret_http.status = 500
        response = client.post("/v1/secrets/task-canary", headers=headers)
        assert response.status_code == 503
        assert response.json() == {"error": "Secret provider is unavailable"}
        assert "synthetic-private-canary" not in caplog.text


def test_revocation_during_provider_read_prevents_secret_delivery(environment, secret_http):
    with TestClient(environment.app) as client:
        grant_id, headers = secret_grant(client)
        secret_http.on_read = lambda: environment.authority.store.revoke(grant_id, principal="service:grafana")
        response = client.post("/v1/secrets/task-canary", headers=headers)
        assert response.status_code == 403
        assert "synthetic-private-canary" not in response.text


def test_loom_bearer_alone_cannot_exchange_runtime_authority(environment):
    with TestClient(environment.app) as client:
        parent = create_grant(client)
        headers = exchange(client, parent, "loom")
        run_id = reserve_loom_run(environment, headers)
        payload = {
            "resource": "/diagnostics/",
            "run_id": run_id,
            "session_id": environment.loom.runs[run_id]["session_id"],
        }
        assert client.post("/v1/loom/credentials", headers=headers, json=payload).status_code == 403
        assert (
            client.post(
                "/v1/loom/credentials", headers={**headers, "X-Marin-Loom-Callback": "wrong-service"}, json=payload
            ).status_code
            == 403
        )
        assert (
            environment.authority.store.grant_for_request(parent_id=parent, idempotency_key=f"loom-runtime:{parent}")
            is None
        )
        trusted_headers = {**headers, "X-Marin-Loom-Callback": environment.config.loom.callback_token_file.read_text()}
        environment.loom.runs[run_id]["shared_grant_id"] = "another-grant"
        assert client.post("/v1/loom/credentials", headers=trusted_headers, json=payload).status_code == 403
        assert (
            environment.authority.store.grant_for_request(parent_id=parent, idempotency_key=f"loom-runtime:{parent}")
            is None
        )
        assert runtime_credentials(client, environment, headers).status_code == 200
