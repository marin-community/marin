# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import time
from dataclasses import dataclass

import jwt
import pytest
from rigging.grant_auth import GrantAuthService, create_app
from rigging.grant_database import SqliteGrantDatabase
from rigging.grants import GrantDenied, GrantStore, Permission, PrincipalAuthority, permissions_cover
from rigging.token_authority import JwksVerifier, generate_ed25519_keypair
from starlette.testclient import TestClient

from infra.auth.iris import IrisAttemptBroker, IrisAttemptRequest, IrisBrokerConfig, mount_iris_attempts


class IdentityDirectory:
    def verify(self, token):
        if token == "controller-identity":
            return "controller:staging"
        if token == "session-identity":
            return "loom:session"
        raise ValueError("Unknown synthetic identity")


class Signer:
    def __init__(self, clock):
        self.key = generate_ed25519_keypair()
        self.clock = clock

    def mint(self, claims, *, audience, ttl_seconds):
        now = int(self.clock[0])
        return jwt.encode(
            {**claims, "iss": "https://auth.example", "aud": audience, "iat": now, "exp": now + ttl_seconds},
            self.key.private_pem,
            algorithm="EdDSA",
            headers={"typ": "at+jwt"},
        )


class Policy:
    permissions = (
        Permission("iris", frozenset({"job.submit"}), "/clusters/staging/jobs/research/"),
        Permission("cw_s3", frozenset({"s3:GetObject"}), "s3://auth-proof/allowed/"),
    )

    def allows_grant(self, principal, permissions):
        return principal == "owner" and permissions_cover(self.permissions, permissions)

    def check(self, principal, service, action, resource):
        return self.allows_grant(principal, (Permission(service, frozenset({action}), resource),))


@dataclass
class BrokerEnvironment:
    authority: GrantAuthService
    broker: IrisAttemptBroker
    request: IrisAttemptRequest
    clock: list[float]


@pytest.fixture
def iris_broker(tmp_path):
    clock = [time.time()]
    store = GrantStore(
        SqliteGrantDatabase(tmp_path / "grants.sqlite"),
        [PrincipalAuthority("owner", "research", 600, 3, "policy-1")],
        Policy(),
        now=lambda: clock[0],
    )
    signer = Signer(clock)
    authority = GrantAuthService(
        store,
        IdentityDirectory(),
        signer,
        JwksVerifier(
            issuers={"https://auth.example": [signer.key.public_pem]},
            expected_audiences=frozenset({"cw_s3", "iris"}),
            leeway_seconds=0,
        ),
        max_token_ttl=60,
        refresh_audiences=frozenset({"cw_s3"}),
        now=lambda: clock[0],
    )
    grant = store.create("owner", actor="loom:session", permissions=Policy.permissions, expires_at=clock[0] + 300)
    broker = IrisAttemptBroker(
        authority,
        IrisBrokerConfig(
            controller_clusters={"controller:staging": "staging"},
            permissions=[{"service": "cw_s3", "actions": ["s3:GetObject"], "resource": "s3://auth-proof/allowed/"}],
            audience="cw_s3",
            max_attempt_ttl=60,
        ),
        auth_url="https://auth.example",
        now=lambda: clock[0],
    )
    request = IrisAttemptRequest(
        grant_id=grant.id,
        resource="/clusters/staging/jobs/research/proof/",
        task_id="/research/proof/0",
        attempt_uid="0123456789abcdef",
    )
    yield BrokerEnvironment(authority, broker, request, clock)
    store.close()


def test_attempt_http_credentials_are_bounded_and_retirement_survives_broker_restart(iris_broker):
    environment = iris_broker
    app = create_app(environment.authority)
    mount_iris_attempts(app, environment.broker)
    with TestClient(app) as client:
        client.headers["Authorization"] = "Bearer controller-identity"
        response = client.post("/v1/iris/attempts/credentials", json=environment.request.model_dump())
        assert response.status_code == 200, response.text
        credentials = response.json()
        token = credentials["access_token"]
        grant = environment.authority.authorize(
            token, audience="cw_s3", action="s3:GetObject", resource="s3://auth-proof/allowed/"
        )
        assert grant.actor != "loom:session"
        assert grant.parent_id == environment.request.grant_id
        assert credentials["expires_at"] <= environment.clock[0] + 60
        for audience, action, resource in [
            ("iris", "job.submit", environment.request.resource),
            ("cw_s3", "s3:PutObject", "s3://auth-proof/allowed/"),
            ("cw_s3", "s3:GetObject", "s3://auth-proof/sibling/"),
        ]:
            with pytest.raises(GrantDenied):
                environment.authority.authorize(token, audience=audience, action=action, resource=resource)
        replay = client.post("/v1/iris/attempts/credentials", json=environment.request.model_dump())
        assert replay.json()["grant_id"] == grant.id
        retired = client.post("/v1/iris/attempts/retire", json=environment.request.model_dump())
        assert retired.status_code == 200
        with pytest.raises(GrantDenied):
            environment.authority.refresh(token, audience="cw_s3")
        # Reconstruct the broker over the same ledger, as a replacement replica does.
        replacement = IrisAttemptBroker(
            environment.authority,
            environment.broker.config,
            auth_url="https://auth.example",
            now=lambda: environment.clock[0],
        )
        assert replacement.retire("controller-identity", environment.request) == {"retired": True}
        with pytest.raises(GrantDenied):
            replacement.issue("controller-identity", environment.request)
        assert environment.authority.store.active(environment.request.grant_id).actor == "loom:session"


def test_attempt_credentials_reject_cross_cluster_parent_and_rebound_attempt(iris_broker):
    environment = iris_broker
    for updates in [
        {"resource": "/clusters/another/jobs/research/proof/"},
        {"resource": "/clusters/staging/jobs/other/proof/", "task_id": "/other/proof/0"},
    ]:
        with pytest.raises(GrantDenied):
            environment.broker.issue("controller-identity", environment.request.model_copy(update=updates))
    with pytest.raises(GrantDenied):
        environment.broker.issue("session-identity", environment.request)
    credentials = environment.broker.issue("controller-identity", environment.request)
    for updates in [
        {"task_id": "/research/proof/1"},
        {"resource": "/clusters/staging/jobs/research/sibling/", "task_id": "/research/sibling/0"},
    ]:
        with pytest.raises(GrantDenied):
            environment.broker.retire("controller-identity", environment.request.model_copy(update=updates))
    assert environment.authority.store.active(credentials["grant_id"]).actor == credentials["actor"]


def test_attempt_retirement_before_issue_blocks_resurrection_and_hard_expiry_bounds_outages(iris_broker):
    environment = iris_broker
    assert environment.broker.retire("controller-identity", environment.request) == {"retired": True}
    with pytest.raises(GrantDenied):
        environment.broker.issue("controller-identity", environment.request)
    next_attempt = environment.request.model_copy(update={"attempt_uid": "fedcba9876543210"})
    credentials = environment.broker.issue("controller-identity", next_attempt)
    environment.clock[0] += 61
    with pytest.raises(GrantDenied):
        environment.authority.refresh(credentials["access_token"], audience="cw_s3")
    with pytest.raises(GrantDenied):
        environment.broker.issue("controller-identity", next_attempt)
    assert environment.authority.store.active(environment.request.grant_id).id == environment.request.grant_id
