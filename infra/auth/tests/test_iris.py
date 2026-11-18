# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import time
from dataclasses import dataclass, replace

import jwt
import pytest
from rigging.grant_auth import GrantAuthService, create_app
from rigging.grant_database import SqliteGrantDatabase
from rigging.grants import GrantDenied, GrantStore, Permission, PrincipalAuthority, permissions_cover
from rigging.testing import MemoryRuntimeGrants
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
    database = SqliteGrantDatabase(tmp_path / "attempts.sqlite")
    store = GrantStore(
        MemoryRuntimeGrants(),
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
        database=database,
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
    database.close()


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
            database=environment.broker.database,
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


def test_controller_renewal_preserves_identity_and_workload_refresh_cannot_extend_lease(iris_broker):
    environment = iris_broker
    initial = environment.broker.issue("controller-identity", environment.request)
    for _ in range(5):
        environment.clock[0] += 20
        renewed = environment.broker.renew("controller-identity", environment.request)
        assert renewed["grant_id"] == initial["grant_id"]
        assert renewed["actor"] == initial["actor"]
        assert renewed["expires_at"] == environment.clock[0] + 60
        assert environment.broker.issue("controller-identity", environment.request)["grant_id"] == initial["grant_id"]
    # The controller can sustain a lease across many original lease windows.
    assert environment.authority.store.active(initial["grant_id"]).credential_expires_at == initial["expires_at"] + 100
    with pytest.raises(GrantDenied):
        environment.broker.renew("session-identity", environment.request)
    # Workload refresh mints a token inside the lease; it never moves the deadline.
    environment.authority.refresh(initial["access_token"], audience="cw_s3")
    assert environment.authority.store.active(initial["grant_id"]).credential_expires_at == renewed["expires_at"]
    environment.clock[0] = renewed["expires_at"] + 1
    with pytest.raises(GrantDenied):
        environment.authority.refresh(initial["access_token"], audience="cw_s3")
    with pytest.raises(GrantDenied):
        environment.broker.renew("controller-identity", environment.request)


@pytest.mark.parametrize("withdrawal", ["parent", "child", "policy", "retire"])
def test_controller_renewal_cannot_restore_withdrawn_authority(iris_broker, monkeypatch, withdrawal):
    environment = iris_broker
    issued = environment.broker.issue("controller-identity", environment.request)
    environment.clock[0] += 20
    if withdrawal == "parent":
        environment.authority.store.revoke(environment.request.grant_id, principal="loom:session")
    elif withdrawal == "child":
        environment.authority.store.revoke(issued["grant_id"], principal=issued["actor"])
    elif withdrawal == "policy":
        monkeypatch.setattr(Policy, "allows_grant", lambda *args: False)
    else:
        environment.broker.retire("controller-identity", environment.request)
    with pytest.raises(GrantDenied):
        environment.broker.renew("controller-identity", environment.request)
    with pytest.raises(GrantDenied):
        environment.authority.refresh(issued["access_token"], audience="cw_s3")


def test_controller_renewal_never_outlives_parent_deadline(iris_broker):
    environment = iris_broker
    issued = environment.broker.issue("controller-identity", environment.request)
    parent = environment.authority.store.active(environment.request.grant_id)
    while environment.clock[0] + 20 < parent.expires_at:
        environment.clock[0] += 20
        renewed = environment.broker.renew("controller-identity", environment.request)
        assert renewed["expires_at"] <= parent.expires_at
    assert renewed["expires_at"] == parent.expires_at
    environment.clock[0] = parent.expires_at
    with pytest.raises(GrantDenied):
        environment.broker.renew("controller-identity", environment.request)
    with pytest.raises(GrantDenied):
        environment.authority.store.active(issued["grant_id"])


@pytest.mark.parametrize("withdrawal", ["parent", "child", "policy", "retire", "delay"])
def test_in_flight_renewal_cannot_escape_revocation_or_its_original_deadline(iris_broker, monkeypatch, withdrawal):
    environment = iris_broker
    issued = environment.broker.issue("controller-identity", environment.request)
    environment.clock[0] += 20
    deadline = environment.clock[0] + 60
    runtime = environment.authority.store.runtime
    renew = runtime.renew

    def delayed_renew(grant, *, expires_at):
        if withdrawal == "parent":
            environment.authority.store.revoke(environment.request.grant_id, principal="loom:session")
        elif withdrawal == "child":
            environment.authority.store.revoke(grant.id, principal=grant.actor)
        elif withdrawal == "policy":
            monkeypatch.setattr(Policy, "allows_grant", lambda *args: False)
        elif withdrawal == "retire":
            environment.broker.retire("controller-identity", environment.request)
        else:
            environment.clock[0] += 80
        return renew(grant, expires_at=expires_at)

    monkeypatch.setattr(runtime, "renew", delayed_renew)
    with pytest.raises(GrantDenied):
        environment.broker.renew("controller-identity", environment.request)
    assert runtime.get(issued["grant_id"]).lease_expires_at == deadline
    with pytest.raises(GrantDenied):
        environment.authority.store.active(issued["grant_id"])


def test_attempt_idempotency_cannot_adopt_an_unleased_grant(iris_broker):
    environment = iris_broker
    issued = environment.broker.issue("controller-identity", environment.request)
    child = environment.authority.store.active(issued["grant_id"])
    # Model an independently issued grant using the same deterministic identity.
    environment.authority.store.runtime.grants[child.id] = replace(child, lease_expires_at=None)
    with pytest.raises(GrantDenied, match="different grant request"):
        environment.broker.issue("controller-identity", environment.request)
