# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
from rigging.auth import StaticTokenProvider
from rigging.grant_database import SqliteGrantDatabase
from rigging.grants import GrantDenied, GrantStore, Permission, PrincipalAuthority
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from infra.auth.loom import LoomLauncher


class GrantService:
    def __init__(self, store, grant):
        self.store = store
        self.grant = grant
        self.token = "synthetic-shared-grant-bearer"
        self.active_gate = None

    def authorize(self, token, *, audience, action, resource):
        if token != self.token:
            raise GrantDenied("Unknown shared credential")
        return self.store.authorize(
            self.grant.id, actor=self.grant.actor, service=audience, action=action, resource=resource
        )

    def active(self, grant_id, actor):
        if self.active_gate is not None and self.active_gate[0] == grant_id:
            _, started, release = self.active_gate
            self.active_gate = None
            started.set()
            assert release.wait(10)
        try:
            return self.store.active(grant_id).actor == actor
        except GrantDenied:
            return False

    def revoke(self, grant_id, principal):
        self.store.revoke(grant_id, principal=principal)


class LaunchPolicy:
    def __init__(self, permission):
        self.permission = permission

    def allows_grant(self, principal, permissions):
        return principal == "owner" and tuple(permissions) == (self.permission,)

    def check(self, principal, service, action, resource):
        return (
            principal == "owner"
            and service == self.permission.service
            and action in self.permission.actions
            and resource == self.permission.resource
        )


@pytest.fixture
def grant_service(tmp_path):
    permission = Permission("loom", frozenset({"session.create"}), "/review/")
    with closing(
        GrantStore(
            SqliteGrantDatabase(tmp_path / "grants.sqlite"),
            [PrincipalAuthority("owner", "owner", 300, 1, "1")],
            LaunchPolicy(permission),
            now=lambda: 1000,
        )
    ) as store:
        grant = store.create("owner", actor="runtime", permissions=(permission,), expires_at=1200)
        yield GrantService(store, grant)


@dataclass
class LoomBoundary:
    base_url: str = ""
    runs: dict = field(default_factory=dict)
    archived: set = field(default_factory=set)
    redirected: list = field(default_factory=list)
    metadata_override: dict = field(default_factory=dict)
    redirect: bool = False
    drop_response: bool = False
    unavailable: bool = False
    get_responses: dict = field(default_factory=dict)
    get_started: threading.Event | None = None
    release_get: threading.Event | None = None
    authority_at_request: list = field(default_factory=list)


@pytest.fixture
def loom_api(grant_service):
    state = LoomBoundary()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            operation = self.path.removeprefix("/api/")
            state.authority_at_request.append(grant_service.active(grant_service.grant.id, grant_service.grant.actor))
            if state.unavailable:
                self.respond({}, status=503)
                return
            if self.path == "/credential-sink":
                state.redirected.append(self.headers.get("Authorization"))
            if state.redirect and operation == "runs/create":
                self.send_response(307)
                self.send_header("Location", "/credential-sink")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            expected = grant_service.token if operation == "runs/create" else "lifecycle-observer"
            if self.headers.get("Authorization") != f"Bearer {expected}":
                self.respond({}, status=403)
                return
            if operation == "runs/create":
                suffix = f"-{len(state.runs) + 1}" if state.runs else ""
                result = {
                    "id": f"native-run{suffix}",
                    "session_id": f"native-session{suffix}",
                    "status": "running",
                    "actor_subject": grant_service.grant.actor,
                    "shared_grant_id": grant_service.grant.id,
                    "idempotency_key": payload["idempotency_key"],
                    "profile": payload["profile"],
                }
                state.runs[result["id"]] = result
                if state.drop_response:
                    self.close_connection = True
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                    return
                self.respond({**result, **state.metadata_override})
            elif operation == "runs/get":
                status, result = state.get_responses.get(payload["id"], (200, state.runs.get(payload["id"])))
                if state.get_started is not None and not state.get_started.is_set():
                    result = dict(result)
                    state.get_started.set()
                    assert state.release_get.wait(15)
                self.respond(result, status=status)
            elif operation == "runs/list":
                self.respond(list(state.runs.values()))
            elif operation == "sessions/archive":
                state.archived.add(payload["session"])
                self.respond({})
            else:
                self.respond({}, status=404)

        def respond(self, result, status=200):
            content = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        state.base_url = f"http://127.0.0.1:{server.server_port}"
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield state
        finally:
            server.shutdown()
            thread.join()


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.integration)])
def launcher(request, tmp_path, grant_service, loom_api):
    database = (
        SqliteGrantDatabase(tmp_path / "runs.sqlite")
        if request.param == "sqlite"
        else request.getfixturevalue("postgres_auth_database")
    )
    with requests.Session() as session:
        yield LoomLauncher(
            database,
            base_url=loom_api.base_url,
            observer_tokens=StaticTokenProvider("lifecycle-observer"),
            grants=grant_service,
            session=session,
        )


def restarted_launcher(launcher):
    return LoomLauncher(
        launcher.database,
        base_url=launcher.base_url,
        observer_tokens=launcher.observer_tokens,
        grants=launcher.grants,
        session=launcher.session,
    )


def test_native_launch_uses_shared_bearer_and_preserves_run_metadata(launcher, grant_service, loom_api):
    run = launcher.launch(grant_service.token, template="review", title="Review", goal="Read change")
    assert (run.run_id, run.session_id, run.actor, run.grant_id, run.template) == (
        "native-run",
        "native-session",
        grant_service.grant.actor,
        grant_service.grant.id,
        "review",
    )
    assert loom_api.runs[run.run_id]["shared_grant_id"] == run.grant_id
    assert launcher.launch(grant_service.token, template="review", title="Review", goal="Read change") == run
    with launcher.database.transaction() as db:
        records = db.execute(text("SELECT * FROM loom_grant_runs")).mappings().all()
    assert grant_service.token not in repr(records)


@pytest.mark.parametrize("field", ["shared_grant_id", "actor_subject", "profile"])
def test_native_launch_rejects_response_belonging_to_another_grant_actor_or_profile(
    launcher, grant_service, loom_api, field
):
    loom_api.metadata_override[field] = "someone-else"
    with pytest.raises(ValueError):
        launcher.launch(grant_service.token, template="review", title="Review", goal="Read change")
    assert grant_service.active(grant_service.grant.id, grant_service.grant.actor)


def test_revoked_grant_cannot_launch_native_run(launcher, grant_service, loom_api):
    grant_service.revoke(grant_service.grant.id, "owner")
    with pytest.raises(GrantDenied):
        launcher.launch(grant_service.token, template="review", title="Review", goal="Read change")
    assert loom_api.runs == {}


def test_completed_run_revokes_runtime_grant_after_launcher_restart(launcher, grant_service, loom_api):
    run = launcher.launch(grant_service.token, template="review", title="Review", goal="Read change")
    loom_api.runs[run.run_id]["status"] = "completed"
    restarted = restarted_launcher(launcher)
    observed = restarted.reconcile()
    assert observed[0].run_id == run.run_id
    assert not grant_service.active(grant_service.grant.id, grant_service.grant.actor)
    assert restarted.reconcile() == []


def test_revoked_grant_archives_its_actual_native_session(launcher, grant_service, loom_api):
    run = launcher.launch(grant_service.token, template="review", title="Review", goal="Read change")
    grant_service.revoke(grant_service.grant.id, "owner")
    launcher.reconcile()
    assert loom_api.archived == {run.session_id}


def test_redirect_does_not_forward_shared_credential(launcher, grant_service, loom_api):
    loom_api.redirect = True
    with pytest.raises(requests.HTTPError):
        launcher.launch(grant_service.token, template="review", title="Review", goal="Read change")
    assert loom_api.redirected == []
    assert loom_api.runs == {}


def test_lost_native_response_is_recovered_by_grant_metadata_after_restart(launcher, grant_service, loom_api):
    loom_api.drop_response = True
    with pytest.raises(requests.ConnectionError):
        launcher.launch(grant_service.token, template="review", title="Review", goal="Read change")
    assert len(loom_api.runs) == 1
    loom_api.runs["native-run"]["status"] = "completed"
    restarted = restarted_launcher(launcher)
    observed = restarted.reconcile()
    assert (observed[0].run_id, observed[0].grant_id) == ("native-run", grant_service.grant.id)
    assert not grant_service.active(grant_service.grant.id, grant_service.grant.actor)
    assert restarted.reconcile() == []


def test_direct_native_launch_is_observed_and_revoked_on_completion(launcher, grant_service, loom_api):
    response = launcher.session.post(
        loom_api.base_url + "/api/runs/create",
        headers={"Authorization": f"Bearer {grant_service.token}"},
        json={"profile": "review", "idempotency_key": f"direct:{grant_service.grant.id}"},
        timeout=5,
    )
    response.raise_for_status()
    native = response.json()
    loom_api.runs[native["id"]].update(status="creating")
    launcher.observe(grant_service.grant, "review", native["id"], native["session_id"])
    loom_api.runs[native["id"]].update(status="running", session_id=native["session_id"])
    observed = launcher.reconcile()
    assert (observed[0].run_id, observed[0].grant_id) == (native["id"], grant_service.grant.id)
    assert grant_service.active(grant_service.grant.id, grant_service.grant.actor)
    loom_api.runs[native["id"]]["status"] = "completed"
    restarted = restarted_launcher(launcher)
    assert restarted.reconcile()[0].status == "completed"
    assert not grant_service.active(grant_service.grant.id, grant_service.grant.actor)
    assert restarted.reconcile() == []


def test_observed_credential_without_native_run_is_revoked_after_binding_deadline(launcher, grant_service, loom_api):
    clock = [1000]
    observer = LoomLauncher(
        launcher.database,
        base_url=launcher.base_url,
        observer_tokens=launcher.observer_tokens,
        grants=grant_service,
        session=launcher.session,
        now=lambda: clock[0],
    )
    loom_api.unavailable = True
    with pytest.raises(requests.HTTPError):
        observer.launch(grant_service.token, template="review", title="Pending", goal="Read change")
    loom_api.unavailable = False
    observer.reconcile()
    assert grant_service.active(grant_service.grant.id, grant_service.grant.actor)
    clock[0] += 61
    assert observer.reconcile()[0].run_id == ""
    assert not grant_service.active(grant_service.grant.id, grant_service.grant.actor)
    assert loom_api.runs == {}
    assert observer.reconcile() == []


@pytest.mark.parametrize(
    "bad_response",
    [
        (404, {}),
        (200, {}),
        (200, {"status": "cleaned"}),
        (200, {"id": "other-run"}),
        (200, {"session_id": "other-session"}),
    ],
)
def test_bad_first_run_does_not_block_later_completed_grant(launcher, grant_service, loom_api, bad_response):
    first = launcher.launch(grant_service.token, template="review", title="First", goal="Read first change")
    second_grant = grant_service.store.create(
        "owner", actor="runtime-two", permissions=grant_service.grant.permissions, expires_at=1200
    )
    grant_service.grant = second_grant
    second = launcher.launch(grant_service.token, template="review", title="Second", goal="Read second change")
    status, override = bad_response
    loom_api.get_responses[first.run_id] = (
        status,
        {**loom_api.runs[first.run_id], **override} if override else override,
    )
    loom_api.runs[second.run_id]["status"] = "completed"
    launcher.reconcile()
    assert not grant_service.active(first.grant_id, first.actor)
    assert not grant_service.active(second.grant_id, second.actor)
    assert first.session_id in loom_api.archived
    assert launcher.reconcile() == []


def test_unbound_deadline_revokes_authority_during_total_loom_outage(launcher, grant_service, loom_api):
    first = launcher.launch(grant_service.token, template="review", title="Bound run", goal="Read change")
    stale = grant_service.store.create(
        "owner", actor="unbound-runtime", permissions=grant_service.grant.permissions, expires_at=1200
    )
    grant_service.grant = stale
    clock = [1000]
    observer = LoomLauncher(
        launcher.database,
        base_url=launcher.base_url,
        observer_tokens=launcher.observer_tokens,
        grants=grant_service,
        session=launcher.session,
        now=lambda: clock[0],
    )
    loom_api.unavailable = True
    with pytest.raises(requests.HTTPError):
        observer.launch(grant_service.token, template="review", title="Pending", goal="Read change")
    observer.reconcile()
    assert grant_service.active(grant_service.grant.id, grant_service.grant.actor)
    clock[0] += 61
    loom_api.authority_at_request.clear()
    observer.reconcile()
    assert not grant_service.active(grant_service.grant.id, grant_service.grant.actor)
    assert loom_api.authority_at_request[0] is False
    assert grant_service.active(first.grant_id, first.actor)
    loom_api.unavailable = False
    observer.reconcile()
    assert {run.grant_id for run in observer.reconcile()} == {first.grant_id}


@pytest.mark.integration
def test_stale_worker_cannot_resurrect_cleaned_run_after_losing_leadership(
    postgres_auth_database, grant_service, loom_api
):
    with requests.Session() as session:
        launcher = LoomLauncher(
            postgres_auth_database,
            base_url=loom_api.base_url,
            observer_tokens=StaticTokenProvider("lifecycle-observer"),
            grants=grant_service,
            session=session,
        )
        run = launcher.launch(grant_service.token, template="review", title="Review", goal="Read change")
        loom_api.get_started, loom_api.release_get = threading.Event(), threading.Event()
        with ThreadPoolExecutor(max_workers=1) as executor:
            stale = executor.submit(launcher.reconcile)
            try:
                assert loom_api.get_started.wait(10)
                # A primary failover drops advisory-lock sessions while a remote reply is in flight.
                with postgres_auth_database.engine.connect() as connection:
                    connection.execute(
                        text(
                            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = current_database() "
                            "AND pid != pg_backend_pid() "
                            "AND pid IN (SELECT pid FROM pg_locks WHERE locktype = 'advisory')"
                        )
                    )
                    connection.commit()
                loom_api.runs[run.run_id]["status"] = "completed"
                restarted_launcher(launcher).reconcile()
            finally:
                loom_api.release_get.set()
            try:
                stale.result(timeout=15)
            except SQLAlchemyError:
                pass  # The disconnected worker can fail when releasing its former session lock.
        with postgres_auth_database.transaction() as connection:
            status = connection.execute(
                text("SELECT status FROM loom_grant_runs WHERE grant_id = :id"), {"id": run.grant_id}
            ).scalar_one()
        assert status == "cleaned"
        assert not grant_service.active(run.grant_id, run.actor)
        assert launcher.reconcile() == []


@pytest.mark.integration
def test_retry_binding_between_retirement_phases_does_not_revoke_live_run(
    postgres_auth_database, grant_service, loom_api
):
    clock = [1000]
    with requests.Session() as session:
        observer = LoomLauncher(
            postgres_auth_database,
            base_url=loom_api.base_url,
            observer_tokens=StaticTokenProvider("lifecycle-observer"),
            grants=grant_service,
            session=session,
            now=lambda: clock[0],
        )
        pending = grant_service.grant
        loom_api.unavailable = True
        with pytest.raises(requests.HTTPError):
            observer.launch(grant_service.token, template="review", title="Retry", goal="Read change")
        loom_api.unavailable = False
        blocker = grant_service.store.create(
            "owner", actor="other-runtime", permissions=pending.permissions, expires_at=1200
        )
        grant_service.grant = blocker
        clock[0] = 1001
        observer.launch(grant_service.token, template="review", title="Other", goal="Read other change")
        started, release = threading.Event(), threading.Event()
        grant_service.active_gate = (blocker.id, started, release)
        clock[0] = 1059
        with ThreadPoolExecutor(max_workers=1) as executor:
            reconciliation = executor.submit(observer.reconcile)
            try:
                assert started.wait(10)
                clock[0] = 1061
                grant_service.grant = pending
                run = observer.launch(grant_service.token, template="review", title="Retry", goal="Read change")
            finally:
                release.set()
            reconciliation.result(timeout=15)
        assert run.session_id
        assert grant_service.active(pending.id, pending.actor)
        assert observer.reconcile()


def test_stale_callback_cannot_replace_a_known_running_launch(launcher, grant_service, loom_api):
    run = launcher.launch(grant_service.token, template="review", title="Review", goal="Read change")
    # The observer reply is stale; the local ledger already recorded successful provisioning.
    loom_api.runs[run.run_id]["status"] = "creating"
    with pytest.raises(GrantDenied):
        launcher.observe(grant_service.grant, "review", run.run_id, run.session_id)
    loom_api.runs[run.run_id]["status"] = "running"
    assert launcher.reconcile()[0].status == "running"
    assert grant_service.active(run.grant_id, run.actor)
