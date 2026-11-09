# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real database replica tests; set MARIN_AUTH_TEST_POSTGRES_URL and select integration."""

import multiprocessing
import time
from contextlib import closing
from queue import Empty

import pytest
from rigging.grant_database import PostgresGrantDatabase
from rigging.grants import GrantDenied, GrantStore, Permission, PrincipalAuthority
from sqlalchemy import text

pytestmark = pytest.mark.integration
PERMISSIONS = (Permission("iris", frozenset({"job.submit"}), "/test/"),)
AUTHORITIES = (PrincipalAuthority("owner", "owner", 600, 3, "test"),)


class TestPolicy:
    __test__ = False

    def allows_grant(self, principal, permissions):
        return principal == "owner" and tuple(permissions) == PERMISSIONS

    def check(self, principal, service, action, resource):
        return principal == "owner" and service == "iris" and action == "job.submit" and resource.startswith("/test/")


def _store(url):
    return GrantStore(PostgresGrantDatabase(url, ssl_context=None), AUTHORITIES, TestPolicy())


def _create_worker(url, expiry, parent_id, operation, gate, results):
    with closing(_store(url)) as store:
        results.put("ready")
        assert gate.wait(15)
        if operation == "revoke":
            store.revoke(parent_id, principal="owner")
            results.put("revoked")
            return
        try:
            grant = store.create(
                "owner" if parent_id is None else "runtime",
                actor="runtime" if parent_id is None else "task",
                permissions=PERMISSIONS,
                expires_at=expiry,
                parent_id=parent_id,
                idempotency_key="same-request",
            )
            results.put(grant.id)
        except GrantDenied:
            results.put("denied")


def _race(url, expiry, parent_id, operations):
    context = multiprocessing.get_context("spawn")
    gate, results = context.Event(), context.Queue()
    processes = [
        context.Process(target=_create_worker, args=(url, expiry, parent_id, operation, gate, results))
        for operation in operations
    ]
    try:
        for process in processes:
            process.start()
        assert [results.get(timeout=20) for _ in processes] == ["ready"] * len(processes)
        gate.set()
        outcomes = [results.get(timeout=20) for _ in processes]
        for process in processes:
            process.join(timeout=15)
            assert process.exitcode == 0
        return outcomes
    except Empty:
        pytest.fail("A grant replica did not complete its database operation")
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()
        results.close()
        results.join_thread()


def test_separate_replicas_create_one_idempotent_grant_and_preserve_revocation(postgres_database_url):
    expiry = time.time() + 300
    ids = _race(postgres_database_url, expiry, None, ["create"] * 4)
    assert len(set(ids)) == 1
    with closing(_store(postgres_database_url)) as store:
        grant = store.create(
            "owner", actor="runtime", permissions=PERMISSIONS, expires_at=expiry, idempotency_key="same-request"
        )
        assert grant.id == ids[0]
        with store.database.transaction() as connection:
            assert connection.execute(text("SELECT count(*) FROM grants")).scalar_one() == 1
        store.revoke(grant.id, principal="owner")
    with closing(_store(postgres_database_url)) as reopened:
        with pytest.raises(GrantDenied):
            reopened.active(ids[0])
        with pytest.raises(GrantDenied):
            reopened.create(
                "owner", actor="runtime", permissions=PERMISSIONS, expires_at=expiry, idempotency_key="same-request"
            )


def test_concurrent_parent_revocation_never_leaves_an_active_descendant(postgres_database_url):
    expiry = time.time() + 300
    with closing(_store(postgres_database_url)) as store:
        parent = store.create("owner", actor="runtime", permissions=PERMISSIONS, expires_at=expiry)
    outcomes = _race(postgres_database_url, expiry, parent.id, ["create", "create", "revoke", "revoke"])
    assert outcomes.count("revoked") == 2
    child_ids = set(outcomes) - {"revoked", "denied"}
    assert len(child_ids) <= 1
    with closing(_store(postgres_database_url)) as reopened:
        for grant_id in {parent.id, *child_ids}:
            with pytest.raises(GrantDenied):
                reopened.authorize(grant_id, actor="task", service="iris", action="job.submit", resource="/test/job")
        with pytest.raises(GrantDenied):
            reopened.create("runtime", actor="task", permissions=PERMISSIONS, expires_at=expiry, parent_id=parent.id)


def _leader_worker(url, results, release):
    with closing(PostgresGrantDatabase(url, ssl_context=None)) as database:
        with database.leadership("test-lifecycle") as leader:
            results.put(leader)
            assert release.wait(15)


def test_process_exit_releases_lifecycle_leadership(postgres_database_url):
    context = multiprocessing.get_context("spawn")
    results, release = context.Queue(), context.Event()
    process = context.Process(target=_leader_worker, args=(postgres_database_url, results, release))
    process.start()
    try:
        assert results.get(timeout=20) is True
        with closing(PostgresGrantDatabase(postgres_database_url, ssl_context=None)) as replica:
            with replica.leadership("test-lifecycle") as leader:
                assert leader is False
            process.kill()
            process.join(timeout=10)
            with replica.serialize("test-lifecycle"):
                pass
            with replica.leadership("test-lifecycle") as leader:
                assert leader is True
    finally:
        if process.is_alive():
            process.kill()
        process.join()
        results.close()
        results.join_thread()
