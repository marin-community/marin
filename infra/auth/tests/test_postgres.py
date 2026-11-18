# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replica tests requiring MARIN_AUTH_TEST_POSTGRES_URL and MARIN_AUTH_TEST_OPENFGA_EXECUTABLE."""

import multiprocessing
import os
import time
from contextlib import closing
from pathlib import Path
from queue import Empty

import pytest
from rigging.grant_database import PostgresGrantDatabase
from rigging.grants import GrantDenied, GrantStore, Permission, PrincipalAuthority
from sqlalchemy.engine import make_url

from infra.auth.openfga import OpenFGARuntime
from infra.auth.proof_fga import configure_runtime, isolated_openfga

pytestmark = pytest.mark.integration
PERMISSIONS = (Permission("iris", frozenset({"job.submit"}), "/test/"),)
AUTHORITIES = (PrincipalAuthority("owner", "owner", 600, 3, "test"),)


class TestPolicy:
    __test__ = False

    def allows_grant(self, principal, permissions):
        return principal == "owner" and tuple(permissions) == PERMISSIONS

    def check(self, principal, service, action, resource):
        return principal == "owner" and service == "iris" and action == "job.submit" and resource.startswith("/test/")


@pytest.fixture
def runtime_grants_config(postgres_database_url, tmp_path):
    executable = Path(os.environ["MARIN_AUTH_TEST_OPENFGA_EXECUTABLE"])
    database_uri = make_url(postgres_database_url).set(drivername="postgresql").render_as_string(hide_password=False)
    with isolated_openfga(executable, tmp_path, database_uri=database_uri) as operator:
        yield configure_runtime(operator)


def _store(config):
    return GrantStore(OpenFGARuntime(**config), AUTHORITIES, TestPolicy())


def _create_worker(config, expiry, parent_id, operation, gate, results):
    with closing(_store(config)) as store:
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


def _race(config, expiry, parent_id, operations):
    context = multiprocessing.get_context("spawn")
    gate, results = context.Event(), context.Queue()
    processes = [
        context.Process(target=_create_worker, args=(config, expiry, parent_id, operation, gate, results))
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
        pytest.fail("A grant replica did not complete its authority operation")
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()
        results.close()
        results.join_thread()


def test_separate_replicas_create_one_idempotent_grant_and_preserve_revocation(runtime_grants_config):
    expiry = time.time() + 300
    ids = _race(runtime_grants_config, expiry, None, ["create"] * 4)
    assert len(set(ids)) == 1
    with closing(_store(runtime_grants_config)) as store:
        grant = store.create(
            "owner", actor="runtime", permissions=PERMISSIONS, expires_at=expiry, idempotency_key="same-request"
        )
        assert grant.id == ids[0]
        store.revoke(grant.id, principal="owner")
    with closing(_store(runtime_grants_config)) as reopened:
        with pytest.raises(GrantDenied):
            reopened.active(ids[0])
        with pytest.raises(GrantDenied):
            reopened.create(
                "owner", actor="runtime", permissions=PERMISSIONS, expires_at=expiry, idempotency_key="same-request"
            )


def test_concurrent_parent_revocation_never_leaves_an_active_descendant(runtime_grants_config):
    expiry = time.time() + 300
    with closing(_store(runtime_grants_config)) as store:
        parent = store.create("owner", actor="runtime", permissions=PERMISSIONS, expires_at=expiry)
    outcomes = _race(runtime_grants_config, expiry, parent.id, ["create", "create", "revoke", "revoke"])
    assert outcomes.count("revoked") == 2
    child_ids = set(outcomes) - {"revoked", "denied"}
    assert len(child_ids) <= 1
    with closing(_store(runtime_grants_config)) as reopened:
        with pytest.raises(GrantDenied):
            reopened.active(parent.id)
        for grant_id in child_ids:
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
