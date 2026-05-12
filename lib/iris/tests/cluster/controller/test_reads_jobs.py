# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for ``reads/jobs.py`` (Stage 8 of the SA Core migration).

Each test sets up DB state via the legacy submission paths, then calls
both ``JobStore.<method>`` (legacy SQL through ``QuerySnapshot``) and
the SA Core port (``reads.jobs.<method>`` through a fresh
``read_snapshot``) and asserts the two paths return equal results.
This proves the SA Core reads can replace the legacy reads call-by-
call without behavioral drift. The actual call-site switchover lands
in Stage 13.
"""

from __future__ import annotations

import pytest
from iris.cluster.controller import db_v2
from iris.cluster.controller.reads import jobs as reads_jobs
from iris.cluster.types import JobName
from iris.rpc import job_pb2

from .conftest import make_job_request, submit_job

# --- Setup helpers ----------------------------------------------------------


@pytest.fixture
def job_tree(state):
    """Submit a small hierarchical job tree for descendant/subtree tests.

    Layout::

        /test-user/parent
        /test-user/parent/child1
        /test-user/parent/child2
        /test-user/parent/child1/grandchild
    """
    submit_job(state, "parent", make_job_request("parent"))
    submit_job(state, "/test-user/parent/child1", make_job_request("child1"))
    submit_job(state, "/test-user/parent/child2", make_job_request("child2"))
    submit_job(state, "/test-user/parent/child1/grandchild", make_job_request("grandchild"))
    return {
        "parent": JobName.from_string("/test-user/parent"),
        "child1": JobName.from_string("/test-user/parent/child1"),
        "child2": JobName.from_string("/test-user/parent/child2"),
        "grandchild": JobName.from_string("/test-user/parent/child1/grandchild"),
    }


# --- Parity assertions ------------------------------------------------------


def test_get_state_parity(state):
    submit_job(state, "j", make_job_request("j"))
    jid = JobName.from_string("/test-user/j")

    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.jobs.get_state(legacy_tx, jid)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_jobs.get_state(sa_tx, jid)

    assert legacy == sa
    assert legacy is not None

    # Missing job: both paths return None.
    missing = JobName.from_string("/test-user/does-not-exist")
    with state._db.read_snapshot() as legacy_tx:
        assert state._store.jobs.get_state(legacy_tx, missing) is None
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        assert reads_jobs.get_state(sa_tx, missing) is None


def test_get_root_submitted_at_ms_parity(state):
    submit_job(state, "j", make_job_request("j"))
    jid = JobName.from_string("/test-user/j")

    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.jobs.get_root_submitted_at_ms(legacy_tx, jid)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_jobs.get_root_submitted_at_ms(sa_tx, jid)

    assert legacy == sa
    assert legacy is not None


def test_get_preemption_info_parity(state):
    submit_job(state, "j", make_job_request("j"))
    jid = JobName.from_string("/test-user/j")

    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.jobs.get_preemption_info(legacy_tx, jid)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_jobs.get_preemption_info(sa_tx, jid)

    assert legacy == sa
    assert legacy is not None

    missing = JobName.from_string("/test-user/missing")
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        assert reads_jobs.get_preemption_info(sa_tx, missing) is None


def test_get_recompute_basis_parity(state):
    submit_job(state, "j", make_job_request("j"))
    jid = JobName.from_string("/test-user/j")

    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.jobs.get_recompute_basis(legacy_tx, jid)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_jobs.get_recompute_basis(sa_tx, jid)

    assert legacy == sa
    assert legacy is not None


def test_get_detail_parity(state):
    submit_job(state, "j", make_job_request("j", cpu=2, memory_bytes=4 * 1024**3))
    jid = JobName.from_string("/test-user/j")

    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.jobs.get_detail(legacy_tx, jid)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_jobs.get_detail(sa_tx, jid)

    assert legacy == sa
    assert legacy is not None

    missing = JobName.from_string("/test-user/missing")
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        assert reads_jobs.get_detail(sa_tx, missing) is None


def test_get_config_parity(state):
    submit_job(state, "j", make_job_request("j"))
    jid = JobName.from_string("/test-user/j")

    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.jobs.get_config(legacy_tx, jid)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_jobs.get_config(sa_tx, jid)

    assert legacy is not None and sa is not None
    # Both return dict-of-column-name → value; keys/values must match.
    assert dict(legacy) == dict(sa)


def test_get_priority_bands_parity(state, job_tree):
    # Submit a job with an explicit priority band so we exercise both the
    # "own band wins" and the "walk up to parent" branches of the CTE.
    submit_job(
        state,
        "/test-user/parent/banded",
        make_job_request("banded", priority_band=int(job_pb2.PRIORITY_BAND_PRODUCTION)),
    )
    banded = JobName.from_string("/test-user/parent/banded")
    inputs = [job_tree["grandchild"], banded, job_tree["parent"]]

    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.jobs.get_priority_bands(legacy_tx, inputs)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_jobs.get_priority_bands(sa_tx, inputs)

    assert legacy == sa
    # The banded child should inherit PRODUCTION; the rest fall back to
    # INTERACTIVE (no band anywhere up the chain).
    assert legacy[banded] == int(job_pb2.PRIORITY_BAND_PRODUCTION)
    assert legacy[job_tree["parent"]] == int(job_pb2.PRIORITY_BAND_INTERACTIVE)
    assert legacy[job_tree["grandchild"]] == int(job_pb2.PRIORITY_BAND_INTERACTIVE)


def test_get_priority_bands_empty_parity(state):
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.jobs.get_priority_bands(legacy_tx, [])
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_jobs.get_priority_bands(sa_tx, [])
    assert legacy == sa == {}


def test_list_descendants_parity(state, job_tree):
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.jobs.list_descendants(legacy_tx, job_tree["parent"])
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_jobs.list_descendants(sa_tx, job_tree["parent"])

    assert set(legacy) == set(sa)
    assert set(legacy) == {job_tree["child1"], job_tree["child2"], job_tree["grandchild"]}


def test_list_descendants_exclude_reservation_holders_parity(state, job_tree):
    # No reservation holders in the tree, so the result is identical to the
    # unfiltered variant — but exercising the branch ensures the SQL
    # selection logic is parity-tested.
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.jobs.list_descendants(legacy_tx, job_tree["parent"], exclude_reservation_holders=True)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_jobs.list_descendants(sa_tx, job_tree["parent"], exclude_reservation_holders=True)
    assert set(legacy) == set(sa)


def test_list_subtree_parity(state, job_tree):
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.jobs.list_subtree(legacy_tx, job_tree["parent"])
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_jobs.list_subtree(sa_tx, job_tree["parent"])

    assert set(legacy) == set(sa)
    assert set(legacy) == {
        job_tree["parent"],
        job_tree["child1"],
        job_tree["child2"],
        job_tree["grandchild"],
    }


def test_find_prunable_parity(state):
    # No terminal jobs exist; both paths should return None.
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.jobs.find_prunable(legacy_tx, before_ms=10**18)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_jobs.find_prunable(sa_tx, before_ms=10**18)
    assert legacy == sa is None

    # Force a job into a terminal state with finished_at_ms set.
    submit_job(state, "j", make_job_request("j"))
    jid = JobName.from_string("/test-user/j")
    state._db.execute(
        "UPDATE jobs SET state = ?, finished_at_ms = ? WHERE job_id = ?",
        (int(job_pb2.JOB_STATE_SUCCEEDED), 1000, jid.to_wire()),
    )

    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.jobs.find_prunable(legacy_tx, before_ms=2000)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_jobs.find_prunable(sa_tx, before_ms=2000)
    assert legacy == sa == jid

    # before_ms below the row's finished_at_ms → neither path returns it.
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.jobs.find_prunable(legacy_tx, before_ms=500)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_jobs.find_prunable(sa_tx, before_ms=500)
    assert legacy == sa is None


def test_get_workdir_files_empty_parity(state):
    submit_job(state, "j", make_job_request("j"))
    jid = JobName.from_string("/test-user/j")

    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.jobs.get_workdir_files(legacy_tx, jid)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_jobs.get_workdir_files(sa_tx, jid)
    assert legacy == sa == {}


def test_get_workdir_files_populated_parity(state):
    submit_job(state, "j", make_job_request("j"))
    jid = JobName.from_string("/test-user/j")
    state._db.execute(
        "INSERT INTO job_workdir_files(job_id, filename, data) VALUES (?, ?, ?)",
        (jid.to_wire(), "main.py", b"print('hi')"),
    )
    state._db.execute(
        "INSERT INTO job_workdir_files(job_id, filename, data) VALUES (?, ?, ?)",
        (jid.to_wire(), "data.bin", b"\x00\x01\x02"),
    )

    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.jobs.get_workdir_files(legacy_tx, jid)
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_jobs.get_workdir_files(sa_tx, jid)
    assert legacy == sa
    assert legacy == {"main.py": b"print('hi')", "data.bin": b"\x00\x01\x02"}


def test_has_unfinished_worker_attempts_parity(state, job_tree):
    # Fresh tree: no attempts exist, both paths return False.
    with state._db.read_snapshot() as legacy_tx:
        legacy = state._store.jobs.has_unfinished_worker_attempts(legacy_tx, job_tree["parent"])
    with db_v2.read_snapshot(state._db.sa_read_engine) as sa_tx:
        sa = reads_jobs.has_unfinished_worker_attempts(sa_tx, job_tree["parent"])
    assert legacy == sa is False
