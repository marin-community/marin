# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for migration ``0052_egress_policy``."""

import importlib.util
import sqlite3
from pathlib import Path

from iris.rpc import job_pb2

_MIGRATION = Path(__file__).parents[3] / "src/iris/cluster/controller/migrations/0052_egress_policy.py"
_RETIRED_GVISOR = 5


def _load_migration():
    spec = importlib.util.spec_from_file_location("m0052", _MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_0052_turns_gvisor_into_sandbox_and_resolves_every_egress_policy() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE job_config (job_id TEXT PRIMARY KEY, container_profile INTEGER NOT NULL DEFAULT 0)")
    conn.executemany(
        "INSERT INTO job_config VALUES (?, ?)",
        [
            ("/a/gvisor", _RETIRED_GVISOR),
            ("/a/sandbox", job_pb2.CONTAINER_PROFILE_SANDBOX),
            ("/a/default", job_pb2.CONTAINER_PROFILE_DEFAULT),
            ("/a/unspecified", job_pb2.CONTAINER_PROFILE_UNSPECIFIED),
        ],
    )
    migration = _load_migration()

    migration.migrate(conn)
    migration.migrate(conn)

    rows = dict(
        (job_id, (profile, egress))
        for job_id, profile, egress in conn.execute("SELECT job_id, container_profile, egress_policy FROM job_config")
    )
    sandbox_internet = (job_pb2.CONTAINER_PROFILE_SANDBOX, job_pb2.EGRESS_POLICY_INTERNET)
    assert rows == {
        "/a/gvisor": sandbox_internet,
        "/a/sandbox": sandbox_internet,
        "/a/default": (job_pb2.CONTAINER_PROFILE_DEFAULT, job_pb2.EGRESS_POLICY_CLUSTER),
        "/a/unspecified": (job_pb2.CONTAINER_PROFILE_UNSPECIFIED, job_pb2.EGRESS_POLICY_CLUSTER),
    }
    conn.close()
