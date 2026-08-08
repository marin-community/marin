# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persisted-contract tests for resource schema v2."""

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.persistence import migrate as resource_migration
from iris.cluster.controller.persistence.migrate import (
    MigrationContext,
    ResourceSchemaMigrationError,
    final_schema_fingerprint,
    initialize_or_upgrade_database,
    inspect_schema,
    preflight_database,
)
from iris.cluster.controller.persistence.schema import metadata
from iris.cluster.controller.persistence.schema.version import MERGE_BASE_SCHEMA_FINGERPRINT
from sqlalchemy import Connection, create_engine
from sqlalchemy.exc import IntegrityError

EXPECTED_TABLES = {
    "action_receipts",
    "attempt_runtime_objects",
    "attempts",
    "endpoints",
    "federated_jobs",
    "federated_tasks",
    "federation_changelog",
    "federation_sync_state",
    "job_specs",
    "job_workdir_files",
    "jobs",
    "meta",
    "node_attributes",
    "node_capacity",
    "rpc_node_details",
    "rpc_nodes",
    "scaling_groups",
    "schema_migrations",
    "slice_members",
    "slices",
    "tasks",
    "user_budgets",
}

EXPECTED_INDEXES = {
    "action_receipts_principal",
    "action_receipts_state",
    "action_receipts_target",
    "attempt_runtime_provider_uid",
    "attempts_backend",
    "attempts_node",
    "attempts_task_state",
    "current_rpc_node_logical_id",
    "endpoints_name",
    "endpoints_owner_task",
    "endpoints_peer",
    "federated_jobs_direction_peer",
    "federation_changelog_requester",
    "jobs_backend_state",
    "jobs_execution_state",
    "jobs_owner_state",
    "jobs_parent",
    "jobs_state_submitted",
    "rpc_nodes_scaling_group",
    "slices_scaling_group",
    "tasks_backend_state",
    "tasks_current_attempt",
    "tasks_execution_state",
    "tasks_job_state",
    "tasks_pending",
}

EXPECTED_DDL_FINGERPRINT = "e9400b09d8a01fae04e26ae189cf4095f91abe73092e500fe4f3d12b7db23bee"


def _migration_context() -> MigrationContext:
    return MigrationContext(
        cluster_id="cluster-a",
        backend_kinds={"k8s": "kubernetes"},
        scale_group_to_backend={},
        backend_namespaces={"k8s": "iris-tasks"},
    )


def _merge_base_database(db_dir: Path) -> Path:
    ControllerDB(db_dir).close()
    database_path = db_dir / ControllerDB.DB_FILENAME
    with sqlite3.connect(database_path) as connection:
        # The in-branch legacy startup already materializes this final-schema
        # noun. The release source is the exact pre-PR merge-base without it.
        connection.execute("DROP TABLE action_receipts")
        connection.commit()
    return database_path


def _seed_merge_base_database(database_path: Path) -> None:
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            INSERT INTO jobs (
                job_id, user_id, submitting_user, root_job_id, depth, state,
                submitted_at_ms, root_submitted_at_ms, num_tasks, name,
                backend_id, cluster
            ) VALUES (
                '/owner/job', 'owner', 'submitter', '/owner/job', 0, 1,
                10, 10, 1, 'training', 'k8s', 'local'
            )
            """
        )
        connection.execute("INSERT INTO job_config (job_id, name) VALUES ('/owner/job', 'training')")
        connection.execute(
            """
            INSERT INTO tasks (
                task_id, job_id, task_index, state, submitted_at_ms,
                max_retries_failure, max_retries_preemption, current_attempt_id,
                priority_neg_depth, priority_root_submitted_ms, priority_insertion,
                priority_band, backend_id, cluster
            ) VALUES (
                '/owner/job/0', '/owner/job', 0, 2, 10, 1, 2, 0,
                0, 10, 1, 2, 'k8s', 'local'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO task_attempts (
                task_id, attempt_id, state, created_at_ms, attempt_uid,
                backend_id, pod_name, pod_uid, node_name
            ) VALUES (
                '/owner/job/0', 0, 2, 11, '0123456789abcdef',
                'k8s', 'training-0', 'pod-uid', 'node-a'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO endpoints (
                endpoint_id, name, address, job_id, task_id, metadata_json,
                registered_at_ms, access
            ) VALUES (
                'endpoint', 'http', 'http://task', '/owner/job', '/owner/job/0',
                '{}', 12, 2
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def _source_snapshot(
    database_path: Path,
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]], list[tuple[object, ...]]]:
    connection = sqlite3.connect(database_path)
    try:
        schema = connection.execute(
            """
            SELECT type, name, tbl_name, sql FROM sqlite_schema
            WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name
            """
        ).fetchall()
        rows = connection.execute("SELECT task_id, task_index, backend_id FROM tasks ORDER BY task_id").fetchall()
        migrations = connection.execute("SELECT name, applied_at_ms FROM schema_migrations ORDER BY name").fetchall()
        return schema, rows, migrations
    finally:
        connection.close()


def _create_schema() -> Connection:
    connection = create_engine("sqlite://").connect()
    connection.exec_driver_sql("PRAGMA foreign_keys = ON")
    metadata.create_all(connection)
    connection.commit()
    return connection


def _insert_job(connection: Connection, *, job_uid: str, job_id: str) -> None:
    connection.exec_driver_sql(
        """
        INSERT INTO jobs (
            job_uid, authority_cluster_id, job_id, execution_cluster_id,
            backend_id, placement_state, owner_id, submitting_principal,
            root_job_uid, depth, state, submitted_at_ms, root_submitted_at_ms,
            num_tasks, name
        ) VALUES (?, 'authority', ?, 'execution', 'backend', 'known',
                  'owner', 'principal', ?, 0, 0, 1, 1, 1, 'job')
        """,
        (job_uid, job_id, job_uid),
    )


def test_create_all_materializes_exact_resource_schema() -> None:
    with _create_schema() as connection:
        rows = connection.exec_driver_sql(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_schema
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).all()

    assert {name for kind, name, _, _ in rows if kind == "table"} == EXPECTED_TABLES
    assert {name for kind, name, _, _ in rows if kind == "index"} == EXPECTED_INDEXES

    normalized = [(kind, name, table, " ".join(sql.split())) for kind, name, table, sql in rows]
    encoded = json.dumps(normalized, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == EXPECTED_DDL_FINGERPRINT


def test_schema_enforces_coordinates_and_cascades_runtime_identity() -> None:
    with _create_schema() as connection:
        _insert_job(connection, job_uid="job-uid", job_id="/owner/job")
        connection.commit()

        with pytest.raises(IntegrityError):
            connection.exec_driver_sql(
                """
                INSERT INTO jobs (
                    job_uid, authority_cluster_id, job_id, execution_cluster_id,
                    backend_id, placement_state, owner_id, submitting_principal,
                    root_job_uid, depth, state, submitted_at_ms, root_submitted_at_ms,
                    num_tasks, name
                ) VALUES (
                    'bad-job', 'authority', '/owner/bad', 'execution',
                    'backend', 'pending', 'owner', 'principal', 'bad-job',
                    0, 0, 1, 1, 0, 'bad'
                )
                """
            )
        connection.rollback()

        with pytest.raises(IntegrityError):
            connection.exec_driver_sql(
                """
                INSERT INTO federated_jobs (
                    job_uid, direction, peer_id, owner_principal, handoff_state, handoff_nonce
                ) VALUES ('job-uid', 'sent', 'peer', 'principal', NULL, 'nonce')
                """
            )
        connection.rollback()

        connection.exec_driver_sql(
            """
            INSERT INTO tasks (
                task_uid, authority_cluster_id, task_id, job_uid, task_index,
                execution_cluster_id, backend_id, placement_state, state,
                submitted_at_ms, max_retries_failure, max_retries_preemption,
                priority_band, priority_neg_depth, priority_root_submitted_ms,
                priority_insertion
            ) VALUES (
                'task-uid', 'authority', '/owner/job/0', 'job-uid', 0,
                'execution', 'backend', 'known', 0, 1, 0, 0, 0, 0, 1, 1
            )
            """
        )
        connection.exec_driver_sql(
            """
            INSERT INTO attempts (
                attempt_uid, task_uid, attempt_number, execution_cluster_id,
                backend_id, state, created_at_ms
            ) VALUES ('attempt-uid', 'task-uid', 0, 'execution', 'backend', 0, 1)
            """
        )
        connection.exec_driver_sql(
            """
            UPDATE tasks SET current_attempt_uid = 'attempt-uid'
            WHERE task_uid = 'task-uid'
            """
        )
        connection.exec_driver_sql(
            """
            INSERT INTO attempt_runtime_objects (
                attempt_uid, provider_kind, namespace, name, provider_uid, observed_at_ms
            ) VALUES ('attempt-uid', 'kubernetes', 'namespace', 'pod', 'pod-uid', 1)
            """
        )
        connection.commit()

        connection.exec_driver_sql("DELETE FROM jobs WHERE job_uid = 'job-uid'")
        connection.commit()

        remaining = connection.exec_driver_sql(
            """
            SELECT
                (SELECT count(*) FROM tasks),
                (SELECT count(*) FROM attempts),
                (SELECT count(*) FROM attempt_runtime_objects)
            """
        ).one()
        assert remaining == (0, 0, 0)


def test_fresh_and_exact_merge_base_upgrade_have_identical_schema(tmp_path: Path) -> None:
    context = _migration_context()
    fresh_dir = tmp_path / "fresh"
    ControllerDB(fresh_dir, resource_migration_context=context).close()

    upgraded_dir = tmp_path / "upgraded"
    upgraded_path = _merge_base_database(upgraded_dir)
    _seed_merge_base_database(upgraded_path)
    assert preflight_database(upgraded_path, context=context).accepted

    ControllerDB(upgraded_dir, resource_migration_context=context).close()

    with sqlite3.connect(fresh_dir / ControllerDB.DB_FILENAME) as fresh:
        fresh_status = inspect_schema(fresh)
    with sqlite3.connect(upgraded_path) as upgraded:
        upgraded.row_factory = sqlite3.Row
        upgraded_status = inspect_schema(upgraded)
        job = upgraded.execute("SELECT * FROM jobs").fetchone()
        task = upgraded.execute("SELECT * FROM tasks").fetchone()
        attempt = upgraded.execute("SELECT * FROM attempts").fetchone()
        runtime = upgraded.execute("SELECT * FROM attempt_runtime_objects").fetchone()
        endpoint = upgraded.execute("SELECT * FROM endpoints").fetchone()
        migration = upgraded.execute("SELECT * FROM schema_migrations").fetchone()

    assert fresh_status.accepted and upgraded_status.accepted
    assert fresh_status.schema_fingerprint == upgraded_status.schema_fingerprint == final_schema_fingerprint()
    assert fresh_status.migration_names == upgraded_status.migration_names == ("resource_schema_v2",)
    assert job["authority_cluster_id"] == job["execution_cluster_id"] == "cluster-a"
    assert job["backend_id"] == task["backend_id"] == attempt["backend_id"] == "k8s"
    assert task["current_attempt_uid"] == attempt["attempt_uid"] == "0123456789abcdef"
    assert runtime["namespace"] == "iris-tasks"
    assert runtime["name"] == "training-0"
    assert runtime["provider_uid"] == "pod-uid"
    assert endpoint["owner_task_uid"] == task["task_uid"]
    assert endpoint["access"] == 1
    assert migration["source_fingerprint"] == MERGE_BASE_SCHEMA_FINGERPRINT
    assert not (fresh_dir / ControllerDB.AUTH_DB_FILENAME).exists()
    assert not (upgraded_dir / ControllerDB.AUTH_DB_FILENAME).exists()


def test_fresh_schema_requires_explicit_cluster_and_backend_identity(tmp_path: Path) -> None:
    database_path = tmp_path / ControllerDB.DB_FILENAME
    context = MigrationContext(
        cluster_id=" ",
        backend_kinds={},
        scale_group_to_backend={},
        backend_namespaces={},
    )

    report = preflight_database(database_path, context=context)

    assert not report.accepted
    assert [problem.name for problem in report.problems] == ["cluster_id_missing", "backend_configuration_invalid"]
    with pytest.raises(ResourceSchemaMigrationError, match="cluster_id_missing=1, backend_configuration_invalid=1"):
        initialize_or_upgrade_database(database_path, context=context)
    with sqlite3.connect(database_path) as connection:
        assert (
            connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            == []
        )


def test_upgrade_normalizes_rpc_nodes_slices_and_current_attempt_coordinates(tmp_path: Path) -> None:
    database_path = _merge_base_database(tmp_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("INSERT INTO scaling_groups (name) VALUES ('gpu-group')")
        connection.execute(
            """
            INSERT INTO slices (slice_id, scale_group, lifecycle, worker_ids, created_at_ms)
            VALUES ('slice-a', 'gpu-group', 'initializing', '["worker-a"]', 7)
            """
        )
        connection.execute(
            """
            INSERT INTO workers (
                worker_id, address, md_hostname, md_ip_address, md_disk_bytes,
                md_provenance_json, total_cpu_millicores, total_memory_bytes,
                total_gpu_count, device_type, device_variant, slice_id, scale_group
            ) VALUES (
                'worker-a', 'worker:123', 'host-a', '10.0.0.1', 1000,
                '{"image":"worker"}', 4000, 8000, 1, 'gpu', 'h100',
                'slice-a', 'gpu-group'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO worker_attributes (worker_id, key, value_type, str_value)
            VALUES ('worker-a', 'zone', 'str', 'us-central1-a')
            """
        )
        connection.execute(
            """
            INSERT INTO jobs (
                job_id, user_id, submitting_user, root_job_id, depth, state,
                submitted_at_ms, root_submitted_at_ms, num_tasks, backend_id, cluster
            ) VALUES ('/owner/rpc', 'owner', 'submitter', '/owner/rpc', 0, 2, 8, 8, 1, 'rpc', 'local')
            """
        )
        connection.execute("INSERT INTO job_config (job_id) VALUES ('/owner/rpc')")
        connection.execute(
            """
            INSERT INTO tasks (
                task_id, job_id, task_index, state, submitted_at_ms,
                max_retries_failure, max_retries_preemption, current_attempt_id,
                priority_neg_depth, priority_root_submitted_ms, priority_insertion,
                current_worker_id, backend_id, cluster
            ) VALUES (
                '/owner/rpc/0', '/owner/rpc', 0, 2, 8, 1, 1, 0,
                0, 8, 1, 'worker-a', 'rpc', 'local'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO task_attempts (
                task_id, attempt_id, worker_id, state, created_at_ms,
                attempt_uid, backend_id
            ) VALUES ('/owner/rpc/0', 0, 'worker-a', 2, 9, 'fedcba9876543210', 'rpc')
            """
        )
        connection.commit()

    context = MigrationContext(
        cluster_id="cluster-a",
        backend_kinds={"rpc": "rpc"},
        scale_group_to_backend={"gpu-group": "rpc"},
        backend_namespaces={},
    )
    initialize_or_upgrade_database(database_path, context=context)

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        task = connection.execute("SELECT current_node_uid FROM tasks").fetchone()
        attempt = connection.execute("SELECT node_uid FROM attempts").fetchone()
        node = connection.execute("SELECT n.*, c.* FROM rpc_nodes n JOIN node_capacity c USING (node_uid)").fetchone()
        node_attribute = connection.execute("SELECT * FROM node_attributes").fetchone()
        resource_slice = connection.execute("SELECT * FROM slices").fetchone()
        member = connection.execute("SELECT * FROM slice_members").fetchone()

    assert task["current_node_uid"] == attempt["node_uid"] == node["node_uid"]
    assert node["backend_id"] == "rpc"
    assert node["scaling_group_id"] == "gpu-group"
    assert node["accelerator_kind"] == "gpu"
    assert node["accelerator_variant"] == "h100"
    assert node["accelerator_count"] == 1
    assert node_attribute["str_value"] == "us-central1-a"
    assert resource_slice["lifecycle"] == "creating"
    assert resource_slice["membership_state"] == "observed"
    assert member["provider_node_id"] == "worker-a"


def test_upgrade_derives_federation_authority_execution_and_pending_placement(tmp_path: Path) -> None:
    database_path = _merge_base_database(tmp_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executemany(
            """
            INSERT INTO jobs (
                job_id, user_id, submitting_user, root_job_id, depth, state,
                submitted_at_ms, root_submitted_at_ms, num_tasks, backend_id, cluster
            ) VALUES (?, 'owner', 'submitter', ?, 0, 1, 10, 10, 0, ?, ?)
            """,
            [
                ("/owner/sent", "/owner/sent", "", "peer-b"),
                ("/owner/received", "/owner/received", "k8s", "local"),
            ],
        )
        connection.executemany(
            "INSERT INTO job_config (job_id) VALUES (?)",
            [("/owner/sent",), ("/owner/received",)],
        )
        connection.executemany(
            """
            INSERT INTO federated_jobs (
                job_id, direction, peer_id, owner_principal, handoff_state, handoff_nonce
            ) VALUES (?, ?, ?, 'owner', ?, ?)
            """,
            [
                ("/owner/sent", 0, "peer-b", 3, "sent-nonce"),
                ("/owner/received", 1, "peer-parent", None, "received-nonce"),
            ],
        )
        connection.commit()

    context = MigrationContext(
        cluster_id="cluster-a",
        backend_kinds={"rpc": "rpc", "k8s": "kubernetes"},
        scale_group_to_backend={},
        backend_namespaces={"k8s": "iris-tasks"},
    )
    initialize_or_upgrade_database(database_path, context=context)

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = {
            row["job_id"]: row
            for row in connection.execute(
                """
                SELECT j.*, f.direction, f.peer_id, f.handoff_state
                FROM jobs j JOIN federated_jobs f USING (job_uid)
                """
            )
        }

    sent = rows["/owner/sent"]
    assert sent["authority_cluster_id"] == "cluster-a"
    assert sent["execution_cluster_id"] == sent["peer_id"] == "peer-b"
    assert sent["placement_state"] == "pending"
    assert sent["backend_id"] == ""
    assert sent["direction"] == "sent"
    assert sent["handoff_state"] == "queued"

    received = rows["/owner/received"]
    assert received["authority_cluster_id"] == received["peer_id"] == "peer-parent"
    assert received["execution_cluster_id"] == "cluster-a"
    assert received["placement_state"] == "known"
    assert received["backend_id"] == "k8s"
    assert received["direction"] == "received"
    assert received["handoff_state"] is None


def test_preflight_rejects_malformed_rows_without_changing_source(tmp_path: Path) -> None:
    database_path = _merge_base_database(tmp_path)
    _seed_merge_base_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("UPDATE tasks SET task_index = 0.5 WHERE task_id = '/owner/job/0'")
        connection.commit()
    before = _source_snapshot(database_path)

    report = preflight_database(database_path, context=_migration_context())

    assert not report.accepted
    assert [(problem.name, problem.count) for problem in report.problems] == [("task_index_invalid", 1)]
    with pytest.raises(ResourceSchemaMigrationError, match="task_index_invalid=1"):
        initialize_or_upgrade_database(database_path, context=_migration_context())
    assert _source_snapshot(database_path) == before


def test_exact_source_schema_drift_is_rejected_without_changing_source(tmp_path: Path) -> None:
    database_path = _merge_base_database(tmp_path)
    _seed_merge_base_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE INDEX unexpected_jobs_index ON jobs(name)")
        connection.commit()
    before = _source_snapshot(database_path)

    report = preflight_database(database_path, context=_migration_context())

    assert not report.accepted
    assert report.schema.problems == ("schema_fingerprint_mismatch",)
    with pytest.raises(ResourceSchemaMigrationError, match="schema_fingerprint_mismatch"):
        initialize_or_upgrade_database(database_path, context=_migration_context())
    assert _source_snapshot(database_path) == before


def test_exact_source_migration_ledger_drift_is_rejected_without_changing_source(tmp_path: Path) -> None:
    database_path = _merge_base_database(tmp_path)
    _seed_merge_base_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("DELETE FROM schema_migrations WHERE name = '0050_drop_controller_secrets.py'")
        connection.commit()
    before = _source_snapshot(database_path)

    report = preflight_database(database_path, context=_migration_context())

    assert not report.accepted
    assert report.schema.problems == ("migration_ledger_mismatch",)
    with pytest.raises(ResourceSchemaMigrationError, match="migration_ledger_mismatch"):
        initialize_or_upgrade_database(database_path, context=_migration_context())
    assert _source_snapshot(database_path) == before


def test_upgrade_sqlite_ddl_failure_rolls_back_the_complete_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = _merge_base_database(tmp_path)
    _seed_merge_base_database(database_path)
    before = _source_snapshot(database_path)
    real_connect = sqlite3.connect

    def connect_with_denied_final_table(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = real_connect(*args, **kwargs)
        if args and Path(str(args[0])) == database_path:

            def authorize(action: int, arg1: str | None, _arg2: str | None, _db: str | None, _source: str | None) -> int:
                if action == sqlite3.SQLITE_CREATE_TABLE and arg1 == "jobs":
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            connection.set_authorizer(authorize)
        return connection

    monkeypatch.setattr(resource_migration.sqlite3, "connect", connect_with_denied_final_table)

    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        initialize_or_upgrade_database(database_path, context=_migration_context())

    assert _source_snapshot(database_path) == before
