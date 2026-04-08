# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    return column in columns


def migrate(conn: sqlite3.Connection) -> None:
    columns_to_add = (
        ("resources_proto", "BLOB"),
        ("constraints_proto", "BLOB"),
        ("has_coscheduling", "INTEGER NOT NULL DEFAULT 0"),
        ("coscheduling_group_by", "TEXT NOT NULL DEFAULT ''"),
        ("scheduling_timeout_ms", "INTEGER"),
        ("max_task_failures", "INTEGER NOT NULL DEFAULT 0"),
    )
    for column, ddl in columns_to_add:
        if not _has_column(conn, "jobs", column):
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {ddl}")

    from iris.rpc import job_pb2
    from iris.rpc import controller_pb2

    rows = conn.execute("SELECT job_id, request_proto FROM jobs WHERE request_proto IS NOT NULL").fetchall()
    for job_id, request_blob in rows:
        request = controller_pb2.Controller.LaunchJobRequest()
        request.ParseFromString(request_blob)

        resources_blob = request.resources.SerializeToString() if request.HasField("resources") else None
        constraint_list = job_pb2.ConstraintList()
        constraint_list.constraints.extend(request.constraints)
        constraints_blob = constraint_list.SerializeToString() if request.constraints else None
        has_coscheduling = 1 if request.HasField("coscheduling") else 0
        coscheduling_group_by = request.coscheduling.group_by if has_coscheduling else ""
        scheduling_timeout_ms = (
            int(request.scheduling_timeout.milliseconds)
            if request.HasField("scheduling_timeout") and request.scheduling_timeout.milliseconds > 0
            else None
        )
        max_task_failures = int(request.max_task_failures)

        conn.execute(
            "UPDATE jobs SET resources_proto = ?, constraints_proto = ?, has_coscheduling = ?, "
            "coscheduling_group_by = ?, scheduling_timeout_ms = ?, max_task_failures = ? "
            "WHERE job_id = ?",
            (
                resources_blob,
                constraints_blob,
                has_coscheduling,
                coscheduling_group_by,
                scheduling_timeout_ms,
                max_task_failures,
                job_id,
            ),
        )
