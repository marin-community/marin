# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Split job config columns into a separate job_config table.

Moves all submission-time configuration (resources, constraints, entrypoint,
environment, etc.) from the jobs table into a 1:1 job_config table. Also
extracts remaining fields from request_proto (entrypoint, environment,
bundle_id, ports, retry limits, timeout, preemption_policy, etc.) and
creates a job_workdir_files table for binary workdir files.
"""

import json
import sqlite3


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    return column in columns


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def _device_to_json(device) -> str | None:
    if device.HasField("gpu"):
        return json.dumps({"gpu": {"variant": device.gpu.variant, "count": device.gpu.count}})
    elif device.HasField("tpu"):
        return json.dumps(
            {"tpu": {"variant": device.tpu.variant, "topology": device.tpu.topology, "count": device.tpu.count}}
        )
    return None


def _constraint_to_dict(c) -> dict:
    d: dict = {"key": c.key, "op": int(c.op)}
    if c.HasField("value"):
        v = c.value
        if v.HasField("string_value"):
            d["value"] = {"string_value": v.string_value}
        elif v.HasField("int_value"):
            d["value"] = {"int_value": v.int_value}
        elif v.HasField("float_value"):
            d["value"] = {"float_value": v.float_value}
    if c.values:
        vals = []
        for v in c.values:
            if v.HasField("string_value"):
                vals.append({"string_value": v.string_value})
            elif v.HasField("int_value"):
                vals.append({"int_value": v.int_value})
            elif v.HasField("float_value"):
                vals.append({"float_value": v.float_value})
        d["values"] = vals
    if c.mode:
        d["mode"] = int(c.mode)
    return d


def _entrypoint_to_json(ep) -> str:
    d: dict = {}
    if ep.setup_commands:
        d["setup_commands"] = list(ep.setup_commands)
    if ep.HasField("run_command"):
        d["run_command"] = {"argv": list(ep.run_command.argv)}
    if ep.workdir_file_refs:
        d["workdir_file_refs"] = dict(ep.workdir_file_refs)
    return json.dumps(d)


def _environment_to_json(env) -> str:
    d: dict = {}
    if env.pip_packages:
        d["pip_packages"] = list(env.pip_packages)
    if env.env_vars:
        d["env_vars"] = dict(env.env_vars)
    if env.extras:
        d["extras"] = list(env.extras)
    if env.python_version:
        d["python_version"] = env.python_version
    if env.dockerfile:
        d["dockerfile"] = env.dockerfile
    return json.dumps(d)


def _reservation_to_json(request) -> str | None:
    if not request.HasField("reservation"):
        return None
    res = request.reservation
    entries = []
    for entry in res.entries:
        e: dict = {}
        if entry.HasField("resources"):
            r = entry.resources
            e["resources"] = {
                "cpu_millicores": r.cpu_millicores,
                "memory_bytes": r.memory_bytes,
                "disk_bytes": r.disk_bytes,
            }
            if r.HasField("device"):
                e["resources"]["device"] = json.loads(_device_to_json(r.device) or "{}")
        if entry.constraints:
            e["constraints"] = [_constraint_to_dict(c) for c in entry.constraints]
        entries.append(e)
    return json.dumps({"entries": entries})


def migrate(conn: sqlite3.Connection) -> None:
    # 1. Create job_config and job_workdir_files tables.
    if not _table_exists(conn, "job_config"):
        conn.execute(
            """
            CREATE TABLE job_config (
                job_id TEXT PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
                name TEXT NOT NULL DEFAULT '',
                has_reservation INTEGER NOT NULL DEFAULT 0,
                res_cpu_millicores INTEGER NOT NULL DEFAULT 0,
                res_memory_bytes INTEGER NOT NULL DEFAULT 0,
                res_disk_bytes INTEGER NOT NULL DEFAULT 0,
                res_device_json TEXT,
                constraints_json TEXT,
                has_coscheduling INTEGER NOT NULL DEFAULT 0,
                coscheduling_group_by TEXT NOT NULL DEFAULT '',
                scheduling_timeout_ms INTEGER,
                max_task_failures INTEGER NOT NULL DEFAULT 0,
                entrypoint_json TEXT NOT NULL DEFAULT '{}',
                environment_json TEXT NOT NULL DEFAULT '{}',
                bundle_id TEXT NOT NULL DEFAULT '',
                ports_json TEXT NOT NULL DEFAULT '[]',
                max_retries_failure INTEGER NOT NULL DEFAULT 0,
                max_retries_preemption INTEGER NOT NULL DEFAULT 100,
                timeout_ms INTEGER,
                preemption_policy INTEGER NOT NULL DEFAULT 0,
                existing_job_policy INTEGER NOT NULL DEFAULT 0,
                priority_band INTEGER NOT NULL DEFAULT 0,
                task_image TEXT NOT NULL DEFAULT '',
                reservation_json TEXT,
                fail_if_exists INTEGER NOT NULL DEFAULT 0
            )
        """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_job_config_name ON job_config(name)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_job_config_has_reservation"
            " ON job_config(has_reservation, job_id) WHERE has_reservation = 1"
        )

    if not _table_exists(conn, "job_workdir_files"):
        conn.execute(
            """
            CREATE TABLE job_workdir_files (
                job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                data BLOB NOT NULL,
                PRIMARY KEY (job_id, filename)
            )
        """
        )

    # 2. Backfill job_config from existing jobs columns + request_proto.
    has_request_proto = _has_column(conn, "jobs", "request_proto")
    has_res_columns = _has_column(conn, "jobs", "res_cpu_millicores")
    has_name = _has_column(conn, "jobs", "name")

    if has_request_proto:
        from iris.rpc import controller_pb2

        rows = conn.execute("SELECT job_id, request_proto FROM jobs WHERE request_proto IS NOT NULL").fetchall()
        for job_id, blob in rows:
            req = controller_pb2.Controller.LaunchJobRequest()
            req.ParseFromString(blob)

            res = req.resources if req.HasField("resources") else None
            constraints = [_constraint_to_dict(c) for c in req.constraints] if req.constraints else None

            conn.execute(
                "INSERT OR IGNORE INTO job_config("
                "job_id, name, has_reservation, "
                "res_cpu_millicores, res_memory_bytes, res_disk_bytes, res_device_json, "
                "constraints_json, has_coscheduling, coscheduling_group_by, "
                "scheduling_timeout_ms, max_task_failures, "
                "entrypoint_json, environment_json, bundle_id, ports_json, "
                "max_retries_failure, max_retries_preemption, timeout_ms, "
                "preemption_policy, existing_job_policy, priority_band, "
                "task_image, reservation_json, fail_if_exists"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    req.name,
                    1 if req.HasField("reservation") and req.reservation.entries else 0,
                    int(res.cpu_millicores) if res else 0,
                    int(res.memory_bytes) if res else 0,
                    int(res.disk_bytes) if res else 0,
                    _device_to_json(res.device) if res else None,
                    json.dumps(constraints) if constraints else None,
                    1 if req.HasField("coscheduling") else 0,
                    req.coscheduling.group_by if req.HasField("coscheduling") else "",
                    (
                        int(req.scheduling_timeout.milliseconds)
                        if req.HasField("scheduling_timeout") and req.scheduling_timeout.milliseconds > 0
                        else None
                    ),
                    int(req.max_task_failures),
                    _entrypoint_to_json(req.entrypoint) if req.HasField("entrypoint") else "{}",
                    _environment_to_json(req.environment) if req.HasField("environment") else "{}",
                    req.bundle_id,
                    json.dumps(list(req.ports)) if req.ports else "[]",
                    int(req.max_retries_failure),
                    int(req.max_retries_preemption),
                    (
                        int(req.timeout.milliseconds)
                        if req.HasField("timeout") and req.timeout.milliseconds > 0
                        else None
                    ),
                    int(req.preemption_policy),
                    int(req.existing_job_policy),
                    int(req.priority_band),
                    req.task_image,
                    _reservation_to_json(req),
                    1 if req.fail_if_exists else 0,
                ),
            )

            # Extract workdir_files to separate table.
            if req.HasField("entrypoint") and req.entrypoint.workdir_files:
                for filename, data in req.entrypoint.workdir_files.items():
                    conn.execute(
                        "INSERT OR IGNORE INTO job_workdir_files(job_id, filename, data) VALUES (?, ?, ?)",
                        (job_id, filename, data),
                    )
    elif has_res_columns and has_name:
        # Columns were already extracted by migrations 0027/earlier but request_proto gone.
        # Copy from existing jobs columns.
        conn.execute(
            """
            INSERT OR IGNORE INTO job_config(
                job_id, name, has_reservation,
                res_cpu_millicores, res_memory_bytes, res_disk_bytes, res_device_json,
                constraints_json, has_coscheduling, coscheduling_group_by,
                scheduling_timeout_ms, max_task_failures
            )
            SELECT
                job_id, name, has_reservation,
                res_cpu_millicores, res_memory_bytes, res_disk_bytes, res_device_json,
                constraints_json, has_coscheduling, coscheduling_group_by,
                scheduling_timeout_ms, max_task_failures
            FROM jobs
        """
        )

    # 3. Drop old indexes that reference columns we're about to drop.
    conn.execute("DROP INDEX IF EXISTS idx_jobs_name")
    conn.execute("DROP INDEX IF EXISTS idx_jobs_has_reservation")

    # 4. Drop moved columns from jobs (and request_proto).
    columns_to_drop = [
        "request_proto",
        "name",
        "has_reservation",
        "res_cpu_millicores",
        "res_memory_bytes",
        "res_disk_bytes",
        "res_device_json",
        "constraints_json",
        "has_coscheduling",
        "coscheduling_group_by",
        "scheduling_timeout_ms",
        "max_task_failures",
    ]
    for col in columns_to_drop:
        if _has_column(conn, "jobs", col):
            conn.execute(f"ALTER TABLE jobs DROP COLUMN {col}")
