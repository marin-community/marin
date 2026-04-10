# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sqlite3


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    return column in columns


def _device_to_json(device) -> str | None:
    if device.HasField("gpu"):
        return json.dumps({"gpu": {"variant": device.gpu.variant, "count": device.gpu.count}})
    elif device.HasField("tpu"):
        return json.dumps(
            {"tpu": {"variant": device.tpu.variant, "topology": device.tpu.topology, "count": device.tpu.count}}
        )
    return None


def _constraint_to_dict(c) -> dict:
    d: dict = {"key": c.key, "op": c.op}
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
        d["mode"] = c.mode
    return d


RESOURCE_COLUMNS = (
    ("res_cpu_millicores", "INTEGER NOT NULL DEFAULT 0"),
    ("res_memory_bytes", "INTEGER NOT NULL DEFAULT 0"),
    ("res_disk_bytes", "INTEGER NOT NULL DEFAULT 0"),
    ("res_device_json", "TEXT"),
)

CONSTRAINT_COLUMNS = (("constraints_json", "TEXT"),)


def migrate(conn: sqlite3.Connection) -> None:
    for column, ddl in (*RESOURCE_COLUMNS, *CONSTRAINT_COLUMNS):
        if not _has_column(conn, "jobs", column):
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {ddl}")

    # Backfill only needed for upgrades, not fresh DBs.
    if not _has_column(conn, "jobs", "resources_proto"):
        return

    from iris.rpc import job_pb2

    rows = conn.execute("SELECT job_id, resources_proto FROM jobs WHERE resources_proto IS NOT NULL").fetchall()
    for job_id, blob in rows:
        res = job_pb2.ResourceSpecProto()
        res.ParseFromString(blob)
        conn.execute(
            "UPDATE jobs SET res_cpu_millicores = ?, res_memory_bytes = ?, "
            "res_disk_bytes = ?, res_device_json = ? WHERE job_id = ?",
            (res.cpu_millicores, res.memory_bytes, res.disk_bytes, _device_to_json(res.device), job_id),
        )

    # Backfill constraints_proto
    rows = conn.execute("SELECT job_id, constraints_proto FROM jobs WHERE constraints_proto IS NOT NULL").fetchall()
    for job_id, blob in rows:
        cl = job_pb2.ConstraintList()
        cl.ParseFromString(blob)
        if cl.constraints:
            constraints_json = json.dumps([_constraint_to_dict(c) for c in cl.constraints])
            conn.execute("UPDATE jobs SET constraints_json = ? WHERE job_id = ?", (constraints_json, job_id))

    # Drop old BLOB columns.
    if _has_column(conn, "jobs", "resources_proto"):
        conn.execute("ALTER TABLE jobs DROP COLUMN resources_proto")
    if _has_column(conn, "jobs", "constraints_proto"):
        conn.execute("ALTER TABLE jobs DROP COLUMN constraints_proto")
