# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sqlite3


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    return column in columns


METADATA_COLUMNS = (
    ("md_hostname", "TEXT NOT NULL DEFAULT ''"),
    ("md_ip_address", "TEXT NOT NULL DEFAULT ''"),
    ("md_cpu_count", "INTEGER NOT NULL DEFAULT 0"),
    ("md_memory_bytes", "INTEGER NOT NULL DEFAULT 0"),
    ("md_disk_bytes", "INTEGER NOT NULL DEFAULT 0"),
    ("md_tpu_name", "TEXT NOT NULL DEFAULT ''"),
    ("md_tpu_worker_hostnames", "TEXT NOT NULL DEFAULT ''"),
    ("md_tpu_worker_id", "TEXT NOT NULL DEFAULT ''"),
    ("md_tpu_chips_per_host_bounds", "TEXT NOT NULL DEFAULT ''"),
    ("md_gpu_count", "INTEGER NOT NULL DEFAULT 0"),
    ("md_gpu_name", "TEXT NOT NULL DEFAULT ''"),
    ("md_gpu_memory_mb", "INTEGER NOT NULL DEFAULT 0"),
    ("md_gce_instance_name", "TEXT NOT NULL DEFAULT ''"),
    ("md_gce_zone", "TEXT NOT NULL DEFAULT ''"),
    ("md_git_hash", "TEXT NOT NULL DEFAULT ''"),
    ("md_device_json", "TEXT NOT NULL DEFAULT '{}'"),
)


def _device_to_json(device) -> str:
    """Serialize a DeviceConfig proto to JSON."""
    if device.HasField("gpu"):
        return json.dumps({"gpu": {"variant": device.gpu.variant, "count": device.gpu.count}})
    elif device.HasField("tpu"):
        return json.dumps(
            {
                "tpu": {
                    "variant": device.tpu.variant,
                    "topology": device.tpu.topology,
                    "count": device.tpu.count,
                }
            }
        )
    return "{}"


def migrate(conn: sqlite3.Connection) -> None:
    for column, ddl in METADATA_COLUMNS:
        if not _has_column(conn, "workers", column):
            conn.execute(f"ALTER TABLE workers ADD COLUMN {column} {ddl}")

    # Backfill only needed for upgrades, not fresh DBs.
    if not _has_column(conn, "workers", "metadata_proto"):
        return

    from iris.rpc import job_pb2

    rows = conn.execute("SELECT worker_id, metadata_proto FROM workers WHERE metadata_proto IS NOT NULL").fetchall()
    for worker_id, blob in rows:
        md = job_pb2.WorkerMetadata()
        md.ParseFromString(blob)
        conn.execute(
            "UPDATE workers SET "
            "md_hostname = ?, md_ip_address = ?, md_cpu_count = ?, md_memory_bytes = ?, "
            "md_disk_bytes = ?, md_tpu_name = ?, md_tpu_worker_hostnames = ?, "
            "md_tpu_worker_id = ?, md_tpu_chips_per_host_bounds = ?, "
            "md_gpu_count = ?, md_gpu_name = ?, md_gpu_memory_mb = ?, "
            "md_gce_instance_name = ?, md_gce_zone = ?, md_git_hash = ?, "
            "md_device_json = ? WHERE worker_id = ?",
            (
                md.hostname,
                md.ip_address,
                md.cpu_count,
                md.memory_bytes,
                md.disk_bytes,
                md.tpu_name,
                md.tpu_worker_hostnames,
                md.tpu_worker_id,
                md.tpu_chips_per_host_bounds,
                md.gpu_count,
                md.gpu_name,
                md.gpu_memory_mb,
                md.gce_instance_name,
                md.gce_zone,
                md.git_hash,
                _device_to_json(md.device),
                worker_id,
            ),
        )

    if _has_column(conn, "workers", "metadata_proto"):
        conn.execute("ALTER TABLE workers DROP COLUMN metadata_proto")
