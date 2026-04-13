# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    return column in columns


def migrate(conn: sqlite3.Connection) -> None:
    columns_to_add = (
        ("total_cpu_millicores", "INTEGER NOT NULL DEFAULT 0"),
        ("total_memory_bytes", "INTEGER NOT NULL DEFAULT 0"),
        ("total_gpu_count", "INTEGER NOT NULL DEFAULT 0"),
        ("total_tpu_count", "INTEGER NOT NULL DEFAULT 0"),
        ("device_type", "TEXT NOT NULL DEFAULT ''"),
        ("device_variant", "TEXT NOT NULL DEFAULT ''"),
    )
    for column, ddl in columns_to_add:
        if not _has_column(conn, "workers", column):
            conn.execute(f"ALTER TABLE workers ADD COLUMN {column} {ddl}")

    if not _has_column(conn, "workers", "metadata_proto"):
        return  # Column already removed by later migration; backfill not needed.

    from iris.cluster.types import get_gpu_count, get_tpu_count
    from iris.rpc import job_pb2

    rows = conn.execute("SELECT worker_id, metadata_proto FROM workers WHERE metadata_proto IS NOT NULL").fetchall()
    for worker_id, metadata_blob in rows:
        metadata = job_pb2.WorkerMetadata()
        metadata.ParseFromString(metadata_blob)
        if metadata.device.HasField("gpu"):
            device_type = "gpu"
            device_variant = metadata.device.gpu.variant
        elif metadata.device.HasField("tpu"):
            device_type = "tpu"
            device_variant = metadata.device.tpu.variant
        else:
            device_type = ""
            device_variant = ""
        conn.execute(
            "UPDATE workers SET total_cpu_millicores = ?, total_memory_bytes = ?, "
            "total_gpu_count = ?, total_tpu_count = ?, device_type = ?, device_variant = ? "
            "WHERE worker_id = ?",
            (
                metadata.cpu_count * 1000,
                metadata.memory_bytes,
                get_gpu_count(metadata.device),
                get_tpu_count(metadata.device),
                device_type,
                device_variant,
                worker_id,
            ),
        )
