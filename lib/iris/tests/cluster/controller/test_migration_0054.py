# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persisted single-container jobs migrate without losing startup or files."""

import importlib.util
import json
import sqlite3
from pathlib import Path

from google.protobuf import json_format
from iris.rpc import job_pb2

_MIGRATION = Path(__file__).parents[3] / "src/iris/cluster/controller/migrations/0054_task_containers.py"


def _load_migration():
    spec = importlib.util.spec_from_file_location("m0054", _MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_0054_preserves_startup_resources_and_workdir_files_on_replay():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE job_config (
            job_id TEXT PRIMARY KEY,
            environment_json TEXT NOT NULL,
            entrypoint_json TEXT NOT NULL,
            task_image TEXT NOT NULL,
            res_cpu_millicores INTEGER NOT NULL,
            res_memory_bytes INTEGER NOT NULL,
            res_disk_bytes INTEGER NOT NULL,
            res_device_json TEXT
        );
        CREATE TABLE job_workdir_files (job_id TEXT, filename TEXT, data BLOB);
        """
    )
    environment = {
        "env_vars": {"TASK_INPUT": "primary-value"},
        "secrets": {"SERVICE_KEY": "explicit-value"},
        "setup_scripts": ["prepare-workspace"],
    }
    entrypoint = {
        "run_command": {"argv": ["python", "_callable_runner.py"]},
        "setup_commands": ["prepare-workspace", "install-runtime"],
        "workdir_file_refs": {"weights.bin": "gs://existing-bucket/weights.bin"},
    }
    conn.execute(
        "INSERT INTO job_config VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "/alice/old-job",
            json.dumps(environment),
            json.dumps(entrypoint),
            "example.test/agent:old",
            4000,
            4 * 1024**3,
            4 * 1024**3,
            None,
        ),
    )
    conn.execute(
        "INSERT INTO job_workdir_files VALUES (?, ?, ?)",
        ("/alice/old-job", "_callable_runner.py", b"print('preserved')"),
    )
    migration = _load_migration()

    migration.migrate(conn)
    first = conn.execute("SELECT * FROM job_config").fetchall()
    migration.migrate(conn)

    assert conn.execute("SELECT * FROM job_config").fetchall() == first
    stored = json_format.Parse(first[0][1], job_pb2.EnvironmentConfig())
    assert stored.primary_container == "task"
    primary = stored.containers[0]
    assert primary.name == "task"
    assert primary.image == "example.test/agent:old"
    assert dict(primary.env_vars) == {"TASK_INPUT": "primary-value"}
    assert dict(primary.secrets) == {"SERVICE_KEY": "explicit-value"}
    assert list(primary.setup_scripts) == ["prepare-workspace"]
    assert list(primary.runtime_entrypoint.run_command.argv) == ["python", "_callable_runner.py"]
    assert list(primary.runtime_entrypoint.setup_commands) == ["prepare-workspace", "install-runtime"]
    assert dict(primary.runtime_entrypoint.workdir_file_refs) == {"weights.bin": "gs://existing-bucket/weights.bin"}
    assert primary.resources.cpu_millicores == 4000
    assert primary.resources.memory_bytes == 4 * 1024**3
    assert primary.resources.disk_bytes == 4 * 1024**3
    assert first[0][4:7] == (4000, 4 * 1024**3, 4 * 1024**3)
    assert conn.execute("SELECT * FROM job_workdir_files").fetchall() == [
        ("/alice/old-job", "_callable_runner.py", b"print('preserved')")
    ]
    assert len(stored.containers) == 1
    conn.close()
