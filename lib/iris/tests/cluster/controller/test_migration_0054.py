# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persisted singleton and primary-plus-peer jobs migrate to complete services."""

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest
from google.protobuf import json_format
from iris.rpc import job_pb2

_MIGRATION = Path(__file__).parents[3] / "src/iris/cluster/controller/migrations/0054_task_services.py"


def _load_migration():
    spec = importlib.util.spec_from_file_location("m0054", _MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("peer_startup", [None, "image", "override"])
def test_migration_0054_preserves_startup_resources_and_workdir_files_on_replay(peer_startup):
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
        "primary_container_name": "agent",
    }
    if peer_startup is not None:
        peer = {
            "name": "runtime",
            "image": "example.test/runtime:old",
            "workdir": "/runtime",
            "env_vars": {"SERVICE_INPUT": "peer-value"},
            "resources": {
                "cpu_millicores": 1000,
                "memory_bytes": str(1024**3),
                "disk_bytes": str(1024**3),
            },
        }
        if peer_startup == "override":
            peer["entrypoint"] = {"argv": ["python3"]}
            peer["arguments"] = {"argv": ["server.py"]}
        environment["sandbox_services"] = [peer]
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
    assert stored.primary_service == "agent"
    primary = stored.services[0]
    assert primary.name == "agent"
    assert primary.image == "example.test/agent:old"
    assert dict(primary.env_vars) == {"TASK_INPUT": "primary-value"}
    assert dict(primary.secrets) == {"SERVICE_KEY": "explicit-value"}
    assert list(primary.setup_scripts) == ["prepare-workspace"]
    assert list(primary.runtime_entrypoint.run_command.argv) == ["python", "_callable_runner.py"]
    assert list(primary.runtime_entrypoint.setup_commands) == ["prepare-workspace", "install-runtime"]
    assert dict(primary.runtime_entrypoint.workdir_file_refs) == {"weights.bin": "gs://existing-bucket/weights.bin"}
    primary_cores = 4 if peer_startup is None else 3
    assert primary.resources.cpu_millicores == primary_cores * 1000
    assert primary.resources.memory_bytes == primary_cores * 1024**3
    assert primary.resources.disk_bytes == primary_cores * 1024**3
    assert first[0][4:7] == (4000, 4 * 1024**3, 4 * 1024**3)
    assert conn.execute("SELECT * FROM job_workdir_files").fetchall() == [
        ("/alice/old-job", "_callable_runner.py", b"print('preserved')")
    ]
    if peer_startup is None:
        assert len(stored.services) == 1
    else:
        _, runtime = stored.services
        assert runtime.name == "runtime"
        assert runtime.image == "example.test/runtime:old"
        assert runtime.workdir == "/runtime"
        assert dict(runtime.env_vars) == {"SERVICE_INPUT": "peer-value"}
        assert runtime.resources.cpu_millicores == 1000
        assert runtime.WhichOneof("startup") == "image_command"
        if peer_startup == "override":
            assert list(runtime.image_command.entrypoint.argv) == ["python3"]
            assert list(runtime.image_command.arguments.argv) == ["server.py"]
        else:
            assert not runtime.image_command.HasField("entrypoint")
            assert not runtime.image_command.HasField("arguments")
    conn.close()
