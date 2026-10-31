# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller upgrades retain task containers for dispatch and federation."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from iris.cluster.controller import reads
from iris.cluster.controller.codec import reconstruct_launch_job_request
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.projections.run_templates import RunTemplatesProjection
from iris.cluster.types import JobName
from iris.rpc import job_pb2


@pytest.mark.parametrize("legacy_singleton", [False, True], ids=["services", "legacy-singleton"])
def test_migration_0055_preserves_dispatch_and_federation_on_replay(tmp_path: Path, legacy_singleton: bool) -> None:
    ControllerDB(db_dir=tmp_path).close()
    job_id = JobName.root("alice", "upgrade")
    entrypoint = {
        "run_command": {"argv": ["python", "_callable_runner.py"]},
        "setup_commands": ["prepare-workspace", "install-runtime"],
        "workdir_file_refs": {"weights.bin": "existing-content-id"},
    }
    environment = {
        "env_vars": {"TASK_INPUT": "primary-value"},
        "secrets": {"SERVICE_KEY": "explicit-value"},
        "setup_scripts": ["prepare-workspace"],
    }
    primary_resources = {"cpu_millicores": 2000, "memory_bytes": 1024, "disk_bytes": 2048}
    primary = {
        "name": "task",
        "image": "example.test/agent:old",
        "workdir": "/app",
        "resources": primary_resources,
        **environment,
        "runtime_entrypoint": entrypoint,
    }
    daemon = {
        "name": "daemon",
        "image": "example.test/daemon:old",
        "workdir": "/daemon",
        "env_vars": {"DAEMON_MODE": "enabled"},
        "resources": {"cpu_millicores": 500, "memory_bytes": 512, "disk_bytes": 1024},
        "image_command": {"entrypoint": {"argv": ["daemon"]}, "arguments": {"argv": ["--debug"]}},
    }
    if not legacy_singleton:
        environment.update(services=[daemon, primary], primary_service="task")
    resources = (
        primary_resources
        if legacy_singleton
        else {
            "cpu_millicores": 2500,
            "memory_bytes": 1536,
            "disk_bytes": 3072,
        }
    )
    runner = b"print('preserved')"
    with closing(sqlite3.connect(tmp_path / ControllerDB.DB_FILENAME)) as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, user_id, root_job_id, depth, state, submitted_at_ms, "
            "root_submitted_at_ms, num_tasks) VALUES (?, 'alice', ?, 0, ?, 1, 1, 1)",
            (job_id.to_wire(), job_id.to_wire(), job_pb2.JOB_STATE_PENDING),
        )
        conn.execute(
            "INSERT INTO job_config (job_id, environment_json, entrypoint_json, task_image, "
            "res_cpu_millicores, res_memory_bytes, res_disk_bytes) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                job_id.to_wire(),
                json.dumps(environment),
                json.dumps(entrypoint),
                primary["image"],
                resources["cpu_millicores"],
                resources["memory_bytes"],
                resources["disk_bytes"],
            ),
        )
        conn.execute(
            "INSERT INTO job_workdir_files VALUES (?, ?, ?)",
            (job_id.to_wire(), "_callable_runner.py", runner),
        )
        conn.execute("DELETE FROM schema_migrations WHERE name = '0055_task_containers.py'")
        if legacy_singleton:
            conn.execute("DELETE FROM schema_migrations WHERE name = '0054_task_services.py'")
        conn.commit()

    expected_containers = [primary] if legacy_singleton else [daemon, primary]
    first_stored = None
    for replay in range(2):
        if replay:
            # Replay after a completed row update whose ledger entry was not
            # committed, as can happen when a migration is interrupted.
            with closing(sqlite3.connect(tmp_path / ControllerDB.DB_FILENAME)) as conn:
                conn.execute("DELETE FROM schema_migrations WHERE name = '0055_task_containers.py'")
                conn.commit()
        with closing(ControllerDB(db_dir=tmp_path)) as db:
            projection = RunTemplatesProjection(db)
            with db.read_snapshot() as tx:
                job = reads.get_job_detail(tx, job_id)
                assert job is not None
                stored = json.loads(job.environment_json)
                assert stored == {
                    "env_vars": environment["env_vars"],
                    "secrets": environment["secrets"],
                    "setup_scripts": environment["setup_scripts"],
                    "containers": expected_containers,
                    "primary_container": "task",
                }
                template = projection.get(tx, job_id)
                launch = reconstruct_launch_job_request(job, workdir_files=reads.get_workdir_files(tx, job_id))
                assert template is not None
                assert template.environment == launch.environment
                assert [container.name for container in template.environment.containers] == [
                    container["name"] for container in expected_containers
                ]
                assert template.environment.primary_container == "task"
                assert dict(template.environment.containers[-1].runtime_entrypoint.workdir_files) == {
                    "_callable_runner.py": runner
                }
                assert template.resources == job_pb2.ResourceSpecProto(**resources)
                if not legacy_singleton:
                    assert list(template.environment.containers[0].image_command.arguments.argv) == ["--debug"]
                if first_stored is None:
                    first_stored = job.environment_json
                else:
                    assert job.environment_json == first_stored
