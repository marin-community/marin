from argparse import Namespace
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

import fsspec
import pytest

from cloud.iris import task_runtime
from cloud.iris.ray_storage import RaySpillBackend
from cloud.iris.task_runtime import sync_debug_artifacts
from marinskyrl.environment_contract import (
    COLLECTIVE_PHASE_DIAGNOSTICS_ENV,
    DEBUG_ARTIFACT_DIR_ENV,
    RUN_ID_ENV,
    DebugMode,
    EnvVarManager,
    EnvVarScope,
    write_process_manifest,
)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize(
    ("mode", "explicit_root"),
    [(DebugMode.LIGHT, False), (DebugMode.LIGHT, True), (DebugMode.OFF, False)],
)
def test_supervisors_share_worker_diagnostics_with_uploader(tmp_path: Path, monkeypatch, rank, mode, explicit_root):
    job_name = f"debug-wiring-{uuid.uuid4().hex}"
    monkeypatch.setenv(RUN_ID_ENV, job_name)
    monkeypatch.setenv(DEBUG_ARTIFACT_DIR_ENV, str(tmp_path / "explicit-debug") if explicit_root else "")
    monkeypatch.setenv("IRIS_TASK_ID", f"/test/{job_name}/{rank}:0")
    monkeypatch.setenv("IRIS_NUM_TASKS", "2")
    monkeypatch.setenv("IRIS_ADVERTISE_HOST", "192.0.2.1")
    monkeypatch.delenv(COLLECTIVE_PHASE_DIAGNOSTICS_ENV, raising=False)
    default_root = Path(
        EnvVarManager.for_debug_launch(job_name=job_name).environment_for(EnvVarScope.TASK_RUNTIME)[
            DEBUG_ARTIFACT_DIR_ENV
        ]
    )
    args = Namespace(
        ray_port=6379,
        ray_log_dir=None,
        rendezvous_dir=str(tmp_path),
        rendezvous_timeout=1,
        ray_spill_backend=RaySpillBackend.LOCAL,
        ray_spill_dir=str(tmp_path / "spill"),
    )
    task_runtime.write_rendezvous(args.rendezvous_dir, "192.0.2.1", args.ray_port, "test-gang")

    # Stop at the external Ray launch, where child processes inherit the task environment.
    def stop_at_ray_start(command, **_kwargs):
        raise subprocess.CalledProcessError(99, command)

    monkeypatch.setattr(task_runtime.subprocess, "run", stop_at_ray_start)
    monkeypatch.setattr(task_runtime.signal, "signal", lambda *_args: None)
    destination = f"memory://{job_name}"
    filesystem = fsspec.filesystem("memory")
    try:
        options = {"debug_mode": mode} if mode is DebugMode.OFF else {}
        with pytest.raises(subprocess.CalledProcessError):
            if rank == 0:
                task_runtime.run_head(args, tmp_path / "config.yaml", **options)
            else:
                task_runtime.run_worker(args, **options)
        trainer = dict(debug_mode=mode.value, ckpt_path=str(tmp_path / "ckpt"), collective_phase_diagnostics=False)
        worker_environment = dict(os.environ)
        EnvVarManager.from_config({"trainer": trainer}).apply_to_process(
            EnvVarScope.RAY_WORKER, environ=worker_environment
        )
        if mode is not DebugMode.OFF:
            manifest_path = write_process_manifest("worker", environment=worker_environment)
            if explicit_root:
                assert manifest_path.is_relative_to(tmp_path / "explicit-debug")
        sync_debug_artifacts(destination, "node-0", "final")
        if mode is DebugMode.OFF:
            assert not filesystem.exists(f"/{job_name}/debug_artifacts")
        else:
            relative_path = manifest_path.relative_to(Path(os.environ[DEBUG_ARTIFACT_DIR_ENV])).as_posix()
            receipt = json.loads(filesystem.cat(f"/{job_name}/debug_artifacts/node-0/{relative_path}"))
            assert receipt["role"] == "worker"
            assert COLLECTIVE_PHASE_DIAGNOSTICS_ENV not in receipt["environment"]
    finally:
        if default_root.exists():
            shutil.rmtree(default_root)


def test_debug_sync_persists_files_and_complete_manifest(tmp_path: Path, monkeypatch):
    artifact_root = tmp_path / "debug"
    (artifact_root / "flight_recorder").mkdir(parents=True)
    (artifact_root / "processes").mkdir()
    (artifact_root / "flight_recorder" / "nccl_fr_rank_0").write_bytes(b"flight")
    (artifact_root / "processes" / "rank0.json").write_text('{"rank": 0}\n')
    monkeypatch.setenv(DEBUG_ARTIFACT_DIR_ENV, str(artifact_root))

    sync_debug_artifacts("memory://debug-contract", "node-0", "test")

    filesystem = fsspec.filesystem("memory")
    base = "/debug-contract/debug_artifacts/node-0"
    assert filesystem.cat(f"{base}/flight_recorder/nccl_fr_rank_0") == b"flight"
    manifest = json.loads(filesystem.cat(f"{base}/sync-manifest.json"))
    assert {item["path"] for item in manifest["copied"]} == {
        "flight_recorder/nccl_fr_rank_0",
        "processes/rank0.json",
    }
    assert manifest["skipped"] == []


def test_debug_sync_records_files_rejected_by_budget(tmp_path: Path, monkeypatch):
    artifact_root = tmp_path / "debug"
    artifact_root.mkdir()
    (artifact_root / "oversized.bin").write_bytes(b"12345")
    monkeypatch.setenv(DEBUG_ARTIFACT_DIR_ENV, str(artifact_root))
    monkeypatch.setattr("cloud.iris.task_runtime.DEBUG_SYNC_MAX_FILE_BYTES", 4)

    sync_debug_artifacts("memory://debug-budget", "node-1", "test")

    filesystem = fsspec.filesystem("memory")
    manifest = json.loads(filesystem.cat("/debug-budget/debug_artifacts/node-1/sync-manifest.json"))
    assert manifest["copied"] == []
    assert manifest["skipped"] == [{"bytes": 5, "path": "oversized.bin", "reason": "budget"}]
