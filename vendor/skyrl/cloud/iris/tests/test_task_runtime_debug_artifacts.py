import json
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from cloud.iris import task_runtime
from iris.cluster.runtime.env import IRIS_OUTPUT_DIR_ENV
from marinskyrl.environment_contract import (
    COLLECTIVE_PHASE_DIAGNOSTICS_ENV,
    DEBUG_ARTIFACT_DIR_ENV,
    FR_DUMP_TEMP_FILE_ENV,
    DebugMode,
    EnvVarManager,
    EnvVarScope,
    write_process_manifest,
)


@pytest.mark.parametrize("mode", [DebugMode.LIGHT, DebugMode.DISTRIBUTED, DebugMode.OFF])
def test_worker_debug_artifacts_use_iris_attempt_outputs(tmp_path, monkeypatch, mode):
    monkeypatch.setenv(IRIS_OUTPUT_DIR_ENV, str(tmp_path))
    monkeypatch.setenv("IRIS_TASK_ID", "/test/debug/0:0")
    monkeypatch.delenv(DEBUG_ARTIFACT_DIR_ENV, raising=False)
    monkeypatch.delenv(COLLECTIVE_PHASE_DIAGNOSTICS_ENV, raising=False)
    # Restore the process environment after applying the diagnostic config.
    monkeypatch.setattr(task_runtime.os, "environ", dict(os.environ))
    config = OmegaConf.create({"trainer": {"debug_mode": mode.value, "collective_phase_diagnostics": False}})

    task_runtime.configure_task_debug_artifacts(config.trainer)

    if mode is DebugMode.OFF:
        assert not list(tmp_path.iterdir())
        return

    worker_environment = dict(os.environ)
    EnvVarManager.from_config(config).apply_to_process(EnvVarScope.RAY_WORKER, environ=worker_environment)
    artifact_root = tmp_path / "skyrl-debug"
    manifest = write_process_manifest("worker", environment=worker_environment)
    assert manifest.is_relative_to(artifact_root)
    assert Path(worker_environment[FR_DUMP_TEMP_FILE_ENV]).parent == artifact_root / "flight_recorder"
    receipt = json.loads(manifest.read_text())
    assert receipt["role"] == "worker"
    assert COLLECTIVE_PHASE_DIAGNOSTICS_ENV not in receipt["environment"]
