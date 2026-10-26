# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
from pathlib import Path

import cloudpickle
import pytest
from fray.iris_backend import resolve_coscheduling
from marin.external_dependencies import TPU_INFERENCE_FORK_REQUIREMENT, VLLM_FORK_REQUIREMENT
from marin.inference.vllm_server import IsolatedTpuVllm
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.serving_parity import launch
from experiments.evaluation.serving_parity.launch import PROFILES, Arm, isolated_runtime
from experiments.evaluation.serving_parity.settings import PROBE


def test_gang_entrypoint_preserves_the_arm_argument():
    arm = Arm("canary", "v4-32", "gs://bucket/fresh", "gs://bucket/model")
    fn, args, kwargs = cloudpickle.loads(arm.entrypoint().workdir_files["_callable.pkl"])
    bound = inspect.signature(fn).bind(*args, **kwargs)
    assert bound.arguments == {"arm": arm}
    resource = arm.resources()
    assert resource.replicas == 4
    assert resolve_coscheduling(resource, resource.replicas).group_by == "tpu-name"


def test_profiles_use_shipped_pin_without_schedule_override():
    for hardware, profile in PROFILES.items():
        arm = Arm("calibration", hardware, "gs://bucket/output", "gs://bucket/model")
        service = arm.service()
        assert "--hf-overrides" not in service.engine.extra_args
        assert service.model.max_model_len == PROBE.max_model_len
        assert service.engine.max_num_batched_tokens == PROBE.max_num_batched_tokens
        assert service.engine.max_num_seqs == PROBE.max_num_seqs
        if profile.is_tpu:
            assert arm.resources().replicas == arm.resources().device.vm_count()
    assert Arm("v5p", "v5p-8", "gs://bucket/out", "gs://bucket/model").service().model.tensor_parallel_size == 4


def test_isolated_runtime_restores_command_and_environment_on_failure(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PARITY_TPU_VARIANT", "ignored-inherited-value")
    before = IsolatedTpuVllm.command
    arm = Arm("v5p", "v5p-8", "gs://bucket/out", "gs://bucket/model")
    with pytest.raises(RuntimeError, match="forced"):
        with isolated_runtime(arm):
            command = IsolatedTpuVllm(
                vllm_ref=VLLM_FORK_REQUIREMENT, tpu_inference_ref=TPU_INFERENCE_FORK_REQUIREMENT
            ).command()
            assert "--config" in command
            config = json.loads(Path(command[command.index("--config") + 1]).read_text())
            assert config["hardware"] == "v5p-8"
            assert config["tensor_parallel_size"] == 4
            raise RuntimeError("forced")
    assert IsolatedTpuVllm.command is before
    assert not list(tmp_path.glob(".parity-*"))


def test_startup_failure_archives_logs_before_close(monkeypatch, tmp_path):

    calls = []
    monkeypatch.setattr(launch.VllmEnvironment, "_report_failure_and_close", lambda env, message: calls.append("closed"))
    original = launch.VllmEnvironment._report_failure_and_close

    class FailedEnvironment:
        def logs(self):
            return "load failed https://private.example/capability-token"

    with launch.capture_backend(StoragePath(str(tmp_path))):
        launch.VllmEnvironment._report_failure_and_close(FailedEnvironment(), "failure")
        assert calls == ["closed"]
        assert (tmp_path / "startup-failure.json").exists()
        assert (tmp_path / "server.log").read_text() == "load failed <redacted-url>"
    assert launch.VllmEnvironment._report_failure_and_close is original


def test_startup_cleanup_survives_artifact_write_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(launch.VllmEnvironment, "_report_failure_and_close", lambda env, message: calls.append("closed"))

    class FailedStorage:
        def __truediv__(self, name):
            return self

        def write_text(self, text):
            raise OSError("upload unavailable")

    class FailedEnvironment:
        def logs(self):
            return "diagnostics"

    with launch.capture_backend(FailedStorage()):
        launch.VllmEnvironment._report_failure_and_close(FailedEnvironment(), "failure")
    assert calls == ["closed"]
