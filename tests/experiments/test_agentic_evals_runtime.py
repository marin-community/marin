# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
from types import SimpleNamespace

import huggingface_hub
import pytest

from experiments.agentic_evals.backends.iris import IrisBackend
from experiments.agentic_evals.launch import DEFAULT_GPU_DISK, DEFAULT_TPU_DISK, _normalize, create_parser
from experiments.agentic_evals.results.hf_upload import HFResultSink
from experiments.agentic_evals.runtime.vllm_server import execute_model_timeout_seconds


def test_execute_model_timeout_exceeds_health_check_window():
    assert execute_model_timeout_seconds(100, 30) == 4_500


def test_launcher_selects_gpu_disk_default(tmp_path):
    harbor_config = tmp_path / "harbor.yaml"
    harbor_config.write_text("agents:\n  - name: terminus-2\n")
    args = create_parser().parse_args(
        [
            "--harbor_config",
            str(harbor_config),
            "--model",
            "Qwen/Qwen3-32B",
            "--dataset",
            "terminal-bench@2.0",
            "--gpu",
            "H100x8",
        ]
    )

    _normalize(args)

    assert args.disk == DEFAULT_GPU_DISK


def test_launcher_selects_tpu_disk_default(tmp_path):
    harbor_config = tmp_path / "harbor.yaml"
    harbor_config.write_text("agents:\n  - name: terminus-2\n")
    args = create_parser().parse_args(
        [
            "--harbor_config",
            str(harbor_config),
            "--model",
            "Qwen/Qwen3-32B",
            "--dataset",
            "terminal-bench@2.0",
        ]
    )

    _normalize(args)

    assert args.disk == DEFAULT_TPU_DISK


def test_iris_backend_submits_requested_gpu_resources(monkeypatch):
    captured = {}

    class FakeClient:
        def submit(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(job_id="/user/eval")

    class FakeTunnel:
        def __exit__(self, *_args):
            return False

    backend = IrisBackend(cluster_config="cluster.yaml")
    monkeypatch.setattr(backend, "_resolve_controller", lambda _path: (FakeClient(), FakeTunnel(), "unused"))

    env_vars = {"_iris_extras": ["harbor"], "HARBOR_ENV": "daytona"}
    backend.submit(
        command=["python", "-m", "agentic_evals.run_eval"],
        job_name="eval",
        env_vars=env_vars,
        accelerator="H100x8",
        cpu=8,
        memory="256GB",
        disk="512GB",
        no_wait=True,
    )

    resources = captured["resources"].to_proto()
    assert resources.cpu_millicores == 8_000
    assert resources.memory_bytes > 0
    assert resources.disk_bytes > 0
    assert resources.device.gpu.variant == "H100"
    assert resources.device.gpu.count == 8
    assert captured["environment"].env_vars == {"HARBOR_ENV": "daytona"}
    assert env_vars == {"_iris_extras": ["harbor"], "HARBOR_ENV": "daytona"}


def test_hf_result_sink_uses_current_harbor_export_api(monkeypatch, tmp_path):
    if importlib.util.find_spec("harbor") is None:
        pytest.skip("Harbor is an optional dependency outside the Harbor worker environment.")
    traces_utils = importlib.import_module("harbor.utils.traces_utils")

    captured = {}

    class FakeApi:
        def __init__(self, token):
            captured["api_token"] = token

        def create_repo(self, **kwargs):
            captured["repo"] = kwargs

    def fake_export_traces(**kwargs):
        captured["export"] = kwargs
        captured["export_token"] = __import__("os").environ["HUGGINGFACE_TOKEN"]

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(traces_utils, "export_traces", fake_export_traces)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    url = HFResultSink("marin/test", hf_token="test-token", hf_private=True).publish(
        job_dir=job_dir,
        job_name="job",
        model_name=None,
        benchmark_name=None,
    )

    assert captured["repo"] == {
        "repo_id": "marin/test",
        "repo_type": "dataset",
        "private": True,
        "exist_ok": True,
    }
    assert captured["export"] == {
        "root": job_dir,
        "episodes": "last",
        "repo_id": "marin/test",
        "push": True,
    }
    assert captured["export_token"] == "test-token"
    assert url == "https://huggingface.co/datasets/marin/test"
