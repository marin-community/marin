# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
import subprocess
from types import SimpleNamespace

import huggingface_hub
import pytest

from experiments.agentic_evals.backends.iris import IrisBackend
from experiments.agentic_evals.backends.federated_iris import (
    build_marin_serve_command,
    build_wait_and_mint_command,
    mint_external_api_base,
)
from experiments.agentic_evals.launch import (
    DEFAULT_GPU_DISK,
    DEFAULT_TPU_DISK,
    _normalize,
    build_worker_command,
    create_parser,
)
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


def test_grug_external_profile_uses_parent_federation_and_durable_jobs(tmp_path):
    harbor_config = tmp_path / "harbor.yaml"
    harbor_config.write_text("agents:\n  - name: opencode\n")
    args = create_parser().parse_args(
        [
            "--harbor_config",
            str(harbor_config),
            "--dataset",
            "DCAgent/dev_set_v2",
            "--job_name",
            "grug-r10",
            "--external-profile",
            "grug",
        ]
    )

    _normalize(args)

    assert args.target_cluster == "cw-us-east-02a"
    assert args.gpu == "H100x8"
    assert args.n_concurrent == 256
    assert "--jobs-dir=s3://marin-us-east-02a/iris/grug-r10/trace_jobs" in args.harbor_extra_arg
    assert "--model-loader-extra-config={\"distributed\":true}" not in build_marin_serve_command(args)
    assert "--idle-timeout-hours" in build_marin_serve_command(args)


def test_external_endpoint_worker_uses_environment_reference_not_capability_url(tmp_path):
    harbor_config = tmp_path / "harbor.yaml"
    harbor_config.write_text("agents:\n  - name: opencode\n")
    args = create_parser().parse_args(
        [
            "--harbor_config",
            str(harbor_config),
            "--model",
            "vllm/example",
            "--dataset",
            "DCAgent/dev_set_v2",
            "--job_name",
            "external-r10",
            "--external-endpoint",
            "/serve/existing",
            "--target-cluster",
            "cw-us-east-02a",
            "--harbor_extra_arg=--jobs-dir=s3://bucket/run",
        ]
    )

    _normalize(args)
    command = build_worker_command(args)

    assert "EXTERNAL_AGENT_API_BASE" in command
    assert "scoped-token" not in command
    assert "--harbor_extra_arg=--jobs-dir=s3://bucket/run" in command


def test_wait_and_mint_command_is_parent_scoped_and_keeps_url_out_of_errors():
    args = SimpleNamespace(
        iris_bin="iris",
        external_parent_cluster="marin",
        external_endpoint_name="/serve/existing",
        external_ttl_hours=24.0,
        external_ready_timeout_seconds=1800.0,
        external_parent_ingress_host="https://iris.oa.dev",
    )
    command = build_wait_and_mint_command(args)
    assert command[:5] == ["iris", "--cluster", "marin", "endpoints", "wait-and-mint"]
    assert "--require-peer" in command

    api_base = mint_external_api_base(
        args,
        run=lambda *_args, **_kwargs: subprocess.CompletedProcess(
            command, 0, stdout="https://iris.oa.dev/proxy/t/scoped-token/serve.existing/\n", stderr=""
        ),
    )
    assert api_base == "https://iris.oa.dev/proxy/t/scoped-token/serve.existing/v1"


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
