# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/workflows/iris_monitor.py."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "scripts" / "workflows" / "iris_monitor.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("iris_monitor", _MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


iris_monitor = _load_module()


@pytest.fixture
def fake_run(monkeypatch):
    """Stub `_run` with a per-test command -> CompletedProcess mapping."""
    handlers: dict[tuple, subprocess.CompletedProcess] = {}
    calls: list[list[str]] = []

    def runner(cmd: list[str]) -> subprocess.CompletedProcess:
        joined = " ".join(cmd)
        calls.append(list(cmd))
        for matcher, result in handlers.items():
            if all(token in joined for token in matcher):
                return result
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(iris_monitor, "_run", runner)

    def register(match: tuple, *, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        handlers[match] = subprocess.CompletedProcess(list(match), returncode, stdout, stderr)

    runner.register = register  # type: ignore[attr-defined]
    runner.calls = calls  # type: ignore[attr-defined]
    return runner


def test_collect_gcp_writes_controller_and_worker_logs(fake_run, tmp_path: Path) -> None:
    fake_run.register(
        ("gcloud", "compute", "instances", "list"),
        stdout=(
            "iris-pr-1-controller-0,us-central1-a,"
            "iris-pr-1-controller=true;iris-pr-1-managed=true\n"
            "iris-pr-1-worker-0,us-central1-a,iris-pr-1-managed=true\n"
            "iris-pr-1-worker-1,us-east5-b,iris-pr-1-managed=true\n"
        ),
    )
    fake_run.register(("iris", "process", "logs"), stdout="controller process logs\n")
    fake_run.register(("iris", "job", "list"), stdout='[{"job_id": "/jobs_foo", "state": "JOB_STATE_FAILED"}]')
    fake_run.register(("gcloud", "compute", "ssh"), stdout="docker ps -a\n")

    out_dir = tmp_path / "diag"
    iris_monitor.collect_diagnostics(
        "/jobs_foo",
        out_dir,
        "gcp",
        iris_config=None,
        controller_url=None,
        project="hai-gcp-models",
        controller_label="iris-pr-1-controller",
        managed_label="iris-pr-1-managed",
        service_account="iris-controller@hai-gcp-models.iam.gserviceaccount.com",
        ssh_key=None,
        namespace=None,
        kubeconfig=None,
        include_cluster_context=False,
        repo_root=_REPO_ROOT,
    )

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["provider"] == "gcp"
    assert summary["missing_required_files"] == []
    files = set(summary["files"])
    assert "controller-iris-pr-1-controller-0.log" in files
    assert "worker-iris-pr-1-worker-0.log" in files
    assert "worker-iris-pr-1-worker-1.log" in files
    assert "job-tree.json" in files


def test_collect_gcp_raises_when_no_controller_logs(fake_run, tmp_path: Path) -> None:
    fake_run.register(("gcloud", "compute", "instances", "list"), stdout="")
    fake_run.register(("iris", "process", "logs"), stdout="")
    fake_run.register(("iris", "job", "list"), stdout="[]")

    out_dir = tmp_path / "diag"
    with pytest.raises(RuntimeError, match="Required gcp diagnostics missing"):
        iris_monitor.collect_diagnostics(
            "/jobs_foo",
            out_dir,
            "gcp",
            iris_config=None,
            controller_url=None,
            project="proj",
            controller_label="iris-x-controller",
            managed_label=None,
            service_account=None,
            ssh_key=None,
            namespace=None,
            kubeconfig=None,
            include_cluster_context=False,
            repo_root=_REPO_ROOT,
        )

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["missing_required_files"] == ["controller-*.log"]


def test_collect_coreweave_writes_pod_and_worker_logs(fake_run, tmp_path: Path) -> None:
    fake_run.register(("iris", "process", "logs"))
    fake_run.register(("iris", "job", "list"), stdout="[]")
    fake_run.register(
        ("kubectl", "get", "pods", "-l=iris.job_id=jobs-foo"),
        stdout='{"items":[]}',
    )
    fake_run.register(("kubectl", "get", "pods", "-l", "iris-iris-ci-managed=true"), stdout="pod/worker-a\n")
    fake_run.register(("kubectl", "logs"), stdout="log line\n")
    fake_run.register(("kubectl", "describe"), stdout="describe\n")
    fake_run.register(("kubectl", "get", "events"), stdout="evt\n")

    out_dir = tmp_path / "diag"
    iris_monitor.collect_diagnostics(
        "/jobs_foo",
        out_dir,
        "coreweave",
        iris_config=None,
        controller_url="http://localhost:1",
        project=None,
        controller_label=None,
        managed_label="iris-iris-ci-managed",
        service_account=None,
        ssh_key=None,
        namespace="iris-ci",
        kubeconfig=None,
        include_cluster_context=False,
        repo_root=_REPO_ROOT,
    )

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["missing_required_files"] == []
    files = set(summary["files"])
    assert "kubernetes-pods.json" in files
    assert "controller.log" in files
    assert "controller-previous.log" in files
    assert "controller-describe.txt" in files
    assert "events.txt" in files
    assert "pod-worker-a.log" in files
    assert "pod-worker-a-describe.txt" in files


def test_collect_coreweave_with_cluster_context(fake_run, tmp_path: Path) -> None:
    fake_run.register(("iris", "process", "logs"))
    fake_run.register(("iris", "job", "list"), stdout="[]")
    fake_run.register(
        ("kubectl", "get", "pods", "-l=iris.job_id=jobs-foo"),
        stdout='{"items":[]}',
    )
    fake_run.register(("kubectl", "logs"), stdout="x\n")
    fake_run.register(("kubectl", "describe"), stdout="x\n")
    fake_run.register(("kubectl", "get", "events"), stdout="x\n")
    fake_run.register(("kubectl", "get", "nodepools.compute.coreweave.com"), stdout="np-data\n")
    fake_run.register(("kubectl", "get", "nodes"), stdout="nodes\n")
    fake_run.register(("iris", "rpc", "controller", "get-autoscaler-status"), stdout="autoscaler\n")
    fake_run.register(("iris", "rpc", "controller", "get-scheduler-state"), stdout="scheduler\n")

    out_dir = tmp_path / "diag"
    iris_monitor.collect_diagnostics(
        "/jobs_foo",
        out_dir,
        "coreweave",
        iris_config=None,
        controller_url="http://localhost:1",
        project=None,
        controller_label=None,
        managed_label=None,
        service_account=None,
        ssh_key=None,
        namespace="iris-ci",
        kubeconfig=None,
        include_cluster_context=True,
        repo_root=_REPO_ROOT,
    )

    files = set(json.loads((out_dir / "summary.json").read_text())["files"])
    assert "nodepools.txt" in files
    assert "nodepools.yaml" in files
    assert "nodes.txt" in files
    assert "autoscaler-status.txt" in files
    assert "scheduler-state.txt" in files


def test_collect_coreweave_missing_pods_artifact_raises(fake_run, tmp_path: Path) -> None:
    fake_run.register(("iris", "process", "logs"))
    fake_run.register(("iris", "job", "list"), stdout="[]")
    fake_run.register(
        ("kubectl", "get", "pods", "-l=iris.job_id=jobs-foo"),
        returncode=1,
        stderr="connection refused",
    )

    out_dir = tmp_path / "diag"
    with pytest.raises(RuntimeError, match="Required coreweave diagnostics missing"):
        iris_monitor.collect_diagnostics(
            "/jobs_foo",
            out_dir,
            "coreweave",
            iris_config=None,
            controller_url=None,
            project=None,
            controller_label=None,
            managed_label=None,
            service_account=None,
            ssh_key=None,
            namespace="iris-ci",
            kubeconfig=None,
            include_cluster_context=False,
            repo_root=_REPO_ROOT,
        )

    summary = json.loads((out_dir / "summary.json").read_text())
    assert "kubernetes-pods.json" in summary["missing_required_files"]


def test_status_cli_reports_state(fake_run) -> None:
    fake_run.register(
        ("iris", "job", "list"),
        stdout=json.dumps([{"job_id": "/jobs_foo", "state": "JOB_STATE_SUCCEEDED"}]),
    )
    runner = CliRunner()
    result = runner.invoke(iris_monitor.cli, ["status", "--job-id", "/jobs_foo"])
    assert result.exit_code == 0, result.output
    assert "JOB_STATE_SUCCEEDED" in result.output


def test_wait_cli_writes_github_output(fake_run, tmp_path: Path, monkeypatch) -> None:
    fake_run.register(
        ("iris", "job", "list"),
        stdout=json.dumps([{"job_id": "/jobs_foo", "state": "JOB_STATE_FAILED", "error": "boom"}]),
    )
    output_path = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    runner = CliRunner()
    result = runner.invoke(
        iris_monitor.cli,
        ["wait", "--job-id", "/jobs_foo", "--poll-interval", "0.01", "--github-output"],
    )
    assert result.exit_code == 1, result.output
    body = output_path.read_text()
    assert "state=JOB_STATE_FAILED" in body
    assert "succeeded=false" in body


def test_port_forward_writes_github_env(monkeypatch, tmp_path: Path) -> None:
    def fake_run_kubectl(cmd: list[str]) -> subprocess.CompletedProcess:
        if "rollout" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "rolled out", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    class FakeProcess:
        def __init__(self, *, alive: bool = True) -> None:
            self.pid = 42
            self._alive = alive

        def poll(self) -> int | None:
            return None if self._alive else 0

        def terminate(self) -> None:
            self._alive = False

    monkeypatch.setattr(iris_monitor, "_run", fake_run_kubectl)
    monkeypatch.setattr(iris_monitor, "_start_port_forward", lambda *a, **kw: FakeProcess())
    monkeypatch.setattr(iris_monitor, "_free_local_port", lambda: 31337)

    class FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: FakeResp(),
    )

    env_path = tmp_path / "github_env"
    monkeypatch.setenv("GITHUB_ENV", str(env_path))

    runner = CliRunner()
    result = runner.invoke(
        iris_monitor.cli,
        [
            "port-forward",
            "--namespace",
            "iris-ci",
            "--service",
            "iris-ci-controller-svc",
            "--rollout-deployment",
            "iris-controller",
            "--timeout",
            "5",
            "--poll-interval",
            "0.01",
        ],
    )
    assert result.exit_code == 0, result.output
    body = env_path.read_text()
    assert "IRIS_CONTROLLER_URL=http://localhost:31337" in body
    assert "LOCAL_PORT=31337" in body
    assert "PF_PID=42" in body


def test_port_forward_times_out_when_unhealthy(monkeypatch) -> None:
    monkeypatch.setattr(iris_monitor, "_run", lambda cmd: subprocess.CompletedProcess(cmd, 0, "", ""))

    class FakeProcess:
        pid = 7

        def poll(self):
            return None

        def terminate(self):
            pass

    monkeypatch.setattr(iris_monitor, "_start_port_forward", lambda *a, **kw: FakeProcess())
    monkeypatch.setattr(iris_monitor, "_free_local_port", lambda: 12345)

    def always_refused(*args, **kwargs):
        raise ConnectionError("nope")

    monkeypatch.setattr("urllib.request.urlopen", always_refused)

    with pytest.raises(TimeoutError):
        iris_monitor.port_forward_until_healthy(
            "iris-ci",
            "svc-x",
            target_port=10000,
            timeout=0.05,
            poll_interval=0.01,
            kubeconfig=None,
            rollout_deployment="iris-controller",
            rollout_timeout=10,
            health_path="/health",
        )
