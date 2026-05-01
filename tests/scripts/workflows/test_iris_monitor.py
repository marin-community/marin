# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/workflows/iris_monitor.py and its internal helpers.

All tests are integration-style: they validate observable behavior (correct output,
correct subprocess command construction, correct exit codes) rather than internal state.
No real iris/gcloud/kubectl is invoked; all subprocess calls are intercepted via
injectable callable seams.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

# ---------------------------------------------------------------------------
# Module loading — scripts/workflows/ is not an installed package, so we load
# the modules directly from their file paths and register them in sys.modules
# so that iris_monitor.py's relative imports work correctly.
# ---------------------------------------------------------------------------

_WORKFLOWS_DIR = Path(__file__).parents[3] / "scripts" / "workflows"


def _load_module(name: str) -> object:
    path = _WORKFLOWS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_iris_cli = _load_module("_iris_cli")
_iris_diagnostics_gcp = _load_module("_iris_diagnostics_gcp")
_iris_diagnostics_coreweave = _load_module("_iris_diagnostics_coreweave")
_iris_monitor = _load_module("iris_monitor")

# Re-export the symbols we test against.
IrisJobState = _iris_cli.IrisJobState
IrisJobStatus = _iris_cli.IrisJobStatus
DiagnosticsRequest = _iris_cli.DiagnosticsRequest
iris_command = _iris_cli.iris_command
job_status = _iris_cli.job_status
wait_for_job = _iris_cli.wait_for_job
collect_diagnostics = _iris_monitor.collect_diagnostics
k8s_job_label = _iris_diagnostics_coreweave.k8s_job_label
cli = _iris_monitor.cli

_FAKE_REPO = Path("/fake/repo")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(stdout: str = "", stderr: str = "", returncode: int = 0) -> callable:
    """Return a run callable that always returns the given values.

    Accepts and ignores keyword arguments so it can be used as a drop-in for
    both subprocess.run (which receives capture_output, text, check) and the
    injectable collect_diagnostics run parameter (which receives only cmd).
    """
    calls: list[list[str]] = []

    def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=returncode, stdout=stdout, stderr=stderr)

    run.calls = calls  # type: ignore[attr-defined]
    return run


def _iris_json(rows: list[dict]) -> str:
    return json.dumps(rows)


# ---------------------------------------------------------------------------
# job_status: parsing
# ---------------------------------------------------------------------------


class TestJobStatus:
    def _run_for(self, rows: list[dict], returncode: int = 0):
        """Build a fake run that returns the given rows as iris JSON output."""
        stdout = _iris_json(rows)
        captured: list[list[str]] = []

        def run_impl(cmd):
            captured.append(cmd)
            return subprocess.CompletedProcess(args=cmd, returncode=returncode, stdout=stdout, stderr="")

        with patch.object(_iris_cli, "subprocess") as mock_subprocess:
            mock_subprocess.run.side_effect = run_impl
            return job_status("job-a", iris_config=None, repo_root=_FAKE_REPO), captured

    def test_single_matching_job(self):
        rows = [{"job_id": "job-a", "state": "JOB_STATE_RUNNING", "error": None}]
        stdout = _iris_json(rows)

        def run(cmd, **kwargs):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

        with patch.object(_iris_cli.subprocess, "run", side_effect=run):
            s = job_status("job-a", iris_config=None, repo_root=_FAKE_REPO)

        assert s.job_id == "job-a"
        assert s.state == IrisJobState.RUNNING
        assert s.error is None

    def test_multiple_jobs_selects_by_job_id(self):
        """When multiple jobs are returned, the correct one is selected by job_id, not position."""
        rows = [
            {"job_id": "job-b", "state": "JOB_STATE_RUNNING", "error": None},
            {"job_id": "job-a", "state": "JOB_STATE_SUCCEEDED", "error": None},
        ]
        stdout = _iris_json(rows)

        def run(cmd, **kwargs):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

        with patch.object(_iris_cli.subprocess, "run", side_effect=run):
            s = job_status("job-a", iris_config=None, repo_root=_FAKE_REPO)

        assert s.state == IrisJobState.SUCCEEDED

    def test_missing_job_raises_lookup_error(self):
        rows = [{"job_id": "job-other", "state": "JOB_STATE_RUNNING", "error": None}]
        stdout = _iris_json(rows)

        def run(cmd, **kwargs):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

        with patch.object(_iris_cli.subprocess, "run", side_effect=run):
            with pytest.raises(LookupError, match="job-a"):
                job_status("job-a", iris_config=None, repo_root=_FAKE_REPO)

    def test_malformed_json_raises_runtime_error(self):
        def run(cmd, **kwargs):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="not json", stderr="")

        with patch.object(_iris_cli.subprocess, "run", side_effect=run):
            with pytest.raises(RuntimeError, match="malformed JSON"):
                job_status("job-a", iris_config=None, repo_root=_FAKE_REPO)

    def test_nonzero_iris_exit_raises_runtime_error_with_stderr(self):
        def run(cmd, **kwargs):
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="connection refused")

        with patch.object(_iris_cli.subprocess, "run", side_effect=run):
            with pytest.raises(RuntimeError, match="connection refused"):
                job_status("job-a", iris_config=None, repo_root=_FAKE_REPO)

    def test_error_field_is_propagated(self):
        rows = [{"job_id": "job-a", "state": "JOB_STATE_FAILED", "error": "OOM killed"}]
        stdout = _iris_json(rows)

        def run(cmd, **kwargs):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

        with patch.object(_iris_cli.subprocess, "run", side_effect=run):
            s = job_status("job-a", iris_config=None, repo_root=_FAKE_REPO)

        assert s.error == "OOM killed"


# ---------------------------------------------------------------------------
# wait_for_job
# ---------------------------------------------------------------------------


def _make_status_sequence(*states: IrisJobState, error: str | None = None):
    """Return a job_status stub that yields successive states on each call."""
    it = iter(states)

    def stub(job_id, *, iris_config, prefix, repo_root, controller_url=None):
        state = next(it)
        return IrisJobStatus(job_id=job_id, state=state, error=error if state != IrisJobState.SUCCEEDED else None)

    return stub


class TestWaitForJob:
    def _wait(self, status_stub, *, poll_interval=10.0, timeout=None):
        sleeps: list[float] = []
        clock = [0.0]

        def sleep(secs):
            sleeps.append(secs)
            clock[0] += secs

        def monotonic():
            return clock[0]

        with patch.object(_iris_cli, "job_status", side_effect=status_stub):
            result = wait_for_job(
                "job-a",
                iris_config=None,
                prefix=None,
                poll_interval=poll_interval,
                timeout=timeout,
                repo_root=_FAKE_REPO,
                sleep=sleep,
                monotonic=monotonic,
            )
        return result, sleeps

    def test_succeeds_on_succeeded_state(self):
        stub = _make_status_sequence(IrisJobState.SUCCEEDED)
        result, _ = self._wait(stub)
        assert result.state == IrisJobState.SUCCEEDED

    def test_fails_on_failed_state(self):
        stub = _make_status_sequence(IrisJobState.FAILED)
        result, _ = self._wait(stub)
        assert result.state == IrisJobState.FAILED

    def test_fails_on_cancelled_state(self):
        stub = _make_status_sequence(IrisJobState.CANCELLED)
        result, _ = self._wait(stub)
        assert result.state == IrisJobState.CANCELLED

    def test_polls_multiple_times(self):
        """Three statuses (RUNNING, RUNNING, SUCCEEDED) should produce exactly two sleeps."""
        stub = _make_status_sequence(IrisJobState.RUNNING, IrisJobState.RUNNING, IrisJobState.SUCCEEDED)
        _, sleeps = self._wait(stub, poll_interval=5.0)
        assert len(sleeps) == 2
        assert all(s == 5.0 for s in sleeps)

    def test_timeout_raises_timeout_error(self):
        # Always returns RUNNING — will never reach terminal state.
        def always_running(job_id, *, iris_config, prefix, repo_root, controller_url=None):
            return IrisJobStatus(job_id=job_id, state=IrisJobState.RUNNING, error=None)

        sleeps: list[float] = []
        clock = [0.0]

        def sleep(secs):
            # Advance time by more than the poll interval to trigger timeout.
            sleeps.append(secs)
            clock[0] += secs + 50.0  # jump forward in time

        def monotonic():
            return clock[0]

        with patch.object(_iris_cli, "job_status", side_effect=always_running):
            with pytest.raises(TimeoutError):
                wait_for_job(
                    "job-a",
                    iris_config=None,
                    prefix=None,
                    poll_interval=10.0,
                    timeout=30.0,
                    repo_root=_FAKE_REPO,
                    sleep=sleep,
                    monotonic=monotonic,
                )


# ---------------------------------------------------------------------------
# collect_diagnostics — GCP
# ---------------------------------------------------------------------------


class TestCollectDiagnosticsGcp:
    def _make_request(self, output_dir: Path) -> DiagnosticsRequest:
        return DiagnosticsRequest(
            job_id="my-job-123",
            output_dir=output_dir,
            iris_config=None,
            provider="gcp",
            project="my-project",
            controller_label="iris-marin-controller",
            namespace=None,
            service_account="svc@proj.iam.gserviceaccount.com",
            ssh_key=Path("/home/runner/.ssh/google_compute_engine"),
            kubeconfig=None,
        )

    def test_gcp_collects_controller_logs(self, tmp_path):
        """Verify that gcloud SSH is called for each discovered controller instance."""
        instances_csv = "ctrl-vm-1,us-central1-a\nctrl-vm-2,us-east1-b\n"

        calls: list[list[str]] = []

        def run(cmd: list[str]) -> subprocess.CompletedProcess:
            calls.append(cmd)
            # gcloud instances list returns CSV; SSH and iris commands return empty logs.
            if "instances" in cmd and "list" in cmd:
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=instances_csv, stderr="")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="log output", stderr="")

        request = self._make_request(tmp_path)
        collect_diagnostics(request, repo_root=_FAKE_REPO, run=run)

        ssh_calls = [c for c in calls if "compute" in c and "ssh" in c]
        assert len(ssh_calls) == 2

        names_in_calls = {c[c.index("ssh") + 1] for c in ssh_calls}
        assert names_in_calls == {"ctrl-vm-1", "ctrl-vm-2"}

    def test_gcp_ssh_args_include_impersonate_and_key(self, tmp_path):
        calls: list[list[str]] = []

        def run(cmd):
            calls.append(cmd)
            if "instances" in cmd and "list" in cmd:
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ctrl-vm,us-central1-a\n", stderr="")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="logs", stderr="")

        request = self._make_request(tmp_path)
        collect_diagnostics(request, repo_root=_FAKE_REPO, run=run)

        ssh_calls = [c for c in calls if "compute" in c and "ssh" in c]
        assert len(ssh_calls) == 1
        ssh_cmd = ssh_calls[0]
        assert "--impersonate-service-account=svc@proj.iam.gserviceaccount.com" in ssh_cmd
        assert "--ssh-key-file=/home/runner/.ssh/google_compute_engine" in ssh_cmd
        assert "--project=my-project" in ssh_cmd
        assert "--zone=us-central1-a" in ssh_cmd

    def test_gcp_writes_summary_and_controller_logs(self, tmp_path):
        def run(cmd):
            if "instances" in cmd and "list" in cmd:
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ctrl-1,us-central1-a\n", stderr="")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="some log content", stderr="")

        request = self._make_request(tmp_path)
        collect_diagnostics(request, repo_root=_FAKE_REPO, run=run)

        assert (tmp_path / "summary.json").exists()
        assert (tmp_path / "controller-ctrl-1.log").exists()
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert "controller-ctrl-1.log" in summary["files"]

    def test_gcp_raises_when_no_instances_found(self, tmp_path):
        def run(cmd):
            if "instances" in cmd and "list" in cmd:
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        request = self._make_request(tmp_path)
        with pytest.raises(RuntimeError, match="No GCP controller logs"):
            collect_diagnostics(request, repo_root=_FAKE_REPO, run=run)


# ---------------------------------------------------------------------------
# collect_diagnostics — CoreWeave
# ---------------------------------------------------------------------------


class TestCollectDiagnosticsCoreweave:
    def _make_request(self, output_dir: Path) -> DiagnosticsRequest:
        return DiagnosticsRequest(
            job_id="my_job_123",
            output_dir=output_dir,
            iris_config=None,
            provider="coreweave",
            project=None,
            controller_label=None,
            namespace="iris-ci",
            service_account=None,
            ssh_key=None,
            kubeconfig=Path("/home/runner/.kube/coreweave-iris"),
        )

    def test_coreweave_calls_kubectl_get_pods(self, tmp_path):
        calls: list[list[str]] = []

        def run(cmd):
            calls.append(cmd)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout='{"items": []}', stderr="")

        request = self._make_request(tmp_path)
        collect_diagnostics(request, repo_root=_FAKE_REPO, run=run)

        kubectl_calls = [c for c in calls if c[0] == "kubectl"]
        assert len(kubectl_calls) == 1
        kubectl_cmd = kubectl_calls[0]
        assert "--kubeconfig=/home/runner/.kube/coreweave-iris" in kubectl_cmd
        assert "-n" in kubectl_cmd
        assert "iris-ci" in kubectl_cmd
        assert "get" in kubectl_cmd
        assert "pods" in kubectl_cmd

    def test_coreweave_label_sanitization_in_kubectl_call(self, tmp_path):
        calls: list[list[str]] = []

        def run(cmd):
            calls.append(cmd)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout='{"items": []}', stderr="")

        request = self._make_request(tmp_path)
        collect_diagnostics(request, repo_root=_FAKE_REPO, run=run)

        kubectl_calls = [c for c in calls if c[0] == "kubectl"]
        kubectl_cmd = kubectl_calls[0]
        # my_job_123 → my-job-123 (underscore → dash)
        label_arg = next(a for a in kubectl_cmd if "iris.job_id=" in a)
        # Extract just the label value after "iris.job_id="
        label_value = label_arg.split("iris.job_id=", 1)[1]
        assert label_value == "my-job-123"
        assert "_" not in label_value

    def test_coreweave_writes_kubernetes_pods_json(self, tmp_path):
        pods_json = '{"items": [{"metadata": {"name": "task-pod-0"}}]}'

        def run(cmd):
            if "kubectl" in cmd[0]:
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=pods_json, stderr="")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        request = self._make_request(tmp_path)
        collect_diagnostics(request, repo_root=_FAKE_REPO, run=run)

        pods_file = tmp_path / "kubernetes-pods.json"
        assert pods_file.exists()
        data = json.loads(pods_file.read_text())
        assert data["items"][0]["metadata"]["name"] == "task-pod-0"

    def test_coreweave_raises_when_kubectl_fails(self, tmp_path):
        def run(cmd):
            if "kubectl" in cmd[0]:
                return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="no route to host")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        request = self._make_request(tmp_path)
        with pytest.raises(RuntimeError, match=r"kubernetes-pods\.json"):
            collect_diagnostics(request, repo_root=_FAKE_REPO, run=run)


# ---------------------------------------------------------------------------
# Kubernetes label sanitization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "job_id, expected",
    [
        ("simple_job", "simple-job"),
        ("no_underscores", "no-underscores"),
        (
            # 70-char job ID with underscores — result must be ≤63 chars with no underscores
            "a_very_long_job_id_that_exceeds_the_kubernetes_label_limit_of_63_chars",
            "a-very-long-job-id-that-exceeds-the-kubernetes-label-limit-of-6",
        ),
        ("/leading_slash", "leading-slash"),
        ("no-underscores", "no-underscores"),
    ],
)
def test_k8s_job_label(job_id, expected):
    result = k8s_job_label(job_id)
    assert result == expected
    assert len(result) <= 63
    assert "_" not in result


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------


class TestCli:
    def test_wait_help_shows_flags(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["wait", "--help"])
        assert result.exit_code == 0
        assert "--job-id" in result.output
        assert "--iris-config" in result.output
        assert "--controller-url" in result.output
        assert "--poll-interval" in result.output
        assert "--github-output" in result.output

    def test_status_help_shows_flags(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--help"])
        assert result.exit_code == 0
        assert "--job-id" in result.output

    def test_collect_help_shows_flags(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["collect", "--help"])
        assert result.exit_code == 0
        assert "--provider" in result.output
        assert "--output-dir" in result.output

    def test_status_calls_job_status_with_correct_args(self, tmp_path):
        """CLI `status` should pass job_id and iris_config through to job_status."""
        captured: list[dict] = []

        def fake_job_status(job_id, *, iris_config, prefix, repo_root, controller_url=None):
            captured.append({"job_id": job_id, "iris_config": iris_config, "prefix": prefix})
            return IrisJobStatus(job_id=job_id, state=IrisJobState.RUNNING, error=None)

        config_path = tmp_path / "iris.yaml"
        config_path.write_text("controller_url: http://localhost:10000\n")

        runner = CliRunner()
        with patch.object(_iris_monitor, "job_status", side_effect=fake_job_status):
            result = runner.invoke(
                cli,
                ["status", "--job-id", "test-job", "--iris-config", str(config_path)],
            )

        assert result.exit_code == 0
        assert len(captured) == 1
        assert captured[0]["job_id"] == "test-job"
        assert captured[0]["iris_config"] == config_path

    def test_wait_exits_zero_on_succeeded(self):
        def fake_wait(*args, **kwargs):
            return IrisJobStatus(job_id="j", state=IrisJobState.SUCCEEDED, error=None)

        runner = CliRunner()
        with patch.object(_iris_monitor, "wait_for_job", side_effect=fake_wait):
            result = runner.invoke(cli, ["wait", "--job-id", "j"])
        assert result.exit_code == 0

    def test_wait_exits_nonzero_on_failed(self):
        def fake_wait(*args, **kwargs):
            return IrisJobStatus(job_id="j", state=IrisJobState.FAILED, error="OOM")

        runner = CliRunner()
        with patch.object(_iris_monitor, "wait_for_job", side_effect=fake_wait):
            result = runner.invoke(cli, ["wait", "--job-id", "j"])
        assert result.exit_code != 0

    def test_wait_writes_github_output(self, tmp_path, monkeypatch):
        out_file = tmp_path / "github_output"
        out_file.write_text("")
        monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))

        def fake_wait(*args, **kwargs):
            return IrisJobStatus(job_id="j", state=IrisJobState.SUCCEEDED, error=None)

        runner = CliRunner()
        with patch.object(_iris_monitor, "wait_for_job", side_effect=fake_wait):
            result = runner.invoke(cli, ["wait", "--job-id", "j", "--github-output"])

        assert result.exit_code == 0
        contents = out_file.read_text()
        assert "job_id=j" in contents
        assert "state=JOB_STATE_SUCCEEDED" in contents
        assert "succeeded=true" in contents
