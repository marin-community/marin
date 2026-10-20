from __future__ import annotations

import csv
import json
from datetime import UTC, datetime, timedelta
from io import BytesIO, StringIO
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from scripts.iris import coreweave_ops, iris_ops, watch_coreweave_rl
from scripts.iris.iris_ops import (
    MonitorError,
    StyledCell,
    box_table,
    filter_records,
    job_bundle,
    job_id_parts,
    load_bundle_manifest,
    parse_regex_filters,
    render_error_report,
    strip_ansi,
    write_error_report,
    write_bundle_manifest,
)


def test_monitor_defaults_to_the_shared_document_bundle_root(monkeypatch):
    monkeypatch.setattr("sys.argv", ["watch_coreweave_rl.py"])

    bundle = job_bundle(watch_coreweave_rl.parse_args().bundle_root, "cw-rno2a", "/operator/training")

    assert bundle.directory.parts[-5:] == (
        "iris-job-bundles",
        "jobs",
        "cw-rno2a",
        "operator",
        "training",
    )
    assert "experiments" not in bundle.directory.parts


@pytest.mark.parametrize(
    ("iris_bin", "expected"), [(None, "iris"), ("/opt/operator/bin/iris", "/opt/operator/bin/iris")]
)
def test_iris_command_resolves_binary_when_invoked(monkeypatch, iris_bin, expected):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    if iris_bin is None:
        monkeypatch.delenv("IRIS_BIN", raising=False)
    else:
        monkeypatch.setenv("IRIS_BIN", iris_bin)
    monkeypatch.setattr(iris_ops.subprocess, "run", fake_run)

    iris_ops.run_iris_command(["job", "list"], cluster="cw-rno2a")

    assert calls[0][0] == [expected, "--cluster=cw-rno2a", "job", "list"]


@pytest.mark.parametrize(
    ("job_id", "relative_path"),
    [
        ("/benjaminfeuer/glm52-r10", Path("benjaminfeuer/glm52-r10")),
        (
            "/benjaminfeuer/coordinator/gpu-child",
            Path("benjaminfeuer/coordinator/gpu-child"),
        ),
    ],
)
def test_job_bundle_uses_cluster_and_full_iris_identity(tmp_path, job_id, relative_path):
    bundle = job_bundle(tmp_path, "cw-rno2a", job_id)

    assert bundle.directory == tmp_path / "jobs" / "cw-rno2a" / relative_path

    write_bundle_manifest(bundle, {"kind": "harbor", "progress": {"completed": 4}})

    assert json.loads(bundle.manifest_path.read_text())["job_id"] == job_id
    assert load_bundle_manifest(bundle)["progress"] == {"completed": 4}


def test_shared_regex_filters_records():
    records = [
        {"cluster": "cw-rno2a", "state": "running", "name": "glm52"},
        {"cluster": "marin", "state": "running", "name": "qwen"},
        {"cluster": "cw-rno2a", "state": "failed", "name": "glm52"},
    ]
    filters = parse_regex_filters(["cluster=^cw-", "name=glm", "state=running"], {"cluster", "name", "state"})

    assert filter_records(records, filters, lambda record: record) == [records[0]]


def test_box_table_wraps_to_width_and_sanitizes_multiline_cells():
    table = box_table(
        ["Job", "Status"],
        [["a-very-long-job-name-that-must-wrap", "line one\nline two"]],
        max_width=32,
    )

    assert max(len(line) for line in table.splitlines()) <= 32
    assert "line one" in table
    assert "line two" in table


def test_box_table_color_preserves_plain_layout_and_can_be_stripped():
    rows = [[StyledCell("running", "success"), StyledCell("warning", "warning")]]

    plain = box_table(["State", "Health"], rows, max_width=40)
    colored = box_table(["State", "Health"], rows, max_width=40, color=True)

    assert "\x1b[" not in plain
    assert "\x1b[" in colored
    assert strip_ansi(colored) == plain


def test_monitor_error_report_is_separate_stable_and_single_line(tmp_path):
    errors = [
        MonitorError("cw-rno2a/job-a", "Finelog sync", "proxy failed\nTraceback: details"),
        MonitorError("marin", "discovery", "controller unavailable"),
    ]

    report = render_error_report(
        "Iris Harbor monitor errors",
        datetime(2026, 7, 25, 12, tzinfo=UTC),
        errors,
    )
    path = write_error_report(
        tmp_path,
        "20260725T120000Z",
        "Iris Harbor monitor errors",
        datetime(2026, 7, 25, 12, tzinfo=UTC),
        errors,
    )

    assert "## cw-rno2a/job-a" in report
    assert "- **Finelog sync:** proxy failed Traceback: details" in report
    assert path == tmp_path / "20260725T120000Z.errors.md"
    assert path.read_text() == report
    assert (tmp_path / "latest-errors.md").read_text() == report


def test_shared_regex_filter_rejects_unknown_fields_and_invalid_regexes():
    with pytest.raises(ValueError, match="Unknown filter field"):
        parse_regex_filters(["missing=value"], {"state"})
    with pytest.raises(ValueError, match="Invalid regex"):
        parse_regex_filters(["state=["], {"state"})


@pytest.mark.parametrize("job_id", ["glm52-r10", "/benjaminfeuer", "/benjaminfeuer/../bad"])
def test_job_id_parts_rejects_noncanonical_or_unsafe_ids(job_id):
    with pytest.raises(ValueError):
        job_id_parts(job_id)


def test_find_pod_uses_exact_iris_job_label(monkeypatch):
    job_id = "/benjaminfeuer/grug-agentic-eval-v2-65k-harbor394c-r3"
    pod_name = "iris-benjaminfeuer-grug-agentic-eval-v2-65k-harbor39-c309bb6c-0"
    monkeypatch.setattr(
        coreweave_ops,
        "command",
        lambda _args: json.dumps(
            {
                "items": [
                    {
                        "metadata": {
                            "name": pod_name,
                            "labels": {"iris.job_id": "benjaminfeuer.grug-agentic-eval-v2-65k-harbor394c-r3"},
                        },
                        "status": {"phase": "Running"},
                    }
                ]
            }
        ),
    )

    assert coreweave_ops.find_pod(["kubectl"], SimpleNamespace(job=job_id, pod=None)) == pod_name


def test_find_pod_disambiguates_colliding_truncated_job_labels_with_pod_group(monkeypatch):
    job_id = "/benjaminfeuer/rl-tasktrove-dq-sweep-30b-gb200-qwen3-co-20260730-103953-9fed75"
    truncated_label = "benjaminfeuer.rl-tasktrove-dq-sweep-30b-gb200-qwen3-co-20260730"
    target_pod = "iris-benjaminfeuer-rl-tasktrove-dq-80d56912-0-2c1f55e27f6f9556"
    sibling_pod = "iris-benjaminfeuer-rl-tasktrove-dq-a4983364-0-4c4098b97cc5a138"
    monkeypatch.setattr(
        coreweave_ops,
        "command",
        lambda _args: json.dumps(
            {
                "items": [
                    {
                        "metadata": {
                            "name": sibling_pod,
                            "labels": {
                                "iris.job_id": truncated_label,
                                "kueue.x-k8s.io/pod-group-name": "iris-pg-f5f5efdf484bc2a9-0",
                            },
                        },
                        "status": {"phase": "Running"},
                    },
                    {
                        "metadata": {
                            "name": target_pod,
                            "labels": {
                                "iris.job_id": truncated_label,
                                "kueue.x-k8s.io/pod-group-name": "iris-pg-cf44f3e17dd34832-0",
                            },
                        },
                        "status": {"phase": "Running"},
                    },
                ]
            }
        ),
    )

    assert coreweave_ops.find_pod(["kubectl"], SimpleNamespace(job=job_id, pod=None)) == target_pod


def test_rl_job_pods_discovers_hashed_names_by_pod_group(monkeypatch):
    cluster = watch_coreweave_rl.Cluster("cw-us-east-08a", Path("/tmp/kubeconfig"), None)
    job = watch_coreweave_rl.IrisRlJob(
        cluster,
        "/benjaminfeuer/rl-tasktrove-dq-sweep-30b-gb200-qwen3-co-20260730-103953-9fed75",
        "running",
        0,
        "",
    )
    target_pod = "iris-benjaminfeuer-rl-tasktrove-dq-80d56912-0-2c1f55e27f6f9556"
    payload = {
        "items": [
            {
                "metadata": {
                    "name": target_pod,
                    "labels": {"kueue.x-k8s.io/pod-group-name": "iris-pg-cf44f3e17dd34832-0"},
                },
                "status": {"phase": "Running"},
            }
        ]
    }
    monkeypatch.setattr(
        watch_coreweave_rl.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr=""),
    )

    assert watch_coreweave_rl.job_pods(job) == [(target_pod, "Running")]


class CapturingInput(BytesIO):
    def close(self) -> None:
        self.closed_by_sync = True


class TarProcess:
    def __init__(self, stdout: bytes, stderr: bytes = b"", return_code: int = 0, *, capture_input: bool = False):
        self.stdin = CapturingInput() if capture_input else None
        self.stdout = BytesIO(stdout)
        self.stderr = BytesIO(stderr)
        self.return_code = return_code
        self.killed = False

    def wait(self) -> int:
        return self.return_code

    def kill(self) -> None:
        self.killed = True


def test_save_ray_logs_reports_empty_tar_stream_as_sync_error(monkeypatch, tmp_path):
    process = TarProcess(b"", b"tar: session log vanished", 1)
    monkeypatch.setattr(coreweave_ops.subprocess, "Popen", lambda *_args, **_kwargs: process)

    with pytest.raises(RuntimeError, match="Could not archive Ray/vLLM logs"):
        coreweave_ops.save_ray_logs(
            ["kubectl"],
            "pod",
            "task",
            [{"path": "worker-1.out", "size": 10}],
            100,
            tmp_path,
        )
    assert (tmp_path / "ray-vllm-sync-error.txt").read_text() == "tar: session log vanished"


def test_save_ray_logs_retries_transient_kubectl_html(monkeypatch, tmp_path):
    archive_bytes = BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        member = tarfile.TarInfo("worker-1.out")
        payload = b"ray log\n"
        member.size = len(payload)
        archive.addfile(member, BytesIO(payload))

    processes = iter(
        [
            TarProcess(b"", b"<!doctype html><html>temporary proxy error", 1),
            TarProcess(archive_bytes.getvalue(), b"", 0),
        ]
    )
    delays: list[int] = []
    monkeypatch.setattr(coreweave_ops.subprocess, "Popen", lambda *_args, **_kwargs: next(processes))
    monkeypatch.setattr(coreweave_ops.time, "sleep", delays.append)

    saved, skipped = coreweave_ops.save_ray_logs(
        ["kubectl"],
        "pod",
        "task",
        [{"path": "worker-1.out", "size": 10}],
        100,
        tmp_path,
    )

    assert saved == [{"path": "worker-1.out", "size": 10}]
    assert skipped == []
    assert delays == [coreweave_ops.DNS_INITIAL_BACKOFF]
    assert (tmp_path / "worker-1.out").read_bytes() == b"ray log\n"


def _ray_delta_archive(entries: list[tuple[str, int, bytes]]) -> bytes:
    archive_bytes = BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        for name, offset, payload in entries:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            member.pax_headers = {"otagent.offset": str(offset)}
            archive.addfile(member, BytesIO(payload))
    return archive_bytes.getvalue()


def test_save_ray_logs_incrementally_appends_and_replaces_rotated_logs(monkeypatch, tmp_path):
    commands: list[list[str]] = []
    inputs: list[CapturingInput] = []
    archives = iter(
        [
            _ray_delta_archive([("worker-1.out", 0, b"abc")]),
            _ray_delta_archive([("worker-1.out", 3, b"def")]),
            _ray_delta_archive([("worker-1.out", 0, b"xy")]),
        ]
    )

    def fake_popen(command, **_kwargs):
        commands.append(command)
        process = TarProcess(next(archives), capture_input=True)
        assert process.stdin is not None
        inputs.append(process.stdin)
        return process

    monkeypatch.setattr(coreweave_ops.subprocess, "Popen", fake_popen)
    initial = [{"path": "worker-1.out", "size": 3, "inode": 41}]
    saved, skipped = coreweave_ops.save_ray_logs(
        ["kubectl"],
        "pod",
        "task",
        initial,
        100,
        tmp_path,
        incremental=True,
        python_executable="python",
    )
    assert (saved, skipped) == (initial, [])
    assert (tmp_path / "worker-1.out").read_bytes() == b"abc"

    grown = [{"path": "worker-1.out", "size": 6, "inode": 41}]
    coreweave_ops.save_ray_logs(
        ["kubectl"],
        "pod",
        "task",
        grown,
        100,
        tmp_path,
        incremental=True,
        python_executable="python",
    )
    assert (tmp_path / "worker-1.out").read_bytes() == b"abcdef"
    assert b'"offset": 3' in inputs[1].getvalue()
    assert getattr(inputs[1], "closed_by_sync", False)

    # An unchanged file does not open another kubectl exec stream.
    coreweave_ops.save_ray_logs(
        ["kubectl"],
        "pod",
        "task",
        grown,
        100,
        tmp_path,
        incremental=True,
        python_executable="python",
    )
    assert len(commands) == 2

    rotated = [{"path": "worker-1.out", "size": 2, "inode": 99}]
    coreweave_ops.save_ray_logs(
        ["kubectl"],
        "pod",
        "task",
        rotated,
        100,
        tmp_path,
        incremental=True,
        python_executable="python",
    )
    assert (tmp_path / "worker-1.out").read_bytes() == b"xy"
    assert b'"offset": 0' in inputs[2].getvalue()


def test_save_ray_logs_retries_only_unfinished_files_after_partial_stream(monkeypatch, tmp_path):
    inputs: list[CapturingInput] = []
    archives = iter(
        [
            TarProcess(_ray_delta_archive([("a.log", 0, b"abc"), ("b.log", 0, b"xyz")]), capture_input=True),
            TarProcess(
                _ray_delta_archive([("a.log", 3, b"def")]),
                b"stream error",
                1,
                capture_input=True,
            ),
            TarProcess(_ray_delta_archive([("b.log", 3, b"uvw")]), capture_input=True),
        ]
    )

    def fake_popen(_command, **_kwargs):
        process = next(archives)
        assert process.stdin is not None
        inputs.append(process.stdin)
        return process

    monkeypatch.setattr(coreweave_ops.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(coreweave_ops.time, "sleep", lambda _delay: None)
    initial = [
        {"path": "a.log", "size": 3, "inode": 41},
        {"path": "b.log", "size": 3, "inode": 42},
    ]
    grown = [
        {"path": "a.log", "size": 6, "inode": 41},
        {"path": "b.log", "size": 6, "inode": 42},
    ]
    coreweave_ops.save_ray_logs(
        ["kubectl"], "pod", "task", initial, 100, tmp_path, incremental=True, python_executable="python"
    )
    coreweave_ops.save_ray_logs(
        ["kubectl"], "pod", "task", grown, 100, tmp_path, incremental=True, python_executable="python"
    )

    assert (tmp_path / "a.log").read_bytes() == b"abcdef"
    assert (tmp_path / "b.log").read_bytes() == b"xyzuvw"
    assert json.loads(inputs[2].getvalue()) == [{"path": "b.log", "size": 6, "inode": 42, "offset": 3}]


def test_save_ray_logs_preserves_completed_files_after_local_append_conflict(monkeypatch, tmp_path):
    inputs: list[CapturingInput] = []
    processes: list[TarProcess] = []
    archives = iter(
        [
            _ray_delta_archive([("a.log", 0, b"abc"), ("b.log", 0, b"xyz")]),
            _ray_delta_archive([("a.log", 3, b"def"), ("b.log", 3, b"uvw")]),
            _ray_delta_archive([("b.log", 0, b"xyzuvw")]),
        ]
    )

    def fake_popen(_command, **_kwargs):
        process = TarProcess(next(archives), capture_input=True)
        assert process.stdin is not None
        inputs.append(process.stdin)
        processes.append(process)
        return process

    monkeypatch.setattr(coreweave_ops.subprocess, "Popen", fake_popen)
    initial = [
        {"path": "a.log", "size": 3, "inode": 41},
        {"path": "b.log", "size": 3, "inode": 42},
    ]
    grown = [
        {"path": "a.log", "size": 6, "inode": 41},
        {"path": "b.log", "size": 6, "inode": 42},
    ]
    coreweave_ops.save_ray_logs(
        ["kubectl"], "pod", "task", initial, 100, tmp_path, incremental=True, python_executable="python"
    )
    (tmp_path / "b.log").write_bytes(b"local edit")
    with pytest.raises(RuntimeError, match="Local Ray/vLLM log changed before append"):
        coreweave_ops.save_ray_logs(
            ["kubectl"], "pod", "task", grown, 100, tmp_path, incremental=True, python_executable="python"
        )
    coreweave_ops.save_ray_logs(
        ["kubectl"], "pod", "task", grown, 100, tmp_path, incremental=True, python_executable="python"
    )

    assert (tmp_path / "a.log").read_bytes() == b"abcdef"
    assert (tmp_path / "b.log").read_bytes() == b"xyzuvw"
    assert json.loads(inputs[2].getvalue()) == [{"path": "b.log", "size": 6, "inode": 42, "offset": 0}]
    assert processes[1].killed


@pytest.mark.parametrize(
    ("stderr", "arguments"),
    [
        ("<!doctype html><html>temporary proxy error", ["kubectl", "get", "pods"]),
        ("command terminated with exit code 1", ["kubectl", "exec", "pod", "--", "true"]),
    ],
)
def test_coreweave_command_retries_transient_transport_failures(monkeypatch, stderr, arguments):
    results = iter(
        [
            subprocess.CompletedProcess([], 1, stderr=stderr),
            subprocess.CompletedProcess([], 0, stdout="ok\n"),
        ]
    )
    delays: list[int] = []
    monkeypatch.setattr(coreweave_ops.subprocess, "run", lambda *_args, **_kwargs: next(results))
    monkeypatch.setattr(coreweave_ops.time, "sleep", delays.append)

    assert coreweave_ops.command(arguments) == "ok\n"
    assert delays == [coreweave_ops.DNS_INITIAL_BACKOFF]


def test_complete_ray_log_sync_inventories_logs_with_the_frozen_task_python(monkeypatch, tmp_path):
    frozen_python = "/app/.venv/bin/python"
    inventory_interpreters: list[str] = []

    def fake_kubectl(arguments, **_kwargs):
        exec_argv = arguments[arguments.index("--") + 1 :]
        if exec_argv[0] == "sh":
            return f"{frozen_python}\n"
        inventory_interpreters.append(exec_argv[0])
        return "[]"

    monkeypatch.setattr(coreweave_ops, "command", fake_kubectl)

    assert watch_coreweave_rl.fetch_complete_ray_logs(["kubectl"], "pod", tmp_path) == 0
    assert inventory_interpreters == [frozen_python]


def test_rl_sync_warning_never_renders_proxy_html():
    warning = watch_coreweave_rl.sync_warning(
        ("pod Ray/vLLM: Could not archive logs (<!doctype html><html>proxy body)",)
    )

    assert warning == "Ray/vLLM log sync unavailable; local diagnostic saved"


def test_rl_sync_warning_preserves_non_ray_error_summary():
    assert watch_coreweave_rl.sync_warning(("trace sync failed",)) == "trace sync failed"


def _rl_report_row(tmp_path, finelog: str = "", state: str = "running", artifacts=None):
    if finelog:
        (tmp_path / "finelog.log").write_text(finelog)
    cluster = watch_coreweave_rl.Cluster("cw-rno2a", Path("/tmp/kubeconfig"), None)
    job = watch_coreweave_rl.IrisRlJob(cluster, "/user/rl-job", state, 0, "", dataset="DCAgent/tasks")
    if artifacts is None:
        artifacts = watch_coreweave_rl.ArtifactResult("synced", "synced", "synced", "synced", None, None, ())
    return watch_coreweave_rl.report_row(job, artifacts, tmp_path)


def test_rl_status_only_refreshes_current_state_from_log_tail(monkeypatch, tmp_path):
    cluster = watch_coreweave_rl.Cluster("cw-rno2a", Path("/tmp/kubeconfig"), None)
    job = watch_coreweave_rl.IrisRlJob(cluster, "/user/rl-job", "running", 0, "", dataset="DCAgent/tasks")
    calls: list[list[str]] = []

    def fake_run_iris(_cluster, arguments, **_kwargs):
        calls.append(arguments)
        step, reward = (16, 0.91) if "--tail" in arguments else (4, 0.8003)
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=(
                f"Training Step Progress: {step} / 80\n"
                f'WANDB_MIRROR kind=train step={step} metrics={{"reward/avg_raw_reward": {reward}}}\n'
            ),
            stderr="",
        )

    monkeypatch.setattr(watch_coreweave_rl, "run_iris", fake_run_iris)

    artifacts, directory = watch_coreweave_rl.sync_iris_job(job, tmp_path, scope="status")
    row = watch_coreweave_rl.report_row(job, artifacts, directory)

    assert len(calls) == 1
    assert row[3] == "16/80"
    assert row[4] == "0.91"


def test_rl_report_row_uses_fresh_pod_metrics_when_finelog_omits_step(tmp_path):
    pod_dir = tmp_path / "pod_logs"
    pod_dir.mkdir()
    pod_log = pod_dir / "current-pod.log"
    pod_log.write_text(
        "Training Batches Processed: 3/128\n"
        'WANDB_MIRROR kind=train step=1 metrics={"reward/avg_raw_reward": 0.7712, '
        '"policy/policy_loss": 1.05}\n'
    )
    (tmp_path / "finelog.log").write_text("Training Batches Processed: 3/128\n")
    artifacts = watch_coreweave_rl.ArtifactResult(
        "synced", "synced", "synced", "synced", None, None, (), pod_log_files=(pod_log,)
    )

    row = _rl_report_row(tmp_path, artifacts=artifacts)

    assert row[3] == "1/128"
    assert row[4] == "0.7712"
    assert row[5] == "1.05"


def test_rl_report_row_uses_newer_finelog_step_over_pod_metrics(tmp_path):
    pod_log = tmp_path / "pod_logs" / "current-pod.log"
    pod_log.parent.mkdir()
    pod_log.write_text('WANDB_MIRROR kind=train step=1 metrics={"reward/avg_raw_reward": 0.7}\n')
    artifacts = watch_coreweave_rl.ArtifactResult(
        "synced", "synced", "synced", "synced", None, None, (), pod_log_files=(pod_log,)
    )

    row = _rl_report_row(
        tmp_path,
        'Training Step Progress: 2 / 128\nWANDB_MIRROR kind=train step=2 metrics={"reward/avg_raw_reward": 0.8}\n',
        artifacts=artifacts,
    )

    assert row[3] == "2/128"
    assert row[4] == "0.8"


def test_rl_status_only_does_not_reuse_old_pod_metrics(tmp_path):
    pod_log = tmp_path / "pod_logs" / "old-pod.log"
    pod_log.parent.mkdir()
    pod_log.write_text('WANDB_MIRROR kind=train step=7 metrics={"reward/avg_raw_reward": 0.9}\n')

    row = _rl_report_row(tmp_path, "Training Step Progress: 1 / 128\n")

    assert row[3] == "1/128"
    assert row[4] == "—"


def test_rl_report_row_keeps_artifact_exceptions_out_of_trend(tmp_path):
    artifacts = watch_coreweave_rl.ArtifactResult(
        "unavailable",
        "unavailable",
        "unavailable",
        "unavailable",
        None,
        None,
        ("Ray/vLLM: raw proxy exception body",),
    )

    row = _rl_report_row(tmp_path, artifacts=artifacts)

    assert "raw proxy exception body" not in repr(row)
    assert row[2].value == "running"


def test_rl_report_row_replaces_traceback_signal_with_error_report_pointer(tmp_path):
    row = _rl_report_row(tmp_path, "Traceback (most recent call last)\nraw details\n", state="failed")

    assert "Traceback" not in repr(row)
    assert row[-1].value == "workload error detected; see error report"


def test_workload_signal_ignores_warning_traceback(tmp_path):
    finelog = tmp_path / "finelog.log"
    finelog.write_text(
        "[17:01:44] task=/user/job/0 | (Worker_TP0_EP0) WARNING 08-11 17:01:44 "
        "[import_utils.py:408] Traceback (most recent call last):\n"
        "[17:01:44] task=/user/job/0 | (Worker_TP0_EP0) WARNING 08-11 17:01:44 "
        "[import_utils.py:408] AssertionError\n"
    )

    assert watch_coreweave_rl.terminal_signal(finelog) is None


def test_workload_signal_reports_driver_exception_with_context(tmp_path):
    finelog = tmp_path / "finelog.log"
    line = (
        "[15:09:16] task=/user/job/0 | (skyrl_entrypoint pid=14143) "
        "ray.exceptions.RayTaskError(AssertionError): ray::MegatronPolicyWorkerBase.ppo_train()"
    )
    finelog.write_text(f"{line}\n")

    assert watch_coreweave_rl.terminal_signal(finelog) == line


def test_workload_signal_ignores_prior_attempt(tmp_path):
    finelog = tmp_path / "finelog.log"
    finelog.write_text(
        "[15:09:16] task=/user/job/0 | ray.exceptions.RayTaskError(AssertionError)\n"
        "[17:06:08] task=/user/job/0 | [iris setup] step 1/2\n"
        "[17:07:00] task=/user/job/0 | healthy startup\n"
    )

    assert watch_coreweave_rl.terminal_signal(finelog) is None


def test_rl_main_degrades_unexpected_job_sync_failure_into_error_report(monkeypatch, tmp_path, capsys):
    cluster = watch_coreweave_rl.Cluster("cw-rno2a", Path("/tmp/kubeconfig"), None)
    job = watch_coreweave_rl.IrisRlJob(
        cluster,
        "/user/rl-job",
        "running",
        1,
        "",
        dataset="DCAgent/tasks",
    )
    monkeypatch.setattr(
        watch_coreweave_rl,
        "parse_args",
        lambda: SimpleNamespace(
            max_non_log_bytes=0,
            trace_sync_limit=0,
            hours=24.0,
            all_users=False,
            user="user",
            filter=[],
            bundle_root=tmp_path,
            quiet_progress=True,
            status_only=True,
            jupiter_only=False,
            jupiter_run=[],
            jupiter_host="Jupiter",
            jupiter_trace_sync_limit=20,
        ),
    )
    monkeypatch.setattr(watch_coreweave_rl, "CLUSTERS", (cluster,))
    monkeypatch.setattr(
        watch_coreweave_rl,
        "discover_rl_jobs",
        lambda *_args, **_kwargs: ([job], []),
    )
    monkeypatch.setattr(
        watch_coreweave_rl,
        "sync_iris_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(LookupError("unexpected raw sync exception")),
    )
    monkeypatch.setattr(watch_coreweave_rl, "write_job_manifest", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(watch_coreweave_rl, "fetch_user_budgets", lambda *_args, **_kwargs: [])

    assert watch_coreweave_rl.main() == 0

    stdout = capsys.readouterr().out
    assert "unexpected raw sync exception" not in stdout
    errors = (tmp_path / "reports/rl/latest-errors.md").read_text()
    assert "unexpected raw sync exception" in errors


def test_rl_discovery_skips_iris_preamble_and_parses_terminal_states(
    monkeypatch,
):
    cluster = watch_coreweave_rl.Cluster("cw-rno2a", Path("/tmp/kubeconfig"), None)
    output = """I20260722 controller tunnel ready
job_id,state,submitted_at_ms,finished_at_ms,entrypoint_json,task_state
/benjaminfeuer/rl-live,3,1000,,"task_runtime.py --train_data '[\\"live\\"]'",3
/benjaminfeuer/rl-failed,5,900,1500,"task_runtime.py --train_data '[\\"failed\\"]'",
"""

    def fake_run_iris(_cluster, arguments, **_kwargs):
        return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr="")

    monkeypatch.setattr(watch_coreweave_rl, "run_iris", fake_run_iris)

    jobs, errors = watch_coreweave_rl.discover_rl_jobs(
        cluster,
        "benjaminfeuer",
        submitted_since_ms=1200,
    )

    assert errors == []
    assert [job.short_name for job in jobs] == ["rl-live", "rl-failed"]
    assert [job.is_terminal for job in jobs] == [False, True]


def test_rl_discovery_accepts_large_entrypoint_csv_field(monkeypatch):
    cluster = watch_coreweave_rl.Cluster("cw-rno2a", Path("/tmp/kubeconfig"), None)
    entrypoint = "task_runtime.py " + "x" * 150_000 + " --train_data '[\"live\"]'"
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(("job_id", "state", "submitted_at_ms", "finished_at_ms", "entrypoint_json", "task_state"))
    writer.writerow(("/benjaminfeuer/rl-large", 3, 1000, "", entrypoint, 3))

    monkeypatch.setattr(
        watch_coreweave_rl,
        "run_iris",
        lambda _cluster, arguments, **_kwargs: subprocess.CompletedProcess(
            arguments, 0, stdout=output.getvalue(), stderr=""
        ),
    )

    jobs, errors = watch_coreweave_rl.discover_rl_jobs(cluster, "benjaminfeuer")

    assert errors == []
    assert len(jobs) == 1
    assert jobs[0].job_id == "/benjaminfeuer/rl-large"
    assert jobs[0].entrypoint == entrypoint


def test_rl_discovery_parses_active_task_state_column(monkeypatch):
    """The task_state column drives placement: same root state=3, different task state."""
    cluster = watch_coreweave_rl.Cluster("cw-rno2a", Path("/tmp/kubeconfig"), None)
    output = (
        "job_id,state,submitted_at_ms,finished_at_ms,entrypoint_json,task_state\n"
        '/benjaminfeuer/rl-placed,3,1000,,"task_runtime.py --train_data \'[\\"a\\"]\'",3\n'
        '/benjaminfeuer/rl-queued,3,900,,"task_runtime.py --train_data \'[\\"b\\"]\'",1\n'
    )

    monkeypatch.setattr(
        watch_coreweave_rl,
        "run_iris",
        lambda _cluster, arguments, **_kwargs: subprocess.CompletedProcess(arguments, 0, stdout=output, stderr=""),
    )

    jobs, _errors = watch_coreweave_rl.discover_rl_jobs(cluster, "benjaminfeuer")
    by_name = {job.short_name: job for job in jobs}

    assert by_name["rl-placed"].task_state == "running"
    assert by_name["rl-queued"].task_state == "pending"


@pytest.mark.parametrize(
    ("state", "task_state", "expected"),
    [
        ("pending", None, "awaiting placement"),
        ("building", None, "awaiting placement"),
        ("running", "pending", "awaiting placement"),
        ("running", "building", "awaiting placement"),
        ("running", "running", "running"),
        ("running", None, "running"),
        ("succeeded", None, "succeeded"),
        ("failed", None, "failed"),
    ],
)
def test_rl_effective_state_classifies_placement(state, task_state, expected):
    cluster = watch_coreweave_rl.Cluster("cw-rno2a", Path("/tmp/kubeconfig"), None)
    job = watch_coreweave_rl.IrisRlJob(
        cluster, "/user/rl-job", state, 0, "", dataset="DCAgent/tasks", task_state=task_state
    )

    assert watch_coreweave_rl.effective_state(job) == expected


def test_fetch_user_budget_parses_spent_limit_and_band(monkeypatch):
    cluster = watch_coreweave_rl.Cluster("cw-rno2a", Path("/tmp/kubeconfig"), None)
    # The CLI emits controller/tunnel preamble lines before the key:value block.
    output = (
        "I20260803 controller tunnel ready\n"
        "User:      benjaminfeuer\n"
        "Limit:     1000000\n"
        "Spent:     487818\n"
        "Max band:  interactive\n"
    )
    monkeypatch.setattr(
        watch_coreweave_rl,
        "run_iris",
        lambda _cluster, arguments, **_kwargs: subprocess.CompletedProcess(arguments, 0, stdout=output, stderr=""),
    )

    budget = watch_coreweave_rl.fetch_user_budget(cluster, "benjaminfeuer")

    assert budget == watch_coreweave_rl.UserBudget(
        "benjaminfeuer", "cw-rno2a", spent=487818, limit=1000000, max_band="interactive"
    )


def test_fetch_user_budget_reports_no_budget_set_when_unset(monkeypatch):
    cluster = watch_coreweave_rl.Cluster("cw-us-east-02a", Path("/tmp/kubeconfig"), None)
    monkeypatch.setattr(
        watch_coreweave_rl,
        "run_iris",
        lambda _cluster, arguments, **_kwargs: subprocess.CompletedProcess(
            arguments,
            1,
            stdout="",
            stderr="Traceback ...\nconnectrpc.errors.ConnectError: No budget found for user benjaminfeuer\n",
        ),
    )

    budget = watch_coreweave_rl.fetch_user_budget(cluster, "benjaminfeuer")

    assert budget.spent is None and budget.limit is None
    assert budget.note == "no budget set"


def _remote_trace_objects():
    root = "iris/rl/trace_jobs/"
    first = datetime(2026, 7, 22, 10, tzinfo=UTC)
    objects = [
        {"Key": f"{root}z-old/result.json", "Size": 1, "LastModified": first},
        {
            "Key": f"{root}a-new/agent/trajectory.json",
            "Size": 1,
            "LastModified": first + timedelta(hours=3),
        },
        {
            "Key": f"{root}a-new/result.json",
            "Size": 1,
            "LastModified": first + timedelta(hours=2),
        },
        {
            "Key": f"{root}m-middle/result.json",
            "Size": 1,
            "LastModified": first + timedelta(hours=2),
        },
    ]
    return root, first, objects


@pytest.mark.parametrize(
    "entrypoint",
    [
        json.dumps(
            {
                "command": [
                    "python",
                    "cloud.iris.training_driver",
                    "++terminal_bench_config.trials_dir='s3://bucket/tmp/ttl=14d/run/trace_jobs'",
                ]
            }
        ),
        json.dumps(
            {
                "command": [
                    "python",
                    "cloud.iris.training_driver",
                    "--trials-dir",
                    "s3://bucket/tmp/ttl=14d/run/trace_jobs",
                ]
            }
        ),
    ],
)
def test_iris_trials_uri_uses_submitted_trials_directory(entrypoint):
    cluster = watch_coreweave_rl.Cluster("cw-rno2a", Path("/tmp/kubeconfig"), None)
    job = watch_coreweave_rl.IrisRlJob(cluster, "/user/run", "running", 0, entrypoint)

    assert job.trials_uri == "s3://bucket/tmp/ttl=14d/run/trace_jobs"


def test_trace_inventory_and_manifest_use_submitted_trials_directory(monkeypatch, tmp_path):
    trials_uri = "s3://bucket/tmp/ttl=14d/run/trace_jobs"
    entrypoint = json.dumps(
        {
            "command": [
                "python",
                "cloud.iris.training_driver",
                f"++terminal_bench_config.trials_dir='{trials_uri}'",
            ]
        }
    )
    cluster = watch_coreweave_rl.Cluster("cw-rno2a", Path("/tmp/kubeconfig"), None)
    job = watch_coreweave_rl.IrisRlJob(cluster, "/user/run", "running", 0, entrypoint)
    observed_prefixes = []
    modified = datetime(2026, 8, 17, 8, tzinfo=UTC)

    monkeypatch.setattr(watch_coreweave_rl, "kubectl_base", lambda *_args: [])
    monkeypatch.setattr(watch_coreweave_rl, "object_store_client", lambda *_args: object())

    def listed_objects(_client, bucket, prefix):
        observed_prefixes.append((bucket, prefix))
        return [{"Key": f"{prefix}trial-1/result.json", "Size": 1, "LastModified": modified}]

    monkeypatch.setattr(watch_coreweave_rl, "iter_objects", listed_objects)

    inventory = watch_coreweave_rl.collect_trace_inventory(job)
    directory = job_bundle(tmp_path, cluster.name, job.job_id).directory
    directory.mkdir(parents=True)
    artifacts = watch_coreweave_rl.ArtifactResult("synced", "synced", "synced", "synced", 1, 1, ())
    watch_coreweave_rl.write_job_manifest(job, tmp_path, directory, artifacts, 0, 500)
    manifest = load_bundle_manifest(job_bundle(tmp_path, cluster.name, job.job_id))

    assert observed_prefixes == [("bucket", "tmp/ttl=14d/run/trace_jobs/")]
    assert inventory.available == 1
    assert manifest["trials_uri"] == trials_uri


def test_recent_trace_jobs_uses_remote_last_modified_and_preserves_remote_counts():
    root, _first, objects = _remote_trace_objects()

    selected, available, completed = watch_coreweave_rl.recent_trace_jobs(objects, root, trace_sync_limit=2)

    assert [trace.name for trace in selected] == ["a-new", "m-middle"]
    assert available == 3
    assert completed == 3
    full, _, _ = watch_coreweave_rl.recent_trace_jobs(objects, root, trace_sync_limit=0)
    assert [trace.name for trace in full] == ["a-new", "m-middle", "z-old"]


def _fleet_trace_inventories():
    root, first, objects = _remote_trace_objects()
    full, _, _ = watch_coreweave_rl.recent_trace_jobs(objects, root, trace_sync_limit=0)
    cluster = watch_coreweave_rl.Cluster("cw-rno2a", Path("/tmp/kubeconfig"), None)
    first_job = watch_coreweave_rl.IrisRlJob(cluster, "/user/first", "running", 0, "")
    second_job = watch_coreweave_rl.IrisRlJob(cluster, "/user/second", "running", 0, "")
    first_inventory = watch_coreweave_rl.TraceInventory(first_job, "bucket", root, object(), tuple(full), 3, 3)
    second_inventory = watch_coreweave_rl.TraceInventory(
        second_job,
        "bucket",
        root,
        object(),
        (watch_coreweave_rl.TraceJobObjects("other", first + timedelta(hours=4), (), True),),
        1,
        1,
    )
    return first_inventory, second_inventory


def test_fleet_trace_selection_applies_one_global_recency_limit():
    first_inventory, second_inventory = _fleet_trace_inventories()

    fleet_selected = watch_coreweave_rl.select_recent_fleet_traces(
        [first_inventory, second_inventory], trace_sync_limit=2
    )

    assert [trace.name for trace in fleet_selected[("cw-rno2a", "/user/first")]] == ["a-new"]
    assert [trace.name for trace in fleet_selected[("cw-rno2a", "/user/second")]] == ["other"]
    full_fleet = watch_coreweave_rl.select_recent_fleet_traces([first_inventory, second_inventory], trace_sync_limit=0)
    assert sum(len(traces) for traces in full_fleet.values()) == 4


def test_trace_selection_manifest_reports_job_and_fleet_omissions():
    first_inventory, second_inventory = _fleet_trace_inventories()
    fleet_selected = watch_coreweave_rl.select_recent_fleet_traces(
        [first_inventory, second_inventory], trace_sync_limit=2
    )
    manifest = watch_coreweave_rl.trace_selection_manifest(
        fleet_selected[("cw-rno2a", "/user/first")],
        first_inventory,
        2,
        fleet_available=4,
        fleet_selected=2,
    )
    assert manifest["selection"] == "latest_object_store_last_modified_across_active_rl_jobs"
    assert manifest["omitted_traces"] == 2
    assert manifest["fleet_selected_traces"] == 2


def test_recent_trace_jobs_requires_remote_last_modified_metadata():
    with pytest.raises(ValueError, match="LastModified"):
        watch_coreweave_rl.recent_trace_jobs(
            [{"Key": "iris/rl/trace_jobs/trace/result.json", "Size": 1}],
            "iris/rl/trace_jobs/",
            trace_sync_limit=500,
        )


def test_trace_sync_skips_object_that_disappears_after_listing(tmp_path):
    class MissingObjectClient:
        def download_file(self, _bucket, _key, _destination):
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")

    cluster = watch_coreweave_rl.Cluster("cw-rno2a", Path("/tmp/kubeconfig"), None)
    job = watch_coreweave_rl.RlJob(cluster, "/user/rl-job", "running", 0, "")
    root = "iris/rl-job/trace_jobs/"
    trace = watch_coreweave_rl.TraceJobObjects(
        "trial-1",
        datetime(2026, 7, 22, 10, tzinfo=UTC),
        (
            {
                "Key": f"{root}trial-1/result.json",
                "Size": 1,
                "LastModified": datetime(2026, 7, 22, 10, tzinfo=UTC),
            },
        ),
        True,
    )
    inventory = watch_coreweave_rl.TraceInventory(
        job, "bucket", root, MissingObjectClient(), (trace,), available=1, completed=1
    )

    status, available, completed, error = watch_coreweave_rl.sync_trace_inventory(
        inventory,
        tmp_path / "trace_jobs",
        [trace],
        max_non_log_bytes=0,
        trace_sync_limit=500,
        fleet_available=1,
        fleet_selected=1,
    )

    assert status.endswith("0 copied, 1 skipped")
    assert (available, completed, error) == (1, 1, None)
    assert json.loads((tmp_path / "trace_jobs" / "skipped_objects.json").read_text()) == [
        {
            "key": "trial-1/result.json",
            "reason": "missing_after_listing",
            "size": 1,
        }
    ]


def test_rl_report_row_reads_driver_mismatch_and_correction_metrics(tmp_path):
    row = _rl_report_row(
        tmp_path,
        "Training Step Progress: 7 / 80\n"
        'WANDB_MIRROR kind=train step=7 metrics={"policy/policy_entropy": 0.05, '
        '"generate/tis/exact_match_fraction": 0.975, "policy/log_ratio_abs_mean": 0.021, "policy/log_ratio_abs_p99": 0.44, "policy/log_ratio_abs_max": 3.25, "policy/mismatch/pooled/log_ratio_abs_mean": 0.125, "policy/correction/weight_mean": 0.75, "policy/correction/truncated_fraction": 0.1, "policy/correction/masked_fraction": 0.2}\n',
    )

    assert "entropy=0.05" in row[-1].value
    assert "mismatch |log r|=0.125" in row[-1].value
    assert "correction weight=0.75" in row[-1].value
    assert "correction truncated=0.1" in row[-1].value
    assert "correction masked=0.2" in row[-1].value
    assert "TIS exact=0.975" in row[-1].value
    assert "0.021/0.44/3.25" in row[-1].value


def test_rl_report_row_flags_configured_correction_without_diagnostics(tmp_path):
    row = _rl_report_row(
        tmp_path,
        "trainer:\n"
        "  algorithm:\n"
        "    off_policy_correction: tis\n"
        "Training Step Progress: 2 / 80\n"
        'WANDB_MIRROR kind=train step=2 metrics={"policy/policy_entropy": 0.7143}\n',
    )

    assert "correction tis; metrics missing" in row[-1].value


def test_rl_report_row_explains_correction_disabled_by_resolved_config(tmp_path):
    row = _rl_report_row(
        tmp_path,
        "trainer:\n"
        "  algorithm:\n"
        "    off_policy_correction: none\n"
        "Training Step Progress: 2 / 80\n"
        'WANDB_MIRROR kind=train step=2 metrics={"policy/policy_entropy": 0.7143, '
        '"policy/log_ratio_abs_mean": 0.0}\n',
    )

    assert "correction disabled" in row[-1].value
    assert "TIS exact=—" not in row[-1].value
    assert "TIS r=—" not in row[-1].value
