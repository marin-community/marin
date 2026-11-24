# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from threading import Event

import pytest
from marin.rl.native_shutdown import artifact_handoff as handoff
from marin.rl.native_shutdown.stack_safety import identity as process_identity
from marin.rl.native_shutdown.stack_safety import proc_status
from omegaconf import OmegaConf

requires_linux_process_api = pytest.mark.skipif(
    sys.platform != "linux",
    reason="requires_linux_process_api: /proc identity and pidfd readiness",
)


class RecordingStore:
    def __init__(self):
        self.objects = {}
        self.order = []

    def put(self, local, remote):
        self.objects[remote] = Path(local).read_bytes()
        self.order.append(remote)

    @contextlib.contextmanager
    def open(self, path, mode):
        assert mode == "wb"
        stream = io.BytesIO()
        yield stream
        self.objects[path] = stream.getvalue()
        self.order.append(path)


@pytest.fixture
def request_files(tmp_path):
    root = tmp_path.resolve()
    nonce = "a" * 32
    directory = root / "native-shutdown-observer" / f"rank0-pid123-{nonce}"
    directory.mkdir(parents=True, mode=0o700)
    config = {"pid": 123, "starttime": 456, "pgrp": 123, "nonce": nonce, "owner_pid": os.getpid(), "owner_starttime": 1}
    if sys.platform == "linux":
        config["owner_starttime"] = process_identity(os.getpid()).starttime
    for key, name in [
        ("ready_path", "tracer-ready.json"),
        ("ack_path", "ack.json"),
        ("trace_path", "stack.txt"),
        ("lldb_receipt_path", "lldb-receipt.json"),
        ("guardian_receipt_path", "guardian-receipt.json"),
        ("debugger_log_path", "debugger.log"),
        ("cancel_path", "cancel"),
    ]:
        config[key] = str(directory / name)
    identity = {key: config[key] for key in ("pid", "starttime", "nonce")}
    lldb = {"bound_pid": config["pid"], "status": "collected", "errors": [], "native_stack_qualified": True}
    guardian = {**identity, "safe_cleanup": True, "lldb": lldb}
    values = {
        "observer-config.json": config,
        "tracer-ready.json": dict(identity, running=True),
        "ack.json": dict(identity, event="shutdown_ack_written", process_id=0, ack_path=config["ack_path"]),
        "lldb-receipt.json": lldb,
        "guardian-receipt.json": guardian,
    }
    for name, value in values.items():
        (directory / name).write_text(json.dumps(value))
    for name in ("stack.txt", "debugger.log", "guardian.stderr"):
        (directory / name).write_bytes(b"diagnostic text\n")
    task = "/owner/current-run/4:0"
    uid = "physical-attempt"
    handoff.write_route(root, "s3://owned/current/attempts/rendezvous", f"rank4-{socket.gethostname()}", task, uid)
    return {
        "directory": str(directory),
        "config": config,
        "process_id": 0,
        "retired_safe": True,
        "guardian_receipt": guardian,
        "observation_completed": True,
        "cancel": False,
        "retirement_error": None,
        "task_id": task,
        "attempt_uid": uid,
        "hostname": socket.gethostname(),
    }


def test_selected_files_are_hashed_and_committed_last_to_physical_task_route(request_files):
    directory = Path(request_files["directory"])
    (directory / "unrelated-secret.txt").write_text("must not upload")
    store = RecordingStore()
    result = handoff.upload_selected(request_files, lambda uri: (store, "owned/prefix"))
    assert result["status"] == "uploaded" and result["observation_completed"] is True
    assert result["destination"].endswith(
        f"/debug_artifacts/rank4-{socket.gethostname()}/native-shutdown-observer/{directory.name}"
    )
    assert {row["name"] for row in result["files"]} == set(handoff.FILE_LIMITS)
    assert store.order[-1] == "owned/prefix/handoff-commit.json"
    assert len(store.order) == 9 and not any("secret" in name for name in store.objects)
    assert json.loads(store.objects[store.order[-1]]) == result
    assert result["bytes"] + len(store.objects[store.order[-1]]) <= handoff.MAX_TOTAL_BYTES


@pytest.mark.parametrize("bad_file", ["symlink", "hardlink", "fifo", "oversize", "wrong_nonce"])
def test_unsafe_selected_file_is_not_uploaded_or_promoted_to_complete(request_files, tmp_path, bad_file):
    directory = Path(request_files["directory"])
    name = "ack.json" if bad_file == "wrong_nonce" else "stack.txt"
    path = directory / name
    path.unlink()
    if bad_file == "symlink":
        secret = tmp_path / "secret"
        secret.write_text("private payload")
        path.symlink_to(secret)
    elif bad_file == "hardlink":
        source = tmp_path / "hardlinked"
        source.write_text("private payload")
        os.link(source, path)
    elif bad_file == "fifo":
        os.mkfifo(path)
    elif bad_file == "oversize":
        with path.open("wb") as stream:
            stream.truncate(handoff.FILE_LIMITS[name] + 1)
    else:
        path.write_text(json.dumps({"pid": 123, "starttime": 456, "nonce": "b" * 32}))
    store = RecordingStore()
    result = handoff.upload_selected(request_files, lambda uri: (store, "owned/prefix"))
    assert result["status"] == "partial" and result["errors"]
    assert name not in {row["name"] for row in result["files"]}
    assert "owned/prefix/" + name not in store.objects
    assert store.order[-1].endswith("handoff-commit.json")


def test_failed_upload_is_explicit_and_keeps_original_local_operands(request_files):
    class FailedStore(RecordingStore):
        def put(self, local, remote):
            if remote.endswith("stack.txt"):
                raise OSError("storage unavailable")
            super().put(local, remote)

    path = Path(request_files["directory"]) / "stack.txt"
    original = path.read_bytes()
    store = FailedStore()
    result = handoff.upload_selected(request_files, lambda uri: (store, "owned/prefix"))
    assert result["status"] == "partial"
    assert {"file": "stack.txt", "type": "OSError"} in result["errors"]
    assert path.read_bytes() == original


def test_commit_failure_cannot_report_success(request_files):
    class FailedCommit(RecordingStore):
        @contextlib.contextmanager
        def open(self, path, mode):
            yield io.BytesIO()
            raise OSError("commit unavailable")

    store = FailedCommit()
    with pytest.raises(OSError, match="commit unavailable"):
        handoff.upload_selected(request_files, lambda uri: (store, "owned/prefix"))
    assert len(store.objects) == 8 and "owned/prefix/handoff-commit.json" not in store.objects


def test_one_MiB_trace_is_retained_exactly_within_total_quota(request_files):
    content = b"x" * (1024 * 1024)
    (Path(request_files["directory"]) / "stack.txt").write_bytes(content)
    store = RecordingStore()
    result = handoff.upload_selected(request_files, lambda uri: (store, "owned/prefix"))
    assert result["status"] == "uploaded" and store.objects["owned/prefix/stack.txt"] == content
    assert result["bytes"] + len(store.objects["owned/prefix/handoff-commit.json"]) <= handoff.MAX_TOTAL_BYTES


def test_file_mutated_during_snapshot_is_rejected_before_upload(request_files, monkeypatch):
    path = Path(request_files["directory"]) / "stack.txt"
    inode = path.stat().st_ino
    fstat = os.fstat
    reads = 0

    def mutate_on_second_stat(descriptor):
        nonlocal reads
        row = fstat(descriptor)
        if row.st_ino == inode:
            reads += 1
            if reads == 2:
                path.write_bytes(b"changed after read")
                row = fstat(descriptor)
        return row

    monkeypatch.setattr(handoff.os, "fstat", mutate_on_second_stat)
    store = RecordingStore()
    result = handoff.upload_selected(request_files, lambda uri: (store, "owned/prefix"))
    assert result["status"] == "partial" and path.read_bytes() == b"changed after read"
    assert "owned/prefix/stack.txt" not in store.objects


def test_symlinked_attempt_directory_prevents_any_remote_write(request_files):
    directory = Path(request_files["directory"])
    moved = directory.with_name("moved-original")
    directory.rename(moved)
    directory.symlink_to(moved, target_is_directory=True)
    store = RecordingStore()
    with pytest.raises(ValueError, match="canonical"):
        handoff.upload_selected(request_files, lambda uri: (store, "owned/prefix"))
    assert store.objects == {}


def test_unsafe_retirement_uploads_only_bound_failure_summary(request_files):
    request_files.update(
        retired_safe=False, guardian_receipt=None, retirement_error="UnsafeCleanup", observation_completed=False
    )
    store = RecordingStore()
    result = handoff.upload_selected(request_files, lambda uri: (store, "owned/prefix"))
    assert result["files"] == [] and result["status"] == "partial" and result["retired_safe"] is False
    assert store.order == ["owned/prefix/handoff-commit.json"]
    assert result["errors"] == [{"stage": "retirement", "type": "UnsafeCleanup"}]


@pytest.mark.parametrize("change", ["attempt", "task", "directory"])
def test_wrong_route_or_directory_identity_prevents_any_remote_write(request_files, change):
    if change == "attempt":
        request_files["attempt_uid"] = "other-attempt"
    elif change == "task":
        request_files["task_id"] = "/owner/current-run/0:0"
    else:
        request_files["config"]["nonce"] = "c" * 32
    store = RecordingStore()
    with pytest.raises(ValueError, match="identity differs"):
        handoff.upload_selected(request_files, lambda uri: (store, "owned/prefix"))
    assert store.objects == {}


@requires_linux_process_api
def test_hung_uploader_and_descendant_are_killed_without_late_writes(request_files, monkeypatch, tmp_path):
    pid_file = tmp_path / "uploader.pid"
    late_file = tmp_path / "late-write"
    popen = subprocess.Popen
    children = []

    def blocked_upload(command, **kwargs):
        # Replace only the external uploader executable, retaining real process
        # ownership, pidfd timeout, group kill and reap behavior.
        code = (
            f"import os,time,pathlib; pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); "
            f"child=os.fork(); time.sleep(1); pathlib.Path({str(late_file)!r}).write_text('late'); time.sleep(10)"
        )
        process = popen([sys.executable, "-c", code], **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(handoff.subprocess, "Popen", blocked_upload)
    monkeypatch.setattr(handoff, "UPLOAD_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setenv("IRIS_TASK_ID", request_files["task_id"])
    monkeypatch.setenv("IRIS_ATTEMPT_UID", request_files["attempt_uid"])
    started = time.monotonic()
    result = handoff.handoff_artifacts(
        Path(request_files["directory"]),
        request_files["config"],
        0,
        retired_safe=True,
        guardian_receipt=request_files["guardian_receipt"],
        observation_completed=True,
        cancel=False,
        retirement_error=None,
    )
    assert time.monotonic() - started < 2 and result["status"] == "timed_out" and result["uploader_reaped"]
    assert pid_file.exists() and children[0].returncode < 0
    Event().wait(1.1)  # A surviving uploader descendant would now create late_file.
    assert not late_file.exists()
    assert (
        json.loads((Path(request_files["directory"]) / "handoff-parent.json").read_text())["remote_commit"] == "unknown"
    )


@requires_linux_process_api
def test_real_uploader_missing_route_writes_failure_without_storage_import(request_files, monkeypatch):
    directory = Path(request_files["directory"])
    (directory.parent.parent / handoff.ROUTE_NAME).unlink()
    monkeypatch.setenv("IRIS_TASK_ID", request_files["task_id"])
    monkeypatch.setenv("IRIS_ATTEMPT_UID", request_files["attempt_uid"])
    result = handoff.handoff_artifacts(
        directory,
        request_files["config"],
        0,
        retired_safe=True,
        guardian_receipt=request_files["guardian_receipt"],
        observation_completed=True,
        cancel=False,
        retirement_error=None,
    )
    assert result["status"] == "failed" and result["uploader_reaped"] is True
    assert result["uploader"]["error_type"] == "FileNotFoundError"
    assert result["uploader_exit_code"] == 1
    assert json.loads((directory / "handoff-parent.json").read_text())["status"] == "failed"


@requires_linux_process_api
def test_exited_leader_stays_reserved_until_descendant_group_is_killed(request_files, monkeypatch, tmp_path):
    late_file = tmp_path / "late-write"
    popen = subprocess.Popen
    killpg = os.killpg
    killed_reserved_leaders = []

    def exited_upload(command, **kwargs):
        code = r"""
import json, os, pathlib, sys, time
request = json.load(sys.stdin)
child = os.fork()
if child == 0:
    time.sleep(1)
    pathlib.Path(sys.argv[1]).write_text('late')
    os._exit(0)
receipt = {key: request['config'][key] for key in ('pid','starttime','nonce')}
receipt['status'] = 'uploaded'
pathlib.Path(request['directory'], 'handoff-upload.json').write_text(json.dumps(receipt))
os._exit(0)
"""
        return popen([sys.executable, "-c", code, str(late_file)], **kwargs)

    def kill_reserved_group(pgid, sig):
        # A zombie still owns its PID/PGID. This must happen before Popen.wait
        # releases that identity, even though the leader exited successfully.
        assert proc_status(pgid)["State"].startswith("Z")
        killed_reserved_leaders.append(pgid)
        return killpg(pgid, sig)

    monkeypatch.setattr(handoff.subprocess, "Popen", exited_upload)
    monkeypatch.setattr(handoff.os, "killpg", kill_reserved_group)
    monkeypatch.setenv("IRIS_TASK_ID", request_files["task_id"])
    monkeypatch.setenv("IRIS_ATTEMPT_UID", request_files["attempt_uid"])
    result = handoff.handoff_artifacts(
        Path(request_files["directory"]),
        request_files["config"],
        0,
        retired_safe=True,
        guardian_receipt=request_files["guardian_receipt"],
        observation_completed=True,
        cancel=False,
        retirement_error=None,
    )
    assert result["status"] == "uploaded" and result["uploader_exit_code"] == 0 and result["uploader_reaped"]
    assert len(killed_reserved_leaders) == 1 and result["uploader_group_cleanup"] == "signalled"
    Event().wait(1.1)
    assert not late_file.exists()


@pytest.mark.parametrize("matching_owner", [False, True])
def test_isolated_uploader_checks_sibling_owner_without_installed_marin(request_files, matching_owner):
    directory = Path(request_files["directory"])
    (directory.parent.parent / handoff.ROUTE_NAME).unlink()
    # -S removes editable installs as well as PYTHONPATH. On Darwin only the
    # Linux OS boundary is simulated; main and sibling identity parsing are real.
    request_files["config"]["owner_starttime"] = 7654321 if matching_owner else 7654322
    harness = r"""
import ctypes, importlib.util, os, runpy, sys
from pathlib import Path
assert importlib.util.find_spec('marin') is None
parent = os.getppid()
if sys.platform != 'linux':
    class Libc:
        def prctl(self, *args): return 0
    ctypes.CDLL = lambda *args, **kwargs: Libc()
sys.platform = 'linux'
read = Path.read_text
def proc_stat(path, *args, **kwargs):
    if str(path) == f'/proc/{parent}/stat':
        print('sibling owner stat read', file=sys.stderr)
        return f'{parent} (owner) ' + ' '.join(['S', '0', str(parent)] + ['0'] * 16 + ['7654321'])
    return read(path, *args, **kwargs)
Path.read_text = proc_stat
sys.argv = [sys.argv[1], '--parent-pid', str(parent)]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", harness, handoff.__file__],
        input=json.dumps(request_files),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1 and "sibling owner stat read" in result.stderr
    assert "No module named 'marin'" not in result.stderr
    if matching_owner:
        receipt = json.loads((directory / "handoff-upload.json").read_text())
        assert receipt["status"] == "failed" and receipt["error_type"] == "FileNotFoundError"
        assert "cloud" not in result.stderr
    else:
        assert "owner starttime differs" in result.stderr
        assert not (directory / "handoff-upload.json").exists()


@pytest.mark.parametrize("retire_fails", [False, True])
@pytest.mark.parametrize("handoff_raises", [False, True])
def test_public_shutdown_preserves_primary_error_and_unsafe_ownership(
    request_files, monkeypatch, retire_fails, handoff_raises
):
    host = pytest.importorskip("skyrl_train.workers.levanter.levanter_worker")
    primary = TimeoutError("original postACK failure")
    order = []

    class Child:
        def request(self, method, payload):
            order.append("original shutdown")
            raise primary

    class Observer:
        directory = Path(request_files["directory"])
        config = request_files["config"]
        artifact_handoff_attempted = False

        def payload(self):
            return {}

        def retire(self, *, cancel):
            order.append("retirement")
            assert cancel
            if retire_fails:
                raise RuntimeError("unsafe cleanup")
            return request_files["guardian_receipt"]

    def transport(*args, **kwargs):
        order.append("handoff")
        assert kwargs["retired_safe"] is not retire_fails
        if handoff_raises:
            raise OSError("uploader failed")
        return {"status": "partial"}

    monkeypatch.setattr(host, "handoff_artifacts", transport)
    if handoff_raises:

        class BrokenLog:
            def warning(self, *args):
                raise OSError("log sink already gone")

        monkeypatch.setattr(host, "logger", BrokenLog())
    worker = host.LevanterHostWorker.__new__(host.LevanterHostWorker)
    observer = Observer()
    worker._shutdown_observer = observer
    worker.child = Child()
    worker.process_id = 0
    with pytest.raises(TimeoutError) as error:
        worker._shutdown_child()
    assert error.value is primary and order == ["original shutdown", "retirement", "handoff"]
    assert (worker._shutdown_observer is observer) is retire_fails
    assert any("artifact handoff" in note for note in primary.__notes__)
    if retire_fails:
        worker._retire_shutdown_observer(primary, cancel=True)
        assert order == ["original shutdown", "retirement", "handoff", "retirement"]


def test_uploaded_files_do_not_waive_nonqualifying_observer_result(request_files, monkeypatch):
    host = pytest.importorskip("skyrl_train.workers.levanter.levanter_worker")
    store = RecordingStore()

    class Observer:
        directory = Path(request_files["directory"])
        config = request_files["config"]
        artifact_handoff_attempted = False

        def retire(self, *, cancel):
            assert cancel is False
            return request_files["guardian_receipt"]

    def transport(*args, **kwargs):
        request_files.update(observation_completed=kwargs["observation_completed"])
        return handoff.upload_selected(request_files, lambda uri: (store, "owned/prefix"))

    monkeypatch.setattr(host, "handoff_artifacts", transport)
    worker = host.LevanterHostWorker.__new__(host.LevanterHostWorker)
    worker._shutdown_observer = Observer()
    worker.process_id = 0
    with pytest.raises(RuntimeError, match="diagnostic is nonqualifying"):
        worker._retire_shutdown_observer()
    assert worker._shutdown_observer is None
    receipt = json.loads(store.objects["owned/prefix/handoff-commit.json"])
    assert receipt["status"] == "uploaded" and receipt["observation_completed"] is False


def test_runtime_route_uses_physical_identity_and_default_off_does_nothing(tmp_path, monkeypatch):
    runtime = pytest.importorskip("cloud.iris.task_runtime")

    root = tmp_path.resolve()
    config = OmegaConf.create(
        {"skyrl": {"trainer": {"policy": {"levanter_config": {"diagnostic_shutdown_observer": False}}}}}
    )
    runtime.initialize_observer_handoff(config, "s3://owned/route")
    assert list(root.iterdir()) == []
    monkeypatch.setenv("SKYRL_DEBUG_ARTIFACT_DIR", str(root))
    monkeypatch.setenv("IRIS_TASK_ID", "/owned/job/4:0")
    monkeypatch.setenv("IRIS_ATTEMPT_UID", "physical-attempt")
    config.skyrl.trainer.policy.levanter_config.diagnostic_shutdown_observer = True
    runtime.initialize_observer_handoff(config, "s3://owned/route")
    route = json.loads((root / handoff.ROUTE_NAME).read_text())
    assert route["node_id"] == f"rank4-{socket.gethostname()}" and route["task_id"] == "/owned/job/4:0"
    runtime.initialize_observer_handoff(config, "s3://owned/route")
    with pytest.raises(ValueError, match="another task"):
        runtime.initialize_observer_handoff(config, "s3://different/route")


def test_runtime_without_native_config_keeps_existing_no_route_behavior(tmp_path):
    runtime = pytest.importorskip("cloud.iris.task_runtime")

    runtime.initialize_observer_handoff(OmegaConf.create({"skyrl": {}}), "s3://unused")
    assert list(tmp_path.iterdir()) == []
