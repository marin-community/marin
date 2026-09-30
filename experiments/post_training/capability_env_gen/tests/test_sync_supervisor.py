"""A stopped uploader cannot strand an in-flight publisher."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import sync_supervisor
from scripts.sync_supervisor import run_sync


def _wait_for(path: Path) -> int:
    for _ in range(100):
        if path.exists():
            return int(path.read_text())
        time.sleep(0.02)
    raise AssertionError(f"child did not start: {path}")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False).stdout.strip()
    return bool(state) and not state.startswith("Z")


def test_deadline_reaps_sync_process_group(tmp_path: Path) -> None:
    child_file = tmp_path / "child.pid"
    script = (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
        "open(sys.argv[1],'w').write(str(p.pid)); time.sleep(30)"
    )
    assert run_sync([sys.executable, "-c", script, str(child_file)], tmp_path / "capture", deadline_seconds=1) == 124
    child = _wait_for(child_file)
    try:
        assert not _alive(child)
    finally:
        if _alive(child):
            os.kill(child, signal.SIGKILL)


def test_termination_reaps_sync_before_supervisor_exit(tmp_path: Path) -> None:
    child_file = tmp_path / "child.pid"
    script = (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
        "open(sys.argv[1],'w').write(str(p.pid)); time.sleep(30)"
    )
    supervisor = subprocess.Popen(
        [sys.executable, "-c", ("import sys; from pathlib import Path; "
         "from scripts.sync_supervisor import run_sync; "
         "sys.exit(run_sync(sys.argv[1:-1],Path(sys.argv[-1]),deadline_seconds=60))"),
         sys.executable, "-c", script, str(child_file), str(tmp_path / "capture")],
        cwd=Path(__file__).resolve().parents[1],
    )
    child = _wait_for(child_file)
    supervisor.terminate()
    try:
        assert supervisor.wait(timeout=7) == 143
        assert not _alive(child)
    finally:
        if _alive(child):
            os.kill(child, signal.SIGKILL)


def test_exited_leader_does_not_leave_term_ignoring_grandchild(tmp_path: Path) -> None:
    child_file = tmp_path / "child.pid"
    grandchild = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"
    script = (
        "import subprocess,sys; "
        "p=subprocess.Popen([sys.executable,'-c',sys.argv[2]]); "
        "open(sys.argv[1],'w').write(str(p.pid))"
    )
    try:
        assert run_sync(
            [sys.executable, "-c", script, str(child_file), grandchild],
            tmp_path / "capture",
            deadline_seconds=5,
        ) == 0
        child = _wait_for(child_file)
        assert not _alive(child)
    finally:
        if child_file.exists():
            child = int(child_file.read_text())
            if _alive(child):
                os.kill(child, signal.SIGKILL)


def test_project_sync_rechecks_frozen_environment(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "results"
    source.mkdir()
    project = tmp_path / "marin"
    project.mkdir()
    commands = []

    def capture_sync(command, capture, *, deadline_seconds, cwd=None):
        commands.append(command)
        capture.write_text("ok")
        return 0

    monkeypatch.setattr(sync_supervisor, "run_sync", capture_sync)
    monkeypatch.setattr(sys, "argv", [
        "sync_supervisor.py", "--source", str(source),
        "--destination", "s3://bucket/run", "--state", str(tmp_path / "state"),
        "--project", str(project), "--final",
    ])
    assert sync_supervisor.main() == 0
    assert commands[0][:6] == [
        "uv", "run", "--project", str(project), "--frozen", "python3",
    ]


def _run_main(monkeypatch, argv) -> tuple[int, list, list]:
    commands, cwds = [], []

    def capture_sync(command, capture, *, deadline_seconds, cwd=None):
        commands.append(command)
        cwds.append(cwd)
        capture.write_text("ok")
        return 0

    monkeypatch.setattr(sync_supervisor, "run_sync", capture_sync)
    monkeypatch.setattr(sys, "argv", ["sync_supervisor.py", *argv])
    return sync_supervisor.main(), commands, cwds


def test_python_runtime_sync_is_independent_of_project(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "results"
    source.mkdir()
    runtime = tmp_path / "sync-env" / "bin" / "python"
    monkeypatch.chdir(tmp_path)
    rc, commands, cwds = _run_main(monkeypatch, [
        "--source", str(source), "--destination", "s3://bucket/run",
        "--state", str(tmp_path / "state"), "--python", str(runtime), "--final",
    ])
    assert rc == 0
    # The runtime interpreter runs sync_results directly, isolated from PYTHONPATH/user site,
    # from the supervisor's own (copied) directory rather than any project tree.
    assert commands[0][:3] == [str(runtime), "-E", "-s"]
    assert commands[0][3] == str(Path(sync_supervisor.__file__).with_name("sync_results.py"))
    assert commands[0][4:] == [
        "sync", "--source", str(source), "--destination", "s3://bucket/run",
        "--state", str(tmp_path / "state"), "--final",
    ]
    assert "uv" not in commands[0]
    assert cwds[0] == str(Path(sync_supervisor.__file__).resolve().parent)


def test_python_and_project_are_mutually_exclusive(tmp_path: Path, monkeypatch) -> None:
    import pytest

    with pytest.raises(SystemExit):
        _run_main(monkeypatch, [
            "--source", str(tmp_path), "--destination", "s3://b/r", "--state", str(tmp_path / "s"),
            "--python", "/x/python", "--project", str(tmp_path),
        ])


def test_build_runtime_copies_only_probed_distributions(tmp_path: Path, monkeypatch) -> None:
    import shutil
    import subprocess as sp

    import pytest

    if shutil.which("uv") is None:
        pytest.skip("uv is not installed")
    # Exercise the copy/verify mechanics with a small installed distribution instead of the
    # S3 stack (which this project's test environment does not install).
    monkeypatch.setattr(sync_supervisor, "_RUNTIME_PROBE", "import iniconfig\n")
    python = sync_supervisor.build_runtime(tmp_path / "sync-env", tmp_path / "no-config")
    site = next((tmp_path / "sync-env").glob("lib/python3*/site-packages"))
    assert (site / "iniconfig").is_dir()
    assert not (site / "sync_supervisor.py").exists()  # the builder itself is not copied
    assert not (site / "pluggy").exists()  # not probed, not required by iniconfig
    # The verifier fails closed on a module the runtime does not carry.
    monkeypatch.setattr(sync_supervisor, "_RUNTIME_PROBE", "import pluggy\n")
    with pytest.raises(sp.CalledProcessError):
        sync_supervisor.verify_runtime(python, cwd=tmp_path)


def test_build_runtime_refuses_base_interpreter_under_outside(tmp_path: Path, monkeypatch) -> None:
    import os as _os

    import pytest

    monkeypatch.setattr(sync_supervisor, "_RUNTIME_PROBE", "")
    base = Path(_os.path.realpath(getattr(sys, "_base_executable", sys.executable)))
    with pytest.raises(RuntimeError, match="lives under"):
        sync_supervisor.build_runtime(tmp_path / "sync-env", tmp_path, outside=(base.parent,))


def test_vanished_state_directory_is_a_failed_round_not_a_crash(tmp_path: Path, monkeypatch) -> None:
    """2026-09-29 shard-053-h1: the state dir vanished mid-sync and main() died."""
    source = tmp_path / "results"
    source.mkdir()
    state = tmp_path / "state"
    calls = []

    def vanishing_sync(command, capture, *, deadline_seconds, cwd=None):
        calls.append(capture)
        if len(calls) == 1:
            import shutil
            shutil.rmtree(state)  # deleted while the sync ran: no capture left behind
            return 1
        if len(calls) == 2:
            raise FileNotFoundError(capture)  # cannot even open the capture
        capture.write_text("ok")
        raise SystemExit(0)  # end the loop on the third round

    monkeypatch.setattr(sync_supervisor, "run_sync", vanishing_sync)
    monkeypatch.setattr(sync_supervisor.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(sys, "argv", [
        "sync_supervisor.py", "--source", str(source), "--destination", "s3://bucket/run",
        "--state", str(state), "--python", sys.executable, "--loop",
    ])
    with pytest.raises(SystemExit):
        sync_supervisor.main()
    assert len(calls) == 3
    assert state.is_dir()
    receipts = [json.loads(line) for line in
                (source / "controller" / "uploader.log").read_text().splitlines()]
    assert [r["ok"] for r in receipts] == [False, False]
    assert receipts[0]["error_classes"] == "SyncCaptureMissingError"
    assert "SyncCaptureUnavailableError" in receipts[1]["error_classes"]
