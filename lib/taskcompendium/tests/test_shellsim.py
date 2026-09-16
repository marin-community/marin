# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real pinned simulator across its process and VFS boundary."""

import json
import os
import sys
from pathlib import Path

import pytest

from taskcompendium.shellsim import SHELLSIM_REVISION, ShellSimError, ShellSimLimits, ShellSimSession, ShellSimTimeout


def test_session_persists_shell_state_and_binary_files(bridge: str):
    with ShellSimSession(bridge) as session:
        session.mkdir("/work/results")
        session.write_file("/work/results/binary", b"\x00\xffabc")
        first = session.run("cd /work/results; export ANSWER=42; cat > input", b"supplied input\n")
        second = session.run('printf "%s:%s" "$PWD" "$ANSWER"; cat input')
        assert first.return_code == second.return_code == 0
        assert second.stdout == "/work/results:42supplied input\n"
        assert session.read_file("/work/results/binary") == b"\x00\xffabc"
        assert session.list_files("/work/results") == ["/work/results/binary", "/work/results/input"]
        assert session.list_dir("/work/results") == ("binary", "input")
    with ShellSimSession(bridge) as independent:
        assert not independent.is_file("/work/results/binary")


def test_vfs_never_reads_or_executes_host_paths(bridge: str, tmp_path: Path):
    secret = tmp_path / "host-only"
    secret.write_text("host secret")
    with ShellSimSession(bridge) as session:
        assert session.run(f"cat {secret}").return_code != 0
        assert session.run(f"{sys.executable} -c 'print(42)'").return_code == 127
        with pytest.raises(ShellSimError):
            session.read_file(str(secret))
        assert session.run("echo alive").stdout == "alive\n"
    assert secret.read_text() == "host secret"


def test_fuel_is_cumulative_and_stops_infinite_loop(bridge: str):
    with ShellSimSession(bridge, limits=ShellSimLimits(cpu=1000)) as session:
        first = session.run("echo first")
        second = session.run("while true; do :; done")
        third = session.run("echo cannot run")
        assert first.return_code == 0
        assert first.usage.cpu_used > 0
        assert second.return_code == third.return_code == 137
        assert second.stop_reason == third.stop_reason == "cpu_exhausted"
        assert third.stdout == ""
        assert second.usage.cpu_used == third.usage.cpu_used == 1000


def test_disk_quota_rejection_preserves_previous_file(bridge: str):
    with ShellSimSession(bridge, limits=ShellSimLimits(disk=1024)) as session:
        session.write_file("/work/value", b"original")
        with pytest.raises(ShellSimError):
            session.write_file("/work/value", b"x" * 2048)
        assert session.read_file("/work/value") == b"original"


def test_output_quota_is_bounded(bridge: str):
    with ShellSimSession(bridge, limits=ShellSimLimits(output=32)) as session:
        result = session.run("yes many")
        assert result.return_code == 137
        assert result.stop_reason == "output_limit_exceeded"
        assert len(result.stdout.encode()) + len(result.stderr.encode()) <= 32


def test_timeout_kills_bridge_that_stops_reading_requests(tmp_path: Path):
    # The child completes initialization, then stops consuming stdin. A payload
    # larger than the pipe capacity must still honor the operation deadline.
    script = tmp_path / "unresponsive"
    pid_file = tmp_path / "child.pid"
    response = json.dumps({"ok": True, "result": {"revision": SHELLSIM_REVISION}})
    script.write_text(
        f"#!{sys.executable}\nimport os, signal, sys\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "sys.stdin.readline()\n"
        f"print({response!r}, flush=True)\n"
        "signal.pause()\n"
    )
    script.chmod(0o755)
    with ShellSimSession(str(script)) as session:
        with pytest.raises(ShellSimTimeout):
            session.run("x" * 1_000_000, timeout=0.1)
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)
        with pytest.raises(ShellSimError):
            session.run("echo dead")


def test_closed_session_rejects_actions_and_releases_child(bridge: str):
    with ShellSimSession(bridge) as session:
        assert session.run("echo live").stdout == "live\n"
    session.close()
    with pytest.raises(ShellSimError):
        session.run("echo gone")
