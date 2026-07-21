# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the iris.cluster.hooks.respawn_main in-place crash respawner and the
client-side spec that drives it. None of this imports jax."""

from __future__ import annotations

import signal
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from iris.client.client import collect_hooks
from iris.cluster.hooks import respawn_main
from iris.cluster.hooks.nsys import NsysHook
from iris.cluster.hooks.respawn import RespawnHook
from iris.cluster.hooks.respawn_main import run
from iris.cluster.types import EnvironmentSpec, ResourceSpec, gpu_device
from rigging.timing import Duration


@pytest.fixture(autouse=True)
def fast_respawn_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(respawn_main, "_RESPAWN_DELAY", Duration.from_seconds(0.1))


def _crashy_child(tmp_path: Path, *, crash_below: int, sig: int = signal.SIGABRT) -> list[str]:
    """A child that logs its attempt index, then dies from `sig` on attempts < crash_below."""
    log = tmp_path / "attempts.log"
    code = (
        "import os,resource,signal,sys\n"
        "attempt = int(os.environ['IRIS_RESPAWN_ATTEMPT'])\n"
        f"open({str(log)!r}, 'a').write(f'{{attempt}}\\n')\n"
        "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
        f"if attempt < {crash_below}: os.kill(os.getpid(), {int(sig)})\n"
        "sys.exit(0)\n"
    )
    return [sys.executable, "-c", code]


def _attempts(tmp_path: Path) -> list[str]:
    return (tmp_path / "attempts.log").read_text().split()


def test_run_clean_exit_returns_zero(tmp_path: Path) -> None:
    assert run(max_restarts=3, child_argv=_crashy_child(tmp_path, crash_below=0)) == 0
    assert _attempts(tmp_path) == ["0"]


def test_run_restores_callers_signal_handlers(tmp_path: Path) -> None:
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    assert run(max_restarts=3, child_argv=_crashy_child(tmp_path, crash_below=0)) == 0

    assert {sig: signal.getsignal(sig) for sig in previous} == previous


def test_run_propagates_nonzero_exit_without_respawn(tmp_path: Path) -> None:
    # A deliberate exit (Python exception path) is deterministic application
    # failure — the child must run exactly once and its code must propagate.
    log = tmp_path / "attempts.log"
    code = f"open({str(log)!r}, 'a').write('ran\\n'); import sys; sys.exit(7)"
    assert run(max_restarts=3, child_argv=[sys.executable, "-c", code]) == 7
    assert _attempts(tmp_path) == ["ran"]


def test_run_respawns_after_crash_signal(tmp_path: Path) -> None:
    # Attempt 0 dies from SIGSEGV (the JAX coordination-service crash shape);
    # the respawned attempt 1 succeeds and the task exits clean.
    child = _crashy_child(tmp_path, crash_below=1, sig=signal.SIGSEGV)
    assert run(max_restarts=3, child_argv=child) == 0
    assert _attempts(tmp_path) == ["0", "1"]


def test_run_kills_crashed_attempt_descendants_before_respawn(tmp_path: Path) -> None:
    # Attempt 0 leaves a descendant holding a TCP port, then crashes. Attempt 1
    # can bind that port only if the respawner killed the old process group.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    log = tmp_path / "attempts.log"
    descendant = (
        "import signal,socket,sys\n"
        "sock = socket.socket()\n"
        "sock.bind(('127.0.0.1', int(sys.argv[1])))\n"
        "sock.listen()\n"
        "print('READY', flush=True)\n"
        "signal.pause()\n"
    )
    child = (
        "import os,resource,signal,socket,subprocess,sys\n"
        "attempt = int(os.environ['IRIS_RESPAWN_ATTEMPT'])\n"
        f"open({str(log)!r}, 'a').write(f'{{attempt}}\\n')\n"
        "if attempt == 0:\n"
        f"    child = subprocess.Popen([sys.executable, '-c', {descendant!r}, {str(port)!r}], "
        "stdout=subprocess.PIPE, text=True)\n"
        "    assert child.stdout is not None and child.stdout.readline() == 'READY\\n'\n"
        "    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
        "    os.kill(os.getpid(), signal.SIGABRT)\n"
        "sock = socket.socket()\n"
        f"sock.bind(('127.0.0.1', {port}))\n"
    )

    assert run(max_restarts=1, child_argv=[sys.executable, "-c", child]) == 0
    assert _attempts(tmp_path) == ["0", "1"]


def test_run_exhausts_restart_budget(tmp_path: Path) -> None:
    # Always-crashing child with a budget of 1: original + one respawn, then the
    # SIGABRT death propagates as the conventional 134.
    child = _crashy_child(tmp_path, crash_below=99)
    assert run(max_restarts=1, child_argv=child) == 128 + signal.SIGABRT
    assert _attempts(tmp_path) == ["0", "1"]


def test_run_gives_up_on_rapid_crash_loop(tmp_path: Path) -> None:
    # Instant crashes must not burn the whole restart budget: three consecutive
    # rapid deaths give up even though max_restarts would allow far more.
    child = _crashy_child(tmp_path, crash_below=99)
    assert run(max_restarts=100, child_argv=child) == 128 + signal.SIGABRT
    assert _attempts(tmp_path) == ["0", "1", "2"]


def test_run_healthy_uptime_resets_rapid_death_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Each attempt lives past _MIN_UPTIME before dying, so the rapid-death brake
    # never engages and the third respawn (attempt 3) runs to success.
    class HealthyAttemptTimer:
        def elapsed_seconds(self) -> float:
            return respawn_main._MIN_UPTIME

    monkeypatch.setattr(respawn_main, "Timer", HealthyAttemptTimer)
    child = _crashy_child(tmp_path, crash_below=3)
    assert run(max_restarts=100, child_argv=child) == 0
    assert _attempts(tmp_path) == ["0", "1", "2", "3"]


def test_run_sigkill_death_propagates_without_respawn(tmp_path: Path) -> None:
    # SIGKILL is external (kernel OOM kill, teardown): no respawn, and the exit
    # surfaces as 137 so the worker's OOM annotation keeps working.
    child = _crashy_child(tmp_path, crash_below=99, sig=signal.SIGKILL)
    assert run(max_restarts=3, child_argv=child) == 137
    assert _attempts(tmp_path) == ["0"]


def test_external_sigterm_returns_128_plus_signum() -> None:
    # The respawner itself is SIGTERMed (preemption/task stop). It forwards the
    # signal and exits 128+SIGTERM without treating the child's death as a crash
    # to respawn.
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "iris.cluster.hooks.respawn_main",
            "--max-restarts",
            "3",
            "--",
            sys.executable,
            "-c",
            "print('READY', flush=True); import signal; signal.pause()",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if "READY" in line:
            break
    proc.send_signal(signal.SIGTERM)
    proc.communicate(timeout=30)
    assert proc.returncode == 128 + signal.SIGTERM


def test_run_rejects_empty_command() -> None:
    with pytest.raises(ValueError, match="no child command"):
        run(max_restarts=1, child_argv=[])


def test_respawn_hook_wraps_command() -> None:
    wrapped = RespawnHook(max_restarts=5).wrap(["python", "train.py", "--steps", "10"])
    assert wrapped == [
        "python",
        "-m",
        "iris.cluster.hooks.respawn_main",
        "--max-restarts",
        "5",
        "--",
        "python",
        "train.py",
        "--steps",
        "10",
    ]


def test_respawn_hook_requires_positive_restart_budget() -> None:
    with pytest.raises(ValueError, match="max_restarts must be >= 1"):
        RespawnHook(max_restarts=0)


def _gpu_resources(count: int) -> ResourceSpec:
    return ResourceSpec(cpu=4, memory="8GB", disk="16GB", device=gpu_device("H100", count))


def test_collect_hooks_orders_respawn_between_supervisor_and_profiler() -> None:
    """Folded in order, the profiler is outermost, the respawner next (a respawn
    restarts every rank the multigpu supervisor runs), the supervisor innermost."""
    env = EnvironmentSpec(profile=NsysHook(), respawn=RespawnHook())
    cmd = ["python", "train.py"]
    for hook in collect_hooks(env, _gpu_resources(8), processes_per_task=8):
        cmd = hook.wrap(cmd)
    assert cmd.index("iris.cluster.hooks.nsys_main") < cmd.index("iris.cluster.hooks.respawn_main")
    assert cmd.index("iris.cluster.hooks.respawn_main") < cmd.index("iris.cluster.hooks.multigpu_main")


def test_collect_hooks_respawn_alone() -> None:
    cpu_only = ResourceSpec(cpu=4, memory="8GB", disk="16GB", device=None)
    (hook,) = collect_hooks(EnvironmentSpec(respawn=RespawnHook(max_restarts=7)), cpu_only, 1)
    assert hook.wrap(["python", "train.py"])[:6] == [
        "python",
        "-m",
        "iris.cluster.hooks.respawn_main",
        "--max-restarts",
        "7",
        "--",
    ]
