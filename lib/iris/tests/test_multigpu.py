# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the iris.hooks.multigpu_main in-task GPU process supervisor (a user-invoked
runtime helper, not something the scheduler injects). None of this imports jax."""

from __future__ import annotations

import signal
import subprocess
import sys
import textwrap

import pytest
from iris.cluster.types import ResourceSpec, gpu_device
from iris.hooks import multigpu_main as multigpu
from iris.hooks.multigpu import MultiGpuHook, build_multigpu_hook
from iris.hooks.multigpu_main import main, run
from rigging.timing import Duration


def _py(code: str) -> list[str]:
    """A child command that runs `code` with this interpreter."""
    return [sys.executable, "-c", code]


def test_run_all_children_succeed_returns_zero() -> None:
    # Each child sees the global world size the supervisor stamped on it.
    check = _py("import os; assert os.environ['IRIS_MULTIGPU_PROCESS_COUNT'] == '3'")
    assert run(nproc=3, devices_per_proc=1, child_argv=check) == 0


def test_run_propagates_first_child_failure() -> None:
    # The rank-1 child exits 7; siblings exit 0. The supervisor surfaces 7.
    code = "import os,sys; sys.exit(7 if os.environ['IRIS_MULTIGPU_PROCESS_INDEX']=='1' else 0)"
    assert run(nproc=3, devices_per_proc=1, child_argv=_py(code)) == 7


def test_run_terminates_peers_when_one_fails() -> None:
    # Rank 0 fails immediately; the peers would otherwise sleep 30s. The
    # supervisor must tear them down and return promptly with the failure code.
    code = "import os,sys,time; sys.exit(3) if os.environ['IRIS_MULTIGPU_PROCESS_INDEX']=='0' else time.sleep(30)"
    assert run(nproc=3, devices_per_proc=1, child_argv=_py(code)) == 3


def test_run_sigkills_a_peer_that_ignores_sigterm(monkeypatch: pytest.MonkeyPatch) -> None:
    # Rank 0 fails; the peer traps SIGTERM and would sleep 30s. With escalation,
    # the supervisor SIGKILLs it after the grace period and still returns 3.
    monkeypatch.setattr(multigpu, "_TERMINATE_GRACE", Duration.from_seconds(0.5))
    code = (
        "import os,sys,signal,time\n"
        "if os.environ['IRIS_MULTIGPU_PROCESS_INDEX']=='0': sys.exit(3)\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(30)\n"
    )
    assert run(nproc=2, devices_per_proc=1, child_argv=_py(code)) == 3


def test_spawn_failure_kills_already_started_children(monkeypatch: pytest.MonkeyPatch) -> None:
    # Rank 0 spawns; rank 1's Popen raises. The started rank-0 child must be
    # killed (not orphaned) before the error propagates.
    real_popen = subprocess.Popen
    started: list[subprocess.Popen] = []
    calls = {"n": 0}

    def flaky_popen(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated spawn failure")
        proc = real_popen(*args, **kwargs)
        started.append(proc)
        return proc

    monkeypatch.setattr(multigpu.subprocess, "Popen", flaky_popen)
    with pytest.raises(OSError, match="simulated spawn failure"):
        run(nproc=3, devices_per_proc=1, child_argv=_py("import time; time.sleep(30)"))
    assert len(started) == 1
    # Killed by SIGKILL → returncode is the negative signal number, never 0.
    assert started[0].returncode is not None and started[0].returncode < 0


def test_external_sigterm_returns_128_plus_signum() -> None:
    # The supervisor itself is SIGTERMed (preemption/task termination). It
    # forwards the signal to the children, which die reporting negative signal
    # codes (-15). The supervisor must surface 128+SIGTERM (143) — the killed
    # task — not map a child's -15 to an arbitrary failure code (241).
    supervisor_src = textwrap.dedent(
        """
        import sys
        from iris.hooks.multigpu_main import run
        child = [sys.executable, "-c", "print('READY', flush=True); import time; time.sleep(30)"]
        sys.exit(run(nproc=2, devices_per_proc=1, child_argv=child))
        """
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", supervisor_src], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert proc.stdout is not None
    # Signal only once both children are live, so the forwarded SIGTERM is what
    # ends them (rather than racing their startup).
    ready = 0
    for line in proc.stdout:
        if "READY" in line:
            ready += 1
            if ready == 2:
                break
    assert ready == 2, "children did not start"
    proc.send_signal(signal.SIGTERM)
    proc.communicate(timeout=30)
    assert proc.returncode == 128 + signal.SIGTERM


def test_run_rejects_empty_command() -> None:
    with pytest.raises(ValueError, match="no child command"):
        run(nproc=2, devices_per_proc=1, child_argv=[])


def test_wrap_prefixes_each_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """--wrap CMD makes each child run `<CMD> -- <command>` — the process-scope seam."""
    seen: dict[str, object] = {}
    monkeypatch.setattr("iris.hooks.multigpu_main.run", lambda nproc, dpp, child_argv: seen.update(argv=child_argv) or 0)
    main(["--nproc", "2", "--wrap", "python -m iris.hooks.nsys_main --tasks first", "--", "python", "train.py"])
    assert seen["argv"] == [
        "python",
        "-m",
        "iris.hooks.nsys_main",
        "--tasks",
        "first",
        "--",
        "python",
        "train.py",
    ]


def _gpu_resources(count: int) -> ResourceSpec:
    return ResourceSpec(cpu=4, memory="8GB", disk="16GB", device=gpu_device("H100", count))


def test_hook_wrap_builds_the_command_the_entry_point_parses() -> None:
    """The programmatic wrap emits exactly what multigpu_main's parser accepts, so a
    caller can compose it in code or write the same command by hand."""
    wrapped = MultiGpuHook(nproc=8).wrap(["python", "train.py"])
    assert wrapped == [
        "python",
        "-m",
        "iris.hooks.multigpu_main",
        "--nproc",
        "8",
        "--devices-per-proc",
        "1",
        "--",
        "python",
        "train.py",
    ]


def test_hook_wrap_child_composes_a_per_process_wrapper() -> None:
    wrapped = MultiGpuHook(nproc=2, wrap_child="python -m iris.hooks.nsys_main --tasks first").wrap(["cmd"])
    assert wrapped[wrapped.index("--wrap") + 1] == "python -m iris.hooks.nsys_main --tasks first"


def test_build_multigpu_hook_groups_devices() -> None:
    assert build_multigpu_hook(_gpu_resources(8), 4) == MultiGpuHook(nproc=4, devices_per_proc=2)


def test_build_multigpu_hook_requires_gpu() -> None:
    cpu_only = ResourceSpec(cpu=4, memory="8GB", disk="16GB", device=None)
    with pytest.raises(ValueError, match="requires a GPU device"):
        build_multigpu_hook(cpu_only, 2)


def test_build_multigpu_hook_requires_divisible_gpu_count() -> None:
    with pytest.raises(ValueError, match="must divide the GPU count"):
        build_multigpu_hook(_gpu_resources(8), 3)
