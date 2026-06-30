# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the iris.runtime.multigpu in-task GPU process supervisor and the
client-side entrypoint wrapping that drives it. None of this imports jax."""

from __future__ import annotations

import sys

import pytest
from iris.client.client import _wrap_entrypoint_for_multiprocess
from iris.cluster.types import Entrypoint, ResourceSpec, gpu_device
from iris.runtime.multigpu import _child_rank_env, run


def _py(code: str) -> list[str]:
    """A child command that runs `code` with this interpreter."""
    return [sys.executable, "-c", code]


def test_child_rank_env_global_rank_and_device_slice() -> None:
    # task 1 of 3, 4 procs/task, 2 devices/proc, local rank 2:
    # global rank = 1*4 + 2 = 6; world = 3*4 = 12; devices = [2*2, 2*2+1] = 4,5
    env = _child_rank_env(local_rank=2, nproc=4, devices_per_proc=2, task_index=1, num_tasks=3)
    assert env == {
        "JAX_PROCESS_COUNT": "12",
        "JAX_PROCESS_INDEX": "6",
        "JAX_LOCAL_DEVICE_IDS": "4,5",
    }


def test_child_rank_env_one_device_per_proc() -> None:
    env = _child_rank_env(local_rank=0, nproc=8, devices_per_proc=1, task_index=0, num_tasks=1)
    assert env["JAX_PROCESS_COUNT"] == "8"
    assert env["JAX_PROCESS_INDEX"] == "0"
    assert env["JAX_LOCAL_DEVICE_IDS"] == "0"


def test_run_all_children_succeed_returns_zero() -> None:
    check = _py("import os; assert os.environ['JAX_PROCESS_COUNT'] == '3'")
    assert run(nproc=3, devices_per_proc=1, child_argv=check) == 0


def test_run_propagates_first_child_failure() -> None:
    # The rank-1 child exits 7; siblings exit 0. The supervisor surfaces 7.
    code = "import os,sys; sys.exit(7 if os.environ['JAX_PROCESS_INDEX']=='1' else 0)"
    assert run(nproc=3, devices_per_proc=1, child_argv=_py(code)) == 7


def test_run_terminates_peers_when_one_fails() -> None:
    # Rank 0 fails immediately; the peers would otherwise sleep 30s. The
    # supervisor must tear them down and return promptly with the failure code.
    code = "import os,sys,time; sys.exit(3) if os.environ['JAX_PROCESS_INDEX']=='0' else time.sleep(30)"
    assert run(nproc=3, devices_per_proc=1, child_argv=_py(code)) == 3


def test_run_rejects_empty_command() -> None:
    with pytest.raises(ValueError, match="no child command"):
        run(nproc=2, devices_per_proc=1, child_argv=[])


def _gpu_resources(count: int) -> ResourceSpec:
    return ResourceSpec(cpu=4, memory="8GB", disk="16GB", device=gpu_device("H100", count))


def test_wrap_entrypoint_one_process_per_gpu() -> None:
    wrapped = _wrap_entrypoint_for_multiprocess(
        Entrypoint.from_command("python", "train.py", "--steps", "10"), _gpu_resources(8), processes_per_task=8
    )
    assert wrapped.command == [
        "python",
        "-m",
        "iris.runtime.multigpu",
        "--nproc",
        "8",
        "--devices-per-proc",
        "1",
        "--",
        "python",
        "train.py",
        "--steps",
        "10",
    ]


def test_wrap_entrypoint_groups_devices_when_fewer_processes() -> None:
    wrapped = _wrap_entrypoint_for_multiprocess(
        Entrypoint.from_command("python", "train.py"), _gpu_resources(8), processes_per_task=4
    )
    assert wrapped.command[:8] == [
        "python",
        "-m",
        "iris.runtime.multigpu",
        "--nproc",
        "4",
        "--devices-per-proc",
        "2",
        "--",
    ]


def test_wrap_entrypoint_requires_gpu() -> None:
    cpu_only = ResourceSpec(cpu=4, memory="8GB", disk="16GB", device=None)
    with pytest.raises(ValueError, match="requires a GPU device"):
        _wrap_entrypoint_for_multiprocess(Entrypoint.from_command("python", "x.py"), cpu_only, processes_per_task=2)


def test_wrap_entrypoint_requires_divisible_gpu_count() -> None:
    entry = Entrypoint.from_command("python", "x.py")
    with pytest.raises(ValueError, match="must divide the GPU count"):
        _wrap_entrypoint_for_multiprocess(entry, _gpu_resources(8), processes_per_task=3)
