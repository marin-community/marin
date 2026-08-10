# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-process GPU supervisor hook.

``MultiGpuHook`` wraps a command in :mod:`iris.hooks.multigpu_main`, which
spawns one process per GPU group inside a single task. ``build_multigpu_hook``
derives it from a job's GPU count.

This module also defines the ``IRIS_MULTIGPU_*`` rank-env contract the supervisor
stamps on each child; :mod:`iris.runtime.jax_init` and
:mod:`iris.hooks.nsys_main` read it.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from iris.cluster.types import ResourceSpec, get_gpu_count

# Rank env the supervisor stamps on each child.
IRIS_MULTIGPU_PROCESS_COUNT_ENV = "IRIS_MULTIGPU_PROCESS_COUNT"
IRIS_MULTIGPU_PROCESS_INDEX_ENV = "IRIS_MULTIGPU_PROCESS_INDEX"
IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV = "IRIS_MULTIGPU_LOCAL_DEVICE_IDS"

_MULTIGPU_MAIN_MODULE = "iris.hooks.multigpu_main"


@dataclass(frozen=True)
class MultiGpuHook:
    """Run the command under the multi-process GPU supervisor.

    The supervisor spawns ``nproc`` children, each pinned to a contiguous group of
    ``devices_per_proc`` of the task's GPUs, and supervises their lifecycle.

    Attributes:
        nproc: Processes to launch on this host.
        devices_per_proc: Local devices assigned to each process.
        wrap_child: Wrapper prepended to *each child* (e.g. ``NsysHook(...).wrap``
            rendered as a string), for per-process profiling — the wrapper then sees
            each child's own ``IRIS_MULTIGPU_PROCESS_INDEX``. A whole-task wrapper is
            composed around this hook instead.
    """

    nproc: int
    devices_per_proc: int = 1
    wrap_child: str | None = None

    def wrap(self, command: Sequence[str]) -> list[str]:
        argv = [
            "python",
            "-m",
            _MULTIGPU_MAIN_MODULE,
            "--nproc",
            str(self.nproc),
            "--devices-per-proc",
            str(self.devices_per_proc),
        ]
        if self.wrap_child:
            argv += ["--wrap", self.wrap_child]
        argv.append("--")
        return [*argv, *command]


def build_multigpu_hook(resources: ResourceSpec, processes_per_task: int) -> MultiGpuHook:
    """Build the hook for *processes_per_task* processes over the job's GPUs.

    Requires a GPU device whose count is divisible by ``processes_per_task``; each
    child then gets ``gpu_count // processes_per_task`` devices.
    """
    device = resources.device
    gpu_count = get_gpu_count(device) if device is not None and device.HasField("gpu") else 0
    if gpu_count <= 0:
        raise ValueError("processes_per_task > 1 requires a GPU device")
    if gpu_count % processes_per_task != 0:
        raise ValueError(f"processes_per_task ({processes_per_task}) must divide the GPU count ({gpu_count})")
    return MultiGpuHook(nproc=processes_per_task, devices_per_proc=gpu_count // processes_per_task)
