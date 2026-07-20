# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The multi-process GPU supervisor hook: client-side spec, the resource-aware builder, and
the supervisor↔runtime rank-env contract.

Everything the multigpu backend contributes to *submission* lives here:

- ``MultiGpuHook`` — the :class:`~iris.cluster.hooks.TaskHook` that wraps the run command with
  ``iris.cluster.hooks.multigpu_main``, which spawns one process per GPU group.
- ``build_multigpu_hook`` — turns a job's GPU count and ``processes_per_task`` into that hook.
- ``IRIS_MULTIGPU_*`` — the env-var names the supervisor stamps on each child and its consumers
  (``iris.runtime.jax_init``, ``iris.runtime.telltale``) read back. Defining them here, in the
  light spec module, lets the run-phase consumers depend on the *contract* rather than importing
  the supervisor implementation.

The run-phase half is :mod:`iris.cluster.hooks.multigpu_main`; it is imported only in-task
(via ``python -m``) so this module stays free of its subprocess/threading machinery.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from iris.cluster.types import ResourceSpec, get_gpu_count

# Module path of the in-task supervisor this hook prepends (``python -m <module> -- <cmd>``).
_MULTIGPU_MAIN_MODULE = "iris.cluster.hooks.multigpu_main"

# The supervisor→child rank contract: the supervisor stamps each child with these, and
# iris.runtime.jax_init reads them to switch initialize_jax into supervised mode (telltale
# reads the index to label its output). They are iris-private (not the JAX_*/framework
# namespace) so an unrelated job that happens to set JAX rank vars never trips the supervised
# path — processes_per_task=1 stays a strict no-op. Defined here (the contract) and imported
# by both the producer (multigpu_main) and the consumers so the names cannot drift.
IRIS_MULTIGPU_PROCESS_COUNT_ENV = "IRIS_MULTIGPU_PROCESS_COUNT"
IRIS_MULTIGPU_PROCESS_INDEX_ENV = "IRIS_MULTIGPU_PROCESS_INDEX"
IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV = "IRIS_MULTIGPU_LOCAL_DEVICE_IDS"


@dataclass(frozen=True)
class MultiGpuHook:
    """Run the command under the multi-process GPU supervisor (see ``iris.cluster.hooks.multigpu_main``).

    The supervisor spawns ``nproc`` children, each pinned to a contiguous group of
    ``devices_per_proc`` of the task's GPUs.
    """

    nproc: int
    devices_per_proc: int

    def setup(self) -> str | None:
        return None

    def wrap(self, command: Sequence[str]) -> list[str]:
        return [
            "python",
            "-m",
            _MULTIGPU_MAIN_MODULE,
            "--nproc",
            str(self.nproc),
            "--devices-per-proc",
            str(self.devices_per_proc),
            "--",
            *command,
        ]


def build_multigpu_hook(resources: ResourceSpec, processes_per_task: int) -> MultiGpuHook:
    """Build the multi-process GPU supervisor hook for *processes_per_task* processes.

    Requires a GPU device whose count is divisible by ``processes_per_task``; each of the
    N children is pinned to a contiguous group of ``gpu_count // N`` GPUs.
    """
    device = resources.device
    gpu_count = get_gpu_count(device) if device is not None and device.HasField("gpu") else 0
    if gpu_count <= 0:
        raise ValueError("processes_per_task > 1 requires a GPU device")
    if gpu_count % processes_per_task != 0:
        raise ValueError(f"processes_per_task ({processes_per_task}) must divide the GPU count ({gpu_count})")
    return MultiGpuHook(nproc=processes_per_task, devices_per_proc=gpu_count // processes_per_task)
