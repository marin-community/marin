# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The multi-process GPU supervisor hook: the client-side spec.

``MultiGpuHook`` wraps the run command with the run-phase supervisor, which lives in
:mod:`iris.runtime.multigpu` (a general in-task runtime whose rank contract
``iris.runtime.jax_init`` consumes — it is not moved under this package). The resource-aware
builder that decides ``nproc``/``devices_per_proc`` from a job's GPU count is
``iris.client.client.build_multigpu_hook``: this class only takes the plain values it needs,
so the hooks package stays a leaf.
"""

from collections.abc import Sequence
from dataclasses import dataclass

# Module path of the in-task supervisor this hook prepends (``python -m <module> -- <cmd>``).
_MULTIGPU_MAIN_MODULE = "iris.runtime.multigpu"


@dataclass(frozen=True)
class MultiGpuHook:
    """Run the command under the multi-process GPU supervisor (see ``iris.runtime.multigpu``).

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
