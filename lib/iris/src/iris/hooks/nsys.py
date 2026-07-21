# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nsight Systems profiling hook.

``NsysHook`` wraps a command in :mod:`iris.hooks.nsys_main`, which runs it
under ``nsys profile`` on the selected units and uploads the report. The ``nsys``
binary is baked into the iris task image, so there is nothing to install.

Scope follows composition. Wrapped around the multigpu supervisor, one report covers
every rank the task runs; passed as that supervisor's ``wrap_child``, each child
writes its own report.
"""

from collections.abc import Sequence
from dataclasses import dataclass

# nsys ``--trace`` default: CUDA kernels + NVTX ranges + cuBLAS. NCCL shows up as CUDA
# kernels plus its own NVTX ranges. CPU sampling and GPU metrics need privileges an
# unprivileged task container lacks, so they are never enabled.
NSYS_DEFAULT_TRACE = "cuda,nvtx,cublas"

_NSYS_MAIN_MODULE = "iris.hooks.nsys_main"


@dataclass(frozen=True)
class NsysHook:
    """Run the command under ``nsys profile`` on the selected units.

    Attributes:
        output_uri: Report directory URI. ``None`` lets the task resolve its cluster's
            temp bucket from its own env (``nsys_main.default_output_uri``) — correct
            even under ``--target-cluster``, where the launcher's cluster is the wrong
            store. The task workdir is an emptyDir, so a report left there dies with it.
        tasks: Which units write a report, by index: ``first``, ``all``, or ``0,7``.
        trace: The nsys ``--trace`` value.
        capture_range: Collect only between ``cuProfilerStart``/``cuProfilerStop`` —
            keeps compile out and aligns multi-unit captures on the same step; the
            application must call the API or nothing is collected.
    """

    output_uri: str | None = None
    tasks: str = "first"
    trace: str = NSYS_DEFAULT_TRACE
    capture_range: bool = False

    def wrap(self, command: Sequence[str]) -> list[str]:
        argv = ["python", "-m", _NSYS_MAIN_MODULE, "--tasks", self.tasks, "--trace", self.trace]
        # Omitted when unset so the wrapper defaults it from the task's own cluster env.
        if self.output_uri is not None:
            argv += ["--output-uri", self.output_uri]
        if self.capture_range:
            argv.append("--capture-range")
        argv.append("--")
        return [*argv, *command]
