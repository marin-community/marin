# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task hooks: pluggable transforms over a task's setup and run command.

A ``TaskHook`` contributes two things, either of which may be a no-op:

- ``setup()`` — a shell script appended to the job's build-phase setup (e.g. installing
  a profiler CLI), or ``None``.
- ``wrap(command)`` — the run command wrapped so the user's process runs *under* the
  hook (e.g. ``python -m iris.runtime.nsys -- <command>``), or the command unchanged.

Hooks are applied in order by the client, and order *is* the nesting: a hook applied
later ends up the outer wrapper. Everything a hook needs at run time — rank selection,
output upload, signal forwarding — lives inside the module its ``wrap`` prepends; the
client only knows "it wraps the command".

This module is deliberately a leaf: the hook classes take plain values, so the
resource- and spec-aware *builders* (which hooks to add, in which order) live in the
client alongside ``collect_hooks``.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from iris.cluster.setup_scripts import nsys_setup_script

# Module paths of the in-task wrappers each hook prepends.
_NSYS_MODULE = "iris.runtime.nsys"
_MULTIGPU_MODULE = "iris.runtime.multigpu"

# nsys ``--trace`` default: CUDA kernels + NVTX ranges + cuBLAS. NCCL shows up as CUDA
# kernels plus its own NVTX ranges. CPU sampling and GPU metrics need privileges an
# unprivileged task container lacks, so they are never enabled.
NSYS_DEFAULT_TRACE = "cuda,nvtx,cublas"


@runtime_checkable
class TaskHook(Protocol):
    """A transform over a task's setup script and run command."""

    def setup(self) -> str | None:
        """A build-phase setup script to append, or ``None`` if the hook installs nothing."""

    def wrap(self, command: Sequence[str]) -> list[str]:
        """Return *command* wrapped to run under this hook."""


@dataclass(frozen=True)
class NsysHook:
    """Run the command under ``nsys profile`` on the selected tasks (see ``iris.runtime.nsys``).

    A launch wrapper by necessity — nsys injects CUDA tracing at ``cuInit`` — so it cannot
    profile a job that is already running, unlike the attach-based py-spy/memray profiler.
    Sitting outside the multi-process GPU supervisor, one report covers every rank a
    selected task runs (nsys traces children): the better artifact for intra-node
    collectives, at the cost of no sub-node GPU selection. Selection, report upload, and
    signal forwarding all happen inside the wrapper module; this only builds its argv and
    names the install script.

    Attributes:
        output_uri: Report directory URI. ``None`` lets the task resolve its cluster's temp
            bucket from its own env (``iris.runtime.nsys.default_output_uri``) — correct even
            under ``--target-cluster``, where the launcher's cluster is the wrong store.
        tasks: Which tasks write a report, by index: ``first``, ``all``, or a list (``0,7``).
        trace: The nsys ``--trace`` value; CPU sampling and GPU metrics need privileges the
            task container lacks and are never enabled.
        capture_range: Collect only between ``cuProfilerStart``/``cuProfilerStop`` — keeps
            compile out and aligns multi-task captures on the same step; the app must call
            the API or nothing is collected.
    """

    output_uri: str | None = None
    tasks: str = "first"
    trace: str = NSYS_DEFAULT_TRACE
    capture_range: bool = False

    def setup(self) -> str | None:
        return nsys_setup_script()

    def wrap(self, command: Sequence[str]) -> list[str]:
        argv = ["python", "-m", _NSYS_MODULE, "--tasks", self.tasks, "--trace", self.trace]
        # Omitted entirely when unset: the wrapper then defaults it from the task's own
        # cluster env, which is the correct store even under federation.
        if self.output_uri is not None:
            argv += ["--output-uri", self.output_uri]
        if self.capture_range:
            argv.append("--capture-range")
        argv.append("--")
        return [*argv, *command]


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
            _MULTIGPU_MODULE,
            "--nproc",
            str(self.nproc),
            "--devices-per-proc",
            str(self.devices_per_proc),
            "--",
            *command,
        ]
