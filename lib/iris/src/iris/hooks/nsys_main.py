# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nsight Systems launch wrapper — a runtime helper the caller composes into the command.

``python -m iris.hooks.nsys_main [--tasks SPEC] [--output-uri URI] -- <argv>`` runs ``<argv>``
under ``nsys profile`` when this unit is selected, and execs ``<argv>`` unchanged
otherwise. iris is a dumb scheduler and does not inject this; the ``nsys`` binary is
baked into the task image, and the caller invokes the wrapper. Without ``--output-uri``
the report goes to the cluster's temp bucket, resolved from the task env
(see :func:`default_output_uri`).

Composed *outside* the multigpu supervisor, ``<argv>`` is the supervisor and one report
covers every GPU rank the task runs (node scope). Composed *inside* it as multigpu's
``--wrap``, each child runs its own wrapper and writes its own report (process scope);
:func:`selection_index` reads the child's ``IRIS_MULTIGPU_PROCESS_INDEX`` in that case.

An unselected unit execs and so costs nothing. A selected unit cannot exec: the report
has to be uploaded once nsys has written it, and the task workdir is an emptyDir that
is destroyed with the pod, so a report left on disk is simply lost. It therefore
supervises the child and forwards signals.

Delivery on termination is best-effort, not guaranteed. Finalizing the report and
uploading it both happen after nsys exits, while the SIGTERM that started the teardown
has already begun a bounded countdown to SIGKILL (10s from the multigpu supervisor;
the pod's own grace period otherwise). A small report makes it; a multi-hundred-MB one
racing a preemption may not. A run whose profile must survive preemption wants
``capture_range`` to keep the report small.

Nsight has to wrap the process at launch: CUDA tracing is injected through
``CUDA_INJECTION64_PATH``, which the driver reads once at ``cuInit``. That is why
this is a submit-time wrapper rather than an arm of the attach-based profiler in
``iris.cluster.runtime.profile`` (py-spy/memray), which can join a live process.

One report is written per profiled task; there is no merged report. Selecting a subset
of tasks is therefore the norm at scale, not an optimization: every task of a 16-node
job would produce 16 multi-hundred-MB reports.

The trace config is fixed to what an unprivileged task container can actually do.
CPU sampling and context-switch tracing need ``perf_event_paranoid <= 2`` (task pods
run at 4), and GPU metrics need a privileged pod, so all three stay off; what remains
is the CUDA/NVTX/NCCL timeline.
"""

import argparse
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from types import FrameType
from typing import NoReturn

from rigging.filesystem import StoragePath
from rigging.filesystem.cluster_config import marin_temp_bucket

from iris.cluster.client.job_info import get_job_info
from iris.hooks.multigpu import IRIS_MULTIGPU_PROCESS_INDEX_ENV

logger = logging.getLogger("iris.nsys")

# Where reports land. Under the workdir so no new mount is needed; one file per rank.
NSYS_OUTPUT_DIR = "nsys"
# Collection stops at cuProfilerStop but the process keeps running, so an app that
# brackets a step window gets exactly that window and nothing else.
_CAPTURE_RANGE_ARGS = ("--capture-range=cudaProfilerApi", "--capture-range-end=stop")
# nsys writes <output>.nsys-rep once the profiled process exits.
_REPORT_SUFFIX = ".nsys-rep"
# Exit status when the command succeeded but nsys produced no report to upload.
_NO_REPORT_EXIT = 1
# Signals forwarded to nsys so a terminated task still finalizes its report.
_FORWARDED_SIGNALS = (signal.SIGINT, signal.SIGTERM)
# Default upload location when no --output-uri is given: the cluster's temp bucket,
# lifecycle-cleaned under tmp/ttl=Nd/, keyed on the job so a run's reports are findable.
_DEFAULT_OUTPUT_TTL_DAYS = 30
_DEFAULT_OUTPUT_SUBDIR = "iris-profiles"


class TaskSelector(StrEnum):
    """Which tasks write a report, selected by task index."""

    FIRST = "first"
    ALL = "all"


def selection_index() -> int:
    """The index ``--tasks`` selects on, from the task env.

    Two scopes, one rule: when this process is a multigpu child (``iris.hooks.multigpu_main``
    stamped ``IRIS_MULTIGPU_PROCESS_INDEX``), that global rank is the unit — one report
    per selected *process*. Otherwise the whole task is the unit (its ``task_index``) —
    one report per selected *task*, covering every rank the task runs. Which one applies
    is set by *where* the wrapper is composed: inside the supervisor (per child, via its
    ``--wrap``) or outside it.

    Raises:
        RuntimeError: If there is no iris task context to take an index from.
    """
    process_index = os.environ.get(IRIS_MULTIGPU_PROCESS_INDEX_ENV)
    if process_index is not None:
        return int(process_index)
    info = get_job_info()
    if info is None:
        raise RuntimeError("no iris job context (IRIS_TASK_ID unset); nsys selection needs one")
    return info.task_index


def default_output_uri() -> str:
    """Resolve the report directory from the *task's own* marin prefix.

    Uses ``MARIN_PREFIX`` (set per-cluster in the task env; region metadata otherwise),
    so reports land on the cluster the task actually runs on. That is correct even when
    the job was federated to a peer — a default computed at submit time from the
    launcher's cluster would name the wrong store, which is why the URI is resolved here
    and not on the client. Keyed on the job so a run's reports are findable and expire
    together under the bucket's ``tmp/ttl=Nd/`` lifecycle rule.

    Raises:
        RuntimeError: If there is no iris task context to key the path on.
    """
    info = get_job_info()
    if info is None:
        raise RuntimeError("no iris job context (IRIS_TASK_ID unset); nsys needs one to default --output-uri")
    prefix = f"{_DEFAULT_OUTPUT_SUBDIR}/{info.job_id.to_wire().lstrip('/')}"
    return marin_temp_bucket(_DEFAULT_OUTPUT_TTL_DAYS, prefix=prefix)


def should_profile(tasks: str, index: int) -> bool:
    """Whether *index* (see :func:`selection_index`) is selected by the ``--tasks`` spec.

    Args:
        tasks: A ``TaskSelector`` value, or a comma-separated list of indices.
        index: This unit's index within the job.

    Raises:
        ValueError: If the spec is neither a selector nor a list of integers.
    """
    if tasks == TaskSelector.ALL:
        return True
    if tasks == TaskSelector.FIRST:
        return index == 0
    try:
        selected = {int(part) for part in tasks.split(",") if part.strip()}
    except ValueError as e:
        options = ", ".join(TaskSelector)
        raise ValueError(f"--tasks must be one of ({options}) or a comma-separated list, got {tasks!r}") from e
    return index in selected


def workdir() -> Path:
    """Return the task workdir, which roots the reports before upload."""
    return Path(os.environ.get("IRIS_WORKDIR", "."))


def resolve_nsys_bin() -> str:
    """Return the ``nsys`` binary from PATH — the iris task image bakes it in.

    Raises:
        RuntimeError: If ``nsys`` is not on PATH (a task image without it baked in).
    """
    on_path = shutil.which("nsys")
    if not on_path:
        raise RuntimeError("no nsys on PATH; run on the iris task image, which bakes it in")
    return on_path


def build_nsys_argv(nsys_bin: str, output_path: Path, trace: str, capture_range: bool) -> list[str]:
    """Build the ``nsys profile`` prefix for a selected rank."""
    argv = [
        nsys_bin,
        "profile",
        f"--trace={trace}",
        "--sample=none",
        "--cpuctxsw=none",
        "--force-overwrite=true",
        "-o",
        str(output_path),
    ]
    if capture_range:
        argv.extend(_CAPTURE_RANGE_ARGS)
    return argv


def report_path(output_dir: Path, index: int) -> Path:
    """Return this unit's report path.

    Every unit uploads into one directory, so the name carries identity: the selection
    index (a task, or a process under multigpu scope) and the host it ran on.
    """
    return output_dir / f"r{index:05d}-{socket.gethostname()}"


def upload_report(report: Path, output_uri: str) -> StoragePath:
    """Copy *report* into the *output_uri* directory and return where it landed.

    Streamed rather than read in one piece: a report can be hundreds of MB. The
    destination is a ``StoragePath`` so the join is structural (a trailing separator on
    the URI cannot double up) and the write routes through rigging's guarded factory,
    which is what puts a finite timeout on an S3 write — an unbounded one would wedge
    the task after the profiled work is already done.
    """
    destination = StoragePath(output_uri) / report.name
    with open(report, "rb") as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return destination


def _supervise(nsys_argv: Sequence[str], command: Sequence[str]) -> int:
    """Run nsys to completion, forwarding termination so it can finalize the report.

    Returns the child's exit code, with a signal death normalized to the conventional
    ``128 + signum``. ``Popen.wait`` reports those as a negative code, which ``sys.exit``
    would turn into a wrapping status (``-15`` becomes 241, not 143) and hide the
    termination behind a bogus application failure. ``iris.hooks.multigpu_main``
    normalizes the same way for the same reason.
    """
    proc = subprocess.Popen([*nsys_argv, *command])

    def forward(signum: int, frame: FrameType | None) -> None:
        proc.send_signal(signum)

    for sig in _FORWARDED_SIGNALS:
        signal.signal(sig, forward)
    returncode = proc.wait()
    return 128 - returncode if returncode < 0 else returncode


def run(tasks: str, trace: str, capture_range: bool, output_uri: str | None, argv: Sequence[str]) -> NoReturn:
    """Run *argv*, profiled by nsys when this task is selected.

    *argv* is the multi-process supervisor (or the command itself at
    ``processes_per_task=1``), so one report covers every GPU rank the task runs —
    nsys traces children. *output_uri* is where the report is uploaded; ``None`` resolves
    the cluster's temp bucket from the task env (see :func:`default_output_uri`).

    An unselected unit execs and never returns. A selected one supervises nsys so it
    can upload the report afterwards, then exits with the command's own status.
    """
    command = list(argv)
    index = selection_index()
    if not should_profile(tasks, index):
        logger.info("unit %d not selected by --tasks=%s; running unprofiled", index, tasks)
        os.execvp(command[0], command)

    # Resolve the destination before profiling, so a bad/unresolvable target fails now
    # rather than after the GPU work and the report are already produced.
    destination_dir = output_uri or default_output_uri()
    output_dir = workdir() / NSYS_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_path(output_dir, index)
    nsys_bin = resolve_nsys_bin()
    nsys_argv = build_nsys_argv(nsys_bin, output_path, trace, capture_range)
    # nsys stages its injection libraries in TMPDIR, and /tmp is mounted noexec.
    os.environ["TMPDIR"] = str(output_dir)
    logger.info("unit %d profiling to %s%s", index, output_path, _REPORT_SUFFIX)

    returncode = _supervise(nsys_argv, command)

    report = output_path.with_name(output_path.name + _REPORT_SUFFIX)
    if not report.exists():
        # A crash before nsys wrote anything is the usual reason, so the command's own
        # failure stays the reported one. But a command that *succeeded* with no report
        # must not pass as a successful profiling run: the unit would be recorded green
        # and its workdir dropped, having produced the one artifact it was asked for.
        logger.error("unit %d wrote no report at %s (command exited %d)", index, report, returncode)
        sys.exit(returncode or _NO_REPORT_EXIT)
    destination = upload_report(report, destination_dir)
    logger.info("unit %d uploaded %s (%.1f MB)", index, destination, report.stat().st_size / 1e6)
    sys.exit(returncode)


def main(argv: list[str] | None = None) -> NoReturn:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--" not in raw:
        raise SystemExit("usage: python -m iris.hooks.nsys_main [--tasks SPEC] [--output-uri URI] -- <command...>")
    split = raw.index("--")
    own_args, command = raw[:split], raw[split + 1 :]

    parser = argparse.ArgumentParser(prog="python -m iris.hooks.nsys_main")
    parser.add_argument("--tasks", default="first", help="'first', 'all', or a comma-separated list of indices")
    parser.add_argument("--trace", default="cuda,nvtx,cublas", help="nsys --trace value")
    parser.add_argument("--output-uri", default=None, help="report directory URI (default: the cluster temp bucket)")
    parser.add_argument(
        "--capture-range",
        action="store_true",
        help="collect only between cuProfilerStart/Stop instead of for the whole run",
    )
    args = parser.parse_args(own_args)
    run(args.tasks, args.trace, args.capture_range, args.output_uri, command)


if __name__ == "__main__":
    main()
