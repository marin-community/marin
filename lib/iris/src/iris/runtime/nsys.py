# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-process Nsight Systems launch wrapper.

``python -m iris.runtime.nsys --ranks SPEC --output-uri URI -- <argv>`` runs ``<argv>``
under ``nsys profile`` when this process's rank is selected, and execs ``<argv>``
unchanged otherwise.

An unselected rank execs and so costs nothing. A selected rank cannot exec: the report
has to be uploaded once nsys has written it, and the task workdir is an emptyDir that
is destroyed with the pod, so a report left on disk is simply lost. It therefore
supervises the child and forwards signals, taking care that a terminated task still
lets nsys finalize its report.

Nsight has to wrap the process at launch: CUDA tracing is injected through
``CUDA_INJECTION64_PATH``, which the driver reads once at ``cuInit``. That is why
this is a submit-time wrapper rather than an arm of the attach-based profiler in
``iris.cluster.runtime.profile`` (py-spy/memray), which can join a live process.

One report is written per profiled process; there is no merged report. Selecting a
subset of ranks is therefore the norm at scale, not an optimization: every rank of a
128-GPU job would produce 128 multi-hundred-MB reports.

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
from dataclasses import dataclass
from enum import StrEnum
from glob import glob
from pathlib import Path
from types import FrameType
from typing import NoReturn

import fsspec.core

from iris.cluster.client.job_info import get_job_info
from iris.cluster.setup_scripts import NSYS_INSTALL_DIR, nsys_bin_glob
from iris.runtime.multigpu import (
    IRIS_MULTIGPU_PROCESS_COUNT_ENV,
    IRIS_MULTIGPU_PROCESS_INDEX_ENV,
)

logger = logging.getLogger("iris.nsys")

# Where reports land. Under the workdir so no new mount is needed; one file per rank.
NSYS_OUTPUT_DIR = "nsys"
# Collection stops at cuProfilerStop but the process keeps running, so an app that
# brackets a step window gets exactly that window and nothing else.
_CAPTURE_RANGE_ARGS = ("--capture-range=cudaProfilerApi", "--capture-range-end=stop")
# nsys writes <output>.nsys-rep once the profiled process exits.
_REPORT_SUFFIX = ".nsys-rep"
# Signals forwarded to nsys so a terminated task still finalizes its report.
_FORWARDED_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class RankSelector(StrEnum):
    """Which ranks write a report."""

    FIRST = "first"
    PER_NODE = "per-node"
    ALL = "all"


@dataclass(frozen=True)
class Rank:
    """This process's identity within the job."""

    global_rank: int
    local_rank: int

    @classmethod
    def from_env(cls) -> "Rank":
        """Read this process's rank from the task environment.

        Rank has two sources. ``iris.runtime.multigpu`` stamps a global
        ``IRIS_MULTIGPU_PROCESS_INDEX`` on each child it spawns, but only when
        ``processes_per_task > 1`` — it is a deliberate no-op at 1. With one process
        per task the task index is the rank, and every process is its own node leader.

        Raises:
            RuntimeError: If there is no iris task context to take a rank from.
        """
        info = get_job_info()
        if info is None:
            raise RuntimeError("no iris job context (IRIS_TASK_ID unset); nsys rank selection needs one")
        process_index = os.environ.get(IRIS_MULTIGPU_PROCESS_INDEX_ENV)
        if process_index is None:
            return cls(global_rank=info.task_index, local_rank=0)
        processes_per_task = int(os.environ[IRIS_MULTIGPU_PROCESS_COUNT_ENV]) // info.num_tasks
        global_rank = int(process_index)
        return cls(global_rank=global_rank, local_rank=global_rank % processes_per_task)


def should_profile(ranks: str, rank: Rank) -> bool:
    """Whether *rank* is selected by the ``--ranks`` spec.

    Args:
        ranks: A ``RankSelector`` value, or a comma-separated list of global ranks.
        rank: This process's identity.

    Raises:
        ValueError: If the spec is neither a selector nor a list of integers.
    """
    if ranks == RankSelector.ALL:
        return True
    if ranks == RankSelector.FIRST:
        return rank.global_rank == 0
    if ranks == RankSelector.PER_NODE:
        return rank.local_rank == 0
    try:
        selected = {int(part) for part in ranks.split(",") if part.strip()}
    except ValueError as e:
        options = ", ".join(RankSelector)
        raise ValueError(f"--ranks must be one of ({options}) or a comma-separated rank list, got {ranks!r}") from e
    return rank.global_rank in selected


def workdir() -> Path:
    """Return the task workdir, which roots both the nsys install and the reports."""
    return Path(os.environ.get("IRIS_WORKDIR", "."))


def resolve_nsys_bin(install_root: Path) -> str:
    """Return the ``nsys`` binary the setup script extracted under *install_root*.

    Raises:
        RuntimeError: If the setup script did not install one.
    """
    bin_glob = nsys_bin_glob(str(install_root))
    matches = sorted(glob(bin_glob))
    if not matches:
        raise RuntimeError(f"no nsys binary at {bin_glob}; was the nsight setup script run?")
    return matches[0]


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


def report_path(output_dir: Path, rank: Rank) -> Path:
    """Return this rank's report path. Ranks share a directory, so the name carries identity."""
    return output_dir / f"rank{rank.global_rank:05d}-{socket.gethostname()}"


def upload_report(report: Path, output_uri: str) -> str:
    """Copy *report* into the *output_uri* directory and return the destination URI.

    Streamed rather than read into memory: a report can be hundreds of MB.
    """
    destination = f"{output_uri.rstrip('/')}/{report.name}"
    with open(report, "rb") as src, fsspec.core.open(destination, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return destination


def _supervise(nsys_argv: Sequence[str], command: Sequence[str]) -> int:
    """Run nsys to completion, forwarding termination so it can finalize the report."""
    proc = subprocess.Popen([*nsys_argv, *command])

    def forward(signum: int, frame: FrameType | None) -> None:
        proc.send_signal(signum)

    for sig in _FORWARDED_SIGNALS:
        signal.signal(sig, forward)
    return proc.wait()


def run(ranks: str, trace: str, capture_range: bool, output_uri: str, argv: Sequence[str]) -> NoReturn:
    """Run *argv*, profiled by nsys when this rank is selected.

    An unselected rank execs and never returns. A selected rank supervises nsys so it
    can upload the report afterwards, then exits with the command's own status.
    """
    command = list(argv)
    rank = Rank.from_env()
    if not should_profile(ranks, rank):
        logger.info("rank %d not selected by --ranks=%s; running unprofiled", rank.global_rank, ranks)
        os.execvp(command[0], command)

    output_dir = workdir() / NSYS_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_path(output_dir, rank)
    nsys_bin = resolve_nsys_bin(workdir() / NSYS_INSTALL_DIR)
    nsys_argv = build_nsys_argv(nsys_bin, output_path, trace, capture_range)
    # nsys stages its injection libraries in TMPDIR, and /tmp is mounted noexec.
    os.environ["TMPDIR"] = str(output_dir)
    logger.info("rank %d profiling to %s%s", rank.global_rank, output_path, _REPORT_SUFFIX)

    returncode = _supervise(nsys_argv, command)

    report = output_path.with_name(output_path.name + _REPORT_SUFFIX)
    if not report.exists():
        # Don't mask the command's own failure; a crash before nsys wrote anything is
        # the usual reason, and the exit code is the more useful signal.
        logger.error("rank %d wrote no report at %s (command exited %d)", rank.global_rank, report, returncode)
        sys.exit(returncode)
    destination = upload_report(report, output_uri)
    logger.info("rank %d uploaded %s (%.1f MB)", rank.global_rank, destination, report.stat().st_size / 1e6)
    sys.exit(returncode)


def main(argv: list[str] | None = None) -> NoReturn:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--" not in raw:
        raise SystemExit("usage: python -m iris.runtime.nsys --ranks SPEC --output-uri URI -- <command...>")
    split = raw.index("--")
    own_args, command = raw[:split], raw[split + 1 :]

    parser = argparse.ArgumentParser(prog="python -m iris.runtime.nsys")
    parser.add_argument("--ranks", required=True, help="'first', 'per-node', 'all', or a comma-separated rank list")
    parser.add_argument("--trace", required=True, help="nsys --trace value (e.g. cuda,nvtx,cublas)")
    parser.add_argument("--output-uri", required=True, help="directory URI to upload each rank's report to")
    parser.add_argument(
        "--capture-range",
        action="store_true",
        help="collect only between cuProfilerStart/Stop instead of for the whole run",
    )
    args = parser.parse_args(own_args)
    run(args.ranks, args.trace, args.capture_range, args.output_uri, command)


if __name__ == "__main__":
    main()
