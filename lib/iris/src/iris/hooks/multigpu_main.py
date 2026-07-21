# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-node multi-process supervisor: run one JAX process per device group.

``python -m iris.hooks.multigpu_main --nproc N [--devices-per-proc D] [--wrap 'CMD'] -- <argv>``
spawns N copies of ``<argv>`` inside a single Iris task, each pinned to a
contiguous group of D local accelerator devices. It is the GPU analogue of
``srun``/``torchrun`` scoped to one host: the children share the pod's IPC
namespace and ``/dev/shm``, so NCCL keeps the intra-node NVLink P2P that
separate single-GPU pods cannot use.

Invoked by the caller (or a launcher) as part of the command — iris is a dumb
scheduler and does not inject it. ``--wrap 'CMD'`` prefixes each child with a
wrapper (e.g. ``python -m iris.hooks.nsys_main --tasks first``) so a per-process
profiler sees each child's own ``IRIS_MULTIGPU_PROCESS_INDEX``; a whole-task
wrapper is composed outside this process instead.

Each child inherits the supervisor's environment plus its rank::

    IRIS_MULTIGPU_PROCESS_COUNT    = num_tasks * nproc           (global world size)
    IRIS_MULTIGPU_PROCESS_INDEX    = task_index * nproc + local  (global rank)
    IRIS_MULTIGPU_LOCAL_DEVICE_IDS = the child's D device ids     ("0", or "2,3")

``iris.runtime.jax_init.initialize_jax`` reads these (their names are the contract
this module defines) and joins the JAX mesh. The names are iris-private (not the
JAX_*/framework namespace) so a job that already sets JAX rank vars never trips
the supervised path.

The supervisor owns child lifecycle: it forwards SIGINT/SIGTERM to every child,
tears the group down and exits non-zero if any child fails, and prefixes each
child's output with its local rank. It deliberately does not import jax — the
children initialize CUDA only after the supervisor has already spawned them, so
no CUDA context exists in the supervisor's address space.
"""

import argparse
import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
from types import FrameType

from rigging.timing import Deadline, Duration

from iris.cluster.client.job_info import get_job_info
from iris.hooks.multigpu import (
    IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV,
    IRIS_MULTIGPU_PROCESS_COUNT_ENV,
    IRIS_MULTIGPU_PROCESS_INDEX_ENV,
)

logger = logging.getLogger("iris.multigpu")

_SHUTDOWN_SIGNALS = (signal.SIGINT, signal.SIGTERM)
# How often to re-poll child liveness while waiting for the group (seconds).
_REAP_POLL_INTERVAL = 1.0
# Grace period after a SIGTERM before escalating to SIGKILL, so a child that
# traps or ignores SIGTERM cannot wedge the supervisor (and hence the task).
_TERMINATE_GRACE = Duration.from_seconds(10.0)


def _child_rank_env(
    local_rank: int, nproc: int, devices_per_proc: int, task_index: int, num_tasks: int
) -> dict[str, str]:
    """Compute the rank env vars handed to one child process.

    Device ids are a contiguous slice so rank ``i`` owns devices
    ``[i*D, (i+1)*D)``; the common case ``D == 1`` gives one device per process.
    """
    begin = local_rank * devices_per_proc
    device_ids = ",".join(str(d) for d in range(begin, begin + devices_per_proc))
    return {
        IRIS_MULTIGPU_PROCESS_COUNT_ENV: str(num_tasks * nproc),
        IRIS_MULTIGPU_PROCESS_INDEX_ENV: str(task_index * nproc + local_rank),
        IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV: device_ids,
    }


def _pump_output(local_rank: int, stream, write_lock: threading.Lock) -> None:
    """Forward a child's merged stdout/stderr, line-prefixed with its rank."""
    prefix = f"[rank{local_rank}] "
    for line in iter(stream.readline, ""):
        with write_lock:
            sys.stdout.write(prefix + line)
            sys.stdout.flush()
    stream.close()


def _signal_children(children: list[subprocess.Popen], ranks, sig: int) -> None:
    """Send ``sig`` to every still-running child in ``ranks``."""
    for rank in ranks:
        child = children[rank]
        if child.poll() is None:
            child.send_signal(sig)


def _spawn_children(
    nproc: int, devices_per_proc: int, task_index: int, num_tasks: int, child_argv: list[str]
) -> tuple[list[subprocess.Popen], list[threading.Thread]]:
    """Spawn all children, or SIGKILL any that started and re-raise on failure.

    A partial spawn (rank 0 up, rank 3 fails on ``EAGAIN`` / a missing binary /
    thread-creation failure) would otherwise leave the started ranks orphaned,
    holding GPUs after the supervisor exits.
    """
    children: list[subprocess.Popen] = []
    pumps: list[threading.Thread] = []
    write_lock = threading.Lock()
    try:
        for local_rank in range(nproc):
            rank_env = _child_rank_env(local_rank, nproc, devices_per_proc, task_index, num_tasks)
            child_env = {**os.environ, **rank_env}
            logger.info(
                "launch local rank %d: %s=%s %s=%s",
                local_rank,
                IRIS_MULTIGPU_PROCESS_INDEX_ENV,
                rank_env[IRIS_MULTIGPU_PROCESS_INDEX_ENV],
                IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV,
                rank_env[IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV],
            )
            child = subprocess.Popen(
                child_argv,
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            children.append(child)
            pump = threading.Thread(target=_pump_output, args=(local_rank, child.stdout, write_lock), daemon=True)
            pump.start()
            pumps.append(pump)
    except BaseException:
        logger.error("spawn failed after starting %d/%d child(ren); killing them", len(children), nproc)
        _signal_children(children, range(len(children)), signal.SIGKILL)
        for child in children:
            try:
                child.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass
        raise
    return children, pumps


def run(nproc: int, devices_per_proc: int, child_argv: list[str]) -> int:
    """Spawn ``nproc`` children, supervise them, return the process exit code.

    ``128 + signum`` if a SIGINT/SIGTERM was delivered to the supervisor (an
    external termination/preemption); this takes precedence because the children
    it forwards the signal to report their own (negative) signal codes, which
    must not be surfaced as a task failure. Otherwise the first non-zero child
    exit code (after tearing the rest down), or 0 once every child exits cleanly.
    """
    if nproc < 1:
        raise ValueError(f"--nproc must be >= 1, got {nproc}")
    if devices_per_proc < 1:
        raise ValueError(f"--devices-per-proc must be >= 1, got {devices_per_proc}")
    if not child_argv:
        raise ValueError("no child command given after '--'")

    job_info = get_job_info()
    num_tasks = job_info.num_tasks if job_info else 1
    task_index = job_info.task_index if job_info else 0
    logger.info(
        "supervising %d process(es) x %d device(s) each; task_index=%d num_tasks=%d; command=%s",
        nproc,
        devices_per_proc,
        task_index,
        num_tasks,
        " ".join(child_argv),
    )

    children, pumps = _spawn_children(nproc, devices_per_proc, task_index, num_tasks, child_argv)

    terminating = threading.Event()
    shutdown_signum: int | None = None
    # Deadline after which surviving children are SIGKILLed; ``None`` until a
    # teardown begins. Shared between the signal handler and the reap loop, both
    # of which run in the main thread (no lock needed).
    kill_deadline: Deadline | None = None

    def _forward_signal(signum: int, _frame: FrameType | None) -> None:
        nonlocal shutdown_signum, kill_deadline
        if shutdown_signum is None:
            shutdown_signum = signum
        terminating.set()
        logger.warning("received signal %d; forwarding to children", signum)
        _signal_children(children, range(nproc), signum)
        if kill_deadline is None:
            kill_deadline = Deadline.from_now(_TERMINATE_GRACE)

    for sig in _SHUTDOWN_SIGNALS:
        signal.signal(sig, _forward_signal)

    first_failure: int | None = None
    pending = set(range(nproc))
    while pending:
        # Escalate to SIGKILL if a requested teardown overran its grace period,
        # so a child that traps or ignores SIGTERM cannot wedge the supervisor.
        if kill_deadline is not None and kill_deadline.expired():
            survivors = [r for r in pending if children[r].poll() is None]
            if survivors:
                logger.error("SIGTERM grace expired; SIGKILL %d straggler(s): %s", len(survivors), survivors)
                _signal_children(children, survivors, signal.SIGKILL)
            kill_deadline = None  # SIGKILL is final; do not re-fire each tick
        for local_rank in list(pending):
            child = children[local_rank]
            try:
                code = child.wait(timeout=_REAP_POLL_INTERVAL)
            except subprocess.TimeoutExpired:
                continue
            pending.discard(local_rank)
            logger.info("local rank %d exited with code %d", local_rank, code)
            if code != 0 and first_failure is None:
                first_failure = code
            # A failure tears the whole group down: a collective job cannot make
            # progress once one rank is gone. A clean exit does not — peers run to
            # their own completion.
            if code != 0 and not terminating.is_set():
                terminating.set()
                logger.error("local rank %d failed (code %d); terminating %d peer(s)", local_rank, code, len(pending))
                _signal_children(children, pending, signal.SIGTERM)
                kill_deadline = Deadline.from_now(_TERMINATE_GRACE)

    for pump in pumps:
        pump.join(timeout=5.0)

    # An external signal wins over child exit codes: the children we forwarded it
    # to report negative (signal) codes that are an artifact of the teardown, not
    # a task failure. A child failing on its own never sets shutdown_signum, so
    # its genuine non-zero code surfaces.
    if shutdown_signum is not None:
        return 128 + shutdown_signum
    if first_failure is not None:
        return first_failure
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--" not in raw:
        raise SystemExit(
            "usage: python -m iris.hooks.multigpu_main --nproc N [--devices-per-proc D] "
            "[--wrap 'CMD'] -- <command...>"
        )
    split = raw.index("--")
    own_args, child_argv = raw[:split], raw[split + 1 :]

    parser = argparse.ArgumentParser(prog="python -m iris.hooks.multigpu_main")
    parser.add_argument("--nproc", type=int, required=True, help="number of processes to launch on this host")
    parser.add_argument(
        "--devices-per-proc", type=int, default=1, help="local accelerator devices assigned to each process"
    )
    parser.add_argument(
        "--wrap",
        default=None,
        help="wrap each child in this command (e.g. 'python -m iris.hooks.nsys_main --tasks first') — for "
        "per-process profiling, where the wrapper sees each child's own IRIS_MULTIGPU_PROCESS_INDEX",
    )
    args = parser.parse_args(own_args)
    # A per-child wrapper (process-scope) goes *inside* the supervisor: each child runs
    # `<wrap> -- <command>`. A whole-task wrapper (node-scope) is composed the other way,
    # outside this process, so it needs nothing here.
    if args.wrap:
        child_argv = [*shlex.split(args.wrap), "--", *child_argv]
    return run(args.nproc, args.devices_per_proc, child_argv)


if __name__ == "__main__":
    sys.exit(main())
