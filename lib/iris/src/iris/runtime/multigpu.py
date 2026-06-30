# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-node multi-process supervisor: run one JAX process per device group.

``python -m iris.runtime.multigpu --nproc N [--devices-per-proc D] -- <argv>``
spawns N copies of ``<argv>`` inside a single Iris task, each pinned to a
contiguous group of D local accelerator devices. It is the GPU analogue of
``srun``/``torchrun`` scoped to one host: the children share the pod's IPC
namespace and ``/dev/shm``, so NCCL keeps the intra-node NVLink P2P that
separate single-GPU pods cannot use.

Each child inherits the supervisor's environment plus its rank::

    JAX_PROCESS_COUNT    = num_tasks * nproc           (global world size)
    JAX_PROCESS_INDEX    = task_index * nproc + local  (global rank)
    JAX_LOCAL_DEVICE_IDS = the child's D device ids     ("0", or "2,3")

``iris.runtime.jax_init.initialize_jax`` reads these and joins the JAX mesh.

The supervisor owns child lifecycle: it forwards SIGINT/SIGTERM to every child,
tears the group down and exits non-zero if any child fails, and prefixes each
child's output with its local rank. It deliberately does not import jax — the
children initialize CUDA only after the supervisor has already spawned them, so
no CUDA context exists in the supervisor's address space.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
from types import FrameType

from iris.cluster.client.job_info import get_job_info

logger = logging.getLogger("iris.multigpu")

_SHUTDOWN_SIGNALS = (signal.SIGINT, signal.SIGTERM)
# How often to re-poll child liveness while waiting for the group (seconds).
_REAP_POLL_INTERVAL = 1.0


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
        "JAX_PROCESS_COUNT": str(num_tasks * nproc),
        "JAX_PROCESS_INDEX": str(task_index * nproc + local_rank),
        "JAX_LOCAL_DEVICE_IDS": device_ids,
    }


def _pump_output(local_rank: int, stream, write_lock: threading.Lock) -> None:
    """Forward a child's merged stdout/stderr, line-prefixed with its rank."""
    prefix = f"[rank{local_rank}] "
    for line in iter(stream.readline, ""):
        with write_lock:
            sys.stdout.write(prefix + line)
            sys.stdout.flush()
    stream.close()


def _terminate(children: list[subprocess.Popen], ranks) -> None:
    for rank in ranks:
        child = children[rank]
        if child.poll() is None:
            child.terminate()


def run(nproc: int, devices_per_proc: int, child_argv: list[str]) -> int:
    """Spawn ``nproc`` children, supervise them, return the process exit code.

    Returns the first non-zero child exit code (after tearing the rest down), or
    0 once every child exits cleanly.
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

    children: list[subprocess.Popen] = []
    pumps: list[threading.Thread] = []
    write_lock = threading.Lock()

    for local_rank in range(nproc):
        rank_env = _child_rank_env(local_rank, nproc, devices_per_proc, task_index, num_tasks)
        child_env = {**os.environ, **rank_env}
        logger.info(
            "launch local rank %d: JAX_PROCESS_INDEX=%s JAX_LOCAL_DEVICE_IDS=%s",
            local_rank,
            rank_env["JAX_PROCESS_INDEX"],
            rank_env["JAX_LOCAL_DEVICE_IDS"],
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

    terminating = threading.Event()

    def _forward_signal(signum: int, _frame: FrameType | None) -> None:
        terminating.set()
        logger.warning("received signal %d; forwarding to children", signum)
        for child in children:
            if child.poll() is None:
                child.send_signal(signum)

    for sig in _SHUTDOWN_SIGNALS:
        signal.signal(sig, _forward_signal)

    first_failure: int | None = None
    pending = set(range(nproc))
    while pending:
        for local_rank in list(pending):
            child = children[local_rank]
            try:
                code = child.wait(timeout=_REAP_POLL_INTERVAL)
            except subprocess.TimeoutExpired:
                continue
            pending.discard(local_rank)
            logger.info("local rank %d exited with code %d", local_rank, code)
            # A failure tears the whole group down: a collective job cannot make
            # progress once one rank is gone. A clean exit does not — peers run to
            # their own completion.
            if code != 0 and first_failure is None:
                first_failure = code
                if not terminating.is_set():
                    terminating.set()
                    logger.error(
                        "local rank %d failed (code %d); terminating %d peer(s)", local_rank, code, len(pending)
                    )
                    _terminate(children, pending)

    for pump in pumps:
        pump.join(timeout=5.0)

    return first_failure or 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--" not in raw:
        raise SystemExit("usage: python -m iris.runtime.multigpu --nproc N [--devices-per-proc D] -- <command...>")
    split = raw.index("--")
    own_args, child_argv = raw[:split], raw[split + 1 :]

    parser = argparse.ArgumentParser(prog="python -m iris.runtime.multigpu")
    parser.add_argument("--nproc", type=int, required=True, help="number of processes to launch on this host")
    parser.add_argument(
        "--devices-per-proc", type=int, default=1, help="local accelerator devices assigned to each process"
    )
    args = parser.parse_args(own_args)
    return run(args.nproc, args.devices_per_proc, child_argv)


if __name__ == "__main__":
    sys.exit(main())
