# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run several interp commands on one 8-GPU node, one GPU each.

Usage: ``python -m experiments.grug.fast_track.interp.fanout "<module> <args...>" "<module> <args...>" ...``. Command
``i`` runs ``python -m experiments.grug.fast_track.interp.<module> <args...>`` with ``CUDA_VISIBLE_DEVICES=i % 8``
(more than 8 commands queue for the next free GPU). Every line of a command's output is prefixed with its index. A
failed command is logged but does not fail the job (a job retry would rerun every command).
"""

import logging
import os
import queue
import shlex
import subprocess
import sys
import threading

logger = logging.getLogger(__name__)

NUM_GPUS = 8
PACKAGE = "experiments.grug.fast_track.interp"


def _pump(index: int, proc: subprocess.Popen) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(f"[{index}] {line}")
        sys.stdout.flush()


def _run(index: int, gpu: int, command: str) -> int:
    module, *args = shlex.split(command)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    proc = subprocess.Popen(
        [sys.executable, "-m", f"{PACKAGE}.{module}", *args],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _pump(index, proc)
    code = proc.wait()
    logger.info("command %d (gpu %d) exited %d: %s", index, gpu, code, command)
    return code


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    commands = sys.argv[1:]
    if not commands:
        raise SystemExit("usage: fanout '<module> <args...>' ...")
    free: queue.Queue[int] = queue.Queue()
    for gpu in range(NUM_GPUS):
        free.put(gpu)
    codes: dict[int, int] = {}

    def worker(index: int, command: str) -> None:
        gpu = free.get()
        try:
            codes[index] = _run(index, gpu, command)
        finally:
            free.put(gpu)

    threads = [threading.Thread(target=worker, args=(i, c)) for i, c in enumerate(commands)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    failed = {i: c for i, c in codes.items() if c != 0}
    logger.info("fanout done: %d commands, failed %s", len(commands), failed)


if __name__ == "__main__":
    main()
