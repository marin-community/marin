# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared runner for the single-GPU hang probes under ``experiments/grug/recovery``.

A probe is a script that runs one arm in one process on one GPU and prints a ``ProbeResult``
line. This module supplies the two halves every probe needs: a watchdog that turns a stalled
iteration into a reported hang and a process exit, and a parent that fans the arms out over the
visible GPUs, streams the children's output, enforces a per-arm time budget, and aggregates the
results. The parent never imports JAX, because initializing it would preallocate every GPU's
memory away from the children.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

RESULT_PREFIX = "RESULT_JSON "
HANG_EXIT_CODE = 3
_TAIL_LINES = 30


@dataclass(frozen=True)
class ProbeResult:
    """One arm's outcome on one GPU.

    ``outcome`` is ``ok``, ``hang``, or ``exit <code>`` for a child that died without reporting.
    Counters that an arm does not measure stay at their zero defaults.
    """

    arm: str
    gpu: str
    outcome: str
    iterations: int = 0
    hangs: int = 0
    mismatches: int = 0
    nonfinite: int = 0
    worst_ratio: float = 0.0
    seconds_per_iter: float = float("nan")
    tail: tuple[str, ...] = field(default_factory=tuple)

    def line(self) -> str:
        return RESULT_PREFIX + json.dumps(asdict(self))

    @classmethod
    def parse(cls, line: str) -> ProbeResult:
        payload = json.loads(line[len(RESULT_PREFIX) :])
        payload["tail"] = tuple(payload.get("tail", ()))
        return cls(**payload)


def current_gpu() -> str:
    return os.environ.get("CUDA_VISIBLE_DEVICES", "?")


class Progress:
    """Iteration progress shared with the watchdog thread.

    Call :meth:`touch` before and after every device-synchronizing step. If ``timeout`` seconds
    pass without a touch, the watchdog prints a hang result for the current iteration and exits
    the process with :data:`HANG_EXIT_CODE`, so a deadlocked kernel becomes a counted event.
    """

    def __init__(self, arm: str, timeout: float):
        self._arm = arm
        self._timeout = timeout
        self._time = time.time()
        self.iteration = -1
        self._stop = threading.Event()
        threading.Thread(target=self._watch, daemon=True).start()

    def touch(self, iteration: int | None = None) -> None:
        if iteration is not None:
            self.iteration = iteration
        self._time = time.time()

    def finish(self) -> None:
        self._stop.set()

    def _watch(self) -> None:
        while not self._stop.wait(1.0):
            if time.time() - self._time > self._timeout:
                print(
                    ProbeResult(self._arm, current_gpu(), "hang", iterations=self.iteration, hangs=1).line(), flush=True
                )
                os._exit(HANG_EXIT_CODE)


def visible_gpus() -> list[int]:
    listing = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], check=True, text=True, capture_output=True
    )
    return [int(line) for line in listing.stdout.split() if line.strip()]


def _run_child(gpu: int, arm: str, command: Sequence[str], budget: float) -> ProbeResult:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # The child's own watchdog normally ends a hung arm; this timer is the backstop for a child
    # that stops producing output without exiting, which the stdout loop below would wait on forever.
    killer = threading.Timer(budget, proc.kill)
    killer.start()
    result = None
    tail: list[str] = []
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            tail = [*tail, line][-_TAIL_LINES:]
            if line.startswith(RESULT_PREFIX):
                result = ProbeResult.parse(line)
            print(f"[gpu{gpu} {arm}] {line}", flush=True)
        code = proc.wait()
    finally:
        killer.cancel()
    if result is None:
        return ProbeResult(arm, str(gpu), f"exit {code}", tail=tuple(tail[-10:]))
    return result


def run_arms(
    arms: Sequence[str],
    build_command: Callable[[int, str], list[str]],
    *,
    gpus: Sequence[int] | None = None,
    budget: float,
) -> list[ProbeResult]:
    """Run every arm on every GPU: arms sequentially per GPU, GPUs in parallel.

    ``build_command(gpu, arm)`` returns the child's argv. ``budget`` is the per-arm wall-clock
    limit after which the child is killed.
    """
    gpus = visible_gpus() if gpus is None else list(gpus)
    results: list[ProbeResult] = []
    lock = threading.Lock()

    def run_gpu(gpu: int) -> None:
        for arm in arms:
            result = _run_child(gpu, arm, build_command(gpu, arm), budget)
            with lock:
                results.append(result)

    threads = [threading.Thread(target=run_gpu, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def print_summary(results: Sequence[ProbeResult], arms: Sequence[str]) -> None:
    print("SUMMARY_JSON " + json.dumps([asdict(r) for r in results]), flush=True)
    for arm in arms:
        rows = [r for r in results if r.arm == arm]
        times = sorted(r.seconds_per_iter for r in rows if r.outcome == "ok")
        median = times[len(times) // 2] if times else float("nan")
        lo = times[0] if times else float("nan")
        hi = times[-1] if times else float("nan")
        print(
            f"ARM {arm}: runs={len(rows)} hangs={sum(r.hangs for r in rows)} "
            f"mismatches={sum(r.mismatches for r in rows)} nonfinite={sum(r.nonfinite for r in rows)} "
            f"other={sum(1 for r in rows if r.outcome not in ('ok', 'hang'))} "
            f"median_seconds_per_iter={median:.5f} min={lo:.5f} max={hi:.5f}",
            flush=True,
        )


def child_argv(script: str, extra: Sequence[str]) -> list[str]:
    return [sys.executable, os.path.abspath(script), *extra]
