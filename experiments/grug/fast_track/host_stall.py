# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample every thread's Python stack while the train loop's post-step host work runs, and log where a slow stretch
spent its time. Single-process runs (one process driving all GPUs) see multi-second host stalls that the JAX profiler
window rarely catches; this names the frames that were running during each one."""

import collections
import logging
import sys
import threading
import time
import traceback

logger = logging.getLogger(__name__)

SAMPLE_INTERVAL = 0.02
TOP_STACKS = 3
STACK_DEPTH = 12
# Innermost frames of a thread that is blocked, not running: those stacks are left out of the report.
_IDLE_FRAMES = frozenset({"wait", "get", "select", "poll", "sleep", "_wait_for_tstate_lock", "accept", "recv", "read"})


_CPU_STAT = "/sys/fs/cgroup/cpu.stat"


def _cgroup_throttle() -> tuple[int, int] | None:
    """``(nr_throttled, throttled_usec)`` of this container's cgroup (v2), or None where it isn't readable."""
    try:
        with open(_CPU_STAT) as f:
            fields = dict(line.split() for line in f)
    except OSError:
        return None
    return int(fields.get("nr_throttled", 0)), int(fields.get("throttled_usec", 0))


def _stack_key(frame) -> str:
    frames = traceback.extract_stack(frame)[-STACK_DEPTH:]
    return " <- ".join(f"{f.name}({f.filename.rsplit('/', 1)[-1]}:{f.lineno})" for f in reversed(frames))


class HostStallSampler:
    """While armed, a daemon thread records each thread's innermost frames every ``SAMPLE_INTERVAL``; ``disarm`` logs
    the most frequent stacks when the armed stretch exceeded ``threshold`` seconds."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self._armed_at: float | None = None
        self._throttle_at_arm: tuple[int, int] | None = None
        self._samples: collections.Counter[tuple[str, str]] = collections.Counter()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="host-stall-sampler", daemon=True)

    def __enter__(self) -> "HostStallSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def arm(self) -> None:
        with self._lock:
            self._samples.clear()
            self._armed_at = time.perf_counter()
            self._throttle_at_arm = _cgroup_throttle()

    def disarm(self, step: int) -> None:
        with self._lock:
            armed_at, self._armed_at = self._armed_at, None
            samples = self._samples.copy()
        if armed_at is None or time.perf_counter() - armed_at <= self.threshold:
            return
        by_thread: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
        for (thread, stack), n in samples.items():
            by_thread[thread][stack] += n
        lines = [f"host stall {time.perf_counter() - armed_at:.3f} s after step {step}"]
        now, before = _cgroup_throttle(), self._throttle_at_arm
        if now is not None and before is not None:
            lines[0] += f"; cgroup throttled {now[0] - before[0]} periods, {(now[1] - before[1]) / 1e6:.3f} s"
        for thread, stacks in by_thread.items():
            ticks = sum(stacks.values())
            busy = [(stack, n) for stack, n in stacks.most_common() if stack.split("(", 1)[0] not in _IDLE_FRAMES]
            for stack, n in busy[:TOP_STACKS]:
                lines.append(f"  [{thread}] {100 * n / ticks:.0f}% of {ticks} samples: {stack}")
        logger.warning("\n".join(lines))

    def _run(self) -> None:
        own = threading.get_ident()
        names = {}
        while not self._stop.wait(SAMPLE_INTERVAL):
            with self._lock:
                if self._armed_at is None:
                    continue
            names.update({t.ident: t.name for t in threading.enumerate()})
            frames = sys._current_frames()
            with self._lock:
                if self._armed_at is None:
                    continue
                for ident, frame in frames.items():
                    if ident != own:
                        self._samples[(names.get(ident, str(ident)), _stack_key(frame))] += 1
