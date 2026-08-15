# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""In-pod supervisor that runs a trainer subprocess and recovers it on a hang.

The supervisor is the long-lived parent process inside an Iris task. It never
creates a CUDA context (it makes no JAX device calls, so the child subprocess
owns the GPUs); it spawns the trainer as a subprocess
(``python -m levanter.recovery.child``), watches it two ways —

- exit-code contract (``levanter.recovery.types.classify_returncode``): the XLA
  hang watchdog's ``LOG(FATAL)`` shows up as a fatal signal, the sticky-error
  sentinel exits 125, the progress watchdog exits 124;
- a deadman on a heartbeat file the trainer touches every step, which catches a
  frozen (``SIGSTOP``-style) process that never exits on its own —

and on a recoverable fault kills the child's process group, gates on GPU health,
and re-execs it. The re-exec'd trainer restores from the host snapshot on its
own, so a fault costs a process restart, not a scheduler round-trip.

This implements the single-node, full-restart tier. The multi-process
survivor-in-place tier layers on top via the recoverability recipe in
``levanter.recovery.detection`` and cross-node coordination; it is out of scope
here.
"""

from __future__ import annotations

import logging
import os
import pickle
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

from levanter.recovery.detection import DetectionConfig, recovery_xla_env
from levanter.recovery.faults import ENV_FAULT_KIND, ENV_FAULT_STEP
from levanter.recovery.snapshot import remove_snapshot
from levanter.recovery.types import (
    DEFAULT_TMPFS_DIR,
    RECOVERABLE,
    ChildSpec,
    FaultClass,
    FaultRecord,
    RunOutcome,
    RunResult,
    classify_returncode,
)


logger = logging.getLogger(__name__)

ENV_HEARTBEAT_PATH = "LEVANTER_HEARTBEAT_PATH"
ENV_SNAPSHOT_TMPFS_BASE = "LEVANTER_SNAPSHOT_TMPFS_BASE"
ENV_RUN_ID = "LEVANTER_RUN_ID"
ENV_ATTEMPT = "LEVANTER_RECOVERY_ATTEMPT"

_CHILD_MODULE = "levanter.recovery.child"
_KILL_ESCALATION_GRACE = 10.0


class GPUHangSupervisor:
    """Context manager that runs and recovers trainer subprocesses.

    Reuse one supervisor across many ``run()`` calls to sweep ablations under a
    single warm pod: each call spawns a fresh subprocess (so process-start env
    vars like ``XLA_FLAGS`` take effect) while the host snapshot and compilation
    cache persist on the node, so successive runs skip the scheduler and cold
    compile.
    """

    def __init__(
        self,
        *,
        detection: DetectionConfig,
        deadman_timeout: float,
        snapshot_tmpfs_base: str = DEFAULT_TMPFS_DIR,
        poll_interval: float = 2.0,
        startup_timeout: float = 1800.0,
        max_restarts_per_run: int = 3,
        gpu_health_gate: bool = True,
        work_dir: str | None = None,
    ):
        if deadman_timeout <= 0:
            raise ValueError("deadman_timeout must be positive")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self.detection = detection
        self.deadman_timeout = deadman_timeout
        self.snapshot_tmpfs_base = snapshot_tmpfs_base
        self.poll_interval = poll_interval
        self.startup_timeout = startup_timeout
        self.max_restarts_per_run = max_restarts_per_run
        self.gpu_health_gate = gpu_health_gate
        self._owns_work_dir = work_dir is None
        self._work_dir = work_dir or tempfile.mkdtemp(prefix="levanter-supervisor-")
        os.makedirs(self._work_dir, exist_ok=True)
        self._active_child: subprocess.Popen | None = None

    def __enter__(self) -> "GPUHangSupervisor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._kill_active_child()
        if self._owns_work_dir:
            shutil.rmtree(self._work_dir, ignore_errors=True)

    # -- public API ------------------------------------------------------

    def run(
        self,
        entrypoint: Callable[[Any], None],
        config: Any,
        *,
        label: str,
        env: dict[str, str] | None = None,
        deadman_timeout: float | None = None,
    ) -> RunResult:
        """Run ``entrypoint(config)`` in a subprocess, recovering hangs.

        ``entrypoint`` must be an importable top-level function; only its module
        and qualified name cross the process boundary (``config`` is pickled).
        ``env`` overlays the child environment (ablation knobs, fault injection).
        ``deadman_timeout`` overrides the supervisor default for this run (e.g. to
        isolate the XLA execution deadman by disabling the host deadman).
        Returns when the run completes, fails unrecoverably, or exhausts its
        restart budget.
        """
        deadman_timeout = deadman_timeout if deadman_timeout is not None else self.deadman_timeout
        heartbeat_path = os.path.join(self._work_dir, f"{label}.heartbeat")
        spec_path = self._write_spec(entrypoint, config, label)
        started = time.monotonic()
        faults: list[FaultRecord] = []
        attempt = 0
        last_step: int | None = None

        while True:
            attempt += 1
            self._clear_heartbeat(heartbeat_path)
            child_env = self._build_child_env(env, heartbeat_path, label, attempt)
            logger.info("run[%s] attempt %d: spawning trainer subprocess", label, attempt)
            child = subprocess.Popen(
                [sys.executable, "-m", _CHILD_MODULE, "--spec", spec_path],
                env=child_env,
                start_new_session=True,
            )
            self._active_child = child

            returncode, deadman_fired = self._monitor(child, heartbeat_path, deadman_timeout)
            last_step = _read_heartbeat_step(heartbeat_path) or last_step
            self._active_child = None
            fault_class = classify_returncode(returncode, deadman=deadman_fired)
            wall = time.monotonic() - started

            if fault_class is FaultClass.NONE:
                logger.info("run[%s] completed cleanly at step %s (attempt %d)", label, last_step, attempt)
                # Free the host snapshot now the run is done, so a completed ablation
                # does not hold multi-GiB of tmpfs while the next label runs.
                remove_snapshot(label, self.snapshot_tmpfs_base)
                return RunResult(
                    label=label,
                    outcome=RunOutcome.COMPLETED,
                    attempts=attempt,
                    final_step=last_step,
                    total_wall_time=wall,
                    faults=faults,
                    restored_from_snapshot=attempt > 1,
                )

            faults.append(
                FaultRecord(
                    attempt=attempt,
                    fault_class=fault_class,
                    returncode=returncode,
                    wall_time=wall,
                    last_step=last_step,
                    detail="deadman" if deadman_fired else "",
                )
            )
            logger.warning(
                "run[%s] attempt %d faulted: class=%s returncode=%d last_step=%s",
                label,
                attempt,
                fault_class,
                returncode,
                last_step,
            )

            if fault_class is FaultClass.HARD:
                logger.error("run[%s] unrecoverable application failure; not restarting", label)
                return self._failed(label, RunOutcome.FAILED, attempt, last_step, wall, faults)

            if fault_class in RECOVERABLE:
                if self.gpu_health_gate and not _gpu_is_healthy():
                    logger.error("run[%s] GPU health gate failed; escalating to scheduler", label)
                    return self._failed(label, RunOutcome.UNHEALTHY_GPU, attempt, last_step, wall, faults)
                if attempt > self.max_restarts_per_run:
                    logger.error("run[%s] restart budget (%d) exhausted", label, self.max_restarts_per_run)
                    return self._failed(label, RunOutcome.FAILED, attempt, last_step, wall, faults)
                logger.info("run[%s] warm-restarting from snapshot (attempt %d)", label, attempt + 1)
                continue

            return self._failed(label, RunOutcome.FAILED, attempt, last_step, wall, faults)

    # -- monitoring ------------------------------------------------------

    def _monitor(self, child: subprocess.Popen, heartbeat_path: str, deadman_timeout: float) -> tuple[int, bool]:
        """Block until the child exits or the deadman fires.

        Returns ``(returncode, deadman_fired)``. During startup (before the first
        heartbeat) the deadman is disabled but a ``startup_timeout`` bounds a
        child that never begins stepping.
        """
        spawn_time = time.monotonic()
        seen_heartbeat = False
        while True:
            returncode = child.poll()
            if returncode is not None:
                return returncode, False

            hb_mtime = _heartbeat_mtime(heartbeat_path)
            now = time.monotonic()
            if hb_mtime is not None:
                seen_heartbeat = True
                age = time.time() - hb_mtime
                if age > deadman_timeout:
                    deadman_fired = self._kill_process_group(child)
                    if deadman_fired:
                        logger.warning("deadman: no heartbeat for %.1fs (> %.1fs); killed child", age, deadman_timeout)
                    return child.wait(), deadman_fired
            elif not seen_heartbeat and now - spawn_time > self.startup_timeout:
                deadman_fired = self._kill_process_group(child)
                if deadman_fired:
                    logger.warning("startup: no first heartbeat within %.1fs; killed child", self.startup_timeout)
                return child.wait(), deadman_fired

            time.sleep(self.poll_interval)

    # -- process lifecycle ----------------------------------------------

    def _kill_process_group(self, child: subprocess.Popen) -> bool:
        """Terminate the child's group and report whether the supervisor sent a signal."""
        if child.poll() is not None:
            return False
        try:
            pgid = os.getpgid(child.pid)
        except ProcessLookupError:
            return False
        sent_signal = False
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return sent_signal
            sent_signal = True
            try:
                child.wait(timeout=_KILL_ESCALATION_GRACE)
                return sent_signal
            except subprocess.TimeoutExpired:
                logger.warning("child did not exit on %s; escalating", sig.name)
        logger.error("child survived SIGKILL escalation; giving up on clean kill")
        return sent_signal

    def _kill_active_child(self) -> None:
        if self._active_child is not None and self._active_child.poll() is None:
            self._kill_process_group(self._active_child)
        self._active_child = None

    # -- env / spec ------------------------------------------------------

    def _build_child_env(
        self, ablation_env: dict[str, str] | None, heartbeat_path: str, label: str, attempt: int
    ) -> dict[str, str]:
        base = {**os.environ, **(ablation_env or {})}
        # A restart must not re-fire the injected fault, or it would loop forever;
        # the point of a restart is a clean run from the snapshot.
        if attempt > 1:
            base.pop(ENV_FAULT_KIND, None)
            base.pop(ENV_FAULT_STEP, None)
        env = recovery_xla_env(self.detection, base)
        env[ENV_HEARTBEAT_PATH] = heartbeat_path
        env[ENV_SNAPSHOT_TMPFS_BASE] = self.snapshot_tmpfs_base
        env[ENV_RUN_ID] = label
        env[ENV_ATTEMPT] = str(attempt)
        return env

    def _write_spec(self, entrypoint: Callable[[Any], None], config: Any, label: str) -> str:
        spec = ChildSpec(
            entry_module=entrypoint.__module__,
            entry_qualname=entrypoint.__qualname__,
            config=config,
        )
        spec_path = os.path.join(self._work_dir, f"{label}.spec.pkl")
        with open(spec_path, "wb") as f:
            pickle.dump(spec, f)
        return spec_path

    @staticmethod
    def _clear_heartbeat(path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    @staticmethod
    def _failed(
        label: str,
        outcome: RunOutcome,
        attempt: int,
        last_step: int | None,
        wall: float,
        faults: list[FaultRecord],
    ) -> RunResult:
        return RunResult(
            label=label,
            outcome=outcome,
            attempts=attempt,
            final_step=last_step,
            total_wall_time=wall,
            faults=faults,
        )


def _heartbeat_mtime(path: str) -> float | None:
    try:
        return os.stat(path).st_mtime
    except FileNotFoundError:
        return None


def _read_heartbeat_step(path: str) -> int | None:
    try:
        with open(path) as f:
            content = f.read().split()
    except (FileNotFoundError, ValueError):
        return None
    if not content:
        return None
    try:
        return int(content[0])
    except ValueError:
        return None


def _gpu_is_healthy(timeout: float = 30.0) -> bool:
    """Cheap host-side GPU liveness gate.

    ``nvidia-smi -L`` hanging or erroring is itself a strong unhealthy signal (a
    wedged driver stops answering). A poisoned but live GPU (Xid 13, RESTART_APP
    class) still answers, so a passing gate correctly permits a same-node
    restart.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        logger.warning("nvidia-smi not found; skipping GPU health gate")
        return True
    except subprocess.TimeoutExpired:
        logger.error("nvidia-smi -L timed out after %.0fs; driver may be wedged", timeout)
        return False
    if result.returncode != 0:
        logger.error("nvidia-smi -L returned %d: %s", result.returncode, result.stderr.decode(errors="replace"))
        return False
    return True
