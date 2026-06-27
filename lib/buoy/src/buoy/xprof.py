# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-run ``xprof`` subprocess lifecycle.

The viewer embeds the real xprof UI, so the service runs one xprof process per
profile, each reading a LOCAL copy of the run's ``jax_profile`` logdir (xprof
cannot read ``gs://``). Processes are evicted by **last-proxied-request** time;
at the configured cap with everything active, an open raises
:class:`XprofCapacityError` (HTTP 503) rather than killing a live session.

Cold startup — download the logdir (~hundreds of MB) + spawn xprof — can exceed
the controller proxy's 30s request cap, so it never runs inside a request:
:meth:`start_prepare` does it on a background thread and :meth:`prepare_status`
is polled by the SPA, which only injects the iframe once a profile is ``ready``.
By then :meth:`ensure` (called per proxied request) hits the warm fast path.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

from iris.cluster.backends.types import find_free_port, wait_for_port

from buoy import cache
from buoy.config import BuoyConfig

logger = logging.getLogger("buoy.xprof")

# A process touched within this window is considered active and is never evicted.
EVICT_GRACE = 30.0


class XprofCapacityError(RuntimeError):
    """Raised when every xprof process is active and none can be evicted."""


@dataclass
class _Proc:
    proc: subprocess.Popen
    port: int
    last_used: float


@dataclass
class _PrepState:
    state: str  # "preparing" | "ready" | "error"
    error: str | None = None


class XprofManager:
    def __init__(self, cfg: BuoyConfig) -> None:
        self._cfg = cfg
        self._guard = threading.Lock()
        self._procs: dict[str, _Proc] = {}
        self._launching: set[str] = set()
        self._launch_locks: dict[str, threading.Lock] = {}
        self._prepare: dict[str, _PrepState] = {}

    # ---- async prepare (the path the SPA drives) -------------------------------

    def start_prepare(self, run_key: str, gcs_logdir: str) -> None:
        """Kick off (download + launch) on a background thread; idempotent."""
        with self._guard:
            running = run_key in self._procs and self._procs[run_key].proc.poll() is None
            if running:
                self._prepare[run_key] = _PrepState("ready")
                return
            current = self._prepare.get(run_key)
            if current and current.state == "preparing":
                return
            self._prepare[run_key] = _PrepState("preparing")
        threading.Thread(target=self._prepare_worker, args=(run_key, gcs_logdir), daemon=True).start()

    def _prepare_worker(self, run_key: str, gcs_logdir: str) -> None:
        try:
            self.ensure(run_key, gcs_logdir)
            state = _PrepState("ready")
        except XprofCapacityError as exc:
            state = _PrepState("error", str(exc))
        except Exception as exc:
            logger.exception("xprof prepare failed for %s", run_key)
            state = _PrepState("error", str(exc))
        with self._guard:
            self._prepare[run_key] = state

    def prepare_status(self, run_key: str) -> dict:
        if self.port_if_running(run_key) is not None:
            return {"state": "ready"}
        with self._guard:
            current = self._prepare.get(run_key)
        if current is None:
            return {"state": "absent"}
        out = {"state": current.state}
        if current.error:
            out["error"] = current.error
        return out

    # ---- warm path (called per proxied request) --------------------------------

    def port_if_running(self, run_key: str) -> int | None:
        with self._guard:
            existing = self._procs.get(run_key)
            if existing and existing.proc.poll() is None:
                existing.last_used = time.time()
                return existing.port
        return None

    def ensure(self, run_key: str, gcs_logdir: str) -> int:
        """Return a live xprof port for ``run_key``, launching one if needed."""
        port = self.port_if_running(run_key)
        if port is not None:
            return port
        if not self._cfg.xprof_bin:
            raise RuntimeError("xprof binary not found; set BUOY_XPROF_BIN or install xprof")

        with self._launch_lock_for(run_key):
            port = self.port_if_running(run_key)  # another caller may have won the race
            if port is not None:
                return port
            local = self._materialize(run_key, gcs_logdir)
            # Reserve a slot atomically: capacity counts live procs + in-flight
            # launches, so two cold opens of different runs can't both exceed the cap.
            with self._guard:
                self._evict_locked()
                self._launching.add(run_key)
            try:
                proc, port = self._spawn(local)
            except BaseException:
                with self._guard:
                    self._launching.discard(run_key)
                raise
            with self._guard:
                self._launching.discard(run_key)
                self._procs[run_key] = _Proc(proc, port, time.time())
            logger.info("launched xprof for %s on :%d (logdir=%s)", run_key, port, local)
            return port

    def shutdown(self) -> None:
        with self._guard:
            for entry in self._procs.values():
                entry.proc.terminate()
            self._procs.clear()

    def _launch_lock_for(self, run_key: str) -> threading.Lock:
        with self._guard:
            return self._launch_locks.setdefault(run_key, threading.Lock())

    def _materialize(self, run_key: str, gcs_logdir: str) -> str:
        safe = run_key.replace("/", "_")
        local = os.path.join(self._cfg.local_profile_dir, safe)
        done = local + ".complete"
        # Reuse only a download that finished — a non-empty dir may be a partial
        # logdir from an interrupted pull, which would feed xprof bad data.
        if os.path.exists(done) and os.path.isdir(local):
            return local
        if os.path.isdir(local):
            shutil.rmtree(local)
        if os.path.exists(done):
            os.remove(done)
        cache.download_tree(gcs_logdir, local)
        open(done, "w").close()
        return local

    def _spawn(self, logdir: str) -> tuple[subprocess.Popen, int]:
        port = find_free_port()
        proc = subprocess.Popen(
            [self._cfg.xprof_bin, "--logdir", logdir, "--port", str(port)],  # type: ignore[list-item]
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not wait_for_port(port, "127.0.0.1", 60.0):
            proc.terminate()
            raise RuntimeError(f"xprof did not bind on :{port}")
        return proc, port

    def _evict_locked(self) -> None:
        for key in [k for k, v in self._procs.items() if v.proc.poll() is not None]:
            del self._procs[key]
        if len(self._procs) + len(self._launching) < self._cfg.max_xprof_procs:
            return
        if not self._procs:
            raise XprofCapacityError("all xprof slots are launching; try again shortly")
        victim_key = min(self._procs, key=lambda k: self._procs[k].last_used)
        victim = self._procs[victim_key]
        if time.time() - victim.last_used < EVICT_GRACE:
            raise XprofCapacityError("all xprof sessions are active; try again shortly")
        victim.proc.terminate()
        del self._procs[victim_key]
        logger.info("evicted idle xprof %s", victim_key)
