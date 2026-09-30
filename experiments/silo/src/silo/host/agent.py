# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host agent: the lifecycle of nested-container sandboxes on one Iris host job.

This module holds the logic the brief's invariants live in, kept free of HTTP and
of any particular container daemon so it can be tested against a fake runtime:

  * fresh, unique, never-recycled sandbox ids            (brief section 3.2)
  * deletion that is asynchronous and observably so      (brief section 3.5)
  * real cgroup limits: a CFS quota AND a cpuset         (brief section 3.7)
  * the polled session model -- polling never consumes output (brief section 2.2)
  * capacity that is explicit, so a full host says so instead of stalling

Allocation is DERIVED from the sandboxes this agent holds (plus creates in
flight), never kept as running counters. Counters drifted: the reaper re-ran a
deletion that was still in progress, both runs released the sandbox's resources,
and after days of churn the oldest hosts reported negative allocation -- so the
broker saw them as the emptiest hosts and sent them nearly every create until
``nerdctl run`` timed out under the load (2026-09-29, ~4,800 timeouts in 8 h).
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from silo.errors import SiloConflictError, SiloError, SiloNotFoundError, SiloRateLimitError
from silo.host.runtime import (
    BUSYBOX,
    RUNTIME_RUNSC,
    RUNTIMES,
    ContainerRuntime,
    ContainerSpec,
    ExecOutcome,
)
from silo.model import ImagePlan, ResourceProfile, SandboxRecord, utcnow

logger = logging.getLogger(__name__)

# Daytona-visible sandbox states. "started" is what a usable sandbox reports;
# "destroying" is the window in which a deleted sandbox is still observable.
STATE_STARTED = "started"
STATE_DESTROYING = "destroying"


@dataclass(frozen=True)
class HostConfig:
    host_id: str
    cpu_budget: int
    memory_budget_bytes: int
    disk_budget_bytes: int
    cpu_ids: tuple[int, ...]
    work_dir: Path
    # How many sandbox cores may share one physical core. 1.0 is honest: each
    # sandbox's cpuset is disjoint until the host is full.
    cpu_oversubscribe: float = 1.0
    default_runtime: str = RUNTIME_RUNSC

    def __post_init__(self) -> None:
        if not self.cpu_ids:
            raise ValueError("host has no cpus to allocate")
        if self.default_runtime not in RUNTIMES:
            raise ValueError(f"unknown runtime {self.default_runtime!r}")


class CpusetAllocator:
    """Hands out cpusets from the host's cores, least-loaded first.

    With oversubscription 1.0 the sets are disjoint until the host is full. Above
    that, cores are shared, but always spread so no core carries more sandboxes
    than any other by more than one.
    """

    def __init__(self, cpu_ids: tuple[int, ...]) -> None:
        self._load = {cpu: 0 for cpu in cpu_ids}
        self._lock = threading.Lock()

    def allocate(self, count: int) -> tuple[int, ...]:
        with self._lock:
            if count > len(self._load):
                raise SiloRateLimitError(f"sandbox needs {count} cpus; host has {len(self._load)}")
            chosen = sorted(self._load, key=lambda cpu: (self._load[cpu], cpu))[:count]
            for cpu in chosen:
                self._load[cpu] += 1
            return tuple(sorted(chosen))

    def release(self, cpuset: tuple[int, ...]) -> None:
        with self._lock:
            for cpu in cpuset:
                self._load[cpu] = max(0, self._load[cpu] - 1)


@dataclass
class _SessionCommand:
    cmd_id: str
    command: str
    stdout_path: Path
    stderr_path: Path
    process: subprocess.Popen[bytes] | None = None
    exit_code: int | None = None
    finished: threading.Event = field(default_factory=threading.Event)


@dataclass
class _Session:
    session_id: str
    commands: dict[str, _SessionCommand] = field(default_factory=dict)


@dataclass
class _Sandbox:
    record: SandboxRecord
    container_name: str
    runtime: str
    cpuset: tuple[int, ...]
    expires_at: float
    sessions: dict[str, _Session] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Set while a _finish_delete runs, so the reaper never starts a second one.
    deleting: bool = False


class HostAgent:
    def __init__(
        self,
        config: HostConfig,
        runtime: ContainerRuntime,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._runtime = runtime
        self._clock = clock
        self._cpusets = CpusetAllocator(config.cpu_ids)
        self._sandboxes: dict[str, _Sandbox] = {}
        # Every id this host has ever issued, including deleted ones. A lookup of
        # a deleted id is a not-found forever, never a fresh sandbox.
        self._retired: set[str] = set()
        self._lock = threading.Lock()
        # Creates past the capacity check but not yet in _sandboxes.
        self._creating: dict[str, ResourceProfile] = {}
        # Images this host has pulled or built, reported to the broker so it can
        # place (and rebuild FROM silo.local/...) where an image already is.
        self._images: set[str] = set()
        config.work_dir.mkdir(parents=True, exist_ok=True)

    @property
    def runtime(self) -> ContainerRuntime:
        return self._runtime

    # ------------------------------------------------------------------ #
    # Capacity
    # ------------------------------------------------------------------ #

    def _cpu_ceiling(self) -> float:
        return self.config.cpu_budget * self.config.cpu_oversubscribe

    def _allocated_locked(self) -> tuple[int, int, int]:
        """(cpu, memory bytes, disk bytes) held by sandboxes, destroying ones and creates in flight."""
        profiles = [s.record.profile for s in self._sandboxes.values()] + list(self._creating.values())
        return (
            sum(p.cpu for p in profiles),
            sum(p.memory_bytes for p in profiles),
            sum(p.disk_bytes for p in profiles),
        )

    def slots_for(self, profile: ResourceProfile) -> int:
        with self._lock:
            return self._slots_for_locked(profile)

    def _slots_for_locked(self, profile: ResourceProfile) -> int:
        cpu, memory, disk = self._allocated_locked()
        free_cpu = self._cpu_ceiling() - cpu
        free_mem = self.config.memory_budget_bytes - memory
        free_disk = self.config.disk_budget_bytes - disk
        return max(
            0,
            int(
                min(
                    free_cpu // profile.cpu,
                    free_mem // profile.memory_bytes,
                    free_disk // profile.disk_bytes,
                )
            ),
        )

    def capacity(self) -> dict[str, object]:
        with self._lock:
            live = sum(1 for s in self._sandboxes.values() if s.record.state == STATE_STARTED)
            cpu, memory, disk = self._allocated_locked()
            return {
                "host_id": self.config.host_id,
                "sandboxes_live": live,
                "cpu": {
                    "budget": self.config.cpu_budget,
                    "oversubscribe": self.config.cpu_oversubscribe,
                    "allocated": cpu,
                },
                "memory_bytes": {"budget": self.config.memory_budget_bytes, "allocated": memory},
                "disk_bytes": {
                    "budget": self.config.disk_budget_bytes,
                    "allocated": disk,
                    # Stated, not implied: disk is accounted against the
                    # budget but there is no per-sandbox quota on the
                    # node's filesystem. The pipeline does not read disk
                    # back, so this is not gate-relevant.
                    "enforcement": "accounted_not_enforced",
                },
                "default_runtime": self.config.default_runtime,
            }

    # ------------------------------------------------------------------ #
    # Sandboxes
    # ------------------------------------------------------------------ #

    def create_sandbox(
        self,
        *,
        sandbox_id: str,
        snapshot_name: str,
        image_ref: str,
        plan: ImagePlan,
        profile: ResourceProfile,
        runtime: str | None = None,
        labels: Mapping[str, str] | None = None,
        ttl_minutes: int = 180,
    ) -> SandboxRecord:
        runtime = runtime or self.config.default_runtime
        with self._lock:
            if sandbox_id in self._sandboxes or sandbox_id in self._retired:
                raise SiloConflictError("sandbox", sandbox_id, "ids are never reused")
            if self._slots_for_locked(profile) < 1:
                raise SiloRateLimitError(
                    f"sandbox capacity exhausted on host {self.config.host_id}",
                    retry_after_seconds=10,
                )
            # Reserve before the slow part so concurrent creates cannot both fit
            # into the last slot.
            self._creating[sandbox_id] = profile
            # Claim the id immediately; a failed create still retires it.
            self._retired.add(sandbox_id)

        cpuset: tuple[int, ...] = ()
        try:
            cpuset = self._cpusets.allocate(profile.cpu)
            self.ensure_image(image_ref)
            spec = ContainerSpec(
                name=sandbox_id,
                image=image_ref,
                runtime=runtime,
                cpus=profile.cpu,
                cpuset=cpuset,
                memory_bytes=profile.memory_bytes,
                network_none=True,
                user=plan.user,
                workdir=plan.workdir,
                env=dict(plan.env),
                labels={"silo.snapshot": snapshot_name, **(labels or {})},
            )
            self._runtime.start(spec)
        except BaseException:
            with self._lock:
                self._creating.pop(sandbox_id, None)
            self._release_cpuset(cpuset)
            raise

        record = SandboxRecord(
            id=sandbox_id,
            snapshot=snapshot_name,
            state=STATE_STARTED,
            profile=profile,
            network_block_all=True,
            host_id=self.config.host_id,
            created_at=utcnow(),
            labels=dict(labels or {}),
            ttl_minutes=ttl_minutes,
        )
        sandbox = _Sandbox(
            record=record,
            container_name=sandbox_id,
            runtime=runtime,
            cpuset=cpuset,
            expires_at=self._clock() + ttl_minutes * 60,
        )
        with self._lock:
            # One step, so the allocation is never counted twice or dropped.
            self._creating.pop(sandbox_id, None)
            self._sandboxes[sandbox_id] = sandbox
        logger.info(
            "sandbox created id=%s snapshot=%s runtime=%s cpuset=%s mem=%dGB",
            sandbox_id,
            snapshot_name,
            runtime,
            ",".join(map(str, cpuset)),
            profile.memory_gb,
        )
        return record

    def _release_cpuset(self, cpuset: tuple[int, ...]) -> None:
        if cpuset:
            self._cpusets.release(cpuset)

    # -- images ----------------------------------------------------------------

    def ensure_image(self, ref: str) -> None:
        self._runtime.ensure_image(ref)
        with self._lock:
            self._images.add(ref)

    def build_image(self, tag: str, dockerfile: str) -> None:
        self._runtime.build(tag, dockerfile)
        with self._lock:
            self._images.add(tag)

    def images(self) -> list[str]:
        with self._lock:
            return sorted(self._images)

    def _get(self, sandbox_id: str) -> _Sandbox:
        with self._lock:
            sandbox = self._sandboxes.get(sandbox_id)
        if sandbox is None:
            raise SiloNotFoundError("sandbox", sandbox_id)
        return sandbox

    def get_sandbox(self, sandbox_id: str) -> SandboxRecord:
        return self._get(sandbox_id).record

    def _live(self, sandbox_id: str) -> _Sandbox:
        sandbox = self._get(sandbox_id)
        if sandbox.record.state != STATE_STARTED:
            raise SiloError(f"sandbox {sandbox_id!r} is {sandbox.record.state}", status_code=409)
        return sandbox

    def list_sandboxes(self) -> list[SandboxRecord]:
        with self._lock:
            return [s.record for s in self._sandboxes.values()]

    def delete_sandbox(self, sandbox_id: str) -> None:
        """Begin deletion and return immediately.

        The sandbox stays visible as ``destroying`` until the container is gone,
        then becomes a not-found -- and stays one, without a second delete. That
        is exactly the observation ``wait_for_sandbox_deletion`` makes.
        """
        sandbox = self._get(sandbox_id)
        with sandbox.lock:
            if sandbox.record.state == STATE_DESTROYING:
                return
            sandbox.record = dataclasses.replace(sandbox.record, state=STATE_DESTROYING)
        threading.Thread(target=self._finish_delete, args=(sandbox,), name=f"rm-{sandbox_id}", daemon=True).start()

    def _finish_delete(self, sandbox: _Sandbox) -> None:
        """Remove the container, then forget the sandbox. Single-flight per sandbox.

        The reaper retries stuck deletions every REAP_SECONDS. Without the
        ``deleting`` guard it also re-ran deletions that were merely slow, and
        each run released the sandbox's resources again.
        """
        sandbox_id = sandbox.record.id
        with sandbox.lock:
            if sandbox.deleting:
                return
            sandbox.deleting = True
        try:
            self._kill_sessions(sandbox)
            self._runtime.remove(sandbox.container_name)
        except Exception:
            # Leave it observable as destroying; a later reap retries. Reporting it
            # gone while the container still exists would be the lie §3.5 guards.
            logger.exception("delete failed for sandbox %s; will retry on reap", sandbox_id)
            with sandbox.lock:
                sandbox.deleting = False
            return
        with self._lock:
            forgotten = self._sandboxes.pop(sandbox_id, None) is not None
        if not forgotten:
            return  # already finished by another path; its resources are already free
        self._release_cpuset(sandbox.cpuset)
        shutil.rmtree(self._session_dir(sandbox_id), ignore_errors=True)
        logger.info("sandbox deleted id=%s", sandbox_id)

    def reap(self) -> list[str]:
        """Delete sandboxes past their TTL and retry stuck deletions."""
        now = self._clock()
        reaped: list[str] = []
        with self._lock:
            candidates = list(self._sandboxes.values())
        for sandbox in candidates:
            if sandbox.record.state == STATE_DESTROYING:
                if sandbox.deleting:
                    continue  # still running; not stuck
                threading.Thread(target=self._finish_delete, args=(sandbox,), daemon=True).start()
            elif now >= sandbox.expires_at:
                logger.info("sandbox %s passed its ttl; deleting", sandbox.record.id)
                self.delete_sandbox(sandbox.record.id)
                reaped.append(sandbox.record.id)
        return reaped

    # ------------------------------------------------------------------ #
    # One-shot exec
    # ------------------------------------------------------------------ #

    def exec(
        self,
        sandbox_id: str,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecOutcome:
        """Run ``command`` through the guest's /bin/sh, output combined.

        Daytona's ``process.exec`` returns stdout and stderr interleaved in one
        ``result``; the pipeline reads it that way, so this does too.
        """
        sandbox = self._live(sandbox_id)
        return self._runtime.exec(sandbox.container_name, ["/bin/sh", "-c", command], env=env, cwd=cwd, timeout=timeout)

    # ------------------------------------------------------------------ #
    # Sessions -- the polled model
    # ------------------------------------------------------------------ #

    def _session_dir(self, sandbox_id: str) -> Path:
        return self.config.work_dir / "sessions" / sandbox_id

    def create_session(self, sandbox_id: str, session_id: str) -> None:
        sandbox = self._live(sandbox_id)
        with sandbox.lock:
            if session_id in sandbox.sessions:
                raise SiloConflictError("session", session_id)
            sandbox.sessions[session_id] = _Session(session_id=session_id)
        (self._session_dir(sandbox_id) / session_id).mkdir(parents=True, exist_ok=True)

    def _session(self, sandbox: _Sandbox, session_id: str) -> _Session:
        with sandbox.lock:
            session = sandbox.sessions.get(session_id)
        if session is None:
            raise SiloNotFoundError("session", session_id)
        return session

    def execute_session_command(
        self, sandbox_id: str, session_id: str, command: str, *, run_async: bool = True
    ) -> _SessionCommand:
        """Launch a command and return its id without waiting.

        Output goes to files on the HOST side, and the exit code is recorded by a
        host thread. The guest can neither see nor forge either, and a poll only
        reads the recorded exit code -- it never consumes output, which the
        controller's timeout accounting depends on.

        Each command runs in a fresh shell. Daytona sessions persist shell state
        between commands; the pipeline does not rely on that (every command it
        sends carries its own ``cd`` and ``export`` prefix), so it is not emulated.
        """
        sandbox = self._live(sandbox_id)
        session = self._session(sandbox, session_id)
        cmd_id = uuid.uuid4().hex
        directory = self._session_dir(sandbox_id) / session_id
        directory.mkdir(parents=True, exist_ok=True)
        entry = _SessionCommand(
            cmd_id=cmd_id,
            command=command,
            stdout_path=directory / f"{cmd_id}.out",
            stderr_path=directory / f"{cmd_id}.err",
        )
        stdout = open(entry.stdout_path, "wb")  # noqa: SIM115 - handed to the child
        stderr = open(entry.stderr_path, "wb")  # noqa: SIM115
        try:
            entry.process = self._runtime.spawn(
                sandbox.container_name,
                ["/bin/sh", "-c", command],
                env=None,
                cwd=None,
                user=None,
                stdout=stdout,
                stderr=stderr,
            )
        finally:
            stdout.close()
            stderr.close()
        with sandbox.lock:
            session.commands[cmd_id] = entry
        threading.Thread(target=self._await_command, args=(entry,), name=f"cmd-{cmd_id[:8]}", daemon=True).start()
        if not run_async:
            entry.finished.wait()
        return entry

    @staticmethod
    def _await_command(entry: _SessionCommand) -> None:
        assert entry.process is not None
        entry.exit_code = entry.process.wait()
        entry.finished.set()

    def get_session_command(self, sandbox_id: str, session_id: str, cmd_id: str) -> _SessionCommand:
        sandbox = self._get(sandbox_id)
        session = self._session(sandbox, session_id)
        with sandbox.lock:
            entry = session.commands.get(cmd_id)
        if entry is None:
            raise SiloNotFoundError("command", cmd_id)
        return entry

    def get_session_command_logs(self, sandbox_id: str, session_id: str, cmd_id: str) -> tuple[bytes, bytes]:
        entry = self.get_session_command(sandbox_id, session_id, cmd_id)
        return entry.stdout_path.read_bytes(), entry.stderr_path.read_bytes()

    def delete_session(self, sandbox_id: str, session_id: str) -> None:
        sandbox = self._get(sandbox_id)
        with sandbox.lock:
            session = sandbox.sessions.pop(session_id, None)
        if session is None:
            raise SiloNotFoundError("session", session_id)
        self._terminate(sandbox, session)
        shutil.rmtree(self._session_dir(sandbox_id) / session_id, ignore_errors=True)

    def _terminate(self, sandbox: _Sandbox, session: _Session) -> None:
        for entry in session.commands.values():
            if entry.process is not None and entry.process.poll() is None:
                # Killing the host-side client does not stop the guest process;
                # the pipeline's own `timeout N` wrapper is what bounds that. The
                # container's removal is the backstop for anything left.
                with contextlib.suppress(Exception):
                    entry.process.kill()

    def _kill_sessions(self, sandbox: _Sandbox) -> None:
        with sandbox.lock:
            sessions = list(sandbox.sessions.values())
            sandbox.sessions.clear()
        for session in sessions:
            self._terminate(sandbox, session)

    # ------------------------------------------------------------------ #
    # Filesystem
    # ------------------------------------------------------------------ #

    def upload_file(self, sandbox_id: str, path: str, data: bytes) -> None:
        """Write ``data`` to ``path`` in the guest through exec stdin.

        Uses the bind-mounted busybox rather than the guest's tools, so it works
        on images with no shell utilities. Proven byte-exact for runc and runsc by
        the Phase 0b spike (4 MiB random roundtrip, sha256 match).
        """
        sandbox = self._live(sandbox_id)
        script = f'{BUSYBOX} mkdir -p "$({BUSYBOX} dirname "$1")" && {BUSYBOX} cat > "$1"'
        outcome = self._runtime.exec(
            sandbox.container_name, [BUSYBOX, "sh", "-c", script, "silo-upload", path], stdin=data, timeout=600
        )
        if outcome.exit_code != 0:
            raise SiloError(
                f"upload to sandbox {sandbox_id!r} path {path!r} failed: "
                f"{outcome.output.decode(errors='replace')[-400:]}"
            )

    def download_file(self, sandbox_id: str, path: str) -> bytes:
        sandbox = self._live(sandbox_id)
        # Test for the file first so a missing path is a clean not-found rather
        # than stderr text mixed into the payload.
        probe = self._runtime.exec(sandbox.container_name, [BUSYBOX, "test", "-f", path], timeout=60)
        if probe.exit_code != 0:
            raise SiloNotFoundError("file", path, f"in sandbox {sandbox_id}")
        outcome = self._runtime.exec(sandbox.container_name, [BUSYBOX, "cat", path], timeout=600, merge_stderr=False)
        if outcome.exit_code != 0:
            raise SiloError(f"download from sandbox {sandbox_id!r} path {path!r} failed")
        return outcome.output
