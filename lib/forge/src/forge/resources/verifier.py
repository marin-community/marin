# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Round-owned verifier leases; task machines remain owned by individual rollouts."""

import asyncio
import json
import tarfile
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import override

from rolloutengine.models import GradedRollout, Model, TaskRolloutSpec
from shellbox.machine import (
    Backend,
    Command,
    ExecutionRequirements,
    ExitReason,
    Machine,
    MachineFactory,
    MachineSpec,
    MachineTerminated,
    Result,
)

from forge.resources.execution import exec_machine_lost
from forge.resources.rollouts import Engine

# Preserve the machine's original supervisor processes, including its idle child.
# PID start times distinguish a surviving supervisor from a reused PID.
PROCESS_RESET = """
import json, os, signal, sys
from pathlib import Path
processes = {}
for path in Path('/proc').glob('[0-9]*/stat'):
    try:
        fields = path.read_text().rsplit(')', 1)[1].split()
        processes[int(path.parent.name)] = (int(fields[1]), fields[19])
    except (FileNotFoundError, ProcessLookupError):
        pass
keep = set()
pid = os.getpid()
while pid in processes and pid not in keep:
    keep.add(pid)
    pid = processes[pid][0]
if len(sys.argv) > 1:
    baseline = json.loads(sys.argv[1])
    for pid, (_, started) in processes.items():
        if pid not in keep and baseline.get(str(pid)) != started:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
else:
    print(json.dumps({str(pid): started for pid, (_, started) in processes.items()}))
"""
CLEAR_ROOTS = """
for root do
    if test -L "$root" || { test -e "$root" && ! test -d "$root"; }; then
        rm -f -- "$root" || exit
    elif test -d "$root"; then
        find "$root" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + || exit
    fi
done
"""
"""Clear contents without unlinking directories that may be mount points."""

RESET_ROOTS = frozenset({"/app", "/workspace", "/tests", "/tmp"})
SYSTEM_ROOTS = frozenset(
    {"/", "/bin", "/dev", "/etc", "/lib", "/lib32", "/lib64", "/proc", "/sbin", "/sys", "/usr", "/var", "/opt"}
)


class LeaseTrace:
    def __init__(self, fresh: bool) -> None:
        self.fresh = fresh
        self.index: int | None = None


TRACE: ContextVar[LeaseTrace | None] = ContextVar("forge_verifier_lease", default=None)


class VerifierPool:
    """One cached machine, with an exclusive lease and bounded round cleanup."""

    def __init__(self, cleanup_timeout: float) -> None:
        self.cleanup_timeout = cleanup_timeout
        self.lock = asyncio.Lock()
        self.cached: VerifierLease | None = None
        self.index = 0

    async def close(self) -> None:
        async with self.lock:
            await self._discard()

    async def _discard(self) -> None:
        cached, self.cached = self.cached, None
        if cached is not None:
            async with asyncio.timeout(self.cleanup_timeout):
                await cached.machine.close()

    async def acquire(self, factory: MachineFactory, spec: MachineSpec) -> Machine:
        await self.lock.acquire()
        trace = TRACE.get()
        fresh = trace is not None and trace.fresh
        lease: VerifierLease | None = None
        acquired = False
        try:
            if self.cached is not None and (self.cached.spec != spec or self.cached.backend != factory.backend):
                await self._discard()
            lease = None if fresh else self.cached
            if lease is None:
                machine = await factory.create(spec)
                lease = VerifierLease(self, machine, factory.backend, spec, fresh)
                # Own the machine before any initialization that can fail.
                if not fresh:
                    self.cached = lease
                await lease.initialize()
            lease.released = False
            await lease.reset()
            self.index += 1
            if trace is not None:
                trace.index = self.index
            acquired = True
            return lease
        finally:
            if not acquired:
                try:
                    if fresh and lease is not None:
                        async with asyncio.timeout(self.cleanup_timeout):
                            await lease.machine.close()
                    else:
                        await self._discard()
                finally:
                    self.lock.release()


class VerifierLease(Machine):
    def __init__(self, pool: VerifierPool, machine: Machine, backend: Backend, spec: MachineSpec, fresh: bool) -> None:
        self.pool, self.machine, self.backend, self.spec, self.fresh = pool, machine, backend, spec, fresh
        self.roots = set(RESET_ROOTS)
        self.baseline = ""
        self.broken = False
        self.released = True
        self.reusable = True
        self.track_path(spec.workdir)

    def track_path(self, path: str) -> None:
        parts = PurePosixPath(path).parts
        root = "/" + parts[1] if len(parts) > 1 and parts[0] == "/" else "/"
        if root in SYSTEM_ROOTS:
            self.reusable = False
        else:
            self.roots.add(root)

    async def checked(self, command: Command) -> Result:
        result = await self.run(command)
        if result.exit_code != 0 or result.reason != ExitReason.EXITED:
            self.broken = True
            raise MachineTerminated(f'Verifier reset failed: {result.reason}: {result.stderr.decode(errors="replace")}')
        return result

    async def initialize(self) -> None:
        if self.backend != Backend.SHELLSIM:
            result = await self.checked(Command(("python3", "-c", PROCESS_RESET), cwd="/", user="0", timeout=30))
            self.baseline = json.dumps(json.loads(result.stdout))

    async def reset(self) -> None:
        if self.baseline:
            await self.checked(Command(("python3", "-c", PROCESS_RESET, self.baseline), cwd="/", user="0", timeout=30))
        roots = tuple(sorted(self.roots - {"/tmp"}))
        await self.checked(Command(("sh", "-c", CLEAR_ROOTS, "forge-clear", *roots), cwd="/", user="0", timeout=30))
        # Iris captures this very command under /tmp/.shellbox-*. Preserve both
        # its current capture prefix and a dedicated runtime directory if present.
        await self.checked(
            Command(
                (
                    "find",
                    "/tmp",
                    "-mindepth",
                    "1",
                    "-maxdepth",
                    "1",
                    "!",
                    "-name",
                    ".shellbox-*",
                    "!",
                    "-name",
                    ".shellbox",
                    "-exec",
                    "rm",
                    "-rf",
                    "--",
                    "{}",
                    "+",
                ),
                cwd="/",
                user="0",
                timeout=30,
            )
        )
        await self.checked(
            Command(("mkdir", "-p", *sorted(RESET_ROOTS), self.spec.workdir), cwd="/", user="0", timeout=30)
        )

    @override
    async def run(self, command: Command) -> Result:
        if command.cwd and command.cwd != "/":
            self.track_path(command.cwd)
        try:
            result = await self.machine.run(command)
        except (MachineTerminated, asyncio.CancelledError, TimeoutError):
            self.broken = True
            raise
        except RuntimeError as error:
            if exec_machine_lost(error):
                self.broken = True
                raise MachineTerminated(f"Verifier container cannot be exec'd: {error}") from error
            raise
        if result.reason == ExitReason.TIMED_OUT:
            self.broken = True
        return result

    @override
    async def upload(self, source: Path, target: str) -> None:
        # TaskCompendium supplies a new archive for every grade, including candidate files.
        if target == "/tmp/taskcompendium-grading.tar":
            previous_roots = set(self.roots)
            with tarfile.open(source) as archive:
                for member in archive.getmembers():
                    self.track_path("/" + member.name.lstrip("/"))
            new_roots = tuple(sorted(self.roots - previous_roots))
            if new_roots:
                await self.checked(
                    Command(("sh", "-c", CLEAR_ROOTS, "forge-clear", *new_roots), cwd="/", user="0", timeout=30)
                )
        try:
            await self.machine.upload(source, target)
        except (MachineTerminated, asyncio.CancelledError, TimeoutError):
            self.broken = True
            raise
        except RuntimeError as error:
            if exec_machine_lost(error):
                self.broken = True
                raise MachineTerminated(f"Verifier container cannot be exec'd: {error}") from error
            raise

    @override
    async def download(self, source: str, target: Path) -> None:
        try:
            await self.machine.download(source, target)
        except (MachineTerminated, asyncio.CancelledError, TimeoutError):
            self.broken = True
            raise
        except RuntimeError as error:
            if exec_machine_lost(error):
                self.broken = True
                raise MachineTerminated(f"Verifier container cannot be exec'd: {error}") from error
            raise

    @override
    async def close(self) -> None:
        if self.released:
            return
        self.released = True
        try:
            if self.fresh:
                async with asyncio.timeout(self.pool.cleanup_timeout):
                    await self.machine.close()
            elif self.broken or not self.reusable:
                await self.pool._discard()
        finally:
            self.pool.lock.release()


class VerifierFactory(MachineFactory):
    def __init__(self, pool: VerifierPool, factory: MachineFactory) -> None:
        self.pool, self.factory = pool, factory

    @property
    @override
    def backend(self) -> Backend:
        return self.factory.backend

    @override
    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        self.factory.check_environment(requirements, spec)

    @override
    async def create(self, spec: MachineSpec) -> Machine:
        return await self.pool.acquire(self.factory, spec)


class VerifierEngine(Engine):
    def __init__(self, engine: Engine, *, fresh: bool = False) -> None:
        self.engine, self.fresh = engine, fresh

    @override
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        trace = LeaseTrace(self.fresh)
        token = TRACE.set(trace)
        try:
            if spec.verifier_machine is not None:
                spec = spec.model_copy(
                    update={
                        "verifier_machine": spec.verifier_machine.model_copy(
                            update={"backend": "verifier:" + spec.verifier_machine.backend}
                        )
                    }
                )
            result = await self.engine.run_task(spec, model, run_id=run_id)
            return replace(
                result,
                grade=replace(
                    result.grade,
                    diagnostics={
                        **result.grade.diagnostics,
                        "verifier_lease": trace.index,
                        "verifier_fresh": self.fresh,
                    },
                ),
            )
        finally:
            TRACE.reset(token)
