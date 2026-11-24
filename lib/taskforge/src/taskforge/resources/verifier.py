# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One round-owned verifier, acquired through the engine's prepared-machine hook."""

import asyncio
import logging
from dataclasses import dataclass, replace

from rolloutengine.engine import RolloutEngine
from rolloutengine.models import GradedRollout, Model, TaskRolloutSpec
from shellbox.machine import Machine, MachineSpec, MachineTerminated, ResettableMachine

from taskforge.resources.execution import Notify, close_resource


class RoundVerifier:
    """A verifier machine shared by a round's grades, reset between leases and replaced when it dies.

    Callers use it through one of two engines. ``rollouts`` leases the machine when the engine starts grading, so
    solver rollouts overlap each other; the wait for the lease counts against the engine's verifier and startup
    deadlines. ``grades`` leases it before the attempt starts, for attempts that only replay a candidate, so their
    wait is outside every engine deadline. Either lease is held until the engine returns the attempt, including
    its task machine's start (``grades`` only) and its cleanup.
    """

    def __init__(self, engine: RolloutEngine, cleanup_timeout: float) -> None:
        self.engine, self.cleanup_timeout = engine, cleanup_timeout
        self.machine: Machine | None = None
        self.key: tuple[str, MachineSpec] | None = None
        self.lock = asyncio.Lock()
        self.index = 0
        self.notify: Notify = logging.warning
        self.rollouts = LeaseAtGrade(self)
        self.grades = LeaseFirst(self)

    async def close(self) -> None:
        machine, self.machine, self.key = self.machine, None, None
        if machine is not None:
            await close_resource(machine.close, self.notify, timeout=self.cleanup_timeout)

    async def leased(self, spec: TaskRolloutSpec, model: Model, run_id: str | None, *, locked: bool) -> GradedRollout:
        """Run ``spec`` on the shared verifier; ``locked`` means the caller already holds ``lock``."""
        selection = spec.verifier_machine
        if selection is None:
            return await self.engine.run_task(spec, model, run_id=run_id)
        leased = False
        lease = 0

        async def acquire(prepared: MachineSpec) -> Machine:
            nonlocal leased, lease
            if not locked:
                await self.lock.acquire()
            leased = True
            key = (selection.backend, prepared)
            if self.machine is not None:
                if self.key != key or not isinstance(self.machine, ResettableMachine):
                    await self.close()
                else:
                    try:
                        await self.machine.reset()
                    except MachineTerminated:
                        await self.close()
            if self.machine is None:
                self.machine = await self.engine.machine_factories[selection.backend].create(prepared)
                self.key = key
                if isinstance(self.machine, ResettableMachine):
                    await self.machine.reset()
            self.index += 1
            lease = self.index
            return self.machine

        try:
            result = await self.engine.run_task(spec, model, run_id=run_id, verifier=acquire)
        except BaseException:
            if leased:
                await self.close()
            raise
        finally:
            if leased and not locked:
                self.lock.release()
        return replace(
            result, grade=replace(result.grade, diagnostics={**result.grade.diagnostics, "verifier_lease": lease})
        )


@dataclass(frozen=True, slots=True)
class LeaseAtGrade:
    """Solver rollouts: lease when grading starts, so the rollout phase does not hold the verifier."""

    verifier: RoundVerifier

    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        return await self.verifier.leased(spec, model, run_id, locked=False)


@dataclass(frozen=True, slots=True)
class LeaseFirst:
    """Grade-only attempts (controls, adversary claims): lease before the engine starts its deadlines."""

    verifier: RoundVerifier

    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        if spec.verifier_machine is None:
            return await self.verifier.engine.run_task(spec, model, run_id=run_id)
        async with self.verifier.lock:
            return await self.verifier.leased(spec, model, run_id, locked=True)
