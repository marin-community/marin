# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One round-owned verifier, acquired through the engine's prepared-machine hook."""

import asyncio
import logging
from dataclasses import replace

from rolloutengine.engine import RolloutEngine
from rolloutengine.models import GradedRollout, Model, TaskRolloutSpec
from shellbox.machine import Machine, MachineSpec, MachineTerminated, ResettableMachine

from forge.resources.execution import Notify, close_resource


class RoundVerifier:
    def __init__(self, engine: RolloutEngine, cleanup_timeout: float) -> None:
        self.engine, self.cleanup_timeout = engine, cleanup_timeout
        self.machine: Machine | None = None
        self.key: tuple[str, MachineSpec] | None = None
        self.lock = asyncio.Lock()
        self.index = 0
        self.notify: Notify = logging.warning

    async def close(self) -> None:
        machine, self.machine, self.key = self.machine, None, None
        if machine is not None:

            async def close() -> None:
                async with asyncio.timeout(self.cleanup_timeout):
                    await machine.close()

            await close_resource(close, self.notify)

    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        selection = spec.verifier_machine
        if selection is None:
            return await self.engine.run_task(spec, model, run_id=run_id)
        leased = False
        lease = 0

        async def acquire(prepared: MachineSpec) -> Machine:
            nonlocal leased, lease
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
            if leased:
                self.lock.release()
        return replace(
            result, grade=replace(result.grade, diagnostics={**result.grade.diagnostics, "verifier_lease": lease})
        )
