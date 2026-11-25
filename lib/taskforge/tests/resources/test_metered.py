# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from typing import cast, override

import pytest
from shellbox.machine import Backend, ExecutionRequirements, Machine, MachineFactory, MachineSpec, RegistryImage

from taskforge.effects.events import Event
from taskforge.resources.sandbox.metered import MachineMeter, MachineStarted, MeteredFactory

MACHINE = cast(Machine, object())


class ScriptedFactory(MachineFactory):
    backend = Backend.GVISOR

    def __init__(self, failure: Exception | None) -> None:
        self.failure = failure

    @override
    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        raise AssertionError("not called")

    @override
    async def create(self, spec: MachineSpec) -> Machine:
        if self.failure is not None:
            raise self.failure
        return MACHINE


def recording_meter() -> tuple[MachineMeter, list[Event[None]]]:
    meter, events = MachineMeter(), []
    meter.notify = events.append
    return meter, events


SPEC = MachineSpec(source=RegistryImage("docker.io/library/python:3.12"), cpus=1, memory_mb=8192)


async def test_started_machine_is_returned_unchanged_and_recorded_with_its_request() -> None:
    meter, events = recording_meter()
    assert await MeteredFactory(ScriptedFactory(None), meter).create(SPEC) is MACHINE
    (event,) = events
    assert isinstance(event, MachineStarted)
    assert (event.machine, event.backend, event.image) == ("", "gvisor", "docker.io/library/python:3.12")
    assert (event.cpus, event.memory_mb, event.error) == (1, 8192, None)


async def test_failed_start_is_recorded_and_still_raised() -> None:
    meter, events = recording_meter()
    with pytest.raises(TimeoutError, match="did not start"):
        await MeteredFactory(ScriptedFactory(TimeoutError("did not start")), meter).create(SPEC)
    (event,) = events
    assert isinstance(event, MachineStarted)
    assert event.error is not None and "did not start" in event.error
