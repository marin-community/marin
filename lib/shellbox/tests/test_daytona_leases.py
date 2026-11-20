# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("daytona")

from daytona import DaytonaNotFoundError
from shellbox.backends.daytona.leases import SnapshotLease
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import MachineSpec


@pytest.mark.parametrize("creation_failed", [False, True])
def test_snapshot_lease_survives_until_sandbox_deletion_is_observed(creation_failed):
    async def exercise():
        deletion_observed = asyncio.Event()
        allow_absence = asyncio.Event()
        events = []

        class Leaser:
            async def acquire(self, spec):
                events.append("acquired")
                return SnapshotLease("owned-snapshot", "lease-token")

            async def release(self, lease):
                assert lease == SnapshotLease("owned-snapshot", "lease-token")
                events.append("released")

        class Client:
            deleting = False

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                events.append("client-closed")

            async def create(self, params, timeout):
                assert params.snapshot == "owned-snapshot"
                assert params.auto_stop_interval == params.auto_pause_interval == 0
                if creation_failed:
                    raise RuntimeError("Creation response lost after provider acceptance")
                return sandbox

            async def get(self, name):
                if self.deleting:
                    deletion_observed.set()
                    await allow_absence.wait()
                    raise DaytonaNotFoundError("Sandbox no longer exists", status_code=404)
                return sandbox

            async def delete(self, instance):
                self.deleting = True
                events.append("delete-accepted")

        async def execute(*args, **kwargs):
            return SimpleNamespace(exit_code=0, result="/")

        sandbox = SimpleNamespace(process=SimpleNamespace(exec=execute))
        client = Client()
        factory = DaytonaMachineFactory(client_factory=lambda: client, snapshot_leaser=Leaser())
        spec = MachineSpec(source=RegistryImage("fixture/image:1"), workdir="")
        if creation_failed:
            closing = asyncio.create_task(factory.create(spec))
        else:
            machine = await factory.create(spec)
            closing = asyncio.create_task(machine.close())
        await deletion_observed.wait()
        assert events == ["acquired", "delete-accepted"]
        allow_absence.set()
        if creation_failed:
            with pytest.raises(RuntimeError, match="Creation response lost"):
                await closing
        else:
            await closing
        assert events == ["acquired", "delete-accepted", "released", "client-closed"]

    asyncio.run(exercise())
