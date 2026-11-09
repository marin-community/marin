# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An Iris machine factory for the live verifier probes that records every machine it creates."""

import os
from typing import override

from shellbox.backends.iris.machine import IrisMachine, IrisMachineFactory
from shellbox.machine import MachineSpec


class RecordingIrisFactory(IrisMachineFactory):
    def __init__(self) -> None:
        super().__init__(controller_url=os.environ["IRIS_CONTROLLER_URL"], scheduling_timeout=1200)
        self.created: list[IrisMachine] = []

    @override
    async def create(self, spec: MachineSpec) -> IrisMachine:
        machine = await super().create(spec)
        self.created.append(machine)
        return machine
