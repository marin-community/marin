# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host-owned snapshot admission, outside sandbox authority."""

from dataclasses import dataclass
from typing import Protocol

from shellbox.machine import MachineSpec


@dataclass(frozen=True)
class SnapshotLease:
    name: str
    token: str


class SnapshotLeaser(Protocol):
    async def acquire(self, spec: MachineSpec) -> SnapshotLease: ...

    async def release(self, lease: SnapshotLease) -> None: ...
