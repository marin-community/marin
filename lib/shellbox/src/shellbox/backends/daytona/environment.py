# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Translate task-level environment inputs into a Daytona machine request."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from shellbox.backends.daytona.snapshot import (
    SnapshotRequest,
    snapshot_resources,
    task_snapshot_source,
)
from shellbox.image import DockerfileSource, RegistryImage
from shellbox.machine import MachineSpec, NetworkPolicy


@dataclass(frozen=True)
class DaytonaTaskEnvironment:
    """The task inputs that select one Daytona snapshot and sandbox.

    Harbor supplies these values from its task configuration; the resulting
    source selection, resource rounding and snapshot identity stay in Shellbox.
    """

    environment_dir: Path
    docker_image: str | None = None
    workdir: str = ""
    env: Mapping[str, str] = field(default_factory=dict)
    allow_internet: bool = False
    cpus: int | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None
    gpus: int = 0
    startup_timeout: float | None = None

    def source(self) -> RegistryImage | DockerfileSource:
        """Select and validate the task's registry image or Dockerfile."""
        return task_snapshot_source(self.docker_image, self.environment_dir)

    def machine_spec(self) -> MachineSpec:
        return MachineSpec(
            source=self.source(),
            workdir=self.workdir,
            env=dict(self.env),
            network=(NetworkPolicy.ALLOW if self.allow_internet else NetworkPolicy.DENY),
            cpus=self.cpus,
            memory_mb=self.memory_mb,
            storage_mb=self.storage_mb,
            gpus=self.gpus,
            startup_timeout=self.startup_timeout,
        )

    def snapshot_request(self) -> SnapshotRequest:
        """The snapshot this environment creates without starting a sandbox."""
        return SnapshotRequest(
            source=self.source(),
            resources=snapshot_resources(
                cpus=self.cpus,
                memory_mb=self.memory_mb,
                storage_mb=self.storage_mb,
                gpus=self.gpus,
            ),
        )
