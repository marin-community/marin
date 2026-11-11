# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Translate task-level environment inputs into a Daytona machine request."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from shellbox.backends.daytona.snapshot import (
    SnapshotRequest,
    snapshot_request_for_spec,
    task_snapshot_source,
)
from shellbox.image import DockerfileSource, RegistryImage
from shellbox.machine import MachineSpec, NetworkPolicy


@dataclass(frozen=True)
class DaytonaTaskEnvironment:
    """Task inputs that select a Daytona snapshot and sandbox."""

    environment_dir: Path
    docker_image: str | None = None
    workdir: str = ""
    env: Mapping[str, str] = field(default_factory=dict)
    allow_internet: bool = False
    cpus: int | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None
    gpus: int = 0

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
        )

    def snapshot_request(self) -> SnapshotRequest:
        """The snapshot this environment creates without starting a sandbox."""
        return snapshot_request_for_spec(self.machine_spec())
