# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Translate task-level environment inputs into a Daytona machine request."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, cast

from shellbox.backends.daytona.snapshot import (
    SnapshotRequest,
    snapshot_request_for_spec,
    task_snapshot_source,
)
from shellbox.image import DockerfileSource, RegistryImage
from shellbox.machine import MachineSpec, NetworkPolicy


class DaytonaTaskEnvironmentSpec(TypedDict):
    """JSON interchange for snapshot preflight and creation."""

    environment_dir: str
    docker_image: str | None
    cpus: int | None
    memory_mb: int | None
    storage_mb: int | None
    gpus: int


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

    @classmethod
    def from_spec(cls, spec: Mapping[str, object]) -> "DaytonaTaskEnvironment":
        """Parse the JSON interchange emitted by ``as_spec``."""
        environment_dir = spec.get("environment_dir")
        if environment_dir is not None and not isinstance(environment_dir, str):
            raise ValueError("environment_dir must be a string")
        return cls(
            environment_dir=Path(environment_dir) if environment_dir else Path("."),
            docker_image=cast(str | None, spec.get("docker_image")),
            cpus=cast(int | None, spec.get("cpus")),
            memory_mb=cast(int | None, spec.get("memory_mb")),
            storage_mb=cast(int | None, spec.get("storage_mb")),
            gpus=cast(int, spec.get("gpus", 0)),
        )

    def as_spec(self) -> DaytonaTaskEnvironmentSpec:
        """Return the effective inputs consumed by snapshot preflight."""
        return {
            "environment_dir": str(self.environment_dir),
            "docker_image": self.docker_image,
            "cpus": self.cpus,
            "memory_mb": self.memory_mb,
            "storage_mb": self.storage_mb,
            "gpus": self.gpus,
        }

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
