# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Infrastructure capacity available to independently released applications."""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from marina.manifest import AppJob

RUNTIME_CONFIG_ENV = "MARINA_RUNTIME_CONFIG"
DEFAULT_RUNTIME_CONFIG = Path(__file__).resolve().parents[2] / "runtime.toml"


@dataclass(frozen=True)
class JobRunner:
    name: str
    schedule: str
    timeout: int
    cpu: int
    memory_gib: int
    secrets: tuple[str, ...]


@dataclass(frozen=True)
class JobPermission:
    runner: str
    secrets: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeConfig:
    required_apps: dict[str, UUID] = field(default_factory=dict)
    runners: tuple[JobRunner, ...] = ()
    job_triggers: dict[str, str] = field(default_factory=dict)
    job_permissions: dict[UUID, JobPermission] = field(default_factory=dict)
    data_urls: dict[str, tuple[str, ...]] = field(default_factory=dict)
    named_readers: tuple[str, ...] = ()

    def runner(self, name: str) -> JobRunner:
        for runner in self.runners:
            if runner.name == name:
                return runner
        raise ValueError(f"No base-runtime runner named {name!r}")

    def validate_jobs(self, applet_id: UUID, jobs: tuple[AppJob, ...]) -> None:
        for job in jobs:
            runner = self.runner(job.runner)
            permission = self.job_permissions.get(applet_id)
            if permission is None or permission.runner != runner.name:
                raise ValueError(f"Application {applet_id} has no permission to use runner {runner.name}")
            if job.timeout > runner.timeout:
                raise ValueError(f"Job {job.name} exceeds runner {runner.name}'s timeout")
            if set(job.secrets) - set(runner.secrets):
                raise ValueError(f"Job {job.name} requests secrets unavailable to runner {runner.name}")
            if set(job.secrets) - set(permission.secrets):
                raise ValueError(f"Application {applet_id} has no permission to use the requested job secrets")

    def require_registrations(self, actual: dict[str, UUID]) -> None:
        missing = [name for name, applet_id in self.required_apps.items() if actual.get(name) != applet_id]
        if missing:
            raise RuntimeError("Release named applications before runtime cutover: " + ", ".join(missing))


def load_runtime_config(path: Path | None = None) -> RuntimeConfig:
    selected = path or Path(os.environ.get(RUNTIME_CONFIG_ENV, DEFAULT_RUNTIME_CONFIG))
    with selected.open("rb") as source:
        raw = tomllib.load(source)
    return RuntimeConfig(
        required_apps={name: UUID(value) for name, value in raw.get("required_apps", {}).items()},
        runners=tuple(
            JobRunner(name=name, **{**values, "secrets": tuple(values.get("secrets", []))})
            for name, values in raw.get("runners", {}).items()
        ),
        job_triggers=raw.get("job_triggers", {}),
        job_permissions={
            UUID(applet_id): JobPermission(values["runner"], tuple(values["secrets"]))
            for applet_id, values in raw.get("job_permissions", {}).items()
        },
        data_urls={name: tuple(prefixes) for name, prefixes in raw.get("data_urls", {}).items()},
        named_readers=tuple(raw.get("named_readers", [])),
    )
