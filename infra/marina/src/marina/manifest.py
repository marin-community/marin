# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The app manifest: what ``apps/<name>/app.toml`` declares and how the kernel finds it.

An app is a directory under the apps root holding an ``app.toml``. The manifest carries
what Marina needs to serve and operate the app: its display name, the origins its page may
fetch, how to build its frontend, and scheduled commands. Unknown keys are an error so a
typo cannot silently disable something.
"""

import re
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

MANIFEST_FILE = "app.toml"
DIST_DIR = "dist"
APP_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
KNOWN_KEYS = frozenset({"title", "description", "connect_src", "build_command", "jobs"})
JOB_KEYS = frozenset({"name", "runner", "schedule", "command", "timeout", "cpu", "memory_gib", "secrets"})
ENV_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
RUNNER_OVERHEAD = 300


@dataclass(frozen=True)
class AppJob:
    """One command an app assigns to a scheduled runner."""

    name: str
    runner: str
    schedule: str
    command: tuple[str, ...]
    timeout: int
    cpu: int
    memory_gib: int
    secrets: tuple[str, ...] = ()


@dataclass(frozen=True)
class AppManifest:
    name: str
    title: str
    description: str
    root: Path
    connect_src: tuple[str, ...] = ()
    build_command: str | None = None
    jobs: tuple[AppJob, ...] = ()

    @property
    def path(self) -> str:
        """The URL prefix the app is served under."""
        return f"/{self.name}/"

    @property
    def dist(self) -> Path:
        """Where the built frontend lives; served verbatim with index.html as the SPA fallback."""
        return self.root / DIST_DIR


@dataclass(frozen=True)
class BoundJob:
    """An app job paired with the app directory that owns it."""

    app: AppManifest
    job: AppJob

    @property
    def qualified_name(self) -> str:
        return f"{self.app.name}.{self.job.name}"

    @property
    def resource_env(self) -> str:
        """Environment variable naming the Cloud Run resource for this job's runner."""
        return f"{self.app.name}_{self.job.name}_JOB".replace("-", "_").upper()


@dataclass(frozen=True)
class JobRunner:
    """App jobs that share one Cloud Run job and Cloud Scheduler trigger."""

    name: str
    schedule: str
    jobs: tuple[BoundJob, ...]

    @property
    def timeout(self) -> int:
        return sum(bound.job.timeout for bound in self.jobs) + RUNNER_OVERHEAD

    @property
    def cpu(self) -> int:
        return max(bound.job.cpu for bound in self.jobs)

    @property
    def memory_gib(self) -> int:
        return max(bound.job.memory_gib for bound in self.jobs)


def _string(raw: object, field: str, manifest_path: Path) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{manifest_path}: job {field!r} must be a non-empty string")
    return raw


def _job(raw: object, manifest_path: Path) -> AppJob:
    if not isinstance(raw, dict):
        raise ValueError(f"{manifest_path}: each job must be a table")
    unknown = set(raw) - JOB_KEYS
    if unknown:
        raise ValueError(f"{manifest_path}: job has unknown keys {sorted(unknown)}")
    missing = JOB_KEYS - {"secrets"} - set(raw)
    if missing:
        raise ValueError(f"{manifest_path}: job is missing required keys {sorted(missing)}")

    name = _string(raw["name"], "name", manifest_path)
    runner = _string(raw["runner"], "runner", manifest_path)
    for field, value in (("name", name), ("runner", runner)):
        if not APP_NAME_PATTERN.fullmatch(value):
            raise ValueError(f"{manifest_path}: job {field} {value!r} must match {APP_NAME_PATTERN.pattern}")

    schedule = _string(raw["schedule"], "schedule", manifest_path)
    if len(schedule.split()) != 5:
        raise ValueError(f"{manifest_path}: job schedule {schedule!r} must have five cron fields")

    command = raw["command"]
    if not isinstance(command, list) or not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError(f"{manifest_path}: job command must be a non-empty array of non-empty strings")
    positive_integers: dict[str, int] = {}
    for field in ("timeout", "cpu", "memory_gib"):
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{manifest_path}: job {field} must be a positive integer")
        positive_integers[field] = value
    secrets = raw.get("secrets", [])
    if not isinstance(secrets, list) or any(
        not isinstance(secret, str) or not ENV_NAME_PATTERN.fullmatch(secret) for secret in secrets
    ):
        raise ValueError(f"{manifest_path}: job secrets must be environment-variable names")
    if len(secrets) != len(set(secrets)):
        raise ValueError(f"{manifest_path}: job secrets must not contain duplicates")
    return AppJob(
        name,
        runner,
        schedule,
        tuple(command),
        positive_integers["timeout"],
        positive_integers["cpu"],
        positive_integers["memory_gib"],
        tuple(secrets),
    )


def load_manifest(app_dir: Path) -> AppManifest:
    """Parse ``app_dir/app.toml``; raise ValueError on a missing, unknown, or malformed key."""
    name = app_dir.name
    if not APP_NAME_PATTERN.match(name):
        raise ValueError(f"app directory {name!r} must match {APP_NAME_PATTERN.pattern}")
    manifest_path = app_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise ValueError(f"{app_dir} has no {MANIFEST_FILE}")
    raw = tomllib.loads(manifest_path.read_text())
    unknown = set(raw) - KNOWN_KEYS
    if unknown:
        raise ValueError(f"{manifest_path}: unknown keys {sorted(unknown)}")
    for key in ("title", "description"):
        if key not in raw:
            raise ValueError(f"{manifest_path}: missing required key {key!r}")
    raw_jobs = raw.get("jobs", [])
    if not isinstance(raw_jobs, list):
        raise ValueError(f"{manifest_path}: jobs must be an array of tables")
    jobs = tuple(_job(job, manifest_path) for job in raw_jobs)
    names = [job.name for job in jobs]
    if len(names) != len(set(names)):
        raise ValueError(f"{manifest_path}: job names must be unique")
    return AppManifest(
        name=name,
        title=raw["title"],
        description=raw["description"],
        root=app_dir,
        connect_src=tuple(raw.get("connect_src", [])),
        build_command=raw.get("build_command"),
        jobs=jobs,
    )


def discover_apps(apps_dir: Path) -> list[AppManifest]:
    """Load every app under ``apps_dir`` in name order.

    Directories starting with ``_`` or ``.`` are not apps. Any other directory without a
    manifest is an error rather than silently skipped.
    """
    if not apps_dir.is_dir():
        raise ValueError(f"apps directory {apps_dir} does not exist")
    return [
        load_manifest(child)
        for child in sorted(apps_dir.iterdir())
        if child.is_dir() and not child.name.startswith(("_", "."))
    ]


def job_runners(apps: Sequence[AppManifest]) -> tuple[JobRunner, ...]:
    """Group app jobs by runner and reject conflicting schedules."""
    grouped: dict[str, list[BoundJob]] = {}
    for app in apps:
        for job in app.jobs:
            grouped.setdefault(job.runner, []).append(BoundJob(app, job))

    runners = []
    for name, jobs in sorted(grouped.items()):
        schedules = {bound.job.schedule for bound in jobs}
        if len(schedules) != 1:
            raise ValueError(f"runner {name!r} has conflicting schedules {sorted(schedules)}")
        ordered = tuple(sorted(jobs, key=lambda bound: bound.qualified_name))
        runners.append(JobRunner(name, schedules.pop(), ordered))
    return tuple(runners)
