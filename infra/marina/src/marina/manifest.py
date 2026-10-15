# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Application metadata and source-directory inspection for Marina packages."""

import re
from dataclasses import dataclass
from pathlib import Path

MANIFEST_FILE = "applet.toml"
APP_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
KERNEL_PREFIXES = frozenset({"a", "api", "healthz", "marina", "public"})
ENV_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
MODULE_PATTERN = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")


@dataclass(frozen=True)
class AppJob:
    name: str
    runner: str
    module: str
    timeout: int
    args: tuple[str, ...] = ()
    secrets: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentPanelManifest:
    profile: str
    repository: str
    starters: tuple[str, ...] = ()


def app_jobs(raw: object) -> tuple[AppJob, ...]:
    """Parse jobs whose modules run from the selected immutable package."""
    if not isinstance(raw, list):
        raise ValueError("jobs must be an array of tables")
    jobs = []
    for job in raw:
        if not isinstance(job, dict) or set(job) - {"name", "runner", "module", "timeout", "args", "secrets"}:
            raise ValueError("job contains unknown fields")
        for key in ("name", "runner"):
            if not isinstance(job.get(key), str) or not APP_NAME_PATTERN.fullmatch(job[key]):
                raise ValueError(f"job {key} must be a lowercase application name")
        if not isinstance(job.get("module"), str) or not MODULE_PATTERN.fullmatch(job["module"]):
            raise ValueError("job module must be a Python module within the package")
        timeout = job.get("timeout")
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
            raise ValueError("job timeout must be a positive integer")
        args = job.get("args", [])
        secrets = job.get("secrets", [])
        if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
            raise ValueError("job args must be strings")
        if not isinstance(secrets, list) or any(
            not isinstance(secret, str) or not ENV_NAME_PATTERN.fullmatch(secret) for secret in secrets
        ):
            raise ValueError("job secrets must be environment variable names")
        jobs.append(AppJob(job["name"], job["runner"], job["module"], timeout, tuple(args), tuple(secrets)))
    if len({job.name for job in jobs}) != len(jobs):
        raise ValueError("job names must be unique")
    return tuple(jobs)


def agent_panel(raw: object) -> AgentPanelManifest:
    if not isinstance(raw, dict) or set(raw) - {"profile", "repository", "starters"}:
        raise ValueError("agent must contain profile, repository and optional starters")
    for field in ("profile", "repository"):
        if not isinstance(raw.get(field), str) or not raw[field].strip():
            raise ValueError(f"agent {field} must be a nonempty string")
    starters = raw.get("starters", [])
    if (
        not isinstance(starters, list)
        or len(starters) > 3
        or any(not isinstance(starter, str) or not 1 <= len(starter.strip()) <= 160 for starter in starters)
    ):
        raise ValueError("agent starters must contain at most three strings of 1 to 160 characters")
    return AgentPanelManifest(raw["profile"], raw["repository"], tuple(starters))


@dataclass(frozen=True)
class AppManifest:
    """A source directory used by local builds and browser journeys."""

    name: str
    title: str
    description: str
    root: Path
    connect_src: tuple[str, ...] = ()
    build_command: str | None = None
    data_url: str | None = None
    jobs: tuple[AppJob, ...] = ()
    agent: AgentPanelManifest | None = None
    python_entrypoint: str | None = None

    @property
    def path(self) -> str:
        return f"/{self.name}/"

    @property
    def dist(self) -> Path:
        return self.root / "dist"
