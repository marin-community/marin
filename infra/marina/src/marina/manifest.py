# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Application metadata and source-directory inspection for Marina packages."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import ConfigDict, Field, StrictInt, StrictStr, StringConstraints, TypeAdapter
from pydantic.dataclasses import dataclass as validated_dataclass

MANIFEST_FILE = "applet.toml"
APP_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
KERNEL_PREFIXES = frozenset({"a", "api", "healthz", "marina", "public"})
ENV_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
MODULE_PATTERN = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")


AppName = Annotated[StrictStr, StringConstraints(pattern=APP_NAME_PATTERN)]
ModuleName = Annotated[StrictStr, StringConstraints(pattern=MODULE_PATTERN)]
EnvironmentName = Annotated[StrictStr, StringConstraints(pattern=ENV_NAME_PATTERN)]
NonemptyString = Annotated[StrictStr, StringConstraints(strip_whitespace=True, min_length=1)]
Starter = Annotated[StrictStr, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]


@validated_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class AppJob:
    name: AppName
    runner: AppName
    module: ModuleName
    timeout: Annotated[StrictInt, Field(gt=0)]
    args: tuple[StrictStr, ...] = ()
    secrets: tuple[EnvironmentName, ...] = ()


@validated_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class AgentPanelManifest:
    profile: NonemptyString
    repository: NonemptyString
    starters: Annotated[tuple[Starter, ...], Field(max_length=3)] = ()


JOBS = TypeAdapter(tuple[AppJob, ...])
AGENT_PANEL = TypeAdapter(AgentPanelManifest)


def app_jobs(raw: object) -> tuple[AppJob, ...]:
    """Parse jobs whose modules run from the selected immutable package."""
    jobs = JOBS.validate_python(raw)
    if len({job.name for job in jobs}) != len(jobs):
        raise ValueError("job names must be unique")
    return jobs


def agent_panel(raw: object) -> AgentPanelManifest:
    return AGENT_PANEL.validate_python(raw)


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

    @property
    def path(self) -> str:
        return f"/{self.name}/"

    @property
    def dist(self) -> Path:
        return self.root / "dist"
