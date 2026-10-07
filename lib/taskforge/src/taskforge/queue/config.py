# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The configuration of one unattended run, read from a JSON file in which every field is required.

The GLM endpoint is a typed choice: ``LaptopGlm`` (a base URL and a token file, for a port-forward)
or ``RelayGlm`` (an Iris relay job and the name of the environment variable that holds the token).
The builders' Parallel key is a file or an environment variable name in the same way. No config
object holds a secret, so a config file can be committed and shown; ``queue.job`` reads secrets at
the boundary. The GLM pool is always named, because the token binds which router pool serves a
request.

A config file looks like ``docs/policy.example.json``::

    {"run_id": ..., "root": ..., "host": "laptop" | "iris", "image_cache": "<laptop directory>" | null,
     "glm": {"kind": "laptop", "base_url": ..., "token_file": ..., "pool": "high" | "bulk"}
          | {"kind": "relay", "relay_job": ..., "token_env": ..., "pool": "high" | "bulk"},
     "web": null | {"kind": "key_file", "path": ...} | {"kind": "key_env", "env": ...},
     "policy": <loop.policy.POLICY>,
     "engine": {"max_turns": ..., "command_timeout": ..., "cleanup_timeout": ...,
                "conventions": [{"type": "PlainText", "convention": {"id": ...}}, ...]},
     "width": ..., "restore_from": null | "<archived run root>"}
"""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from shellbox.machine import MachineFactory
from taskcompendium.environment import EnvironmentKind
from taskcompendium.submission import SubmissionConvention

from taskforge.build.run import CONVENTION_TYPES
from taskforge.llm.client import Pool
from taskforge.loop.policy import POLICY, LoopPolicy
from taskforge.sandbox.factories import FactoryCapabilities, MachineHost
from taskforge.validate.trials import EngineSettings

ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")
RUN_FIELDS = frozenset(
    {"run_id", "root", "host", "image_cache", "glm", "web", "policy", "engine", "width", "restore_from"}
)


class GlmKind(StrEnum):
    """The ``kind`` tag of a config's ``glm`` object."""

    LAPTOP = "laptop"
    RELAY = "relay"


@dataclass(frozen=True)
class LaptopGlm:
    """GLM through a reachable base URL (a port-forward); the token is on a ``GLM_API_TOKEN=`` line of a file."""

    base_url: str
    token_file: Path
    pool: Pool


@dataclass(frozen=True)
class RelayGlm:
    """GLM through an Iris relay job, resolved inside the task; the token is in environment variable ``token_env``.

    ``relay_job`` is explicit per cluster; nothing falls back to a default relay.
    """

    relay_job: str
    token_env: str
    pool: Pool

    def __post_init__(self) -> None:
        if not ENV_NAME.fullmatch(self.token_env):
            raise ValueError("RelayGlm.token_env names an environment variable, for example GLM_API_TOKEN")


type GlmConfig = LaptopGlm | RelayGlm


class WebKind(StrEnum):
    """The ``kind`` tag of a config's ``web`` object."""

    KEY_FILE = "key_file"
    KEY_ENV = "key_env"


@dataclass(frozen=True)
class ParallelKeyFile:
    """The Parallel API key for the builders' web tools, on a ``PARALLEL_KEY=`` line of ``path``."""

    path: Path


@dataclass(frozen=True)
class ParallelKeyEnv:
    """The Parallel API key for the builders' web tools, in environment variable ``env``."""

    env: str

    def __post_init__(self) -> None:
        if not ENV_NAME.fullmatch(self.env):
            raise ValueError("ParallelKeyEnv.env names an environment variable, for example PARALLEL_KEY")


type WebConfig = ParallelKeyFile | ParallelKeyEnv


@dataclass(frozen=True)
class EngineConfig:
    """The run-wide RolloutEngine settings; the factories come from the host at run time.

    ``conventions`` are what tasks may be presented with, in preference order; each draft is
    validated under its own convention only.
    """

    max_turns: int
    command_timeout: float
    cleanup_timeout: float
    conventions: tuple[SubmissionConvention, ...]

    def settings(
        self,
        factories: Mapping[EnvironmentKind, MachineFactory],
        capabilities: Mapping[EnvironmentKind, FactoryCapabilities],
    ) -> EngineSettings:
        return EngineSettings(
            factories=factories,
            capabilities=capabilities,
            max_turns=self.max_turns,
            command_timeout=self.command_timeout,
            cleanup_timeout=self.cleanup_timeout,
            conventions=self.conventions,
        )


@dataclass(frozen=True)
class RunConfig:
    """One unattended run.

    Attributes:
        run_id: Names the run in the Finelog ledger mirror and in ``summary.json``.
        root: The run root (``items/``, ``cache/``, ``ledger/``, ``policy.json``, ``summary.json``).
            On Iris it is relative and placed under ``$IRIS_OUTPUT_DIR``, which Iris archives per attempt.
        host: Where machine factories come from.
        image_cache: Where the laptop Docker factory keeps the images it prepares; required on a
            laptop, None on Iris (``sandbox.factories.machine_factories``).
        glm: The GLM endpoint, resolved once by ``queue.job``.
        web: Where the builders' Parallel key comes from, or None for builders without web tools.
        policy: Every bound of the run.
        engine: The run-wide RolloutEngine settings.
        width: Concurrent phases across items (``LoopServices.slots``); not a request limit.
        restore_from: A previous attempt's archived run root, copied in before resuming.
    """

    run_id: str
    root: Path
    host: MachineHost
    image_cache: Path | None
    glm: GlmConfig
    web: WebConfig | None
    policy: LoopPolicy
    engine: EngineConfig
    width: int
    restore_from: str | None

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError(f"width must be at least 1, got {self.width}")
        if (self.host is MachineHost.LAPTOP) != (self.image_cache is not None):
            raise ValueError(f"a {self.host} run takes an image_cache only on a laptop, got {self.image_cache}")
        if self.host is MachineHost.IRIS and self.root.is_absolute():
            raise ValueError(f"an Iris run root is relative to $IRIS_OUTPUT_DIR, got {self.root}")


def _fields(obj: Mapping[str, Any], names: frozenset[str], where: str) -> None:
    if set(obj) != names:
        missing, unknown = sorted(names - set(obj)), sorted(set(obj) - names)
        raise ValueError(f"{where}: missing {missing}, unknown {unknown}")


def _kind(obj: Mapping[str, Any], where: str) -> str:
    if "kind" not in obj:
        raise ValueError(f"{where}: missing ['kind']")
    return obj["kind"]


def glm_config(obj: Mapping[str, Any]) -> GlmConfig:
    if GlmKind(_kind(obj, "glm")) is GlmKind.LAPTOP:
        _fields(obj, frozenset({"kind", "base_url", "token_file", "pool"}), "glm")
        return LaptopGlm(obj["base_url"], Path(obj["token_file"]).expanduser(), Pool(obj["pool"]))
    _fields(obj, frozenset({"kind", "relay_job", "token_env", "pool"}), "glm")
    return RelayGlm(obj["relay_job"], obj["token_env"], Pool(obj["pool"]))


def web_config(obj: Mapping[str, Any] | None) -> WebConfig | None:
    if obj is None:
        return None
    if WebKind(_kind(obj, "web")) is WebKind.KEY_FILE:
        _fields(obj, frozenset({"kind", "path"}), "web")
        return ParallelKeyFile(Path(obj["path"]).expanduser())
    _fields(obj, frozenset({"kind", "env"}), "web")
    return ParallelKeyEnv(obj["env"])


def engine_config(obj: Mapping[str, Any]) -> EngineConfig:
    _fields(obj, frozenset({"max_turns", "command_timeout", "cleanup_timeout", "conventions"}), "engine")
    return EngineConfig(
        max_turns=obj["max_turns"],
        command_timeout=obj["command_timeout"],
        cleanup_timeout=obj["cleanup_timeout"],
        conventions=tuple(convention(c) for c in obj["conventions"]),
    )


def convention(obj: Mapping[str, Any]) -> SubmissionConvention:
    _fields(obj, frozenset({"type", "convention"}), "convention")
    if obj["type"] not in CONVENTION_TYPES:
        raise ValueError(f"convention: unknown type {obj['type']!r}, known {sorted(CONVENTION_TYPES)}")
    return CONVENTION_TYPES[obj["type"]].model_validate(obj["convention"])


def run_config(obj: Mapping[str, Any]) -> RunConfig:
    """A ``RunConfig`` from its JSON object; a missing or unknown field raises ``ValueError``."""
    _fields(obj, RUN_FIELDS, "run config")
    return RunConfig(
        run_id=obj["run_id"],
        root=Path(obj["root"]).expanduser(),
        host=MachineHost(obj["host"]),
        image_cache=None if obj["image_cache"] is None else Path(obj["image_cache"]).expanduser(),
        glm=glm_config(obj["glm"]),
        web=web_config(obj["web"]),
        policy=POLICY.validate_python(obj["policy"]),
        engine=engine_config(obj["engine"]),
        width=obj["width"],
        restore_from=obj["restore_from"],
    )


def load_run_config(path: Path) -> RunConfig:
    return run_config(json.loads(path.read_text()))
