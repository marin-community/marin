# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The configuration of one unattended run, read from a JSON file in which every field is required.

No config object holds a secret, so a config file can be committed and shown. Local paths (a laptop
root, the image cache) are absolute: the reader does not expand ``~`` or resolve against the working
directory, so a config means the same thing wherever it is launched from.

A config file looks like ``docs/policy.example.json``::

    {"run_id": ..., "root": ..., "host": "laptop" | "iris", "image_cache": "<laptop directory>" | null,
     "policy": <loop.policy.POLICY>,
     "engine": {"max_turns": ..., "command_timeout": ..., "tool_turn_timeout": ...,
                "model_turn_timeout": ..., "cleanup_timeout": ...},
     "width": ...}
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shellbox.machine import MachineFactory

from taskforge.loop.policy import POLICY, LoopPolicy
from taskforge.sandbox.factories import FactoryCapabilities, MachineHost
from taskforge.validate.trials import EngineSettings

RUN_FIELDS = frozenset({"run_id", "root", "host", "image_cache", "policy", "engine", "width"})


def _require_absolute(path: Path, name: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"{name} is an absolute path (no ~, no working-directory-relative path), got {path}")


@dataclass(frozen=True)
class EngineConfig:
    """The run-wide RolloutEngine session limits (``validate.trials.EngineSettings``); the factories
    come from the host at run time. Each task carries its own answer format.
    """

    max_turns: int
    command_timeout: float
    tool_turn_timeout: float
    model_turn_timeout: float
    cleanup_timeout: float

    def settings(
        self, factories: Mapping[str, MachineFactory], capabilities: Mapping[str, FactoryCapabilities]
    ) -> EngineSettings:
        """These limits over ``factories`` and ``capabilities``, both keyed by shellbox ``Backend`` value."""
        return EngineSettings(
            factories=factories,
            capabilities=capabilities,
            max_turns=self.max_turns,
            command_timeout=self.command_timeout,
            tool_turn_timeout=self.tool_turn_timeout,
            model_turn_timeout=self.model_turn_timeout,
            cleanup_timeout=self.cleanup_timeout,
        )


@dataclass(frozen=True)
class RunConfig:
    """One unattended run.

    Attributes:
        run_id: Names the run in ``summary.json``.
        root: The run root (``items/``, ``cache/``, ``ledger/``, ``policy.json``, ``summary.json``).
            Absolute on a laptop; on Iris it is relative and placed under ``$IRIS_OUTPUT_DIR``, which
            Iris archives per attempt.
        host: Where machine factories come from.
        image_cache: Where the laptop Docker factory keeps the images it prepares; an absolute
            directory on a laptop, None on Iris (``sandbox.factories.machine_factories``).
        policy: Every bound of the run.
        engine: The run-wide RolloutEngine settings.
        width: Concurrent phases across items (``LoopServices.slots``); not a request limit.
    """

    run_id: str
    root: Path
    host: MachineHost
    image_cache: Path | None
    policy: LoopPolicy
    engine: EngineConfig
    width: int

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError(f"width must be at least 1, got {self.width}")
        if (self.host is MachineHost.LAPTOP) != (self.image_cache is not None):
            raise ValueError(
                f"image_cache is a directory on a laptop and null on Iris; this {self.host} run has {self.image_cache}"
            )
        if self.host is MachineHost.IRIS and self.root.is_absolute():
            raise ValueError(f"an Iris run root is relative to $IRIS_OUTPUT_DIR, got {self.root}")
        if self.host is MachineHost.LAPTOP:
            _require_absolute(self.root, "a laptop run root")
            assert self.image_cache is not None
            _require_absolute(self.image_cache, "image_cache")


def _fields(obj: Mapping[str, Any], names: frozenset[str], where: str) -> None:
    if set(obj) != names:
        missing, unknown = sorted(names - set(obj)), sorted(set(obj) - names)
        raise ValueError(f"{where}: missing {missing}, unknown {unknown}")


def engine_config(obj: Mapping[str, Any]) -> EngineConfig:
    _fields(
        obj,
        frozenset({"max_turns", "command_timeout", "tool_turn_timeout", "model_turn_timeout", "cleanup_timeout"}),
        "engine",
    )
    return EngineConfig(
        max_turns=obj["max_turns"],
        command_timeout=obj["command_timeout"],
        tool_turn_timeout=obj["tool_turn_timeout"],
        model_turn_timeout=obj["model_turn_timeout"],
        cleanup_timeout=obj["cleanup_timeout"],
    )


def run_config(obj: Mapping[str, Any]) -> RunConfig:
    """A ``RunConfig`` from its JSON object; a missing or unknown field raises ``ValueError``."""
    _fields(obj, RUN_FIELDS, "run config")
    return RunConfig(
        run_id=obj["run_id"],
        root=Path(obj["root"]),
        host=MachineHost(obj["host"]),
        image_cache=None if obj["image_cache"] is None else Path(obj["image_cache"]),
        policy=POLICY.validate_python(obj["policy"]),
        engine=engine_config(obj["engine"]),
        width=obj["width"],
    )


def load_run_config(path: Path) -> RunConfig:
    return run_config(json.loads(path.read_text()))
