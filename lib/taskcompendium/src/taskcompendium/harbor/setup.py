# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decode a Harbor task package's setup declaration."""

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class HarborSetup:
    command: str
    timeout_sec: float | None


def read_harbor_setup(document: Mapping[str, object], *, source: str) -> HarborSetup | None:
    """Read the optional setup declaration from a parsed Harbor task.toml."""
    environment = document.get("environment")
    if not isinstance(environment, Mapping) or "setup" not in environment:
        return None
    setup = environment["setup"]
    if not isinstance(setup, Mapping):
        raise ValueError(f"{source}: [environment.setup] must be a table")
    if unknown := setup.keys() - {"command", "timeout_sec"}:
        fields = ", ".join(f"environment.setup.{key}" for key in sorted(unknown))
        raise ValueError(f"{source} has unsupported Harbor setup keys: {fields}")
    command = setup.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError(f"{source}: environment.setup.command must be a nonblank string")
    if "timeout_sec" not in setup:
        return HarborSetup(command, None)
    timeout = setup["timeout_sec"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < math.inf:
        raise ValueError(f"{source}: environment.setup.timeout_sec must be a positive finite number")
    return HarborSetup(command, timeout)
