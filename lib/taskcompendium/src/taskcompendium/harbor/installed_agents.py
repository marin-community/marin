# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline installation policy for Harbor's native mini-SWE-agent runner."""

from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.environments.base import BaseEnvironment

MINI_SWE_AGENT_VERSION = "2.4.6"


class PreinstalledMiniSweAgent(MiniSweAgent):
    """Preserve Harbor's agent loop while requiring an image with pinned dependencies."""

    async def install(self, environment: BaseEnvironment) -> None:
        result = await self.exec_as_agent(
            environment,
            command="python3 -c 'from importlib.metadata import version; print(version(\"mini-swe-agent\"))'",
        )
        actual = (result.stdout or "").strip()
        if result.return_code != 0 or actual != MINI_SWE_AGENT_VERSION:
            raise RuntimeError(f"Image must contain mini-swe-agent=={MINI_SWE_AGENT_VERSION}, found {actual!r}")
        if self._version is not None and self._version != actual:
            raise ValueError(f"Requested mini-SWE-agent version {self._version} differs from pinned {actual}")
        self._version = actual
        # The native runner sources this file before invoking its CLI.
        await self.exec_as_agent(environment, command='mkdir -p "$HOME/.local/bin"; touch "$HOME/.local/bin/env"')
