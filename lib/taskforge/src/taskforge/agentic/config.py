# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The settings of an agentic Taskforge session: the models, the limits, the gate, and the tool budget."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from rolloutengine.models import Model, RolloutLimits
from shellbox.machine import MachineFactory

from taskforge.agentic.gate import Gate
from taskforge.llm.client import ModelClient
from taskforge.llm.policy import LLMPolicy


class Web(Protocol):
    """Web search and page reading for the agent's web tools, for example ``ParallelWeb``."""

    async def search(self, objective: str, queries: Sequence[str]) -> dict[str, Any]: ...

    async def extract(self, urls: Sequence[str], objective: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ToolBudget:
    """The most calls that the tools make for one idea. The agent's own tokens fall under ``RolloutLimits``."""

    solver_rollouts: int
    judge_calls: int
    web_calls: int


@dataclass(frozen=True)
class AgenticConfig:
    """The models and limits of a session. One config can serve many ideas at the same time."""

    agent_model: Model
    """The agent that makes the tasks, for example a ``rolloutengine.chat.ChatModel`` on GLM."""
    workspace_factory: MachineFactory
    """The agent's machine: a ``LocalMachineFactory`` (bubblewrap) whose mounts give its shell Python with
    taskcompendium and verifyit, or a ShellSim factory, which has no Python."""
    solver_model: Model
    """The policy of the validation rollouts."""
    judge: ModelClient
    """The independent reviewer that ``submit`` asks last."""
    judge_policy: LLMPolicy
    web: Web
    agent_limits: RolloutLimits
    solver_limits: RolloutLimits
    gate: Gate
    budget: ToolBudget
    command_timeout: float
    """Seconds that one shell command can run, in the agent's workspace and in a solver's."""
    max_concurrent_rollouts: int
    """Solver rollouts that the tools of one session run at the same time."""
