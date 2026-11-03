# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The settings of an agentic Taskforge session: the models, the limits, the gate, and the tool budget."""

from dataclasses import dataclass

from rolloutengine.models import Model, RolloutLimits
from shellbox.machine import MachineFactory

from taskforge.agentic.gate import Gate
from taskforge.llm.client import ModelClient
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.web import Web


@dataclass(frozen=True)
class ToolBudget:
    """The most calls that the tools make for one idea. The agent's own tokens fall under ``RolloutLimits``."""

    solver_rollouts: int | None
    judge_calls: int | None
    web_calls: int | None


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
    solver_factory: MachineFactory
    """The machines of the validation rollouts: a ShellSim factory, or a ``LocalMachineFactory`` (bubblewrap) whose
    shell runs real Linux programs. A task declares the command semantics of this backend."""
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
