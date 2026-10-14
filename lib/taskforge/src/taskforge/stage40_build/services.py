# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What a build uses outside its own directory: machines, images, web tools and the ledger.

The caller builds one ``RLEnvBuildServices`` for a host and passes it to ``run_build``, which puts it
on every step's context. Steps pass it on to the SDK's effects (``sandbox_effects``) and to the
agent loop. Services never enter a step key and never appear in a step output, so ``run_build``
folds the settings that change what a step returns, ``reasoning_replay`` and ``agent_limits``, into
the model digest of every key (``run.build_model_digest``).
"""

from collections.abc import Mapping
from dataclasses import dataclass

from shellbox.machine import MachineFactory

from taskforge.ledger.records import Ledger, LedgerEntry
from taskforge.llm.agent import AgentTool, ReasoningReplay
from taskforge.sandbox.factories import MachineHost
from taskforge.sandbox.images import ImageBuilder


@dataclass(frozen=True)
class AgentLimits:
    """The bounds of a step's model use.

    Attributes:
        max_turns: Model replies one agent loop may take.
        max_tokens: The output budget each turn sends. The client sends the whole budget with every
            turn, so it leaves room in the context window for the conversation the loop builds.
        max_calls: Model calls one step may make in all, counting every agent turn and every
            structured request; a program's own retry loop ends here, as the program's failure.
    """

    max_turns: int
    max_tokens: int
    max_calls: int


@dataclass(frozen=True)
class RLEnvBuildServices:
    """The host resources of a build.

    Attributes:
        host: Where the build runs; machine backends are chosen for it.
        factories: Machine factories keyed by shellbox ``Backend`` value, as
            ``taskforge.sandbox.factories.machine_factories(host, ...)`` returns them.
        images: Publishes task images by digest, or ``None`` when this host cannot.
        web_tools: Web search and fetch tools (``taskforge.llm.web.web_tools``), or empty.
        ledger: Where steps, model calls and tool calls are recorded, or ``None`` to record nothing.
        reasoning_replay: Whether every agent loop replays its assistant turns' reasoning.
        agent_limits: The bounds of every agent loop.
    """

    host: MachineHost
    factories: Mapping[str, MachineFactory]
    images: ImageBuilder | None
    web_tools: tuple[AgentTool, ...]
    ledger: Ledger | None
    reasoning_replay: ReasoningReplay
    agent_limits: AgentLimits

    @property
    def recording(self) -> Ledger:
        """``ledger``, or a ledger that drops every entry when ``ledger`` is ``None``."""
        return _DISCARD if self.ledger is None else self.ledger


class _DiscardingLedger:
    def record(self, entry: LedgerEntry) -> None:
        pass


_DISCARD = _DiscardingLedger()
