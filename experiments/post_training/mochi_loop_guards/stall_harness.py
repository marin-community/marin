# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A scripted looping harness and model for local stall-stop validation.

The local end-to-end run drives the real recorder through this harness: the model repeats
one Terminus command and the harness replies with an identical observation, so the stall
window fills. The task, engine, and rollout-spec fixtures come from the engine's own test
support, so the run exercises the same grading path the engine ships with.
"""

from dataclasses import dataclass, field

from rolloutengine.harness.core import HarnessContext, ModelSession
from rolloutengine.models import ContextRecovery, LimitReached, ModelTurn, Prompt
from rolloutengine.test_support import arithmetic_task, engine, rollout_spec


@dataclass
class ScriptedHarness:
    """A harness that replies a fixed number of times, with per-reply observations."""

    replies: int
    observations: tuple[str, ...] | None = None
    """One observation per reply; a harness that never varies it leaves this ``None``."""
    context_recovery = ContextRecovery.NEVER
    refused: list[str] = field(default_factory=list)
    """The stop reasons of every ``LimitReached`` the harness saw."""

    def check(self, task, limits) -> None:
        pass

    def observation(self, index: int) -> tuple[dict, ...]:
        if self.observations is None:
            return ({"role": "user", "content": "unchanged output"},)
        return ({"role": "user", "content": self.observations[index]},)

    async def run(self, context: HarnessContext, session: ModelSession) -> None:
        await session.start(Prompt(({"role": "user", "content": "Run the check."},), {}))
        for index in range(self.replies):
            try:
                await session.reply(self.observation(index))
            except LimitReached as error:
                self.refused.append(error.stop_reason)
                raise


@dataclass
class ConstantModel:
    """A model that answers every request with the same message."""

    content: str

    async def complete(self, request) -> ModelTurn:
        served = request.continuation.served_token_ids if request.continuation is not None else (10, 11)
        return ModelTurn({"role": "assistant", "content": self.content}, served, (20,), (-0.5,), "stop")


__all__ = ["ConstantModel", "ScriptedHarness", "arithmetic_task", "engine", "rollout_spec"]
