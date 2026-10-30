# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The step harness: an episode gives the first prompt and answers each model turn."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from taskcompendium.models import TaskSpec

from rolloutengine.harness.core import HarnessContext, ModelSession
from rolloutengine.models import ModelTurn, Prompt, RolloutLimits, Transition


class Episode(Protocol):
    """One attempt of a step-style harness: the first prompt, then the response to each model turn."""

    async def prepare(self) -> Prompt: ...

    async def advance(self, turn: ModelTurn) -> Transition: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class StepHarness:
    """A harness for an ``Episode``: ``prepare`` gives the first prompt, and ``advance`` answers each model turn.

    The episode closes in the attempt's cleanup.
    A harness that must reject a task before any machine starts implements ``Harness`` itself,
    and its ``run`` can call this harness's ``run``.
    """

    open_episode: Callable[[HarnessContext], Episode]
    """Starts one episode per attempt, for example an episode class."""

    def check(self, task: TaskSpec, limits: RolloutLimits) -> None:
        pass

    async def run(self, context: HarnessContext, session: ModelSession) -> None:
        episode = self.open_episode(context)
        context.cleanup.push("episode_close", episode.close)
        turn = await session.start(await episode.prepare())
        while True:
            transition = await episode.advance(turn)
            final = transition.observations if transition.done else ()
            session.record(observations=final, data=transition.data, metrics=transition.metrics)
            if transition.done:
                return
            turn = await session.reply(transition.observations, token_ids=transition.token_ids)
