"""Harnesses for SkyRL's direct task episodes."""

from collections.abc import Callable, Mapping
from concurrent.futures import Executor
from dataclasses import dataclass, replace
from functools import partial
from typing import Protocol

from rolloutengine.harness.core import Harness, HarnessContext, ModelSession
from rolloutengine.harness.step import StepHarness
from rolloutengine.models import ModelTurn, Prompt, RolloutLimits, TaskRolloutSpec, Transition
from shellbox.machine import Machine
from taskcompendium.models import TaskSpec
from taskcompendium.grading_result import GradeResult

from skyrl_gym.answer_tasks import (
    grade_aime,
    grade_cat_count,
    grade_sequence,
    grade_gsm8k,
    grade_ifeval,
    grade_mcq,
    grade_nupa,
    grade_prompt_only,
    grade_reasoning_gym,
)
from skyrl_gym.closure_task import STRUCTURE_CLOSURE_SESSION, ClosureReplaySession
from skyrl_gym.nemotron_tasks import NemotronTaskSession
from skyrl_gym.sql_tasks import SQLTaskSession, SeededSQLTaskSession
from skyrl_gym.task_records import TaskTransition
from skyrl_gym.task_sessions import (
    AnswerTaskSession,
    CodeTaskSession,
    MathTaskSession,
    SearchCodeTaskSession,
    SearchTaskSession,
)
from skyrl_gym.tool_protocol_task import ToolProtocolReplaySession


class TaskEpisode(Protocol):
    async def prepare(self) -> Prompt: ...

    async def advance(self, turn: ModelTurn) -> TaskTransition: ...

    async def grade(
        self, messages: tuple[dict, ...], *, stop_reason: str, outputs: dict[str, bytes]
    ) -> GradeResult: ...

    async def close(self) -> None: ...


EpisodeFactory = Callable[[TaskRolloutSpec, Machine | None], TaskEpisode]


@dataclass
class _StepEpisode:
    episode: TaskEpisode

    async def prepare(self) -> Prompt:
        return await self.episode.prepare()

    async def advance(self, turn: ModelTurn) -> Transition:
        transition = await self.episode.advance(turn)
        if transition.done:
            grade = await self.episode.grade((), stop_reason=turn.stop_reason, outputs={})
            transition = replace(transition, task_grade=grade)
        return transition.for_harness()

    async def close(self) -> None:
        await self.episode.close()


@dataclass(frozen=True)
class _NemotronHarness:
    open_episode: Callable[[HarnessContext], TaskEpisode]

    def check(self, task: TaskSpec, limits: RolloutLimits) -> None:
        pass

    async def run(self, context: HarnessContext, session: ModelSession) -> None:
        episode = self.open_episode(context)
        context.cleanup.push("episode_close", episode.close)
        turn = await session.start(await episode.prepare())
        while True:
            transition = await episode.advance(turn)
            if transition.done:
                grade = await episode.grade((), stop_reason=turn.stop_reason, outputs={})
                transition = replace(transition, task_grade=grade)
            session.record(
                observations=transition.observations if transition.done else (),
                data=transition.for_harness().data,
                metrics=transition.metrics,
            )
            if transition.done:
                return
            if transition.reset_conversation is not None:
                turn = await session.start(Prompt(transition.reset_conversation, {}))
            else:
                turn = await session.reply(transition.observations, token_ids=transition.token_ids)


def episode_factories(*, executor: Executor | None = None) -> dict[str, EpisodeFactory]:
    """Return direct task episode constructors keyed by harness name."""
    answer_graders = {
        "aime": grade_aime,
        "cat_count": grade_cat_count,
        "sequence": grade_sequence,
        "gsm8k": grade_gsm8k,
        "ifeval": grade_ifeval,
        "mcq": grade_mcq,
        "nupa": grade_nupa,
        "preference": grade_prompt_only,
        "prompt_only": grade_prompt_only,
        "reasoning_gym": grade_reasoning_gym,
    }
    return {
        **{
            name: partial(AnswerTaskSession, grader=grader, executor=executor)
            for name, grader in answer_graders.items()
        },
        "gsm8k_multi_turn": MathTaskSession,
        "search": partial(SearchTaskSession, executor=executor),
        "searchcode": partial(SearchCodeTaskSession, executor=executor),
        "lcb": CodeTaskSession,
        "nemotron_ultra": partial(NemotronTaskSession, executor=executor),
        "text2sql": partial(SQLTaskSession, executor=executor),
        "text_to_sql": partial(SeededSQLTaskSession, executor=executor),
        "tool_protocol_replay": ToolProtocolReplaySession,
        STRUCTURE_CLOSURE_SESSION: ClosureReplaySession,
    }


def episode_harnesses(spec: TaskRolloutSpec, factories: Mapping[str, EpisodeFactory]) -> dict[str, Harness]:
    """Bind a task's limits to the episodes registered for this attempt."""
    harnesses: dict[str, Harness] = {}
    for name, factory in factories.items():

        def open_episode(context: HarnessContext, factory: EpisodeFactory = factory) -> TaskEpisode:
            return factory(spec, context.machine)

        if name == "nemotron_ultra":
            harnesses[name] = _NemotronHarness(open_episode)
        else:
            harnesses[name] = StepHarness(
                lambda context, open_episode=open_episode: _StepEpisode(open_episode(context))
            )
    return harnesses
