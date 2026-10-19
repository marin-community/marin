"""Explicit factories for the task implementations supplied by SkyRL."""

from collections.abc import Callable
from concurrent.futures import Executor
from functools import partial

from rolloutengine.contracts import TaskSession, TaskSessionContext, TaskSessionFactory
from shellbox.machine import Machine
from rolloutengine.spec import LoweredTaskSpec

from skyrl_gym.answer_tasks import (
    grade_aime,
    grade_cat_count,
    grade_gsm8k,
    grade_ifeval,
    grade_mcq,
    grade_nupa,
    grade_prompt_only,
    grade_reasoning_gym,
)
from skyrl_gym.task_sessions import (
    AnswerTaskSession,
    CodeTaskSession,
    MathTaskSession,
    SearchCodeTaskSession,
    SearchTaskSession,
)
from skyrl_gym.nemotron_tasks import NemotronTaskSession
from skyrl_gym.sql_tasks import SQLTaskSession, SeededSQLTaskSession
from skyrl_gym.tool_protocol_task import ToolProtocolReplaySession
from skyrl_gym.closure_task import STRUCTURE_CLOSURE_SESSION, ClosureReplaySession


def adapt_self_grading_session(factory: Callable[[LoweredTaskSpec, Machine | None], TaskSession]) -> TaskSessionFactory:
    def create(context: TaskSessionContext) -> TaskSession:
        return factory(context.lowered, context.machine)

    return create


def session_factories(*, executor: Executor | None = None) -> dict[str, TaskSessionFactory]:
    """Return factories that apply each lowered task's session settings."""
    answer_graders = {
        "aime": grade_aime,
        "cat_count": grade_cat_count,
        "gsm8k": grade_gsm8k,
        "ifeval": grade_ifeval,
        "mcq": grade_mcq,
        "nupa": grade_nupa,
        "preference": grade_prompt_only,
        "prompt_only": grade_prompt_only,
        "reasoning_gym": grade_reasoning_gym,
    }
    constructors: dict[str, Callable[[LoweredTaskSpec, Machine | None], TaskSession]] = {
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
    return {name: adapt_self_grading_session(factory) for name, factory in constructors.items()}
