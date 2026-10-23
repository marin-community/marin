# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A scripted model for agentic author sessions, the session budgets, and a toy program a session submits.

``ScriptedModel`` is an in-process ``ModelClient`` that answers each request with the next queued
``Completion`` and keeps every request, so a test sets each turn's tool calls and token usage exactly.
``turn`` scripts an assistant turn making tool calls, and ``answer`` one replying without any.
``PROGRAM`` grades 6 * 7 in process and calls no model; ``in_process_loader`` is a session's
``ProgramLoader`` that loads in the test's process.
"""

import json
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from taskforge.llm.client import Completion, FinishReason, StructuredResult, ToolCall, Usage
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.structured import complete_structured, response_format
from taskforge.stage30_author.author import program_manifest
from taskforge.stage30_author.load_process import ProgramManifest
from taskforge.stage40_build_with_program.agentic_author import AgenticAuthorConfig, SameSessionBounds
from taskforge.stage40_build_with_program.program_sdk import load_program

PROGRAM = '''"""Grades 6 * 7 in process."""

from dataclasses import replace
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, Source, TaskSpec
from verifyit.spec import ExactSpec

from taskforge.spec import draft
from taskforge.stage10_proposal.model import TaskProposal
from taskforge.stage30_author.program import rl_env_build_program, rl_env_build_sink, rl_env_build_step
from taskforge.stage40_build_with_program.context import BuildFailure, RLEnvBuildContext
from taskforge.stage40_build_with_program.outputs import Grader, Sources
from taskforge.stage40_build_with_program.sandbox_effects import grade_candidate
from taskforge.stage40_build_with_program.sdk import ANSWER_FORMAT, graded_candidate


@rl_env_build_step
async def sources(context: RLEnvBuildContext, proposal: TaskProposal) -> Sources:
    return Sources(notes="", findings=())


@rl_env_build_step
async def grader(context: RLEnvBuildContext, sources: Sources) -> Grader:
    package = draft.answer_grader(ExactSpec(expected=("42",)))
    reference = await grade_candidate(
        context, draft.NO_MACHINE, package, AnswerType.TEXT, ANSWER_FORMAT, "(instruction)", "42"
    )
    if reference.status != Outcome.GRADED:
        raise BuildFailure(f"the reference was graded {reference.status}")
    return Grader(
        package, "Reply with the number.", "42", (), ("42",), 0.0, (graded_candidate("42", (), reference),)
    )


@rl_env_build_sink
async def assemble(context: RLEnvBuildContext, grader: Grader) -> TaskSpec:
    return draft.assemble(
        context.workpiece_id,
        "What is 6 * 7?",
        AnswerType.TEXT,
        ANSWER_FORMAT,
        grader.package,
        Source(dataset="test", revision="r", row="0", importer_revision="test"),
        environment=draft.requirements(image=None).model_copy(
            update={"package_spec": replace(draft.SHELLSIM_CPYTHON, pypi_packages=None)}
        ),
    )


PROGRAM = rl_env_build_program(sources, grader, sink=assemble)
'''
CONFIG = AgenticAuthorConfig(
    turns=48,
    turn_max_tokens=49_152,
    session_token_cap=262_144,
    revision_same_session=SameSessionBounds(150_000, 36),
    max_continuations=0,
)
STRUCTURED_REPAIRS = 1


@dataclass(frozen=True)
class ScriptedRequest:
    messages: list[Message]
    policy: LLMPolicy
    fields: dict[str, object]


class ScriptedModel:
    """Answers each request with the next queued completion; structured calls go through ``complete``."""

    model = "scripted"

    def __init__(self, *replies: Completion) -> None:
        self.replies: deque[Completion] = deque(replies)
        self.requests: list[ScriptedRequest] = []

    async def complete(
        self, messages: Sequence[Message], policy: LLMPolicy, request_fields: Mapping[str, object] | None = None
    ) -> Completion:
        self.requests.append(ScriptedRequest(list(messages), policy, dict(request_fields or {})))
        return self.replies.popleft()

    async def complete_schema(
        self, messages: Sequence[Message], output_type: type[BaseModel], policy: LLMPolicy
    ) -> Completion:
        return await self.complete(messages, policy, {"response_format": response_format(output_type)})

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        return await complete_structured(self, messages, output_type, policy, STRUCTURED_REPAIRS)


def turn(*calls: tuple[str, Mapping[str, object]], prompt: int = 30_000, out: int = 2_000) -> Completion:
    """An assistant turn making ``calls`` (tool name, arguments)."""
    return Completion(
        content="",
        reasoning="thinking",
        tool_calls=tuple(ToolCall(f"call-{uuid.uuid4().hex}", name, json.dumps(args)) for name, args in calls),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=Usage(prompt_tokens=prompt, completion_tokens=out, reasoning_tokens=out // 2, cached_tokens=prompt - 1000),
        wall_time=2.0,
        ttft=0.1,
        decode_time=1.9,
        continuations=0,
        attempts=(),
    )


def answer(text: str) -> Completion:
    return Completion(text, "", (), FinishReason.STOP, Usage(30_000, 100, 50, 29_000), 1.0, 0.1, 0.9, 0, ())


async def in_process_loader(source: str) -> ProgramManifest:
    """Loads in this process: the tests check the session, and the child loader has its own tests."""
    return ProgramManifest.model_validate(program_manifest(load_program(source)))
