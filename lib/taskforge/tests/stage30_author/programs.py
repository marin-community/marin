# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A small authored program, as source text, for the loader and author tests."""

PROGRAM = '''"""A program that grades 6 * 7 in process."""

from dataclasses import dataclass

from taskcompendium.models import AnswerType, Source, TaskSpec
from verifyit.spec import ExactSpec

from taskforge.spec import draft
from taskforge.stage30_author.program import rl_env_build_program, rl_env_build_sink, rl_env_build_step
from taskforge.stage40_build_with_program.context import RLEnvBuildContext
from taskforge.stage40_build_with_program.outputs import Grader
from taskforge.stage40_build_with_program.sdk import ANSWER_FORMAT


@dataclass(frozen=True)
class Question:
    text: str


@rl_env_build_step
async def question(context: RLEnvBuildContext) -> Question:
    return Question(text="What is 6 * 7?")


@rl_env_build_step
async def grader(context: RLEnvBuildContext) -> Grader:
    package = draft.answer_grader(ExactSpec(expected=("42",)))
    return Grader(package, "Reply with the number.", "42", (), ("42",), 0.0, ())


@rl_env_build_sink
async def assemble(context: RLEnvBuildContext, question: Question, grader: Grader) -> TaskSpec:
    return draft.assemble(
        context.workpiece_id,
        question.text,
        AnswerType.TEXT,
        ANSWER_FORMAT,
        grader.package,
        Source(dataset="test", revision="r", row="0", importer_revision="test"),
        environment=None,
    )


PROGRAM = rl_env_build_program(question, grader, sink=assemble)
'''
