# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import inspect
import sys

import pytest

from taskforge.stage30_author.loader import load_program_source, module_name
from taskforge.stage40_build.template import standard
from tests.stage30_author.programs import PROGRAM


def test_the_standard_template_loads_as_an_authored_program():
    program = load_program_source(inspect.getsource(standard))

    assert program.order == ("sources", "fixtures", "requirements", "grader", "instructions", "assemble")
    assert program.digest == standard.PROGRAM.digest


def test_a_program_loads_once_per_source_with_its_own_output_models():
    program = load_program_source(PROGRAM)

    assert load_program_source(PROGRAM) is program
    assert program.steps["question"].output_type.__module__ == module_name(PROGRAM)


@pytest.mark.parametrize(
    ("edit", "problem"),
    [
        (
            lambda s: s.replace(
                "@dataclass(frozen=True)\nclass Question",
                "def helper():\n    return 1\n\n\n@dataclass(frozen=True)\nclass Question",
            ),
            "the function `helper` is not allowed at module level",
        ),
        (
            lambda s: s.replace(
                "\n\n@dataclass(frozen=True)\nclass Question",
                '\nPROMPT = "What is 6 * 7?"\n\n\n@dataclass(frozen=True)\nclass Question',
            ),
            r"an assignment \(`PROMPT = 'What is 6 \* 7\?'`\) is not allowed",
        ),
        (
            lambda s: s.replace("@rl_env_build_step\nasync def question", "async def question"),
            "the async function `question`, which is not a single-decorator `@rl_env_build_step` step",
        ),
        (
            lambda s: s.replace("\n\nPROGRAM = rl_env_build_program(question, grader, sink=assemble)\n", "\n"),
            "must end with `PROGRAM =",
        ),
        (lambda s: s + "\nprint(PROGRAM)\n", "must end with `PROGRAM ="),
        (lambda s: s.replace("from dataclasses", "import os\nfrom dataclasses"), "may not import 'os'"),
        (
            lambda s: s.replace('return Question(text="What is 6 * 7?")', 'return Question(text=open("q").read())'),
            r"\['open'\]",
        ),
        (
            lambda s: s.replace("question: Question, grader", "questions: Question, grader"),
            r"declares inputs \['questions'\]",
        ),
        (
            lambda s: s.replace(
                "rl_env_build_program(question, grader, sink=assemble)", "rl_env_build_program(question, sink=assemble)"
            ),
            "declares inputs",
        ),
        (
            lambda s: s.replace("async def grader(", "async def grade(")
            .replace("grader: Grader) ->", "grade: Grader) ->")
            .replace("grader.package,", "grade.package,")
            .replace("question, grader, sink=assemble", "question, grade, sink=assemble"),
            "needs a step named 'grader'",
        ),
        (
            lambda s: s.replace("-> TaskSpec:", "-> Question:").replace(
                "    return draft.assemble(", "    return question\n    return draft.assemble("
            ),
            "must return TaskSpec",
        ),
        (
            lambda s: s.replace("from verifyit.spec import ExactSpec\n", ""),
            r"no import, definition, assignment or builtin binds: `ExactSpec` \(line \d+\)",
        ),
        (
            lambda s: s.replace("        ANSWER_FORMAT,\n", "        ANSWER_FROMAT,\n"),
            r"`ANSWER_FROMAT` \(line \d+\); did you mean 'ANSWER_FORMAT'\?",
        ),
        (
            lambda s: s.replace("sdk import ANSWER_FORMAT", "sdk import ANSWER_FROMAT").replace(
                "        ANSWER_FORMAT,\n", "        ANSWER_FROMAT,\n"
            ),
            r"has no name 'ANSWER_FROMAT'; did you mean 'ANSWER_FORMAT' from taskforge\.stage40_build\.sdk\?",
        ),
        (
            lambda s: s.replace(
                "sdk import ANSWER_FORMAT\n",
                "sdk import ANSWER_FORMAT\nfrom taskforge.spec.draft import DOCKER_IMAGE_REQUIREMENTS\n",
            ),
            r"'taskforge\.spec\.draft' has no name 'DOCKER_IMAGE_REQUIREMENTS'; "
            r"did you mean 'DOCKER_IMAGE_REQUIREMENTS' from taskforge\.stage40_build\.sdk\?",
        ),
        (
            lambda s: s.replace(
                "sdk import ANSWER_FORMAT\n",
                "sdk import ANSWER_FORMAT\nfrom taskforge.stage40_build.sandbox_effects import machine\n",
            ),
            r"has no name 'machine'; did you mean 'build_machine' from taskforge\.stage40_build\.sandbox_effects\?",
        ),
        (
            lambda s: s.replace(
                '    return Question(text="What is 6 * 7?")',
                '    await research(context, "Find the product.")\n    return Question(text="What is 6 * 7?")',
            ),
            r"`research` \(line \d+\); did you mean 'research' from taskforge\.stage40_build\.sdk\?",
        ),
        (
            lambda s: s.replace(
                '    return Question(text="What is 6 * 7?")',
                '    await try_grader(context)\n    return Question(text="What is 6 * 7?")',
            ),
            r"`try_grader` \(line \d+\); did you mean 'grade_candidate' "
            r"from taskforge\.stage40_build\.sandbox_effects\?",
        ),
        (
            lambda s: s.replace(
                "sdk import ANSWER_FORMAT\n",
                "sdk import ANSWER_FORMAT\nfrom taskforge.stage40_build.sandbox_effects import try_grader\n",
            ),
            r"has no name 'try_grader'; did you mean 'grade_candidate' from taskforge\.stage40_build\.sandbox_effects\?",
        ),
        (
            lambda s: s.replace("stage40_build.outputs import", "stage40_build.output import"),
            r"may not import 'taskforge.stage40_build.output'; did you mean 'taskforge.stage40_build.outputs'\?",
        ),
        (
            lambda s: s.replace("sink=assemble", "assemble"),
            "pass the final step to `rl_env_build_program` as `sink=`",
        ),
        (
            lambda s: s.replace("@rl_env_build_sink\nasync def assemble", "@rl_env_build_step\nasync def assemble"),
            "must be decorated with `@rl_env_build_sink`",
        ),
    ],
    ids=[
        "helper",
        "constant",
        "undecorated-step",
        "no-program",
        "statement-after-program",
        "import",
        "unsafe-builtin",
        "unknown-input",
        "step-left-out",
        "no-grader",
        "sink-type",
        "unbound-name",
        "misspelt-name",
        "misspelt-import",
        "import-from-wrong-module",
        "close-name-in-the-imported-module",
        "unimported-sdk-name",
        "former-sdk-name",
        "former-sdk-name-import",
        "misspelt-module",
        "no-sink-keyword",
        "step-as-sink",
    ],
)
def test_a_program_outside_the_rules_is_refused_and_unregistered(edit, problem):
    source = edit(PROGRAM)
    assert source != PROGRAM

    with pytest.raises(ValueError, match=problem):
        load_program_source(source)

    assert module_name(source) not in sys.modules
