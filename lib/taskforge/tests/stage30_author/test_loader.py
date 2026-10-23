# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import inspect
import re
import sys

import pytest

from taskforge.stage30_author.loader import SdkExports, load_program_source, module_name
from taskforge.stage40_build_with_program.program_sdk import sdk_exports
from taskforge.stage40_build_with_program.template import standard
from tests.stage30_author.programs import PROGRAM

SDK = sdk_exports()


def test_the_standard_template_loads_as_an_authored_program():
    program = load_program_source(inspect.getsource(standard), SDK)

    assert program.order == ("sources", "fixtures", "requirements", "grader", "instructions", "assemble")
    assert program.digest == standard.PROGRAM.digest


def test_a_program_loads_once_per_source_with_its_own_output_models():
    program = load_program_source(PROGRAM, SDK)

    assert load_program_source(PROGRAM, SDK) is program
    assert program.steps["question"].output_type.__module__ == module_name(PROGRAM)


def test_an_interrupted_load_does_not_stand_in_for_the_next_load():
    """An interrupt mid-load, as a stall watcher or a KeyboardInterrupt raises, leaves no half-run module behind."""
    source = PROGRAM.replace(
        "@dataclass(frozen=True)\nclass Question",
        "class Interrupting:\n    raise KeyboardInterrupt()\n\n\n@dataclass(frozen=True)\nclass Question",
    )
    assert source != PROGRAM

    for _ in range(2):
        with pytest.raises(KeyboardInterrupt):
            load_program_source(source, SDK)
        assert module_name(source) not in sys.modules


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
            r"no import, definition, assignment or builtin binds:\n- `ExactSpec` \(line \d+\)",
        ),
        (
            lambda s: s.replace("        ANSWER_FORMAT,\n", "        ANSWER_FROMAT,\n"),
            r"`ANSWER_FROMAT` \(line \d+\); did you mean 'ANSWER_FORMAT'\?",
        ),
        (
            lambda s: s.replace("sdk import ANSWER_FORMAT", "sdk import ANSWER_FROMAT").replace(
                "        ANSWER_FORMAT,\n", "        ANSWER_FROMAT,\n"
            ),
            r"has no name 'ANSWER_FROMAT'; "
            r"did you mean 'ANSWER_FORMAT' from taskforge\.stage40_build_with_program\.sdk\?",
        ),
        (
            lambda s: s.replace(
                "sdk import ANSWER_FORMAT\n",
                "sdk import ANSWER_FORMAT\nfrom taskforge.spec.draft import DOCKER_IMAGE_REQUIREMENTS\n",
            ),
            r"'taskforge\.spec\.draft' has no name 'DOCKER_IMAGE_REQUIREMENTS'; "
            r"did you mean 'DOCKER_IMAGE_REQUIREMENTS' from taskforge\.stage40_build_with_program\.sdk\?",
        ),
        (
            lambda s: s.replace(
                "sdk import ANSWER_FORMAT\n",
                "sdk import ANSWER_FORMAT\nfrom taskforge.stage40_build_with_program.sandbox_effects import machine\n",
            ),
            r"has no name 'machine'; "
            r"did you mean 'build_machine' from taskforge\.stage40_build_with_program\.sandbox_effects, "
            r"or 'machine' from taskforge\.spec\.draft\?",
        ),
        (
            lambda s: s.replace(
                '    return Question(text="What is 6 * 7?")',
                '    await research(context, "Find the product.")\n    return Question(text="What is 6 * 7?")',
            ),
            r"`research` \(line \d+\); did you mean 'research' from taskforge\.stage40_build_with_program\.sdk\?",
        ),
        (
            lambda s: s.replace(
                '    return Question(text="What is 6 * 7?")',
                '    await try_grader(context)\n    return Question(text="What is 6 * 7?")',
            ),
            r"`try_grader` \(line \d+\); did you mean 'grade_candidate' "
            r"from taskforge\.stage40_build_with_program\.sandbox_effects\?",
        ),
        (
            lambda s: s.replace(
                "sdk import ANSWER_FORMAT\n",
                "sdk import ANSWER_FORMAT\n"
                "from taskforge.stage40_build_with_program.sandbox_effects import try_grader\n",
            ),
            r"has no name 'try_grader'; "
            r"did you mean 'grade_candidate' from taskforge\.stage40_build_with_program\.sandbox_effects\?",
        ),
        (
            lambda s: s.replace("stage40_build_with_program.outputs import", "stage40_build_with_program.output import"),
            r"may not import 'taskforge.stage40_build_with_program.output'; "
            r"did you mean 'taskforge.stage40_build_with_program.outputs'\?",
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
        load_program_source(source, SDK)

    assert module_name(source) not in sys.modules


UNBOUND = re.escape("the program reads names that no import, definition, assignment or builtin binds:")
IMPORT_EACH = re.escape("Import each SDK name the program uses from its module.")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda s: s.replace(
                '    return Question(text="What is 6 * 7?")',
                '    await qzxv()\n    return Question(text="What is 6 * 7?")',
            ),
            rf"{UNBOUND}\n- `qzxv` \(line \d+\)\n{IMPORT_EACH}",
        ),
        (
            lambda s: s.replace(
                '    return Question(text="What is 6 * 7?")',
                '    await research(context, "Find the product.")\n    return Question(text="What is 6 * 7?")',
            ),
            rf"{UNBOUND}\n- `research` \(line \d+\); "
            rf"did you mean 'research' from taskforge\.stage40_build_with_program\.sdk\?\n"
            rf"{IMPORT_EACH}",
        ),
        (
            lambda s: s.replace("from dataclasses", "import os\nfrom dataclasses"),
            r"the program failed to load: ImportError: build programs may not import 'os'\. Allowed: \[.*\]",
        ),
        (
            lambda s: s.replace("stage40_build_with_program.outputs import", "stage40_build_with_program.output import"),
            r"the program failed to load: ImportError: build programs may not import "
            r"'taskforge\.stage40_build_with_program\.output'; "
            r"did you mean 'taskforge\.stage40_build_with_program\.outputs'\? Allowed: \[.*\]",
        ),
    ],
    ids=["unbound-without-suggestion", "unbound-with-suggestion", "import-without-suggestion", "import-with-suggestion"],
)
def test_a_refusal_ends_each_sentence_once_with_or_without_a_suggestion(edit, message):
    with pytest.raises(ValueError) as refusal:
        load_program_source(edit(PROGRAM), SDK)

    assert re.fullmatch(message, str(refusal.value)), str(refusal.value)


def test_the_loader_suggests_only_the_sdk_names_its_caller_passes():
    source = PROGRAM.replace(
        '    return Question(text="What is 6 * 7?")',
        '    await research(context, "Find the product.")\n    return Question(text="What is 6 * 7?")',
    )
    sdk = SdkExports(defining_modules={"research": ("example.sdk",)})

    with pytest.raises(ValueError, match=r"did you mean 'research' from example\.sdk\?") as suggested:
        load_program_source(source, sdk)
    with pytest.raises(ValueError) as unsuggested:
        load_program_source(source, SdkExports(defining_modules={}))

    assert "did you mean" not in str(unsuggested.value)
    assert module_name(source) not in sys.modules, str(suggested.value)
