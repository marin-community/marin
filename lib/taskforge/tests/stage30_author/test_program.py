# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import inspect
import linecache
import sys
import types

import pytest
from taskcompendium.models import TaskSpec

from taskforge.content_hash import sha256_hex
from taskforge.stage30_author.program import (
    RLEnvBuildStep,
    rl_env_build_program,
    rl_env_build_sink,
    rl_env_build_step,
)


@rl_env_build_step
async def fixtures(context) -> str:
    return "fixtures"


@rl_env_build_step
async def research(context) -> str:
    return "research"


@rl_env_build_step
async def grader(context, fixtures: str) -> str:
    return fixtures


@rl_env_build_sink
async def task(context, grader: str, research: str, proposal: dict) -> TaskSpec:
    raise AssertionError("the program tests run no step")


@rl_env_build_sink
async def task_without_grader(context, research: str) -> TaskSpec:
    raise AssertionError("the program tests run no step")


def test_steps_run_after_their_inputs_with_ties_broken_by_declaration_order():
    program = rl_env_build_program(grader, research, fixtures, sink=task)

    assert program.order == ("research", "fixtures", "grader", "task")
    assert list(program.steps) == ["grader", "research", "fixtures"]
    assert program.sink is task


def test_two_steps_with_one_name_are_refused():
    @rl_env_build_step
    async def fixtures(context) -> str:
        return "other"

    with pytest.raises(ValueError, match="Two steps are named 'fixtures'"):
        rl_env_build_program(globals()["fixtures"], fixtures, research, grader, sink=task)


def test_an_input_with_no_step_of_that_name_is_refused():
    with pytest.raises(ValueError, match=r"declares inputs \['fixtures'\]"):
        rl_env_build_program(grader, research, sink=task)


def test_a_cycle_is_refused():
    @rl_env_build_step
    async def first(context, second: str) -> str:
        return second

    @rl_env_build_step
    async def second(context, first: str) -> str:
        return first

    @rl_env_build_sink
    async def task(context, first: str) -> TaskSpec:
        raise AssertionError("the program tests run no step")

    with pytest.raises(ValueError, match=r"\['first', 'second', 'task'\] form a cycle"):
        rl_env_build_program(first, second, sink=task)


def test_a_step_feeding_neither_another_step_nor_the_sink_is_refused():
    with pytest.raises(ValueError, match=r"Steps \['fixtures'\] feed neither another step nor the sink"):
        rl_env_build_program(fixtures, research, sink=task_without_grader)


def test_a_step_taking_the_sink_as_an_input_is_refused():
    @rl_env_build_step
    async def review(context, task_without_grader: TaskSpec) -> str:
        return ""

    with pytest.raises(ValueError, match="takes the sink task_without_grader as an input"):
        rl_env_build_program(research, review, sink=task_without_grader)


def test_a_sink_returning_anything_but_a_task_spec_is_refused():
    async def text(context, research: str) -> str:
        return research

    with pytest.raises(TypeError, match="Sink text must return TaskSpec"):
        rl_env_build_sink(text)  # type: ignore[arg-type]  # a sink annotated as returning str


def test_a_step_named_after_the_proposal_input_is_refused():
    @rl_env_build_step
    async def proposal(context) -> str:
        return ""

    with pytest.raises(ValueError, match="may not be named 'proposal'"):
        rl_env_build_program(proposal, sink=task_without_grader)


PREAMBLE = """from dataclasses import dataclass

from taskforge.stage30_author.program import rl_env_build_step
"""

FIXTURE = """
@dataclass(frozen=True)
class Fixture:
    path: str
"""

NOTE = """
@dataclass(frozen=True)
class Note:
    text: str
"""

STEP = """
@rl_env_build_step
async def fixtures(context) -> Fixture:
    return Fixture(path="workspace/numbers.txt")
"""


@pytest.fixture
def load_step(monkeypatch):
    """Exec a program's source into a fresh module, as a program written by the language model would be loaded."""
    loaded = 0

    def load(source: str) -> RLEnvBuildStep:
        nonlocal loaded
        loaded += 1
        name = f"authored_program_{loaded}"
        filename = f"<{name}>"
        source = PREAMBLE + source
        module = types.ModuleType(name)
        module.__file__ = filename
        monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setitem(linecache.cache, filename, (len(source), None, source.splitlines(True), filename))
        exec(compile(source, filename, "exec"), module.__dict__)
        return module.fixtures

    return load


def test_an_edit_to_an_output_type_defined_beside_the_step_changes_its_source_digest(load_step):
    step = load_step(FIXTURE + NOTE + STEP)
    edited_output = load_step(FIXTURE + "    lines: tuple[str, ...] = ()\n" + NOTE + STEP)
    edited_other_class = load_step(FIXTURE + NOTE + "    author: str = ''\n" + STEP)

    assert edited_output.source_digest != step.source_digest
    assert edited_other_class.source_digest == step.source_digest


HELPER = """
LIMIT = 3


def path_for(name):
    return f"workspace/{name}"
"""

STEP_THROUGH_HELPERS = """
@rl_env_build_step
async def fixtures(context) -> list[Fixture]:
    return [Fixture(path=path_for("numbers.txt"))] * LIMIT
"""

NOTED_FIXTURE = """
@dataclass(frozen=True)
class Fixture:
    path: str
    note: Note
"""


def test_the_source_digest_follows_what_the_step_reaches_in_its_module(load_step):
    """A helper, a constant, an annotation's type argument and a class a class names all enter the key."""
    step = load_step(FIXTURE + HELPER + STEP_THROUGH_HELPERS)

    edited_helper = load_step(FIXTURE + HELPER.replace("workspace/", "work/") + STEP_THROUGH_HELPERS)
    edited_constant = load_step(FIXTURE + HELPER.replace("LIMIT = 3", "LIMIT = 4") + STEP_THROUGH_HELPERS)
    edited_argument = load_step(FIXTURE + "    lines: tuple[str, ...] = ()\n" + HELPER + STEP_THROUGH_HELPERS)
    assert len({edited_helper.source_digest, edited_constant.source_digest, edited_argument.source_digest}) == 3
    assert step.source_digest not in {
        edited_helper.source_digest,
        edited_constant.source_digest,
        edited_argument.source_digest,
    }

    nested = load_step(NOTE + NOTED_FIXTURE + STEP)
    edited_nested = load_step(NOTE + "    author: str = ''\n" + NOTED_FIXTURE + STEP)
    assert edited_nested.source_digest != nested.source_digest


EARLIER_STEP = """
@rl_env_build_step
async def anchors(context) -> Fixture:
    return Fixture(path="workspace/anchors.txt")
"""

STEP_NAMING_A_STEP = """
@rl_env_build_step
async def fixtures(context) -> Fixture:
    assert anchors.name == "anchors"
    return Fixture(path="workspace/numbers.txt")
"""


def test_a_step_that_names_another_step_digests_that_step_by_its_digest(load_step):
    """The other step's object is not text: its repr names its function's address, which changes per load."""
    first = load_step(FIXTURE + EARLIER_STEP + STEP_NAMING_A_STEP)
    second = load_step(FIXTURE + EARLIER_STEP + STEP_NAMING_A_STEP)
    edited = load_step(FIXTURE + EARLIER_STEP.replace("anchors.txt", "anchor.txt") + STEP_NAMING_A_STEP)

    assert first.source_digest == second.source_digest
    assert edited.source_digest != first.source_digest


STEP_REACHING_AN_OBJECT = """
SENTINEL = object()


@rl_env_build_step
async def fixtures(context) -> Fixture:
    assert SENTINEL is not None
    return Fixture(path="workspace/numbers.txt")
"""


def test_a_value_with_no_stable_text_is_refused_by_name(load_step):
    with pytest.raises(TypeError, match="SENTINEL has no text"):
        load_step(FIXTURE + STEP_REACHING_AN_OBJECT)


def test_a_step_returning_a_library_type_digests_its_own_source_only():
    @rl_env_build_step
    async def task(context) -> TaskSpec:
        raise AssertionError("never runs")

    assert task.source_digest == sha256_hex(inspect.getsource(task.fn).encode())
