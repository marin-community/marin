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

    assert step.source_digest == sha256_hex((STEP.lstrip("\n") + FIXTURE.lstrip("\n")).encode())
    assert edited_output.source_digest != step.source_digest
    assert edited_other_class.source_digest == step.source_digest


def test_a_step_returning_a_library_type_digests_its_own_source_only():
    @rl_env_build_step
    async def task(context) -> TaskSpec:
        raise AssertionError("never runs")

    assert task.source_digest == sha256_hex(inspect.getsource(task.fn).encode())
