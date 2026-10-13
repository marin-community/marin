# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from taskforge.stage30_author.program import PROPOSAL_INPUT, rl_env_build_program, rl_env_build_step


@rl_env_build_step()
async def fixtures(context) -> str:
    return "fixtures"


@rl_env_build_step()
async def research(context) -> str:
    return "research"


@rl_env_build_step(inputs=("fixtures",))
async def grader(context, fixtures: str) -> str:
    return fixtures


@rl_env_build_step(inputs=("grader", "research", PROPOSAL_INPUT))
async def task(context, grader: str, research: str, proposal: dict) -> str:
    return grader + research


def test_steps_run_after_their_inputs_with_ties_broken_by_declaration_order():
    program = rl_env_build_program(task, grader, research, fixtures)

    assert program.order == ("research", "fixtures", "grader", "task")
    assert program.sink == "task"


def test_two_steps_with_one_name_are_refused():
    @rl_env_build_step()
    async def fixtures(context) -> str:
        return "other"

    with pytest.raises(ValueError, match="Two steps are named 'fixtures'"):
        rl_env_build_program(globals()["fixtures"], fixtures)


def test_an_input_with_no_step_of_that_name_is_refused():
    with pytest.raises(ValueError, match=r"declares inputs \['fixtures'\]"):
        rl_env_build_program(grader)


def test_a_cycle_is_refused():
    @rl_env_build_step(inputs=("second",))
    async def first(context, second: str) -> str:
        return second

    @rl_env_build_step(inputs=("first",))
    async def second(context, first: str) -> str:
        return first

    with pytest.raises(ValueError, match=r"\['first', 'second'\] form a cycle"):
        rl_env_build_program(fixtures, first, second)


@pytest.mark.parametrize("steps", [(), (fixtures, research)], ids=["no sink", "two sinks"])
def test_a_program_needs_exactly_one_sink(steps):
    with pytest.raises(ValueError, match="exactly one sink"):
        rl_env_build_program(*steps)


def test_a_step_named_after_the_proposal_input_is_refused():
    @rl_env_build_step()
    async def proposal(context) -> str:
        return ""

    with pytest.raises(ValueError, match="may not be named 'proposal'"):
        rl_env_build_program(proposal)


def test_the_decorator_refuses_parameters_that_differ_from_the_declared_inputs():
    with pytest.raises(TypeError, match="declares inputs"):
        rl_env_build_step(inputs=("research",))(grader.fn)
