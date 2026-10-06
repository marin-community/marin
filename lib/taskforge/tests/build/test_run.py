# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from rolloutengine.contracts import ModelRequest, ModelTurn
from rolloutengine.engine import ShellboxRolloutEngine
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from taskcompendium.environment import EnvironmentKind
from taskcompendium.grading_result import Outcome
from taskcompendium.submission import PlainText

from taskforge.build.author import compile_program
from taskforge.build.run import load_draft, run_build
from taskforge.build.sdk import GRADED_RESOURCE_PREFIX, BuildFailure
from taskforge.build.step import CacheStatus


async def build_once(proposal, program, tmp_path, services, invalidate=()):
    async with services() as s:
        return await run_build(program, proposal, tmp_path / "item", tmp_path / "cache", s, invalidate=invalidate)


def statuses(draft) -> dict[str, CacheStatus]:
    return {record.name: record.status for record in draft.provenance.steps}


async def test_program_builds_a_runnable_task_and_records_the_draft(proposal, program_source, tmp_path, services):
    program = compile_program(program_source, proposal.digest)

    draft = await build_once(proposal, program, tmp_path, services)

    assert load_draft(tmp_path / "item" / "draft") == draft
    names = [r.name for r in draft.provenance.resources]
    assert names[-1] == "grader/grade.py"
    assert [n for n in names[:-1] if n.startswith(GRADED_RESOURCE_PREFIX)] == names[:-1] and len(names) == 2
    assert draft.provenance.program_digest == program.digest

    async def answer(request: ModelRequest) -> ModelTurn:
        return ModelTurn({"role": "assistant", "content": "ANSWER = 42"}, (1, 2), (3,), None, "stop")

    engine = ShellboxRolloutEngine(
        answer,
        {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
        max_turns=2,
        command_timeout=30,
        cleanup_timeout=30,
        convention=PlainText(id="plain"),
    )
    rollout = await engine.run(draft.task, execution=draft.execution)
    assert (rollout.grade.status, rollout.grade.reward) == (Outcome.GRADED, 1.0)


async def test_patching_one_step_reuses_the_steps_it_does_not_affect(proposal, program_source, tmp_path, services):
    first = await build_once(proposal, compile_program(program_source, proposal.digest), tmp_path, services)
    assert set(statuses(first).values()) == {CacheStatus.MISS}

    again = await build_once(proposal, compile_program(program_source, proposal.digest), tmp_path, services)
    assert set(statuses(again).values()) == {CacheStatus.HIT}
    assert again.task == first.task
    assert again.provenance.resources == first.provenance.resources

    patched = program_source.replace("timeout=60,", "timeout=90,")
    after = await build_once(proposal, compile_program(patched, proposal.digest), tmp_path, services)
    assert statuses(after) == {
        "machine": CacheStatus.HIT,
        "grader": CacheStatus.MISS,
        "assemble": CacheStatus.MISS,
        "fixed_controls": CacheStatus.MISS,
    }

    # An edit that leaves a step's output unchanged recomputes that step only.
    cosmetic = patched.replace('"Compute the product', '"Compute" " the product')
    last = await build_once(proposal, compile_program(cosmetic, proposal.digest), tmp_path, services)
    assert statuses(last) == {
        "machine": CacheStatus.HIT,
        "grader": CacheStatus.HIT,
        "assemble": CacheStatus.MISS,
        "fixed_controls": CacheStatus.HIT,
    }
    assert last.task == after.task


async def test_invalidated_step_is_recomputed(proposal, program_source, tmp_path, services):
    program = compile_program(program_source, proposal.digest)
    await build_once(proposal, program, tmp_path, services)

    draft = await build_once(proposal, program, tmp_path, services, invalidate=("grader",))

    assert statuses(draft)["grader"] == CacheStatus.INVALIDATED
    assert statuses(draft)["machine"] == CacheStatus.HIT


async def test_failed_check_names_the_step(proposal, program_source, tmp_path, services):
    broken = program_source.replace('"ANSWER = 42")\n    b.check', '"ANSWER = 41")\n    b.check')
    with pytest.raises(BuildFailure) as failure:
        await build_once(proposal, compile_program(broken, proposal.digest), tmp_path, services)
    assert failure.value.step == "grader"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        # The verifier is rebuilt outside the grader step.
        ("verifier=graded.verifier,", 'verifier=spec.answer_verifier(ExactSpec(expected=("42",))),', "GRADER step"),
        # The program adds a control outside the CONTROLS step.
        (
            "controls=await fixed_controls(b, task))",
            "controls=(*await fixed_controls(b, task), "
            'control("late", K.POSITIVE, C.KNOWN_CORRECT, "42", reward_min=1.0)))',
            "CONTROLS step",
        ),
        # The execution settings prepare a stage the task does not have.
        ("execution=EXECUTION, convention=", 'execution=TaskExecution(stages={"extra": {}}), convention=', "execution:"),
        # The convention cannot carry the task's text answer.
        ('CONVENTION = PlainText(id="plain_text")', 'CONVENTION = JsonValueAnswer(id="json")', "convention 'json'"),
        # The control set lacks a shortcut or reward-hack control.
        (
            'control("sum", K.NEGATIVE, C.TASK_SPECIFIC_SHORTCUT',
            'control("sum", K.NEGATIVE, C.PLAUSIBLE_WRONG',
            "controls:",
        ),
    ],
)
async def test_output_that_breaks_a_library_rule_fails_the_build(
    proposal, program_source, tmp_path, services, old, new, message
):
    source = (
        program_source.replace(old, new)
        .replace(
            "from taskcompendium.grading_result import Outcome",
            "from taskcompendium.grading_result import Outcome\nfrom verifyit.spec import ExactSpec",
        )
        .replace("import PlainText", "import JsonValueAnswer, PlainText")
    )
    assert source != program_source
    program = compile_program(source, proposal.digest)
    with pytest.raises(BuildFailure, match=message):
        await build_once(proposal, program, tmp_path, services)


PROTOTYPE = '    reference = await b.try_grader(env, verifier, AnswerType.TEXT, CONVENTION, "question", "ANSWER = 42")\n'


def prototyping_on(program_source: str, candidate: str) -> str:
    """The test program with its grader step also grading ``candidate``."""
    extra = (
        f'    wrong = await b.try_grader(env, verifier, AnswerType.TEXT, CONVENTION, "question", "{candidate}")\n'
        '    b.check(wrong.reward == 0.0, "")\n'
    )
    assert PROTOTYPE in program_source
    return program_source.replace(PROTOTYPE, PROTOTYPE + extra)


async def test_grading_a_control_candidate_during_the_build_fails_it(proposal, program_source, tmp_path, services):
    program = compile_program(prototyping_on(program_source, "ANSWER = 41"), proposal.digest)
    with pytest.raises(BuildFailure, match=r"\['off-by-one'\]"):
        await build_once(proposal, program, tmp_path, services)


async def test_prototyping_on_a_wrong_answer_that_is_not_a_control_builds(proposal, program_source, tmp_path, services):
    program = compile_program(prototyping_on(program_source, "ANSWER = 40"), proposal.digest)
    draft = await build_once(proposal, program, tmp_path, services)
    assert [c.id for c in draft.controls] == ["gold", "empty", "off-by-one", "sum"]
