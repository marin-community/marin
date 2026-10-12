# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
from taskcompendium.grader import GraderPackage
from taskcompendium.models import AnswerType, PlainText, Source, TaskSpec
from verifyit.spec import NumericSpec

from taskforge.builder.run import DRAFT_DIR, load_draft, run_build
from taskforge.builder.sdk import BuildContext, BuildFailure
from taskforge.builder.step import CacheStatus, program, step
from taskforge.ledger.records import EntryKind
from taskforge.llm.client import NoModel
from taskforge.llm.policy import LLMPolicy
from taskforge.proposal.model import TaskProposal, body_sections
from taskforge.spec.draft import answer_grader, assemble, file, requirements

SOURCE = Source(dataset="test", revision="r1", row="0", importer_revision="test")


@step()
async def numbers(b: BuildContext) -> str:
    return "6 7\n"


@step()
async def grader(b: BuildContext) -> GraderPackage:
    return answer_grader(NumericSpec(expected="42", tolerance_abs=0.0, tolerance_rel=0.0))


@step(inputs=("proposal",))
async def instruction(b: BuildContext, proposal: TaskProposal) -> str:
    return body_sections(proposal.body)["Task"] + " Reply with the product alone."


@step(inputs=("numbers", "grader", "instruction"))
async def task(b: BuildContext, numbers: str, grader: GraderPackage, instruction: str) -> TaskSpec:
    return assemble(
        b.item_id,
        instruction,
        AnswerType.TEXT,
        PlainText(),
        grader,
        SOURCE,
        environment=requirements(),
        files=(file("workspace/numbers.txt", numbers),),
    )


PROGRAM = program(numbers, grader, instruction, task)


async def build(tmp_path, built, proposal, ledger):
    return await run_build(built, proposal, tmp_path / "item", tmp_path / "cache", NoModel(), LLMPolicy(), ledger)


def statuses(draft) -> dict[str, CacheStatus]:
    return {record.name: record.status for record in draft.provenance.steps}


async def test_a_rebuild_hits_every_step_and_records_each_input_digest(tmp_path, proposal, ledger):
    first = await build(tmp_path, PROGRAM, proposal, ledger)
    again = await build(tmp_path, PROGRAM, proposal, ledger)

    assert [record.name for record in first.provenance.steps] == ["numbers", "grader", "instruction", "task"]
    assert set(statuses(first).values()) == {CacheStatus.MISS}
    assert set(statuses(again).values()) == {CacheStatus.HIT}
    assert again.task == first.task
    assert load_draft(tmp_path / "item" / DRAFT_DIR) == again
    outputs = {record.name: record.output.digest for record in first.provenance.steps}
    by_name = {record.name: record for record in again.provenance.steps}
    assert by_name["task"].inputs == {name: outputs[name] for name in ("numbers", "grader", "instruction")}
    assert by_name["instruction"].inputs == {"proposal": proposal.digest}
    assert by_name["numbers"].inputs == {}
    steps = [entry for entry in ledger.entries if entry.kind is EntryKind.STEP]
    assert [(entry.step, entry.attrs["cache"]) for entry in steps[4:]] == [
        ("numbers", "hit"),
        ("grader", "hit"),
        ("instruction", "hit"),
        ("task", "hit"),
    ]
    assert [entry.output_hash for entry in steps[4:]] == list(outputs.values())


async def test_a_rewritten_step_recomputes_itself_and_its_consumers_only(tmp_path, proposal, ledger):
    first = await build(tmp_path, PROGRAM, proposal, ledger)

    @step()
    async def numbers(b: BuildContext) -> str:
        return "3 14\n"

    rebuilt = await build(tmp_path, program(numbers, grader, instruction, task), proposal, ledger)

    assert statuses(rebuilt) == {
        "numbers": CacheStatus.MISS,
        "grader": CacheStatus.HIT,
        "instruction": CacheStatus.HIT,
        "task": CacheStatus.MISS,
    }
    assert rebuilt.task.resources.worker != first.task.resources.worker


async def test_an_edited_proposal_reruns_only_the_steps_that_take_it_and_their_consumers(tmp_path, proposal, ledger):
    first = await build(tmp_path, PROGRAM, proposal, ledger)
    edited = replace(proposal, body=proposal.body.replace("Multiply the two numbers", "Multiply both numbers"))

    rebuilt = await build(tmp_path, PROGRAM, edited, ledger)

    assert statuses(rebuilt) == {
        "numbers": CacheStatus.HIT,
        "grader": CacheStatus.HIT,
        "instruction": CacheStatus.MISS,
        "task": CacheStatus.MISS,
    }
    assert rebuilt.provenance.proposal_digest == edited.digest != first.provenance.proposal_digest


async def test_a_failing_sink_names_its_step_and_a_sink_must_return_a_task(tmp_path, proposal, ledger):
    @step(inputs=("numbers",))
    async def broken(b: BuildContext, numbers: str) -> TaskSpec:
        raise BuildFailure("the numbers do not multiply to the answer")

    @step(inputs=("numbers",))
    async def text(b: BuildContext, numbers: str) -> str:
        return numbers

    with pytest.raises(BuildFailure) as failure:
        await build(tmp_path, program(numbers, broken), proposal, ledger)
    assert failure.value.step == "broken"

    with pytest.raises(BuildFailure) as refused:
        await build(tmp_path, program(numbers, text), proposal, ledger)
    assert refused.value.step == "text"


def test_program_refuses_an_unknown_input_a_cycle_two_sinks_and_the_reserved_name():
    @step(inputs=("missing",))
    async def orphan(b: BuildContext, missing: str) -> str:
        return missing

    @step(inputs=("second",))
    async def first(b: BuildContext, second: str) -> str:
        return second

    @step(inputs=("first",))
    async def second(b: BuildContext, first: str) -> str:
        return first

    @step()
    async def proposal(b: BuildContext) -> str:
        return ""

    with pytest.raises(ValueError, match="no step produces"):
        program(orphan)
    with pytest.raises(ValueError, match="cycle"):
        program(first, second, task, numbers, grader, instruction)
    with pytest.raises(ValueError, match="exactly one sink"):
        program(numbers, grader)
    with pytest.raises(ValueError, match="built-in proposal input"):
        program(proposal)
    with pytest.raises(TypeError, match="declares inputs"):
        step(inputs=("numbers",))(numbers.fn)
