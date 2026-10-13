# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A toy program carried through ``run_build``: ``numbers`` and ``grader`` feed the sink ``task``."""

import base64
from collections.abc import Sequence
from dataclasses import replace

import pytest
from pydantic import BaseModel
from taskcompendium.grader import GraderPackage, verifyit_package
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    InlineFile,
    PlainText,
    ResourceGroups,
    Source,
    TaskResource,
    TaskSpec,
    TextMessage,
)
from verifyit.spec import NumericSpec

from taskforge.content_hash import digest
from taskforge.llm.client import Completion, StructuredResult
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.stage30_author.program import PROPOSAL_INPUT, rl_env_build_program, rl_env_build_step
from taskforge.stage40_build.context import BuildFailure, RLEnvBuildContext
from taskforge.stage40_build.run import DRAFT_DIR, load_draft, run_build, run_program
from taskforge.stage40_build.step_store import CacheStatus, RLEnvBuildStepStore, model_digest

ITEM = "toy-0"
PROPOSAL = {"id": "toy/0", "task": "Multiply the two numbers in /workspace/numbers.txt."}
SAMPLING = LLMPolicy()


class Hint(BaseModel):
    text: str


class FakeModel:
    """Answers every structured call with a ``Hint`` and counts the calls."""

    model = "fake-model"

    def __init__(self):
        self.calls = 0

    async def complete(self, messages: Sequence[Message], policy: LLMPolicy) -> Completion:
        raise AssertionError("the steps here make structured calls only")

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        self.calls += 1
        return StructuredResult(output_type.model_validate({"text": f"hint {self.calls}"}), ())


def inline_file(path: str, content: bytes) -> TaskResource:
    return TaskResource(path=path, source=InlineFile(content_base64=base64.b64encode(content).decode()))


@rl_env_build_step()
async def numbers(context: RLEnvBuildContext) -> TaskResource:
    return inline_file("workspace/numbers.txt", b"6 7\n")


@rl_env_build_step()
async def grader(context: RLEnvBuildContext) -> GraderPackage:
    return verifyit_package(NumericSpec(expected="42", tolerance_abs=0.0, tolerance_rel=0.0))


@rl_env_build_step(inputs=("numbers", "grader", PROPOSAL_INPUT))
async def task(context: RLEnvBuildContext, numbers: TaskResource, grader: GraderPackage, proposal: dict) -> TaskSpec:
    return TaskSpec(
        id=context.item_id,
        context=ConversationInput(events=(TextMessage(role="user", content=f"{proposal['task']} Reply alone."),)),
        environment_requirements=EnvironmentRequirements(
            capabilities=("shell", "filesystem"),
            command_semantics=CommandSemantics.SHELL_SIMULATOR,
            working_directory="/workspace",
        ),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=grader.grader,
        source=Source(dataset="taskforge", revision="test", row=context.item_id, importer_revision="test"),
        resources=ResourceGroups(worker=(numbers,), verifier=grader.resources),
    )


PROGRAM = rl_env_build_program(numbers, grader, task)


async def build(tmp_path, program=PROGRAM, proposal=PROPOSAL):
    model = FakeModel()
    return await run_build(
        program,
        proposal,
        ITEM,
        tmp_path / "round",
        str(tmp_path / "store"),
        model,
        SAMPLING,
        model_digest(model, SAMPLING),
        0,
    )


def statuses(draft) -> dict[str, CacheStatus]:
    return {record.name: record.status for record in draft.provenance.steps}


async def test_a_rebuild_hits_every_step_and_records_the_same_task_and_input_digests(tmp_path):
    first = await build(tmp_path)
    again = await build(tmp_path)

    assert [(record.name, record.status) for record in first.provenance.steps] == [
        ("numbers", CacheStatus.MISS),
        ("grader", CacheStatus.MISS),
        ("task", CacheStatus.MISS),
    ]
    assert set(statuses(again).values()) == {CacheStatus.HIT}
    assert again.task == first.task
    assert [record.inputs for record in again.provenance.steps] == [record.inputs for record in first.provenance.steps]
    outputs = {record.name: record.output_digest for record in first.provenance.steps}
    assert again.provenance.steps[-1].inputs == {
        "numbers": outputs["numbers"],
        "grader": outputs["grader"],
        "proposal": digest(PROPOSAL),
    }
    assert load_draft(tmp_path / "round" / DRAFT_DIR) == again


async def test_a_rewritten_step_reruns_itself_and_its_consumers_only(tmp_path):
    first = await build(tmp_path)

    @rl_env_build_step()
    async def numbers(context: RLEnvBuildContext) -> TaskResource:
        return inline_file("workspace/numbers.txt", b"7 6\n")

    rebuilt = await build(tmp_path, rl_env_build_program(numbers, grader, task))

    assert statuses(rebuilt) == {"numbers": CacheStatus.MISS, "grader": CacheStatus.HIT, "task": CacheStatus.MISS}
    assert rebuilt.task.resources.worker != first.task.resources.worker


async def test_a_rewritten_step_with_identical_output_leaves_its_consumers_hit(tmp_path):
    await build(tmp_path)

    @rl_env_build_step()
    async def numbers(context: RLEnvBuildContext) -> TaskResource:
        return inline_file("workspace/numbers.txt", b"6 " + b"7\n")

    rebuilt = await build(tmp_path, rl_env_build_program(numbers, grader, task))

    assert statuses(rebuilt) == {"numbers": CacheStatus.MISS, "grader": CacheStatus.HIT, "task": CacheStatus.HIT}


async def test_a_changed_proposal_reruns_only_the_steps_declaring_it(tmp_path):
    first = await build(tmp_path)
    edited = {**PROPOSAL, "task": "Multiply both numbers in /workspace/numbers.txt."}

    rebuilt = await build(tmp_path, proposal=edited)

    assert statuses(rebuilt) == {"numbers": CacheStatus.HIT, "grader": CacheStatus.HIT, "task": CacheStatus.MISS}
    assert rebuilt.provenance.proposal_digest == digest(edited) != first.provenance.proposal_digest


async def test_a_failing_sink_raises_a_build_failure_carrying_its_step_name(tmp_path):
    @rl_env_build_step(inputs=("numbers",))
    async def task(context: RLEnvBuildContext, numbers: TaskResource) -> TaskSpec:
        raise BuildFailure("the numbers do not multiply to the answer")

    with pytest.raises(BuildFailure) as failure:
        await build(tmp_path, rl_env_build_program(numbers, task))

    assert failure.value.step == "task"
    assert str(failure.value) == "task: the numbers do not multiply to the answer"


async def test_a_sink_returning_anything_but_a_task_spec_is_refused_before_any_step_runs(tmp_path):
    @rl_env_build_step(inputs=("numbers",))
    async def text(context: RLEnvBuildContext, numbers: TaskResource) -> str:
        raise AssertionError("no step runs")

    with pytest.raises(BuildFailure) as refused:
        await build(tmp_path, rl_env_build_program(numbers, text))

    assert refused.value.step == "text"


@rl_env_build_step(inputs=(PROPOSAL_INPUT,))
async def hint(context: RLEnvBuildContext, proposal: dict) -> Hint:
    messages: list[Message] = [{"role": "user", "content": proposal["task"]}]
    result = await context.model.complete_structured(messages, Hint, context.sampling)
    return result.value


async def test_the_model_and_sampling_enter_every_step_key(tmp_path):
    model = FakeModel()
    program = rl_env_build_program(hint)
    store = RLEnvBuildStepStore(str(tmp_path), ITEM)

    async def run(sampling: LLMPolicy):
        context = RLEnvBuildContext(item_id=ITEM, model=model, sampling=sampling)
        return await run_program(program, context, PROPOSAL, store, model_digest(model, sampling))

    first, [first_record] = await run(SAMPLING)
    again, [again_record] = await run(SAMPLING)
    resampled, [resampled_record] = await run(replace(SAMPLING, temperature=0.2))

    assert (first_record.status, again_record.status, resampled_record.status) == (
        CacheStatus.MISS,
        CacheStatus.HIT,
        CacheStatus.MISS,
    )
    assert again == first == Hint(text="hint 1")
    assert resampled == Hint(text="hint 2")
    assert model.calls == 2
