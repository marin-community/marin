# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A toy program carried through ``run_build``: ``numbers`` and ``grader`` feed the sink ``task``."""

import asyncio
import base64
import time
from collections.abc import Mapping, Sequence
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
from taskforge.stage30_author.program import rl_env_build_program, rl_env_build_sink, rl_env_build_step
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

    async def complete(
        self, messages: Sequence[Message], policy: LLMPolicy, request_fields: Mapping[str, object] | None = None
    ) -> Completion:
        raise AssertionError("the steps here make structured calls only")

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        self.calls += 1
        return StructuredResult(output_type.model_validate({"text": f"hint {self.calls}"}), ())


def inline_file(path: str, content: bytes) -> TaskResource:
    return TaskResource(path=path, source=InlineFile(content_base64=base64.b64encode(content).decode()))


@rl_env_build_step
async def numbers(context: RLEnvBuildContext) -> TaskResource:
    return inline_file("workspace/numbers.txt", b"6 7\n")


@rl_env_build_step
async def grader(context: RLEnvBuildContext) -> GraderPackage:
    return verifyit_package(NumericSpec(expected="42", tolerance_abs=0.0, tolerance_rel=0.0))


def toy_task(item_id: str, prompt: str, numbers: TaskResource, grader: GraderPackage) -> TaskSpec:
    return TaskSpec(
        id=item_id,
        context=ConversationInput(events=(TextMessage(role="user", content=prompt),)),
        environment_requirements=EnvironmentRequirements(
            capabilities=("shell", "filesystem"),
            command_semantics=CommandSemantics.SHELL_SIMULATOR,
            working_directory="/workspace",
        ),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=grader.grader,
        source=Source(dataset="taskforge", revision="test", row=item_id, importer_revision="test"),
        resources=ResourceGroups(worker=(numbers,), verifier=grader.resources),
    )


@rl_env_build_sink
async def task(context: RLEnvBuildContext, numbers: TaskResource, grader: GraderPackage, proposal: dict) -> TaskSpec:
    return toy_task(context.item_id, f"{proposal['task']} Reply alone.", numbers, grader)


PROGRAM = rl_env_build_program(numbers, grader, sink=task)


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


async def test_steps_without_a_dependency_between_them_run_together_and_outcomes_keep_the_program_order(tmp_path):
    spans: dict[str, tuple[float, float]] = {}

    async def linger(name: str) -> None:
        started = time.monotonic()
        await asyncio.sleep(0.2)
        spans[name] = (started, time.monotonic())

    @rl_env_build_step
    async def slow_numbers(context: RLEnvBuildContext) -> TaskResource:
        await linger("slow_numbers")
        return inline_file("workspace/numbers.txt", b"6 7\n")

    @rl_env_build_step
    async def slow_grader(context: RLEnvBuildContext) -> GraderPackage:
        await linger("slow_grader")
        return verifyit_package(NumericSpec(expected="42", tolerance_abs=0.0, tolerance_rel=0.0))

    @rl_env_build_sink
    async def joined(
        context: RLEnvBuildContext, slow_numbers: TaskResource, slow_grader: GraderPackage, proposal: dict
    ) -> TaskSpec:
        await linger("joined")
        return toy_task(context.item_id, proposal["task"], slow_numbers, slow_grader)

    program = rl_env_build_program(slow_numbers, slow_grader, sink=joined)
    draft = await build(tmp_path, program)

    assert spans["slow_grader"][0] < spans["slow_numbers"][1], "the two steps did not overlap"
    assert spans["joined"][0] >= max(spans["slow_numbers"][1], spans["slow_grader"][1])
    assert [outcome.memo.name for outcome in draft.provenance.steps] == list(program.order)


async def test_a_failing_step_ends_the_build_and_cancels_the_steps_still_running(tmp_path):
    cancelled = asyncio.Event()

    @rl_env_build_step
    async def doomed(context: RLEnvBuildContext) -> TaskResource:
        await asyncio.sleep(0.05)
        raise BuildFailure("the numbers never arrived")

    @rl_env_build_step
    async def patient(context: RLEnvBuildContext) -> GraderPackage:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise AssertionError("unreachable")

    @rl_env_build_sink
    async def never(context: RLEnvBuildContext, doomed: TaskResource, patient: GraderPackage) -> TaskSpec:
        raise AssertionError("unreachable")

    with pytest.raises(BuildFailure) as failure:
        await build(tmp_path, rl_env_build_program(doomed, patient, sink=never))

    assert (failure.value.step, cancelled.is_set()) == ("doomed", True)


def statuses(draft) -> dict[str, CacheStatus]:
    return {outcome.memo.name: outcome.status for outcome in draft.provenance.steps}


async def test_a_rebuild_hits_every_step_and_records_the_same_task_and_input_digests(tmp_path):
    first = await build(tmp_path)
    again = await build(tmp_path)

    assert [(outcome.memo.name, outcome.status) for outcome in first.provenance.steps] == [
        ("numbers", CacheStatus.MISS),
        ("grader", CacheStatus.MISS),
        ("task", CacheStatus.MISS),
    ]
    assert set(statuses(again).values()) == {CacheStatus.HIT}
    assert again.task == first.task
    first_memos = [outcome.memo for outcome in first.provenance.steps]
    assert [outcome.memo.inputs for outcome in again.provenance.steps] == [memo.inputs for memo in first_memos]
    outputs = {memo.name: memo.output_digest for memo in first_memos}
    assert again.provenance.steps[-1].memo.inputs == {
        "numbers": outputs["numbers"],
        "grader": outputs["grader"],
        "proposal": digest(PROPOSAL),
    }
    assert load_draft(tmp_path / "round" / DRAFT_DIR) == again


async def test_a_rewritten_step_reruns_itself_and_its_consumers_only(tmp_path):
    first = await build(tmp_path)

    @rl_env_build_step
    async def numbers(context: RLEnvBuildContext) -> TaskResource:
        return inline_file("workspace/numbers.txt", b"7 6\n")

    rebuilt = await build(tmp_path, rl_env_build_program(numbers, grader, sink=task))

    assert statuses(rebuilt) == {"numbers": CacheStatus.MISS, "grader": CacheStatus.HIT, "task": CacheStatus.MISS}
    assert rebuilt.task.resources.worker != first.task.resources.worker


async def test_a_rewritten_step_with_identical_output_leaves_its_consumers_hit(tmp_path):
    await build(tmp_path)

    @rl_env_build_step
    async def numbers(context: RLEnvBuildContext) -> TaskResource:
        return inline_file("workspace/numbers.txt", b"6 " + b"7\n")

    rebuilt = await build(tmp_path, rl_env_build_program(numbers, grader, sink=task))

    assert statuses(rebuilt) == {"numbers": CacheStatus.MISS, "grader": CacheStatus.HIT, "task": CacheStatus.HIT}


async def test_a_changed_proposal_reruns_only_the_steps_declaring_it(tmp_path):
    first = await build(tmp_path)
    edited = {**PROPOSAL, "task": "Multiply both numbers in /workspace/numbers.txt."}

    rebuilt = await build(tmp_path, proposal=edited)

    assert statuses(rebuilt) == {"numbers": CacheStatus.HIT, "grader": CacheStatus.HIT, "task": CacheStatus.MISS}
    assert rebuilt.provenance.proposal_digest == digest(edited) != first.provenance.proposal_digest


async def test_a_failing_sink_raises_a_build_failure_carrying_its_step_name(tmp_path):
    @rl_env_build_sink
    async def task(context: RLEnvBuildContext, numbers: TaskResource) -> TaskSpec:
        raise BuildFailure("the numbers do not multiply to the answer")

    with pytest.raises(BuildFailure) as failure:
        await build(tmp_path, rl_env_build_program(numbers, sink=task))

    assert failure.value.step == "task"
    assert str(failure.value) == "task: the numbers do not multiply to the answer"


@rl_env_build_step
async def hint(context: RLEnvBuildContext, proposal: dict) -> Hint:
    messages: list[Message] = [{"role": "user", "content": proposal["task"]}]
    result = await context.model.complete_structured(messages, Hint, context.sampling)
    return result.value


@rl_env_build_sink
async def hinted_task(context: RLEnvBuildContext, numbers: TaskResource, grader: GraderPackage, hint: Hint) -> TaskSpec:
    return toy_task(context.item_id, hint.text, numbers, grader)


def prompt_of(task: TaskSpec) -> str:
    [message] = task.context.events
    assert isinstance(message, TextMessage)
    return message.content


async def test_the_model_and_sampling_enter_every_step_key(tmp_path):
    model = FakeModel()
    program = rl_env_build_program(numbers, grader, hint, sink=hinted_task)
    store = RLEnvBuildStepStore(str(tmp_path), ITEM)

    async def run(sampling: LLMPolicy) -> tuple[str, set[CacheStatus]]:
        context = RLEnvBuildContext(item_id=ITEM, model=model, sampling=sampling)
        task, outcomes = await run_program(program, context, PROPOSAL, store, model_digest(model, sampling))
        return prompt_of(task), {outcome.status for outcome in outcomes}

    first = await run(SAMPLING)
    again = await run(SAMPLING)
    resampled = await run(replace(SAMPLING, temperature=0.2))

    assert first == ("hint 1", {CacheStatus.MISS})
    assert again == ("hint 1", {CacheStatus.HIT})
    assert resampled == ("hint 2", {CacheStatus.MISS})
    assert model.calls == 2
