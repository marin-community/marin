# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A toy program carried through ``run_build``: ``numbers`` and ``grader`` feed the sink ``task``."""

import asyncio
import base64
import json
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
from taskforge.ledger.records import EntryKind
from taskforge.llm.agent import ReasoningReplay
from taskforge.llm.client import Completion, StructuredResult
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.stage10_proposal.model import TaskProposal
from taskforge.stage30_author.program import rl_env_build_program, rl_env_build_sink, rl_env_build_step
from taskforge.stage40_build_with_program import run as run_module
from taskforge.stage40_build_with_program.context import BuildFailure, RLEnvBuildContext
from taskforge.stage40_build_with_program.outputs import ResearchFinding, Sources
from taskforge.stage40_build_with_program.run import (
    DRAFT_DIR,
    ResearchRequirement,
    build_model_digest,
    load_draft,
    run_build,
    run_program,
    step_output,
)
from taskforge.stage40_build_with_program.sdk import research, structured
from taskforge.stage40_build_with_program.services import AgentLimits
from taskforge.stage40_build_with_program.step_store import CacheStatus, RLEnvBuildStepStore
from tests.stage40_build_with_program.conftest import WORKPIECE, SAMPLING, web_tool


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


NUMBERS = inline_file("workspace/numbers.txt", b"6 7\n")
GRADER = verifyit_package(NumericSpec(expected="42", tolerance_abs=0.0, tolerance_rel=0.0))


def toy_task(workpiece_id: str, numbers: TaskResource, grader: GraderPackage, instruction: str) -> TaskSpec:
    return TaskSpec(
        id=workpiece_id,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(
            capabilities=("shell", "filesystem"),
            command_semantics=CommandSemantics.SHELL_SIMULATOR,
            working_directory="/workspace",
        ),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=grader.grader,
        source=Source(dataset="taskforge", revision="test", row=workpiece_id, importer_revision="test"),
        resources=ResourceGroups(worker=(numbers,), verifier=grader.resources),
    )


@rl_env_build_step
async def numbers(context: RLEnvBuildContext) -> TaskResource:
    return inline_file("workspace/numbers.txt", b"6 7\n")


@rl_env_build_step
async def grader(context: RLEnvBuildContext) -> GraderPackage:
    return verifyit_package(NumericSpec(expected="42", tolerance_abs=0.0, tolerance_rel=0.0))


@rl_env_build_sink
async def task(
    context: RLEnvBuildContext, numbers: TaskResource, grader: GraderPackage, proposal: TaskProposal
) -> TaskSpec:
    return toy_task(context.workpiece_id, numbers, grader, proposal.body)


PROGRAM = rl_env_build_program(numbers, grader, sink=task)


async def build(tmp_path, proposal, services, program=PROGRAM, **options):
    model = options.pop("model", FakeModel())
    return await run_build(
        program,
        proposal,
        WORKPIECE,
        tmp_path / "round",
        str(tmp_path / "store"),
        model,
        SAMPLING,
        services,
        0,
        research=options.pop("research", ResearchRequirement.OFF),
        **options,
    )


async def test_steps_without_a_dependency_between_them_run_together_and_outcomes_keep_the_program_order(
    tmp_path, proposal, services
):
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
        context: RLEnvBuildContext, slow_numbers: TaskResource, slow_grader: GraderPackage, proposal: TaskProposal
    ) -> TaskSpec:
        await linger("joined")
        return toy_task(context.workpiece_id, slow_numbers, slow_grader, proposal.body)

    program = rl_env_build_program(slow_numbers, slow_grader, sink=joined)
    draft = await build(tmp_path, proposal, services, program)

    assert spans["slow_grader"][0] < spans["slow_numbers"][1], "the two steps did not overlap"
    assert spans["joined"][0] >= max(spans["slow_numbers"][1], spans["slow_grader"][1])
    assert [outcome.memo.name for outcome in draft.provenance.steps] == list(program.order)


async def test_a_failing_step_ends_the_build_and_cancels_the_steps_still_running(tmp_path, proposal, services):
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
        await build(tmp_path, proposal, services, rl_env_build_program(doomed, patient, sink=never))

    assert (failure.value.step, cancelled.is_set()) == ("doomed", True)


def statuses(draft) -> dict[str, CacheStatus]:
    return {outcome.memo.name: outcome.status for outcome in draft.provenance.steps}


async def test_a_rebuild_hits_every_step_and_records_the_same_task_and_input_digests(tmp_path, proposal, services):
    first = await build(tmp_path, proposal, services)
    again = await build(tmp_path, proposal, services)

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
        "proposal": digest(proposal),
    }
    assert load_draft(tmp_path / "round" / DRAFT_DIR) == again
    assert step_output(again, str(tmp_path / "store"), "numbers", TaskResource) == NUMBERS


async def test_a_rewritten_step_reruns_itself_and_its_consumers_only(tmp_path, proposal, services):
    first = await build(tmp_path, proposal, services)

    @rl_env_build_step
    async def numbers(context: RLEnvBuildContext) -> TaskResource:
        return inline_file("workspace/numbers.txt", b"7 6\n")

    rebuilt = await build(tmp_path, proposal, services, rl_env_build_program(numbers, grader, sink=task))

    assert statuses(rebuilt) == {"numbers": CacheStatus.MISS, "grader": CacheStatus.HIT, "task": CacheStatus.MISS}
    assert rebuilt.task.resources.worker != first.task.resources.worker


async def test_a_rewritten_step_with_identical_output_leaves_its_consumers_hit(tmp_path, proposal, services):
    await build(tmp_path, proposal, services)

    @rl_env_build_step
    async def numbers(context: RLEnvBuildContext) -> TaskResource:
        return inline_file("workspace/numbers.txt", b"6 " + b"7\n")

    rebuilt = await build(tmp_path, proposal, services, rl_env_build_program(numbers, grader, sink=task))

    assert statuses(rebuilt) == {"numbers": CacheStatus.MISS, "grader": CacheStatus.HIT, "task": CacheStatus.HIT}


async def test_a_changed_proposal_reruns_only_the_steps_declaring_it(tmp_path, proposal, services):
    first = await build(tmp_path, proposal, services)
    edited = replace(proposal, body=proposal.body.replace("A clerk", "An auditor"))

    rebuilt = await build(tmp_path, edited, services)

    assert statuses(rebuilt) == {"numbers": CacheStatus.HIT, "grader": CacheStatus.HIT, "task": CacheStatus.MISS}
    assert rebuilt.provenance.proposal_digest == digest(edited) != first.provenance.proposal_digest


async def test_an_invalidated_step_reruns_on_a_hit_and_its_unchanged_output_keeps_consumers_hit(
    tmp_path, proposal, services
):
    await build(tmp_path, proposal, services)

    rebuilt = await build(tmp_path, proposal, services, invalidate=frozenset({"grader"}))

    assert statuses(rebuilt) == {
        "numbers": CacheStatus.HIT,
        "grader": CacheStatus.INVALIDATED,
        "task": CacheStatus.HIT,
    }


async def test_a_failing_sink_raises_a_build_failure_carrying_its_step_name(tmp_path, proposal, services):
    @rl_env_build_sink
    async def task(context: RLEnvBuildContext, numbers: TaskResource) -> TaskSpec:
        raise BuildFailure("the numbers do not multiply to the answer")

    with pytest.raises(BuildFailure) as failure:
        await build(tmp_path, proposal, services, rl_env_build_program(numbers, sink=task))

    assert failure.value.step == "task"
    assert str(failure.value) == "task: the numbers do not multiply to the answer"
    assert not (tmp_path / "round" / DRAFT_DIR).exists()


async def test_a_step_that_never_returns_fails_the_build_as_the_programs(tmp_path, proposal, services, monkeypatch):
    monkeypatch.setattr(run_module, "STEP_TIMEOUT", 0.05)

    @rl_env_build_sink
    async def task(context: RLEnvBuildContext, numbers: TaskResource) -> TaskSpec:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    with pytest.raises(BuildFailure) as failure:
        await build(tmp_path, proposal, services, rl_env_build_program(numbers, sink=task))

    assert failure.value.step == "task"
    assert str(failure.value) == "task: ran for more than 0 seconds without returning"


@rl_env_build_step
async def hint(context: RLEnvBuildContext, proposal: TaskProposal) -> Hint:
    messages: list[Message] = [{"role": "user", "content": proposal.body}]
    result = await context.model.complete_structured(messages, Hint, context.sampling)
    return result.value


@rl_env_build_sink
async def hinted_task(context: RLEnvBuildContext, numbers: TaskResource, grader: GraderPackage, hint: Hint) -> TaskSpec:
    return toy_task(context.workpiece_id, numbers, grader, hint.text)


def prompt_of(task: TaskSpec) -> str:
    [message] = task.context.events
    assert isinstance(message, TextMessage)
    return message.content


async def test_the_model_and_sampling_enter_every_step_key(tmp_path, proposal, services):
    model = FakeModel()
    program = rl_env_build_program(numbers, grader, hint, sink=hinted_task)
    store = RLEnvBuildStepStore(str(tmp_path), WORKPIECE)

    async def run(sampling: LLMPolicy) -> tuple[str, set[CacheStatus]]:
        context = RLEnvBuildContext(WORKPIECE, 0, model, sampling, services, "")
        task, outcomes = await run_program(
            program, context, proposal, store, build_model_digest(model, sampling, services)
        )
        return prompt_of(task), {outcome.status for outcome in outcomes}

    assert await run(SAMPLING) == ("hint 1", {CacheStatus.MISS})
    assert await run(SAMPLING) == ("hint 1", {CacheStatus.HIT})
    assert await run(replace(SAMPLING, temperature=0.2)) == ("hint 2", {CacheStatus.MISS})
    assert model.calls == 2


async def test_reasoning_replay_and_agent_limits_enter_every_step_key_unless_the_caller_overrides(
    tmp_path, proposal, services
):
    keep = await build(tmp_path, proposal, services)
    dropped = replace(services, reasoning_replay=ReasoningReplay.DROP)

    assert set(statuses(await build(tmp_path, proposal, dropped)).values()) == {CacheStatus.MISS}
    shorter = replace(services, agent_limits=AgentLimits(max_turns=2, max_tokens=1024))
    assert set(statuses(await build(tmp_path, proposal, shorter)).values()) == {CacheStatus.MISS}
    overridden = await build(tmp_path, proposal, dropped, model_digest=keep.provenance.model_digest)
    assert set(statuses(overridden).values()) == {CacheStatus.HIT}


@rl_env_build_step
async def recorded_hint(context: RLEnvBuildContext, proposal: TaskProposal) -> Hint:
    return await structured(context, [{"role": "user", "content": proposal.body}], Hint)


@rl_env_build_sink
async def recorded_task(context: RLEnvBuildContext, recorded_hint: Hint) -> TaskSpec:
    return toy_task(context.workpiece_id, NUMBERS, GRADER, recorded_hint.text)


async def test_each_step_records_a_span_under_its_key_and_its_model_calls_under_its_name(
    tmp_path, proposal, services, glm_client, fake_glm, ledger
):
    fake_glm.stream(content=json.dumps({"text": "multiply"}))
    program = rl_env_build_program(recorded_hint, sink=recorded_task)
    store = RLEnvBuildStepStore(str(tmp_path), WORKPIECE)
    context = RLEnvBuildContext(WORKPIECE, 3, glm_client, SAMPLING, services, "")
    model_digest = build_model_digest(glm_client, SAMPLING, services)

    await run_program(program, context, proposal, store, model_digest)
    _, [hint_hit, task_hit] = await run_program(program, context, proposal, store, model_digest)

    steps = [entry for entry in ledger.entries if entry.kind == EntryKind.STEP]
    assert [(e.step, e.round, e.input_hash, e.attrs["cache"]) for e in steps] == [
        ("recorded_hint", 3, hint_hit.memo.key, CacheStatus.MISS),
        ("recorded_task", 3, task_hit.memo.key, CacheStatus.MISS),
        ("recorded_hint", 3, hint_hit.memo.key, CacheStatus.HIT),
        ("recorded_task", 3, task_hit.memo.key, CacheStatus.HIT),
    ]
    calls = [entry for entry in ledger.entries if entry.kind == EntryKind.LLM_CALL]
    assert [(e.step, e.round) for e in calls] == [("recorded_hint", 3)]


def researching(indices: tuple[int, ...], caught: bool = False):
    @rl_env_build_step
    async def sources(context: RLEnvBuildContext, proposal: TaskProposal) -> Sources:
        findings: list[ResearchFinding] = []
        for index in indices:
            try:
                findings.append(await research(context, proposal, index))
            except BuildFailure:
                if not caught:
                    raise
        return Sources(notes="", findings=tuple(findings))

    @rl_env_build_sink
    async def task(context: RLEnvBuildContext, sources: Sources, proposal: TaskProposal) -> TaskSpec:
        return toy_task(context.workpiece_id, NUMBERS, GRADER, proposal.body)

    return rl_env_build_program(sources, sink=task)


def serve_research(fake_glm, *, searched: bool = True) -> None:
    """One research run on the fake server: a web search (unless not ``searched``), then the notes."""
    if searched:
        fake_glm.stream(tool_calls=(("web_search", '{"query": "6 times 7"}'),), finish="tool_calls")
    fake_glm.stream(content="Confirmed at https://example.org/table.")


@pytest.fixture
def two_items(proposal) -> TaskProposal:
    header = proposal.header
    item = replace(header.research[0], purpose="confirm the answer line convention")
    body = proposal.body.replace("Sources: research[0].", "Sources: research[0] and research[1].")
    return replace(proposal, header=replace(header, research=(*header.research, item)), body=body)


async def test_a_program_that_skips_a_research_item_fails_under_the_requirement(
    tmp_path, two_items, services, glm_client, fake_glm
):
    serve_research(fake_glm)
    researched = replace(services, web_tools=(web_tool([]),))

    with pytest.raises(BuildFailure, match=r"research items \[1\] have no finding") as failure:
        await build(
            tmp_path, two_items, researched, researching((0,)), model=glm_client, research=ResearchRequirement.ENFORCED
        )
    assert failure.value.step == "sources"


async def test_a_program_that_catches_a_failed_lookup_still_fails_under_the_requirement(
    tmp_path, proposal, services, glm_client, fake_glm
):
    serve_research(fake_glm, searched=False)
    researched = replace(services, web_tools=(web_tool([]),))

    with pytest.raises(BuildFailure, match=r"research items \[0\] have no finding"):
        await build(
            tmp_path,
            proposal,
            researched,
            researching((0,), caught=True),
            model=glm_client,
            research=ResearchRequirement.ENFORCED,
        )


async def test_research_off_builds_without_every_finding(tmp_path, two_items, services, glm_client, fake_glm):
    serve_research(fake_glm)
    researched = replace(services, web_tools=(web_tool([]),))

    draft = await build(tmp_path, two_items, researched, researching((0,)), model=glm_client)

    sources = step_output(draft, str(tmp_path / "store"), "sources", Sources)
    assert [finding.index for finding in sources.findings] == [0]


async def test_a_program_without_a_sources_step_is_refused_under_the_requirement(tmp_path, proposal, services):
    with pytest.raises(BuildFailure, match="needs a 'sources' step"):
        await build(tmp_path, proposal, services, research=ResearchRequirement.ENFORCED)
