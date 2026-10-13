# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A toy item run through the stage functions directly, then through ``run_item``.

The program has two steps: ``numbers`` writes ``/workspace/numbers.txt``, and the sink ``task``
assembles the ``TaskSpec`` from that file and the proposal. Validation returns a fixed summary and
review accepts it, unless a test passes other functions. Reading top to bottom:

1. the stage functions run on their own: triage accepts, the author returns the program, the
   build writes the draft;
2. ``run_item`` records an event per stage, and a relaunch of the finished item changes no byte;
3. a relaunch with another proposal or budget is refused, as is a program other than the authored one;
4. a build failure costs a revision, and the failure after the last revision rejects the item;
5. an unavailable model fails the item at validation, and a relaunch resumes there;
6. a ``Modify`` decision sends the item back to the author in round 1 with the findings.
"""

import base64
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import pytest
from pydantic import BaseModel
from taskcompendium.grader import verifyit_package
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

from taskforge.llm.client import Completion, ModelUnavailable, StructuredResult
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.stage20_triage.verdict import TriageDecision, accept
from taskforge.stage30_author.author import Revision, RevisionKind, static_author
from taskforge.stage30_author.program import RLEnvBuildProgram, rl_env_build_program, rl_env_build_sink, rl_env_build_step
from taskforge.stage40_build.context import BuildFailure, RLEnvBuildContext
from taskforge.stage40_build.run import DRAFT_DIR, TASK_FILE, TaskDraft, load_draft, run_build
from taskforge.stage40_build.step_store import model_digest
from taskforge.stage60_review.decision import Accept, BandOutcome, Decision, Modify
from taskforge.stage_coordinator.coordinator import (
    DECISION_FILE,
    STORE_DIR,
    VALIDATION_FILE,
    ItemBudget,
    StageCoordinator,
    run_item,
)
from taskforge.stage_coordinator.event_log import InMemoryEventLog, JsonEventStore
from taskforge.stage_coordinator.item_machine import EXIT, Stage, Terminal, derive_state
from taskforge.stage_coordinator.stage_functions import StageFunctions

ITEM = "toy-0"
PROPOSAL = {"id": "toy/0", "task": "Multiply the two numbers in /workspace/numbers.txt."}
BUDGET = ItemBudget(max_build_revisions=1, max_validation_retries=1)
SAMPLING = LLMPolicy()
SUMMARY = {"graded": 4, "solved": 2}


class UnusedModel:
    """The model of a program whose steps call no model."""

    model = "unused"

    async def complete(self, messages: Sequence[Message], policy: LLMPolicy) -> Completion:
        raise AssertionError("no step calls the model")

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        raise AssertionError("no step calls the model")


@rl_env_build_step
async def numbers(context: RLEnvBuildContext) -> TaskResource:
    return TaskResource(
        path="workspace/numbers.txt", source=InlineFile(content_base64=base64.b64encode(b"6 7\n").decode())
    )


@rl_env_build_sink
async def task(context: RLEnvBuildContext, numbers: TaskResource, proposal: dict[str, str]) -> TaskSpec:
    grader = verifyit_package(NumericSpec(expected="42", tolerance_abs=0.0, tolerance_rel=0.0))
    return TaskSpec(
        id=context.item_id,
        context=ConversationInput(events=(TextMessage(role="user", content=proposal["task"]),)),
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


@rl_env_build_sink
async def broken(context: RLEnvBuildContext, proposal: dict[str, str]) -> TaskSpec:
    raise BuildFailure("the numbers do not multiply to the answer")


PROGRAM = rl_env_build_program(numbers, sink=task)
FAILING = rl_env_build_program(sink=broken)


async def validate(draft: TaskDraft) -> dict[str, int]:
    return SUMMARY


def review(summary: object) -> Decision:
    return Accept(band=BandOutcome.IN_BAND)


def no_call(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("this stage does not run")


STAGES = StageFunctions(triage=accept, author=static_author(PROGRAM), build=run_build, validate=validate, review=review)


class RecordingAuthor:
    """Returns ``programs`` in turn, the last one from then on, and records each revision asked for."""

    def __init__(self, *programs: RLEnvBuildProgram):
        self.programs = list(programs)
        self.revisions: list[Revision | None] = []

    def __call__(self, proposal: object, revision: Revision | None) -> RLEnvBuildProgram:
        self.revisions.append(revision)
        return self.programs.pop(0) if len(self.programs) > 1 else self.programs[0]


def coordinator(root: Path, stages: StageFunctions = STAGES, budget: ItemBudget = BUDGET) -> StageCoordinator:
    model = UnusedModel()
    return StageCoordinator(stages, budget, root, JsonEventStore(root), model, SAMPLING, model_digest(model, SAMPLING))


def exits(log: InMemoryEventLog) -> list[tuple[str, str | None]]:
    """Each event's kind with its exit, or with its terminal for ``TERMINAL``."""
    return [(event.kind, event.attrs.get(EXIT, event.attrs.get("terminal"))) for event in log.events()]


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


ACCEPTED_EXITS = [
    ("opened", None),
    ("triaged", "accept"),
    ("authored", "authored"),
    ("built", "built"),
    ("validated", "validated"),
    ("decided", "accept"),
    ("terminal", "accepted"),
]


async def test_the_stage_functions_carry_the_proposal_to_an_accepted_draft_when_called_directly(tmp_path):
    # 1. Each stage function runs without a coordinator.
    assert accept(PROPOSAL) is TriageDecision.ACCEPT
    program = static_author(PROGRAM)(PROPOSAL, None)
    assert (program.order, program.sink.name) == (("numbers", "task"), "task")
    model = UnusedModel()
    draft = await run_build(
        program, PROPOSAL, ITEM, tmp_path, str(tmp_path / STORE_DIR), model, SAMPLING, model_digest(model, SAMPLING), 0
    )
    assert draft.task.id == ITEM
    assert load_draft(tmp_path / DRAFT_DIR) == draft
    assert review(await validate(draft)) == Accept(band=BandOutcome.IN_BAND)


async def test_run_item_records_an_event_per_stage_and_a_relaunch_changes_nothing(tmp_path):
    # 2. The coordinator calls the same functions, one event per stage, and writes the round's files.
    log = InMemoryEventLog()
    assert await run_item(coordinator(tmp_path), ITEM, PROPOSAL, log) is Terminal.ACCEPTED

    assert exits(log) == ACCEPTED_EXITS
    assert JsonEventStore(tmp_path).load(ITEM) == log.events()
    round_dir = tmp_path / "items" / ITEM / "rounds" / "0"
    for name in ("program.json", f"{DRAFT_DIR}/{TASK_FILE}", VALIDATION_FILE, DECISION_FILE):
        assert (round_dir / name).is_file(), name
    assert (tmp_path / STORE_DIR / "items" / ITEM / "steps" / "task").is_dir()

    before = tree_bytes(tmp_path)
    untouched = StageFunctions(no_call, no_call, no_call, no_call, no_call)
    relaunch = await run_item(coordinator(tmp_path, untouched), ITEM, PROPOSAL, InMemoryEventLog())
    assert relaunch is Terminal.ACCEPTED
    assert tree_bytes(tmp_path) == before


@pytest.mark.parametrize(
    ("proposal", "budget", "refusal"),
    [
        ({**PROPOSAL, "task": "Add the two numbers."}, BUDGET, "opened for proposal"),
        (PROPOSAL, replace(BUDGET, max_build_revisions=2), "opened under budget"),
    ],
    ids=["proposal", "budget"],
)
async def test_a_relaunch_with_another_proposal_or_budget_is_refused(tmp_path, proposal, budget, refusal):
    # 3. The OPENED event holds the digests of the proposal and the budget.
    await run_item(coordinator(tmp_path), ITEM, PROPOSAL, InMemoryEventLog())
    before = tree_bytes(tmp_path)

    with pytest.raises(ValueError, match=refusal):
        await run_item(coordinator(tmp_path, budget=budget), ITEM, proposal, InMemoryEventLog())
    assert tree_bytes(tmp_path) == before


async def test_a_program_other_than_the_authored_one_fails_the_item_at_build(tmp_path):
    # 3. The build asks the author again and refuses a program with another digest.
    stages = replace(STAGES, author=RecordingAuthor(PROGRAM, FAILING))
    with pytest.raises(ValueError, match="now returns program"):
        await run_item(coordinator(tmp_path, stages), ITEM, PROPOSAL, InMemoryEventLog())

    state = derive_state(JsonEventStore(tmp_path).load(ITEM) or ())
    assert (state.stage, state.terminal) == (Stage.BUILD, Terminal.FAILED)


async def test_a_build_failure_costs_a_revision_and_the_next_one_rejects_the_item(tmp_path):
    # 4. With one revision in the budget, the second build failure closes the item on its budget.
    author = RecordingAuthor(FAILING)
    log = InMemoryEventLog()
    terminal = await run_item(coordinator(tmp_path, replace(STAGES, author=author)), ITEM, PROPOSAL, log)

    assert terminal is Terminal.REJECTED
    assert exits(log) == [
        ("opened", None),
        ("triaged", "accept"),
        ("authored", "authored"),
        ("build_failed", "revise"),
        ("authored", "authored"),
        ("build_failed", "reject"),
        ("terminal", "rejected"),
    ]
    assert log.events()[-1].attrs["kind"] == "budget"
    asked = [None if revision is None else revision.kind for revision in author.revisions]
    assert asked == [None, None, RevisionKind.BUILD_FAILURE, RevisionKind.BUILD_FAILURE]
    revision = author.revisions[-1]
    assert revision is not None and "the numbers do not multiply to the answer" in revision.detail


class UnavailableOnce:
    """A validate function whose first call finds the model unavailable."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, draft: TaskDraft) -> dict[str, int]:
        self.calls += 1
        if self.calls == 1:
            raise ModelUnavailable("the hold ran out", ())
        return SUMMARY


async def test_an_unavailable_model_fails_the_item_at_validation_and_a_relaunch_resumes_there(tmp_path):
    # 5. ModelUnavailable saves the item as failed in place, and the next run_item re-enters validation.
    validate_once = UnavailableOnce()
    with pytest.raises(ModelUnavailable):
        await run_item(
            coordinator(tmp_path, replace(STAGES, validate=validate_once)), ITEM, PROPOSAL, InMemoryEventLog()
        )
    state = derive_state(JsonEventStore(tmp_path).load(ITEM) or ())
    assert (state.stage, state.terminal) == (Stage.VALIDATE, Terminal.FAILED)

    resumed = replace(STAGES, author=no_call, build=no_call, validate=validate_once)
    log = InMemoryEventLog()
    assert await run_item(coordinator(tmp_path, resumed), ITEM, PROPOSAL, log) is Terminal.ACCEPTED
    assert exits(log) == [*ACCEPTED_EXITS[:4], ("terminal", "failed"), *ACCEPTED_EXITS[4:]]


async def test_a_modify_decision_re_authors_the_item_in_round_one_with_the_findings(tmp_path):
    # 6. The author of the next round receives review's findings.
    decisions: list[Decision] = [Modify(findings=("the file states the product",)), Accept(band=BandOutcome.IN_BAND)]
    author = RecordingAuthor(PROGRAM)
    stages = replace(STAGES, author=author, review=lambda summary: decisions.pop(0))
    log = InMemoryEventLog()

    assert await run_item(coordinator(tmp_path, stages), ITEM, PROPOSAL, log) is Terminal.ACCEPTED
    assert [(event.kind, event.round) for event in log.events()][5:8] == [("decided", 0), ("authored", 1), ("built", 1)]
    assert author.revisions[2:] == [Revision(kind=RevisionKind.MODIFY, detail="the file states the product")] * 2
    assert (tmp_path / "items" / ITEM / "rounds" / "1" / DRAFT_DIR / TASK_FILE).is_file()
