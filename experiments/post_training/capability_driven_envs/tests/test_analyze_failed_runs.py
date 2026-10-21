# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("taskforge")

import json
from collections.abc import Sequence

from pydantic import BaseModel
from rolloutengine.models import (
    ConversationRecord,
    GradedRollout,
    ModelTurn,
    Rollout,
    RolloutFailure,
    RolloutStep,
    Transition,
)
from rolloutengine.test_support import arithmetic_task
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import AnswerCall, TaskSpec
from taskforge.llm.client import StructuredResult
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.structured import StructuredOutputError
from zephyr.context import ZephyrContext
from zephyr.coordinator import ZephyrExecutionResult

from experiments.post_training.capability_driven_envs.analyze_failed_runs.models import (
    Capability,
    Group,
    MaybeIdea,
    RolloutIdeas,
    groups_of,
)
from experiments.post_training.capability_driven_envs.analyze_failed_runs.pipeline import (
    INVALID_REPLIES,
    analyze_failed_runs,
    merge_ideas,
    yield_ideas,
)
from experiments.post_training.capability_driven_envs.analyze_failed_runs.rollouts import FAILURE_OPERATION
from experiments.post_training.capability_driven_envs.generate_tasks import task_workpieces
from experiments.post_training.capability_driven_envs.tests.test_generate_tasks import COMPLETION

LOOKUP_CALL = {"id": "call", "type": "function", "function": {"name": "lookup", "arguments": '{"key": "six"}'}}
STOP_LOOPS = Capability(
    id="failed-runs.tool-use.stop-loops",
    name="Stop repeating a call",
    outcome="Change the approach after a repeated result.",
    includes=("Notice it",),
)
CHECK_WORK = Capability(
    id="failed-runs.verify.check-work",
    name="Check the work",
    outcome="Verify an answer before it is final.",
    includes=("Recompute",),
)
CUT_OFF = StructuredOutputError(ValueError("the reply was cut off"), (COMPLETION,))


def evaluated(
    run_id: str, status: Outcome, reward: float | None, *turns: dict, failure=None, task_id: str | None = None
) -> tuple[TaskSpec, GradedRollout]:
    """A rollout of an arithmetic task whose policy produced ``turns``, graded with ``status`` and ``reward``.

    Each rollout has its own task unless ``task_id`` names a task that other rollouts share.
    """
    steps = tuple(RolloutStep(ModelTurn(turn, (), (), None, "stop"), Transition(done=False), 0, ()) for turn in turns)
    prompt = {"role": "user", "content": "What is six plus six?"}
    conversation = ConversationRecord((prompt, *turns), (), (), (), None, steps)
    rollout = Rollout(run_id, (conversation,), "stop", failure=failure)
    return arithmetic_task(id=task_id or f"task-{run_id}"), GradedRollout(rollout, GradeResult(status, reward))


def loop(run_id: str, task_id: str | None = None) -> tuple[TaskSpec, GradedRollout]:
    turn = {"role": "assistant", "tool_calls": [LOOKUP_CALL]}
    return evaluated(run_id, Outcome.SUBMISSION_FAILURE, None, turn, turn, task_id=task_id)


def wrong(run_id: str, answer: str = "13") -> tuple[TaskSpec, GradedRollout]:
    return evaluated(run_id, Outcome.GRADED, 0.0, {"role": "assistant", "content": answer})


ROLLOUTS = [
    evaluated("pass", Outcome.GRADED, 1.0, {"role": "assistant", "content": "12"}),
    wrong("wrong"),
    loop("loop"),
    evaluated("infra", Outcome.INFRA_ERROR, None, {"role": "assistant", "content": "12"}),
    evaluated(
        "model-error",
        Outcome.GRADED,
        0.0,
        {"role": "assistant", "content": "1"},
        failure=RolloutFailure("ConnectionError", {FAILURE_OPERATION: "model"}),
    ),
    evaluated(
        "deadline",
        Outcome.UNAVAILABLE,
        None,
        {"role": "assistant", "content": "Let me count: one, two, three"},
        failure=RolloutFailure("TimeoutError", {FAILURE_OPERATION: "attempt"}),
    ),
]


class Analyst:
    """Names no skill for a rollout that says "no gap", the loop skill for a rollout with a lookup call, and the check
    skill for the others, and merges the lines with the same name. A rollout that says "cut off", and the first
    ``failed_merges`` merges, fail as a reply at the output limit does. It keeps the messages of its calls."""

    model = "fake"

    def __init__(self, failed_merges: int = 0):
        self.failed_merges = failed_merges
        self.requests: list[str] = []

    async def __aenter__(self) -> "Analyst":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def complete(self, messages: Sequence[Message], policy: LLMPolicy):
        raise AssertionError("the analysis asks only for structured output")

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        text = str(messages[-1]["content"])
        self.requests.append(text)
        if output_type is RolloutIdeas:
            if "cut off" in text:
                raise CUT_OFF
            skills = () if "no gap" in text else (STOP_LOOPS if "lookup" in text else CHECK_WORK,)
            value = RolloutIdeas(ideas=skills)
        else:
            if self.failed_merges:
                self.failed_merges -= 1
                raise CUT_OFF
            lines = [json.loads(line) for line in text.splitlines()]
            names = dict.fromkeys(line["name"] for line in lines)
            groups = (
                Group(
                    id=f"failed-runs.general.skill-{i}",
                    name=name,
                    outcome="An outcome.",
                    includes=("A behavior",),
                    members=tuple(line["id"] for line in lines if line["name"] == name),
                )
                for i, name in enumerate(names)
            )
            value = output_type(capabilities=tuple(groups))
        assert isinstance(value, output_type)
        return StructuredResult(value, (COMPLETION,))


def no_client():
    raise AssertionError("no model call was expected")


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path_factory.mktemp("zephyr"))) as context:
        yield context


def analyze(ctx: ZephyrContext, rollouts, shard_size: int = 100, make_client=Analyst) -> ZephyrExecutionResult:
    plan = analyze_failed_runs(
        rollouts,
        make_client,
        pass_threshold=1.0,
        max_new_ideas=8,
        shard_size=shard_size,
        extract_ideas_llm_policy=LLMPolicy(),
        merge_ideas_llm_policy=LLMPolicy(),
        catalog_version="round-1",
    )
    return ctx.execute(plan)


def names_and_evidence(ideas) -> list[tuple[str, list[str]]]:
    return [(idea.capability["name"], idea.capability["evidence"]) for idea in ideas]


def test_each_idea_cites_the_failed_rollouts_of_its_skills_and_becomes_a_catalog_idea(ctx):
    ideas = analyze(ctx, [*ROLLOUTS, loop("loop-2"), loop("loop-3")]).results

    assert names_and_evidence(ideas) == [
        (STOP_LOOPS.name, ["loop", "loop-2", "loop-3"]),
        (CHECK_WORK.name, ["deadline", "wrong"]),
    ]
    idea = ideas[0]
    assert (idea.catalog_version, idea.capability["kind"]) == ("round-1", "capability")
    assert task_workpieces({idea.capability_id: idea}, 1) == [(f"{idea.capability_id}--00", idea)]


def test_the_ideas_of_several_shards_merge_again(ctx):
    rollouts = [loop(f"loop-{i}") for i in range(4)] + [wrong(f"wrong-{i}") for i in range(2)]

    ideas = analyze(ctx, rollouts, shard_size=3).results

    assert names_and_evidence(ideas) == [
        (STOP_LOOPS.name, [f"loop-{i}" for i in range(4)]),
        (CHECK_WORK.name, ["wrong-0", "wrong-1"]),
    ]


def test_a_rollout_without_a_gap_gives_no_ideas_and_only_an_invalid_reply_is_counted(ctx):
    result = analyze(ctx, [loop("loop-1"), loop("loop-2"), wrong("cut", "cut off"), wrong("calm", "no gap")])

    assert names_and_evidence(result.results) == [(STOP_LOOPS.name, ["loop-1", "loop-2"])]
    assert result.counters[INVALID_REPLIES] == 1


def test_an_idea_needs_rollouts_of_two_tasks_unless_one_task_failed(ctx):
    samples = [loop(f"sample-{i}", task_id="task-a") for i in range(3)]

    ideas = analyze(ctx, [*samples, wrong("wrong-b"), wrong("wrong-c")]).results

    assert names_and_evidence(ideas) == [(CHECK_WORK.name, ["wrong-b", "wrong-c"])]
    assert names_and_evidence(analyze(ctx, samples).results) == [(STOP_LOOPS.name, [f"sample-{i}" for i in range(3)])]


def test_no_failed_rollout_needs_no_model_call(ctx):
    assert analyze(ctx, ROLLOUTS[:1], make_client=no_client).results == []


def test_the_rollout_shows_the_tools_that_the_task_gave_and_a_malformed_tool_call():
    turn = {"role": "assistant", "tool_calls": [{"id": "bad", "type": "function"}]}
    _, graded = evaluated("malformed", Outcome.SUBMISSION_FAILURE, None, turn)
    analyst = Analyst()

    rollout = (arithmetic_task(id="task-malformed", answer_format=AnswerCall()), graded)
    [maybe] = yield_ideas(rollout, make_client=lambda: analyst, policy=LLMPolicy())

    summary, tools, *messages = analyst.requests[0].splitlines()
    assert json.loads(summary)["run_id"] == "malformed" and maybe.evidence == {"malformed"}
    assert [tool["function"]["name"] for tool in json.loads(tools)["tools"]] == ["submit_answer"]
    assert messages[-1] == '[1] assistant: <tool call> malformed {"id": "bad", "type": "function"}'


def test_a_merge_samples_again_after_a_cut_off_reply():
    analyst = Analyst(failed_merges=2)
    ideas = [
        MaybeIdea(STOP_LOOPS, frozenset({"a"}), frozenset({"task-a"})),
        MaybeIdea(STOP_LOOPS, frozenset({"a", "b"}), frozenset({"task-a", "task-b"})),
    ]

    [merged] = merge_ideas(
        ideas, make_client=lambda: analyst, max_new_ideas=8, max_ideas_per_merge=100, policy=LLMPolicy()
    )

    assert (merged.idea.name, merged.num_rollouts, merged.num_tasks, len(analyst.requests)) == (STOP_LOOPS.name, 2, 2, 3)


def test_more_ideas_than_one_merge_call_holds_merge_in_rounds_and_keep_their_evidence():
    analyst = Analyst()
    ideas = [MaybeIdea(STOP_LOOPS, frozenset({f"loop-{i}"}), frozenset({f"task-{i}"})) for i in range(5)]
    ideas.append(MaybeIdea(CHECK_WORK, frozenset({"wrong"}), frozenset({"task-wrong"})))

    merged = merge_ideas(ideas, make_client=lambda: analyst, max_new_ideas=2, max_ideas_per_merge=4, policy=LLMPolicy())

    assert [(candidate.idea.name, candidate.num_rollouts) for candidate in merged] == [
        (STOP_LOOPS.name, 5),
        (CHECK_WORK.name, 1),
    ]
    assert max(len(request.splitlines()) for request in analyst.requests) == 4


def test_a_merge_reply_puts_each_line_in_exactly_one_capability_with_its_own_id():
    reply_type = groups_of(frozenset({"k0", "k1", "k2"}))

    def group(capability_id: str, *members: str) -> Group:
        return Group(id=capability_id, name="A", outcome="B.", includes=("C",), members=members)

    assert len(reply_type(capabilities=(group("a", "k0", "k1"), group("b", "k2"))).capabilities) == 2
    for invalid in [
        (group("a", "k0", "k1"),),
        (group("a", "k0", "k1"), group("b", "k1", "k2")),
        (group("a", "k0", "k1", "k2", "k3"),),
        (group("a", "k0", "k1"), group("a", "k2")),
    ]:
        with pytest.raises(ValueError):
            reply_type(capabilities=invalid)


def test_a_merged_capability_cannot_give_its_member_ids_as_behaviors():
    with pytest.raises(ValueError, match="not the IDs of its members"):
        Group(id="failed-runs.a", name="A", outcome="B.", includes=("k0", "k4"), members=("k0", "k4"))
