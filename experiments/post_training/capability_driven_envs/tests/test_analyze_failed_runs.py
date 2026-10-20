# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("taskforge")

import json
from collections.abc import Sequence
from dataclasses import replace

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
from taskcompendium.grading_result import GradeResult, Outcome
from taskforge.llm.client import StructuredResult
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.structured import StructuredOutputError

from experiments.post_training.capability_driven_envs import analyze_failed_runs as module
from experiments.post_training.capability_driven_envs.analyze_failed_runs import (
    Group,
    Groups,
    RolloutSkills,
    Skill,
    analyze_failed_runs,
    policy_failures,
)
from experiments.post_training.capability_driven_envs.generate_tasks import task_workpieces
from experiments.post_training.capability_driven_envs.tests.test_generate_tasks import COMPLETION

LOOKUP_CALL = {"id": "call", "type": "function", "function": {"name": "lookup", "arguments": '{"key": "six"}'}}
STOP_LOOPS = Skill(
    name="Stop repeating a call", outcome="Change the approach after a repeated result.", includes=("Notice it",)
)
CHECK_WORK = Skill(name="Check the work", outcome="Verify an answer before it is final.", includes=("Recompute",))


def evaluated(
    run_id: str, status: Outcome, reward: float | None, *turns: dict, options=None, failure=None
) -> GradedRollout:
    """A rollout of the arithmetic task whose policy produced ``turns``, graded with ``status`` and ``reward``."""
    steps = tuple(RolloutStep(ModelTurn(turn, (), (), None, "stop"), Transition(done=False), 0, ()) for turn in turns)
    prompt = {"role": "user", "content": "What is six plus six?"}
    conversation = ConversationRecord((prompt, *turns), options or {}, (), (), (), None, steps)
    rollout = Rollout(run_id, (conversation,), "stop", failure=failure)
    return GradedRollout(rollout, GradeResult(status, reward))


def loop(run_id: str) -> GradedRollout:
    turn = {"role": "assistant", "tool_calls": [LOOKUP_CALL]}
    return evaluated(run_id, Outcome.SUBMISSION_FAILURE, None, turn, turn)


def wrong(run_id: str, answer: str = "13") -> GradedRollout:
    return evaluated(run_id, Outcome.GRADED, 0.0, {"role": "assistant", "content": answer})


ROLLOUTS = [
    evaluated("pass", Outcome.GRADED, 1.0, {"role": "assistant", "content": "12"}),
    wrong("wrong"),
    loop("loop"),
    evaluated("infra", Outcome.INFRA_ERROR, None, {"role": "assistant", "content": "12"}),
    evaluated(
        "interrupted",
        Outcome.GRADED,
        0.0,
        {"role": "assistant", "content": "1"},
        failure=RolloutFailure("ConnectionError", {"operation": "model"}),
    ),
]


class Analyst:
    """Names the loop skill for a rollout with a lookup call and the check skill for the others, and merges the lines
    with the same name. A rollout that says "cut off", and a merge of more than ``max_skills`` skills, fail as a reply
    at the output limit does."""

    model = "fake"

    def __init__(self, max_skills: int = 100):
        self.max_skills = max_skills
        self.rollouts: list[str] = []
        self.merges: list[list[dict]] = []

    async def complete(self, messages: Sequence[Message], policy: LLMPolicy):
        raise AssertionError("the analysis asks only for structured output")

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        text = str(messages[-1]["content"])
        if output_type is RolloutSkills:
            self.rollouts.append(text)
            if "cut off" in text:
                raise StructuredOutputError(ValueError("the reply was cut off"), (COMPLETION,))
            value = RolloutSkills(skills=(STOP_LOOPS if "lookup" in text else CHECK_WORK,))
        else:
            lines = [json.loads(line) for line in text.splitlines()]
            self.merges.append(lines)
            if len(lines) > self.max_skills and "rollouts" not in lines[0]:
                raise StructuredOutputError(ValueError("the reply was cut off"), (COMPLETION,))
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
            value = Groups(capabilities=tuple(groups))
        assert isinstance(value, output_type)
        return StructuredResult(value, (COMPLETION,))


def analyze(rollouts: Sequence[GradedRollout], analyst: Analyst):
    return analyze_failed_runs(
        rollouts,
        analyst,
        pass_threshold=1.0,
        max_capabilities=8,
        policy=LLMPolicy(),
        merge_policy=LLMPolicy(),
        catalog_version="round-1",
    )


def test_only_the_rollouts_that_the_policy_failed_are_analyzed():
    failures = policy_failures(ROLLOUTS, pass_threshold=1.0)
    assert [graded.rollout.run_id for graded in failures] == ["wrong", "loop"]


async def test_each_capability_cites_the_failed_rollouts_of_its_skills_and_becomes_a_catalog_idea():
    analyst = Analyst()

    ideas = await analyze([*ROLLOUTS, loop("loop-2")], analyst)

    assert [(idea.capability["name"], idea.capability["evidence"]) for idea in ideas] == [
        (STOP_LOOPS.name, ["loop", "loop-2"]),
        (CHECK_WORK.name, ["wrong"]),
    ]
    assert '[1] assistant: <tool call> lookup({"key": "six"})' in analyst.rollouts[1]
    idea = ideas[0]
    assert (idea.catalog_version, idea.capability["kind"]) == ("round-1", "capability")
    assert task_workpieces({idea.capability_id: idea}, 1) == [(f"{idea.capability_id}--00", idea)]


async def test_a_rollout_shows_recorded_options_for_each_conversation_and_a_malformed_tool_call():
    policy = {"role": "assistant", "tool_calls": [{"id": "bad", "type": "function"}]}
    options = {"tools": [{"type": "function", "function": {"name": "lookup"}}], "tool_choice": "required"}
    graded = evaluated("malformed", Outcome.SUBMISSION_FAILURE, None, policy, options=options)
    first = graded.rollout.conversations[0]
    revised = replace(
        first, messages=({"role": "user", "content": "Try another tool."}, policy), options={"tool_choice": "none"}
    )
    graded = replace(graded, rollout=replace(graded.rollout, conversations=(first, revised)))
    analyst = Analyst()

    await analyze([graded], analyst)

    summary, first_header, first_message, _, revised_header, revised_message, last_message = analyst.rollouts[
        0
    ].splitlines()
    assert json.loads(summary)["run_id"] == "malformed"
    assert json.loads(first_header) == {"conversation": 0, "options": options}
    assert json.loads(revised_header) == {"conversation": 1, "options": {"tool_choice": "none"}}
    assert first_message == "[0] user: What is six plus six?"
    assert revised_message == "[2] user: Try another tool."
    assert last_message == '[3] assistant: <tool call> malformed {"id": "bad", "type": "function"}'


async def test_no_failed_rollout_needs_no_model_call():
    analyst = Analyst()
    assert await analyze(ROLLOUTS[:1], analyst) == []
    assert analyst.rollouts == [] and analyst.merges == []


async def test_a_rollout_whose_skill_call_fails_gives_no_skills_unless_too_many_fail():
    loops = [loop(f"loop-{i}") for i in range(10)]

    [idea] = await analyze([*loops, wrong("cut", "cut off")], Analyst())

    assert idea.capability["evidence"] == [f"loop-{i}" for i in range(10)]
    with pytest.raises(ValueError, match="failed for 1 of 2"):
        await analyze([loops[0], wrong("cut", "cut off")], Analyst())


async def test_a_merge_that_stays_invalid_merges_in_two_levels(monkeypatch):
    monkeypatch.setattr(module, "CHUNK_ROLLOUTS", 2)
    analyst = Analyst(max_skills=2)

    ideas = await analyze([loop("loop-1"), loop("loop-2"), wrong("wrong-1"), wrong("wrong-2")], analyst)

    assert [(idea.capability["name"], idea.capability["evidence"]) for idea in ideas] == [
        (STOP_LOOPS.name, ["loop-1", "loop-2"]),
        (CHECK_WORK.name, ["wrong-1", "wrong-2"]),
    ]
    # Three samples of the merge of all four skills, two chunk merges, and the merge of the four chunk capabilities.
    assert [len(lines) for lines in analyst.merges] == [4, 4, 4, 2, 2, 4]


def test_a_merged_capability_cannot_give_its_member_ids_as_behaviors():
    with pytest.raises(ValueError, match="not the IDs of its members"):
        Group(id="failed-runs.a", name="A", outcome="B.", includes=("k0", "k4"), members=("k0", "k4"))
