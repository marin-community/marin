# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end lane tests on the effect stack with scripted model and grading fakes.

The fakes stand in for the two machines the lane talks to: a scripted completion backend
replays the generator, and a scripted grading engine grades the candidate reply the way an
in-process verifyit grader would. The taskforge effect machinery -- inference effects, grade
effects, event-log replay -- is the real production code.
"""

import asyncio
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from rolloutengine.models import GradedRollout, Rollout
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    ConversationInput,
    ConversationToolCall,
    EnvironmentRequirements,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from taskforge.llm.client import Completion as ProviderCompletion
from taskforge.llm.client import Usage as ProviderUsage
from taskforge.resources.grading import GradingResource
from taskforge.resources.web import FakeWeb
from taskforge.wiring import build_resources
from verifyit.spec import ExactSpec

from experiments.post_training.task_curation.balanced_reasoning.bands import Band, budget_for
from experiments.post_training.task_curation.balanced_reasoning.difficulty import TaskDifficulty
from experiments.post_training.task_curation.balanced_reasoning.graceful import BEST_EFFORT_MARKER
from experiments.post_training.task_curation.balanced_reasoning.lane import (
    RESOLUTION_SOLVED_SHORTEST,
    RESOLUTION_UNSOLVED,
    run_selected_task,
)
from experiments.post_training.task_curation.balanced_reasoning.runtime import lane_policy
from experiments.post_training.task_curation.balanced_reasoning.selection import CandidateTask, SelectedTask

EXPECTED = "12"
EXPECTED_PROMPT = "What is 6 + 6? Answer with the number only."


def make_task(prompt: str, expected: str, task_id: str = "lane-test-task") -> TaskSpec:
    """A minimal machine-less exact-match task: the shape the lane's reasoning round consumes."""
    return TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content=prompt),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=verifyit_package(ExactSpec(expected=(expected,))).grader,
        source=Source(dataset="org/project", revision="pinned-revision", row="0", importer_revision="1"),
    )


def example_task() -> TaskSpec:
    return make_task(EXPECTED_PROMPT, EXPECTED)


def selected_task(
    solve_rate: float = 0.5, task_id: str = "t1", sha: str = "ab" * 32, domain: str = "math"
) -> SelectedTask:
    candidate = CandidateTask(
        task_id=task_id,
        source_id="src",
        task_sha256=sha,
        domain=domain,
        task_path=f"/tasks/{task_id}.json",
    )
    difficulty = TaskDifficulty(
        source_id="src",
        task_sha256=sha,
        task_id=task_id,
        solve_rate=solve_rate,
        scored=3,
        basis="ledger:hosted",
    )
    return SelectedTask(candidate, difficulty, Band.MEDIUM, budget_for(Band.MEDIUM))


@dataclass
class FakeBackend:
    """The ``CompletionBackend`` the taskforge runtime expects, with a scripted reply list."""

    completions: tuple[ProviderCompletion, ...]
    calls: int = 0

    @property
    def model(self) -> str:
        return "glm-5.3"

    async def complete(self, messages, policy, request_fields=None) -> ProviderCompletion:
        if self.calls >= len(self.completions):
            raise RuntimeError("Fake backend script exhausted")
        result = self.completions[self.calls]
        self.calls += 1
        return result


def provider_completion(content: str, *, completion_tokens: int, reasoning: str = "") -> ProviderCompletion:
    return ProviderCompletion(
        content=content,
        reasoning=reasoning,
        tool_calls=(),
        finish_reason="stop",
        usage=ProviderUsage(
            prompt_tokens=50,
            completion_tokens=completion_tokens,
            reasoning_tokens=completion_tokens - 5,
            cached_tokens=0,
        ),
        wall_time=1.0,
        ttft=0.1,
        decode_time=0.9,
        continuations=0,
        attempts=(),
    )


@dataclass
class FakeGradingEngine:
    """Grades the candidate reply the way an in-process exact-match grader would."""

    expected: str = EXPECTED

    async def run_task(self, spec, model, *, run_id=None):
        reply = model.turns[-1]["content"]
        passed = reply == self.expected
        grade = GradeResult(
            status=Outcome.GRADED,
            reward=1.0 if passed else 0.0,
            passed=passed,
            detail={"reply": reply},
        )
        rollout = Rollout(run_id=run_id or "fake", conversations=(), stop_reason="stop")
        return GradedRollout(rollout=rollout, grade=grade)


def lane_resources(policy, backend, expected: str = EXPECTED):
    """The resources bundle with the grading engine swapped for the scripted grader."""
    resources = build_resources(policy, {"author": backend, "solver": backend}, {}, FakeWeb({}, {}))
    grading = GradingResource(FakeGradingEngine(expected), "lane-grader")
    return replace(resources, grading=grading)


def lane_policy_defaults():
    return lane_policy({"sampling.temperature": 0.7, "sampling.max_continuations": 0})


def test_lane_solves_and_selects_shortest(tmp_path: Path) -> None:
    async def run() -> object:
        policy = lane_policy_defaults()
        backend = FakeBackend(
            (
                provider_completion(EXPECTED, completion_tokens=100),
                provider_completion("twelve", completion_tokens=50),
                provider_completion(EXPECTED, completion_tokens=80),
                provider_completion(EXPECTED, completion_tokens=30),
            )
        )
        return await run_selected_task(
            selected_task(), example_task(), policy, lane_resources(policy, backend), tmp_path / "t1"
        )

    outcome = asyncio.run(run())
    assert outcome.resolution_kind == RESOLUTION_SOLVED_SHORTEST
    assert outcome.selected_index == 3, "the shortest correct trace must ship"
    assert outcome.attempts[1].grade_passed is False
    assert outcome.attempts[0].grade_passed is True
    assert outcome.grader_kind == "verifyit"
    document = json.loads((tmp_path / "t1" / "attempt.json").read_text())
    assert document["attempts"][3]["completion_tokens"] == 30
    assert document["band"] == "medium"
    assert (tmp_path / "t1" / "status.json").exists()


def test_lane_graceful_resolution(tmp_path: Path) -> None:
    async def run() -> object:
        policy = lane_policy_defaults()
        backend = FakeBackend(
            (
                provider_completion("thirteen", completion_tokens=40),
                provider_completion("fourteen", completion_tokens=41),
                provider_completion("thirteen", completion_tokens=42),
                provider_completion("fourteen", completion_tokens=43),
                provider_completion(
                    f"{BEST_EFFORT_MARKER} 13 -- the partial sums point here.",
                    completion_tokens=60,
                    reasoning="I could not finish the reduction.",
                ),
            )
        )
        return await run_selected_task(
            selected_task(), example_task(), policy, lane_resources(policy, backend), tmp_path / "t1"
        )

    outcome = asyncio.run(run())
    assert outcome.resolution_kind == "graceful_best_effort"
    assert outcome.selected_index == len(outcome.attempts) - 1
    assert outcome.attempts[-1].graceful_resolution == "graceful_best_effort"
    assert outcome.attempts[-1].kind == "graceful"


def test_lane_unsolved_when_graceful_contract_fails(tmp_path: Path) -> None:
    async def run() -> object:
        policy = lane_policy_defaults()
        backend = FakeBackend(
            (
                *(provider_completion("thirteen", completion_tokens=40 + index) for index in range(4)),
                provider_completion("probably 13", completion_tokens=60),  # no contract closing
            )
        )
        return await run_selected_task(
            selected_task(), example_task(), policy, lane_resources(policy, backend), tmp_path / "t1"
        )

    outcome = asyncio.run(run())
    assert outcome.resolution_kind == RESOLUTION_UNSOLVED
    assert outcome.selected_index is None
    assert outcome.unsolved_reason is not None


def test_lane_replays_recorded_effects_on_relaunch(tmp_path: Path) -> None:
    async def run() -> tuple[object, FakeBackend]:
        policy = lane_policy_defaults()
        script = tuple(provider_completion(EXPECTED, completion_tokens=30 + index) for index in range(4))
        first_backend = FakeBackend(script)
        workdir = str(tmp_path / "t1")
        await run_selected_task(selected_task(), example_task(), policy, lane_resources(policy, first_backend), workdir)
        second_backend = FakeBackend(())  # any call would exhaust the script and fail the test
        outcome = await run_selected_task(
            selected_task(), example_task(), policy, lane_resources(policy, second_backend), workdir
        )
        return outcome, second_backend

    outcome, second_backend = asyncio.run(run())
    assert second_backend.calls == 0, "relaunch must replay recorded model calls, not regenerate"
    assert outcome.resolution_kind == RESOLUTION_SOLVED_SHORTEST


def test_lane_rejects_tool_contexts(tmp_path: Path) -> None:
    async def run() -> None:
        policy = lane_policy_defaults()
        tool_call = AssistantToolCalls(calls=(ConversationToolCall(call_id="c1", name="shell", arguments="{}"),))
        task = TaskSpec.model_construct(
            id="lane-tool-task",
            context=ConversationInput.model_construct(
                events=(TextMessage(role="user", content=EXPECTED_PROMPT), tool_call)
            ),
            environment_requirements=EnvironmentRequirements(),
            answer_type=AnswerType.TEXT,
            answer_format=PlainText(),
            grader=verifyit_package(ExactSpec(expected=(EXPECTED,))).grader,
            source=Source(dataset="org/project", revision="pinned-revision", row="0", importer_revision="1"),
        )
        backend = FakeBackend(())
        await run_selected_task(selected_task(), task, policy, lane_resources(policy, backend), tmp_path / "t1")

    with pytest.raises(ValueError, match="single-turn traces"):
        asyncio.run(run())
