# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic implementations of the Baby RSI interfaces."""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from taskcompendium.grading import exact_answer
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, Source, TaskSpec, TextMessage
from taskcompendium.rollout import ModelRequest, ModelTurn, RolloutEngine, ShellboxRolloutEngine
from taskcompendium.submission import AnswerFormat, SubmissionConvention

from experiments.post_training.baby_rsi.interfaces import (
    AnalysisEngine,
    BabyRsiRound,
    BabyRsiRun,
    PolicyState,
    ProblemGenerator,
    RolloutBuffer,
    RolloutRecord,
    TaskFilter,
    TaskPrompt,
    TaskPromptSet,
    Trainer,
)

ADDITION = "arithmetic.addition"
MULTIPLICATION = "arithmetic.multiplication"
DUMMY_REVISION = "dummy-v1"

_ARITHMETIC = re.compile(r"Calculate\s+(\d+)\s*([+*])\s*(\d+)", re.IGNORECASE)
_PLAIN = SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN)


@dataclass(frozen=True)
class _TaskTemplate:
    instruction: str
    answer: str


_EVALUATION_TEMPLATES = {
    ADDITION: _TaskTemplate("Calculate 19 + 23. Give only the number.", "42"),
    MULTIPLICATION: _TaskTemplate("Calculate 6 * 7. Give only the number.", "42"),
}
_TRAINING_TEMPLATES = {
    ADDITION: _TaskTemplate("Calculate 8 + 5. Give only the number.", "13"),
    MULTIPLICATION: _TaskTemplate("Calculate 3 * 5. Give only the number.", "15"),
}


def _task(
    capability_id: str,
    template: _TaskTemplate,
    *,
    task_id: str,
    importer_revision: str,
    context: str | None = None,
) -> TaskSpec:
    instruction = template.instruction if context is None else f"Practice focus: {context}\n\n{template.instruction}"
    return TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=exact_answer(template.answer),
        metadata={"baby_rsi_capability": capability_id},
        source=Source(
            dataset="baby-rsi-dummy",
            revision=DUMMY_REVISION,
            row=task_id,
            importer_revision=importer_revision,
        ),
    )


def fixed_evaluation_tasks() -> tuple[TaskSpec, ...]:
    """Return tasks that stay outside the generated training set."""
    return tuple(
        _task(
            capability_id,
            template,
            task_id=f"baby-rsi/evaluation/{capability_id}",
            importer_revision=DUMMY_REVISION,
        )
        for capability_id, template in _EVALUATION_TEMPLATES.items()
    )


def initial_policy() -> PolicyState:
    """Return a policy that knows addition but not multiplication."""
    return PolicyState(model_id="dummy-student-v0", learned_capabilities=(ADDITION,))


def teacher_policy() -> PolicyState:
    """Return the inference policy used to produce successful training responses."""
    return PolicyState(model_id="dummy-teacher-v0", learned_capabilities=(ADDITION, MULTIPLICATION))


def accept_all_tasks(specification: TaskSpec) -> bool:
    """Accept every schema-valid TaskSpec."""
    return bool(specification.id)


class DummyProblemGenerator(ProblemGenerator):
    """Map each abstract failure to one different arithmetic training task."""

    def generate_tasks(
        self,
        task_prompts: Sequence[TaskPrompt],
        inference_policy: PolicyState,
        task_filter: TaskFilter,
    ) -> tuple[TaskSpec, ...]:
        tasks = []
        for prompt in task_prompts:
            template = _TRAINING_TEMPLATES.get(prompt.capability_id)
            if template is None:
                raise ValueError(f"The dummy generator has no template for {prompt.capability_id!r}")
            task = _task(
                prompt.capability_id,
                template,
                task_id=f"baby-rsi/training/{prompt.id}",
                importer_revision=f"{DUMMY_REVISION}:{inference_policy.model_id}",
                context=prompt.description,
            )
            if task_filter(task):
                tasks.append(task)
        return tuple(tasks)


class DummyRolloutModel:
    """Return deterministic arithmetic responses through the shared rollout engine."""

    def __init__(self, policy: PolicyState):
        self.policy = policy

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        instruction = self._arithmetic_instruction(request.messages)
        match = _ARITHMETIC.search(instruction)
        if match is None:
            raise ValueError("The dummy rollout model cannot find an arithmetic instruction")
        left, operation, right = match.groups()
        capability_id = ADDITION if operation == "+" else MULTIPLICATION
        if capability_id not in self.policy.learned_capabilities:
            response = "0"
        elif operation == "+":
            response = str(int(left) + int(right))
        else:
            response = str(int(left) * int(right))
        prompt_tokens = tuple(json.dumps(request.messages, sort_keys=True).encode())
        response_tokens = tuple(response.encode())
        return ModelTurn(
            message={"role": "assistant", "content": response},
            prompt_token_ids=prompt_tokens,
            response_token_ids=response_tokens,
            logprobs=(0.0,) * len(response_tokens),
            stop_reason="stop",
            text=response,
        )

    @staticmethod
    def _arithmetic_instruction(messages: tuple[dict[str, Any], ...]) -> str:
        for message in messages:
            content = message.get("content")
            if isinstance(content, str) and _ARITHMETIC.search(content):
                return content
        raise ValueError("The dummy rollout model received no arithmetic instruction")


def dummy_rollout_engine(policy: PolicyState) -> RolloutEngine:
    """Build the shared rollout engine with a deterministic model transport."""
    return ShellboxRolloutEngine(
        DummyRolloutModel(policy),
        {},
        max_turns=1,
        command_timeout=1,
        convention=_PLAIN,
    )


def _collect_rollouts(policy: PolicyState, tasks: tuple[TaskSpec, ...]) -> RolloutBuffer:
    labels = {task.id: _capability_id(task) for task in tasks}
    records = []
    engine = dummy_rollout_engine(policy)
    for rollout in engine.generate(iter(tasks)):
        response = rollout.messages[-1].get("content", "")
        assert isinstance(response, str)
        records.append(
            RolloutRecord(
                task_id=rollout.task_id,
                capability_id=labels[rollout.task_id],
                response=response,
                reward=float(rollout.grade.reward or 0.0),
            )
        )
    return RolloutBuffer(policy_id=policy.model_id, rollouts=tuple(records))


def run_policy(policy: PolicyState, tasks: tuple[TaskSpec, ...]) -> RolloutBuffer:
    """Run a policy through TaskCompendium and collect the feedback fields."""
    return _collect_rollouts(policy, tasks)


def _capability_id(task: TaskSpec) -> str:
    capability_id = task.metadata.get("baby_rsi_capability")
    if not isinstance(capability_id, str):
        raise ValueError(f"Task {task.id!r} has no Baby RSI capability")
    return capability_id


class DummyTrainer(Trainer):
    """Mark capabilities with successful teacher rollouts as learned."""

    def __init__(self, policy: PolicyState):
        self.policy = policy

    def train(self, buffer: RolloutBuffer) -> PolicyState:
        learned = set(self.policy.learned_capabilities)
        learned.update(rollout.capability_id for rollout in buffer.rollouts if rollout.reward == 1.0)
        capabilities = tuple(sorted(learned))
        suffix = "-".join(capability.rsplit(".", 1)[-1] for capability in capabilities)
        return PolicyState(model_id=f"{self.policy.model_id}-trained-{suffix}", learned_capabilities=capabilities)


class DummyAnalysisEngine(AnalysisEngine):
    """Group failures by capability without copying held-out task text."""

    def analyze_failed_runs(self, rollouts: RolloutBuffer) -> TaskPromptSet:
        failed_capabilities = sorted({rollout.capability_id for rollout in rollouts.rollouts if rollout.reward < 1.0})
        return TaskPromptSet(
            prompts=tuple(
                TaskPrompt(
                    id=f"failure-{capability_id}",
                    capability_id=capability_id,
                    description=f"The policy failed tasks that require {capability_id}.",
                )
                for capability_id in failed_capabilities
            )
        )


def run_dummy_rsi(rounds: int) -> BabyRsiRun:
    """Run the finite feedback loop without model inference or training jobs."""
    if rounds <= 0:
        raise ValueError("Baby RSI requires at least one round")

    evaluation_tasks = fixed_evaluation_tasks()
    inference_policy = teacher_policy()
    policy = initial_policy()
    generator = DummyProblemGenerator()
    analysis_engine = DummyAnalysisEngine()

    initial_evaluation = run_policy(policy, evaluation_tasks)
    evaluation_before = initial_evaluation
    completed_rounds = []
    for _ in range(rounds):
        prompts = analysis_engine.analyze_failed_runs(evaluation_before)
        if not prompts.prompts:
            break
        training_tasks = generator.generate_tasks(prompts.prompts, inference_policy, accept_all_tasks)
        training_rollouts = run_policy(inference_policy, training_tasks)
        policy = DummyTrainer(policy).train(training_rollouts)
        evaluation_after = run_policy(policy, evaluation_tasks)
        completed_rounds.append(
            BabyRsiRound(
                evaluation_before=evaluation_before,
                prompts=prompts,
                training_tasks=training_tasks,
                training_rollouts=training_rollouts,
                policy_after=policy,
                evaluation_after=evaluation_after,
            )
        )
        evaluation_before = evaluation_after

    return BabyRsiRun(
        initial_evaluation=initial_evaluation,
        rounds=tuple(completed_rounds),
        final_evaluation=evaluation_before,
    )
