"""Canonical task requirements reach the selected actor and verifier machines."""

from rolloutengine.machines import _machine_spec
from rolloutengine.models import MachineRuntimeSpec, RolloutLimits
from shellbox.machine import NetworkPolicy
import pytest
from skyrl_gym.rollout_specs import rollout_limits, rollout_spec
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    ExecutionRequirements,
    PlainText,
    NoGrader,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)


def test_declared_allocations_and_phase_deadlines_reach_machine_requests():
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_image="fixture@sha256:" + "a" * 64,
    )
    actor = ExecutionRequirements(
        cpus=2,
        memory_mb=4096,
        storage_mb=8192,
        agent_timeout_sec=600,
        verifier_timeout_sec=60,
    )
    verifier = actor.model_copy(update={"cpus": 4, "memory_mb": 8192, "storage_mb": 10240})
    task = TaskSpec(
        id="allocated",
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content="Fix the project."),)),
        environment_requirements=environment,
        execution_requirements=actor,
        verifier_execution_requirements=verifier,
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=ScriptGrader(argv=("bash", "/tests/test.sh"), environment=environment),
    )
    runtime = MachineRuntimeSpec(
        backend="daytona",
        network=NetworkPolicy.DENY,
        cpus=1,
        memory_mb=512,
        storage_mb=1024,
        gpus=0,
        user=None,
        startup_timeout=60,
        cleanup_timeout=60,
    )
    lowered = rollout_spec(
        task,
        {
            "task_session": "shellbox",
            "max_turns": 4,
            "max_total_tokens": None,
            "model_turn_timeout": None,
            "tool_turn_timeout": None,
            "total_turn_timeout": 1200,
            "attempt_timeout": None,
            "verifier_timeout": 300,
            "cleanup_timeout": 60,
        },
        task_machine=runtime,
        verifier_machine=runtime,
    )
    selected_actor = _machine_spec(environment, lowered.task_machine)
    selected_verifier = _machine_spec(environment, lowered.verifier_machine)
    assert (selected_actor.cpus, selected_actor.memory_mb, selected_actor.storage_mb) == (2, 4096, 8192)
    assert (selected_verifier.cpus, selected_verifier.memory_mb, selected_verifier.storage_mb) == (4, 8192, 10240)
    assert lowered.limits.total_turn_timeout == 600
    assert lowered.limits.verifier_timeout == 300


@pytest.mark.parametrize("declared,expected", [(720, 1800), (2400, 2400)])
def test_verifier_floor_preserves_declared_deadlines_and_extends_attempt_budget(declared, expected):
    task = TaskSpec(
        id="deadline",
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content="Solve the task."),)),
        environment_requirements=EnvironmentRequirements(),
        execution_requirements=ExecutionRequirements(
            cpus=1,
            memory_mb=2048,
            storage_mb=2048,
            agent_timeout_sec=600,
            verifier_timeout_sec=declared,
        ),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=ScriptGrader(
            argv=("bash", "/tests/test.sh"),
            environment=EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                docker_image="fixture@sha256:" + "a" * 64,
            ),
        ),
    )
    lowered = rollout_spec(
        task,
        {
            "task_session": "shellbox",
            "max_turns": 4,
            "max_total_tokens": None,
            "model_turn_timeout": None,
            "tool_turn_timeout": None,
            "total_turn_timeout": 600,
            "attempt_timeout": 1000,
            "verifier_timeout": 300,
            "cleanup_timeout": 60,
        },
    )
    limits = rollout_limits(task, lowered.limits, minimum_verifier_timeout=1800)
    assert limits.verifier_timeout == expected
    assert limits.attempt_timeout == 600 + expected + 60
    assert lowered.task == task


def test_verifier_floor_keeps_a_finite_attempt_without_a_separate_actor_limit():
    task = TaskSpec(
        id="harbor-deadline",
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content="Fix the project."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="Harbor owns trial grading"),
    )
    limits = RolloutLimits(
        max_turns=4,
        max_total_tokens=None,
        model_turn_timeout=None,
        tool_turn_timeout=None,
        total_turn_timeout=None,
        attempt_timeout=5400,
        verifier_timeout=None,
        cleanup_timeout=60,
    )
    updated = rollout_limits(task, limits, minimum_verifier_timeout=1800)
    assert updated.attempt_timeout == 5400
    assert updated.total_turn_timeout is None
    assert updated.verifier_timeout == 1800
