"""Canonical task requirements reach the selected actor and verifier machines."""

from rolloutengine.machines import _machine_spec
from rolloutengine.models import MachineRuntimeSpec
from shellbox.machine import NetworkPolicy
from skyrl_gym.rollout_specs import rollout_spec
from taskcompendium.models import (
    AnswerType, CommandSemantics, ConversationInput, EnvironmentRequirements, ExecutionRequirements,
    PlainText, ScriptGrader, Source, TaskSpec, TextMessage,
)


def test_declared_allocations_and_phase_deadlines_reach_machine_requests():
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="fixture@sha256:" + "a" * 64,
    )
    actor = ExecutionRequirements(
        cpus=2, memory_mb=4096, storage_mb=8192, agent_timeout_sec=600, verifier_timeout_sec=60,
    )
    verifier = actor.model_copy(update={"cpus":4, "memory_mb":8192, "storage_mb":10240})
    task = TaskSpec(
        id="allocated", source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content="Fix the project."),)),
        environment_requirements=environment, execution_requirements=actor,
        verifier_execution_requirements=verifier, answer_type=AnswerType.TEXT, answer_format=PlainText(),
        grader=ScriptGrader(argv=("bash", "/tests/test.sh"), environment=environment),
    )
    runtime = MachineRuntimeSpec(
        backend="daytona", network=NetworkPolicy.DENY, cpus=1, memory_mb=512, storage_mb=1024,
        gpus=0, user=None, startup_timeout=60, cleanup_timeout=60,
    )
    lowered = rollout_spec(task, {
        "task_session":"shellbox", "max_turns":4, "max_total_tokens":None,
        "model_turn_timeout":None, "tool_turn_timeout":None, "total_turn_timeout":1200,
        "attempt_timeout":None, "verifier_timeout":300, "cleanup_timeout":60,
    }, task_machine=runtime, verifier_machine=runtime)
    selected_actor = _machine_spec(environment, lowered.task_machine)
    selected_verifier = _machine_spec(environment, lowered.verifier_machine)
    assert (selected_actor.cpus, selected_actor.memory_mb, selected_actor.storage_mb) == (2,4096,8192)
    assert (selected_verifier.cpus, selected_verifier.memory_mb, selected_verifier.storage_mb) == (4,8192,10240)
    assert lowered.limits.total_turn_timeout == 600
    assert lowered.limits.verifier_timeout == 60
