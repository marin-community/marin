# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal native task fixtures for the loop-guard preflight selection.

These stand in for the real lowered tasks (`selection.py`) in the CPU-only preflight: the
plan path reads the selection summary, never the task bodies, so the fixture rows carry the
frozen IDs and the panel protocol without the pinned Harbor packages.
"""

from rolloutengine.models import MachineRuntimeSpec, RolloutLimits, TaskRolloutSpec
from shellbox.machine import Backend, NetworkPolicy
from taskcompendium.models import (
    AgentMachine,
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ExitCodeReward,
    PlainText,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)


def fixture_task(task_id: str) -> TaskSpec:
    """One minimal task whose body is irrelevant to the plan path."""
    return TaskSpec.model_validate(
        {
            "id": task_id,
            "context": ConversationInput(events=(TextMessage(role="user", content="Return 12."),)),
            "environment_requirements": EnvironmentRequirements(),
            "answer_type": AnswerType.NUMBER,
            "answer_format": PlainText(),
            "grader": ScriptGrader(
                argv=("test", "-f", "answer.txt"),
                placement=AgentMachine(),
                reward=ExitCodeReward(),
            ),
            "source": Source(dataset="mochi-loop-guards-fixture", revision="1", row="0", importer_revision="1"),
        }
    )


def fixture_rollout(task_id: str) -> TaskRolloutSpec:
    """A native rollout spec with the panel protocol and a local task machine."""
    machine = MachineRuntimeSpec(
        backend=Backend.LOCAL,
        network=NetworkPolicy.DENY,
        cpus=None,
        memory_mb=None,
        storage_mb=None,
        gpus=0,
        user=None,
        startup_timeout=None,
        cleanup_timeout=120,
    )
    return TaskRolloutSpec(
        task=fixture_task(task_id),
        harness="shellbox",
        task_machine=machine,
        verifier_machine=None,
        limits=RolloutLimits(
            max_turns=1024,
            max_total_tokens=None,
            model_turn_timeout=600,
            tool_turn_timeout=None,
            total_turn_timeout=1800,
            attempt_timeout=8 * 60 * 60,
            verifier_timeout=300,
            cleanup_timeout=120,
        ),
    )
