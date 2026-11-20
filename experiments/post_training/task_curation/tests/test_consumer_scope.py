# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consumer scope follows task contracts, independent of global schema alignment."""

import pytest
from finestore.rl.curation import ConsumerNeed
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    ExecutionRequirements,
    PlainText,
    ProviderRequirement,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from verifyit.spec import JudgeSpec

from experiments.post_training.task_curation.consumer_scope import ConsumerCapacity, consumer_needs

CAPACITY = ConsumerCapacity(cpus=4, memory_mb=8192, storage_mb=10240, agent_timeout=2400, verifier_timeout=300)


@pytest.fixture
def standard_task():
    return TaskSpec(
        id="native",
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content="Fix the program."),)),
        environment_requirements=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            docker_image="fixture@sha256:" + "a" * 64,
        ),
        execution_requirements=ExecutionRequirements(
            cpus=2,
            memory_mb=4096,
            storage_mb=10240,
            agent_timeout_sec=600,
            verifier_timeout_sec=60,
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


def test_standard_harbor_task_with_satisfiable_nondefault_allocation_is_exempt(standard_task):
    assert consumer_needs(standard_task, CAPACITY) == frozenset()


def test_tools_and_service_binding_select_bespoke_scope(standard_task):
    task = standard_task.model_copy(
        update={
            "environment_requirements": standard_task.environment_requirements.model_copy(
                update={
                    "tool_providers": {
                        "retrieval": ProviderRequirement(
                            action_interface="openresearcher-bm25:v1",
                            network_access="service_only",
                            initial_state={},
                        )
                    }
                }
            )
        }
    )
    assert consumer_needs(task, CAPACITY) == {ConsumerNeed.TOOL_PROVIDER, ConsumerNeed.SERVICE}


def test_custom_judge_selects_bespoke_scope(standard_task):
    task = standard_task.model_copy(
        update={
            "grader": standard_task.grader.model_copy(
                update={"judge": JudgeSpec(references=("answer",), question="Question")}
            )
        }
    )
    assert consumer_needs(task, CAPACITY) == {ConsumerNeed.CUSTOM_JUDGE}


def test_distinct_supported_verifier_allocation_selects_scope(standard_task):
    task = standard_task.model_copy(
        update={
            "verifier_execution_requirements": standard_task.execution_requirements.model_copy(
                update={"memory_mb": 8192}
            )
        }
    )
    assert consumer_needs(task, CAPACITY) == {ConsumerNeed.VERIFIER_ALLOCATION}


@pytest.mark.parametrize(
    "update",
    [
        {"cpus": 5},
        {"memory_mb": 8193},
        {"storage_mb": 10241},
        {"agent_timeout_sec": 2401},
        {"verifier_timeout_sec": 301},
    ],
)
def test_allocation_selects_scope_only_when_consumer_cannot_satisfy_it(standard_task, update):
    task = standard_task.model_copy(
        update={"execution_requirements": standard_task.execution_requirements.model_copy(update=update)}
    )
    assert consumer_needs(task, CAPACITY) == {ConsumerNeed.EXECUTION}
