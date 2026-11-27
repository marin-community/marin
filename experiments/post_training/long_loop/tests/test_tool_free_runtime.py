# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""tool_free_single tasks lower to one tool-free completion and grade with host-captured reply status."""

import json

import pytest
from pydantic import ValidationError
from rolloutengine.harness.tool_free import TOOL_FREE_HARNESS, ToolFreeHarness, tool_free_prompt
from rolloutengine.models import TaskRolloutSpec
from skyrl_train.dataset.tasks import canonical_task_spec
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    Backend,
    CommandSemantics,
    ConversationInput,
    ConversationToolCall,
    ConversationTrace,
    DockerBuildContext,
    EnvironmentRequirements,
    FileReward,
    FunctionDefinition,
    GradingAttempt,
    PlainText,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.reply_status import reply_status_files
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.long_loop.campaign_runtime import bind_runtime

STATUS_PATH = "/app/answer_status.json"
VERIFIER = {
    "backend": "daytona",
    "cpus": 1,
    "memory_mb": 1024,
    "storage_mb": 1024,
    "gpus": 0,
    "network": "deny",
    "user": None,
    "startup_timeout": 600,
    "cleanup_timeout": 60,
}
CONFIG = {
    "default_harness": "shellbox",
    "harness_by_dataset": {},
    "max_turns_by_dataset": {},
    "task_machine_by_dataset": {},
    "session": {
        "max_turns": 8,
        "max_total_tokens": None,
        "model_turn_timeout": 60,
        "tool_turn_timeout": 90,
        "total_turn_timeout": 600,
        "attempt_timeout": 900,
        "verifier_timeout": 60,
        "cleanup_timeout": 60,
        "command_timeout": 30,
        "shell_tool": {},
    },
    "image_verifier": VERIFIER,
    "package_verifier": VERIFIER,
}


def tool_free_task(**overrides) -> TaskSpec:
    """The PDBThink shape: system and user text, a script grader reading the reply status."""
    grader = ScriptGrader(
        argv=("python", "/tests/adapter.py"),
        environment=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="python@sha256:" + "0" * 64
        ),
        answer_path=None,
        reply_status_path=STATUS_PATH,
        reward=FileReward(files=(RewardFile(path="/logs/verifier/reward.txt", format=RewardFileFormat.NUMBER),)),
    )
    fields = {
        "id": "tool-free-fixture",
        "source": Source(dataset="fixture/pdbthink", revision="1", row="0", importer_revision="1"),
        "context": ConversationInput(
            events=(TextMessage(role="system", content="Answer without tools."), TextMessage(role="user", content="Q?"))
        ),
        "environment_requirements": EnvironmentRequirements(),
        "completion_protocol": "tool_free_single",
        "answer_type": AnswerType.TEXT,
        "answer_format": PlainText(),
        "grader": grader,
    }
    return TaskSpec(**{**fields, **overrides})


def test_protocol_round_trips_and_lowers_to_one_tool_free_completion():
    task = tool_free_task()
    assert TaskSpec.model_validate_json(task.model_dump_json()) == task
    assert canonical_task_spec(task, CONFIG).harness == TOOL_FREE_HARNESS
    assert tool_free_prompt(task).options == {}
    with pytest.raises(ValidationError, match="cannot advertise or bind tools"):
        tool_free_task(interaction_tools=(FunctionDefinition(name="f", parameters={"type": "object"}),))
    with pytest.raises(ValueError, match="only tool_free_single"):
        ToolFreeHarness().check(tool_free_task(completion_protocol=None), None)


def staged_status(task: TaskSpec, reply, stop_reason) -> dict:
    attempt = GradingAttempt(ConversationTrace(events=(*task.context.events, reply)), stop_reason=stop_reason)
    (staged,) = reply_status_files(task, attempt)
    assert staged.path == STATUS_PATH
    return json.loads(staged.data)


def test_reply_status_reports_truncation_and_tool_events_from_the_host():
    task = tool_free_task()
    answer = TextMessage(role="assistant", content="A:V22")
    assert staged_status(task, answer, "stop") == {
        "origin": "provider",
        "finish_reason": "stop",
        "truncated": False,
        "provider_refusal": False,
        "tool_events": [],
    }
    assert staged_status(task, answer, "length")["truncated"] is True
    call = AssistantToolCalls(calls=(ConversationToolCall(call_id="1", name="search", arguments={}),))
    assert staged_status(task, call, "tool_calls")["tool_events"] == ["reply[0].tool_calls"]
    with pytest.raises(ValueError, match="no stop reason"):
        reply_status_files(task, GradingAttempt(ConversationTrace(events=(*task.context.events, answer))))


def test_dataset_placement_keeps_its_backend_under_runtime_shapes():
    """Per-task-image sources stay on their LOCAL template when the cohort supplies machine shapes."""
    task = tool_free_task()
    local = {**VERIFIER, "backend": "local"}
    config = {**CONFIG, "verifier_machine_by_dataset": {task.source.dataset: local}}
    assert canonical_task_spec(task, config).verifier_machine.backend == "local"
    row = {"task_spec": task.model_dump_json(), "runtime": {"verifier_machine": {"memory_mb": 4096}}}
    lowered = TaskRolloutSpec.model_validate_json(bind_runtime(row, config)["task_rollout_spec"])
    assert (lowered.verifier_machine.backend, lowered.verifier_machine.memory_mb) == ("local", 4096)


def test_required_backend_routes_the_verifier_machine_too():
    """A task that declares its backend routes its grader there as well (the routed x86-only rows)."""
    build = DockerBuildContext(files=(inline_resource("Dockerfile", b"FROM ubuntu:22.04\n"),))
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_build=build,
        required_backend=Backend.DAYTONA,
    )
    task = TaskSpec(
        id="lego-x-terminal-lego-15k-routed-fixture",
        source=Source(dataset="Lego-X/Terminal-Lego-15k", revision="1", row="0", importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content="configure the task"),)),
        environment_requirements=environment,
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=ScriptGrader(
            argv=("cp", "/source-reward.txt", "/logs/verifier/reward.txt"),
            cwd="/",
            environment=EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                docker_build=build,
            ),
            answer_path=None,
        ),
    )
    local = {**VERIFIER, "backend": "local"}
    config = {
        **CONFIG,
        "task_machine_by_dataset": {task.source.dataset: local},
        "verifier_machine_by_dataset": {task.source.dataset: local},
        "task_machine_by_backend": {"daytona": VERIFIER},
    }
    lowered = canonical_task_spec(task, config)
    assert (lowered.task_machine.backend, lowered.verifier_machine.backend) == ("daytona", "daytona")
    with pytest.raises(ValueError, match="required daytona backend"):
        canonical_task_spec(task, {**config, "task_machine_by_backend": {}})
