# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""tool_free_single tasks lower to one tool-free completion and grade with host-captured reply status."""

import json

import pytest
from pydantic import ValidationError
from rolloutengine.harness.tool_free import TOOL_FREE_HARNESS, ToolFreeHarness, tool_free_prompt
from skyrl_train.dataset.tasks import canonical_task_spec
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    CommandSemantics,
    ConversationInput,
    ConversationToolCall,
    ConversationTrace,
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
