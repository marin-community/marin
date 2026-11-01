# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal tasks that test whether an agent follows four protocols, separately from answering well.

Each task declares one ``protocol_check``; grading reports protocol compliance, submission
completeness and answer correctness in ``GradeResult.diagnostics["protocol"]``. The answers are
nonces derived from the task id, so a model cannot guess them without following the protocol.

| Check | What the model must do |
| --- | --- |
| ``required_file`` | Write the answer to a file; a chat answer does not count. |
| ``tool_execution`` | Run a command with the shell tool and report its output. |
| ``thinking_disabled`` | Answer with reasoning turned off, emitting no reasoning. |
| ``url_loading`` | Retrieve a URL with the ``fetch_url`` tool and report its contents. |
"""

import hashlib
import json

from pydantic import JsonValue
from shellbox.machine import ExitReason
from verifyit.spec import DEFAULT_WORKSPACE, ExactSpec

from taskcompendium.fetch import FETCH_TOOL, HttpPage, fetch_requirement
from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    ENABLE_THINKING,
    AnswerType,
    AssistantToolCalls,
    ConversationEvent,
    ConversationInput,
    ConversationToolCall,
    EnvironmentRequirements,
    FunctionDefinition,
    PlainText,
    ProtocolCheck,
    ReasoningApplied,
    ReasoningMode,
    RequiredFileCheck,
    ResourceGroups,
    ShellToolBinding,
    Source,
    TaskSpec,
    TextMessage,
    ThinkingDisabledCheck,
    ToolExecutionCheck,
    ToolResult,
    TurnEvidence,
    UrlLoadingCheck,
)
from taskcompendium.pipeline.models import Transcript
from taskcompendium.runtime.resources import inline_resource

PROBE_DATASET = "protocol-probes"
PROBE_REVISION = "1"
PROBE_TAG = "protocol-probe"
WORKDIR = DEFAULT_WORKSPACE
ANSWER_PATH = f"{WORKDIR}/answer.txt"
NONCE_FILE = "probe/nonce"
PROBE_URL = "https://protocol-probe.invalid/token"
FETCH_REQUIREMENT = "web"
COMMAND_TOOL = FunctionDefinition(
    name="run_command",
    description="Run a Bash command in the workspace and return its exit code and output.",
    parameters={
        "type": "object",
        "properties": {"command": {"type": "string", "description": "The Bash command to run."}},
        "required": ["command"],
        "additionalProperties": False,
    },
)
REQUIRED_FILE_TURNS = 2
TOOL_TURNS = 4
URL_TURNS = 4
THINKING_TURNS = 1


def probe_nonce(task_id: str) -> str:
    """The answer to the probe task ``task_id``; unguessable without following the protocol."""
    return hashlib.sha256(f"{PROBE_DATASET}/{task_id}".encode()).hexdigest()[:16]


def _task(
    task_id: str,
    check: ProtocolCheck,
    instruction: str,
    environment: EnvironmentRequirements,
    turn_budget: int,
    *,
    reasoning: ReasoningMode = ReasoningMode.MODEL_DEFAULT,
    grader_environment: EnvironmentRequirements | None = None,
    answer_type: AnswerType = AnswerType.TEXT,
    output_paths: tuple[str, ...] = (),
    interaction_tools: tuple[FunctionDefinition, ...] = (),
    tool_bindings: dict[str, ShellToolBinding] | None = None,
    resources: ResourceGroups | None = None,
) -> TaskSpec:
    expected = probe_nonce(task_id)
    spec = ExactSpec((expected,), output=ANSWER_PATH) if answer_type == AnswerType.FILE else ExactSpec((expected,))
    package = verifyit_package(spec, environment=grader_environment)
    return TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction.format(nonce=expected)),)),
        environment_requirements=environment,
        interaction_tools=interaction_tools,
        tool_bindings=tool_bindings or {},
        output_paths=output_paths,
        answer_type=answer_type,
        answer_format=PlainText(),
        grader=package.grader,
        source=Source(dataset=PROBE_DATASET, revision=PROBE_REVISION, row=task_id, importer_revision=PROBE_REVISION),
        resources=resources or ResourceGroups(),
        tags=(PROBE_TAG, f"{PROBE_TAG}:{check.kind}"),
        turn_budget=turn_budget,
        reasoning=reasoning,
        protocol_check=check,
    )


def probe_tasks(
    *, task_environment: EnvironmentRequirements, grader_environment: EnvironmentRequirements, variants: int
) -> tuple[TaskSpec, ...]:
    """``variants`` tasks for each protocol check, each with its own nonce.

    ``task_environment`` is the shell machine for the file and tool probes; ``grader_environment``
    grades the answer file. The thinking and URL probes need no machine.
    """
    if variants < 1:
        raise ValueError("At least one variant per check is required")
    shell = task_environment.model_copy(update={"capabilities": ("shell", "filesystem"), "working_directory": WORKDIR})
    tasks = []
    for index in range(variants):
        file_id, tool_id, thinking_id, url_id = (
            f"{check}-{index}" for check in ("required_file", "tool_execution", "thinking_disabled", "url_loading")
        )
        tasks.extend(
            (
                _task(
                    file_id,
                    RequiredFileCheck(),
                    f"Write {{nonce}} to {ANSWER_PATH}. Only the file is graded; an answer in chat does not count.",
                    shell,
                    REQUIRED_FILE_TURNS,
                    grader_environment=grader_environment,
                    answer_type=AnswerType.FILE,
                    output_paths=(ANSWER_PATH,),
                ),
                _task(
                    tool_id,
                    ToolExecutionCheck(tool=COMMAND_TOOL.name),
                    f"Run `cat /{NONCE_FILE}` with the {COMMAND_TOOL.name} tool and reply with only its "
                    "output. The value cannot be guessed.",
                    shell,
                    TOOL_TURNS,
                    interaction_tools=(COMMAND_TOOL,),
                    tool_bindings={COMMAND_TOOL.name: ShellToolBinding(command_parameter="command")},
                    resources=ResourceGroups(worker=(inline_resource(NONCE_FILE, probe_nonce(tool_id).encode()),)),
                ),
                _task(
                    thinking_id,
                    ThinkingDisabledCheck(),
                    "Reply with only this token: {nonce}",
                    EnvironmentRequirements(),
                    THINKING_TURNS,
                    reasoning=ReasoningMode.OFF,
                ),
                _task(
                    url_id,
                    UrlLoadingCheck(tool=FETCH_TOOL.name, url=PROBE_URL),
                    f"Retrieve {PROBE_URL} with the {FETCH_TOOL.name} tool and reply with only the token it "
                    "contains. The token cannot be guessed.",
                    EnvironmentRequirements(
                        tool_providers={
                            FETCH_REQUIREMENT: fetch_requirement(
                                {PROBE_URL: HttpPage(status=200, body=probe_nonce(url_id))}
                            )
                        }
                    ),
                    URL_TURNS,
                    interaction_tools=(FETCH_TOOL,),
                ),
            )
        )
    return tuple(tasks)


def _tool_turn(
    name: str, arguments: dict[str, JsonValue], result: dict[str, JsonValue]
) -> tuple[ConversationEvent, ...]:
    call = ConversationToolCall(call_id="call-0", name=name, arguments=arguments)
    return (AssistantToolCalls(calls=(call,)), ToolResult(call_id=call.call_id, content=json.dumps(result)))


def _executed(stdout: str) -> dict[str, JsonValue]:
    return {"exit_code": 0, "reason": ExitReason.EXITED.value, "stdout": stdout, "stderr": ""}


def golden_transcript(task: TaskSpec) -> Transcript | None:
    """A compliant, correct attempt at a probe task, for the task's golden control."""
    nonce = probe_nonce(task.id)
    answer = TextMessage(role="assistant", content=nonce)
    check = task.protocol_check
    if isinstance(check, RequiredFileCheck):
        write = f"printf %s {nonce} > {ANSWER_PATH}"
        events = (*_tool_turn("shell", {"command": write}, _executed("")), answer)
        return Transcript(events, {ANSWER_PATH: nonce.encode()})
    if isinstance(check, ToolExecutionCheck):
        command: dict[str, JsonValue] = {"command": f"cat /{NONCE_FILE}"}
        return Transcript((*_tool_turn(check.tool, command, _executed(nonce)), answer))
    if isinstance(check, UrlLoadingCheck):
        page: dict[str, JsonValue] = {"url": check.url, "status": 200, "body": nonce}
        return Transcript((*_tool_turn(check.tool, {"url": check.url}, page), answer))
    if isinstance(check, ThinkingDisabledCheck):
        settings: dict[str, JsonValue] = {"chat_template_kwargs": {ENABLE_THINKING: False}}
        turn = TurnEvidence(nonce, settings, ReasoningApplied.OFF, ("<think>", "</think>"))
        return Transcript((answer,), turns=(turn,))
    return None
