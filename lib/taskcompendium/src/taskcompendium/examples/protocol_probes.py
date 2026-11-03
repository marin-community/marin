# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One minimal task per agent protocol check.

Each answer is a nonce derived from the task id, so a model cannot guess it without following the
protocol.
"""

import hashlib

from verifyit.spec import DEFAULT_WORKSPACE, ExactSpec

from taskcompendium.fetch import FETCH_TOOL, HttpPage, fetch_requirement
from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    FunctionDefinition,
    PlainText,
    ProtocolCheck,
    RequiredFileCheck,
    ResourceGroups,
    ShellToolBinding,
    Source,
    TaskSpec,
    TextMessage,
    ThinkingDisabledCheck,
    ToolExecutionCheck,
    UrlLoadingCheck,
)
from taskcompendium.runtime.resources import inline_resource

PROBE_DATASET = "protocol-probes"
PROBE_REVISION = "1"
WORKDIR = DEFAULT_WORKSPACE
ANSWER_PATH = f"{WORKDIR}/answer.txt"
NONCE_FILE = "probe/nonce"
PROBE_URL = "https://protocol-probe.invalid/token"
FETCH_REQUIREMENT = "web"
PROBE_TURNS = 4
"""Model turns enough to call a tool, read its result and answer."""
TASK_ENVIRONMENT = EnvironmentRequirements(
    command_semantics=CommandSemantics.SHELL_SIMULATOR, capabilities=("shell", "filesystem"), working_directory=WORKDIR
)
"""The file and tool probes run in a shell simulator; the thinking and URL probes need no machine."""
SHELLSIM_MACHINE = {
    "backend": "shellsim",
    "network": "deny",
    "cpus": None,
    "memory_mb": None,
    "storage_mb": None,
    "gpus": 0,
    "user": None,
    "startup_timeout": 60,
    "cleanup_timeout": 60,
}
"""The probes' task machine, as a RolloutEngine ``MachineRuntimeSpec`` payload for runtime configuration."""
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


def lock_grader_environment(lock_url: str) -> EnvironmentRequirements:
    """Grade the answer file in a process with the verifyit lock at ``lock_url``."""
    return EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock=lock_url)


def probe_nonce(task_id: str) -> str:
    """The answer to the probe task ``task_id``; unguessable without following the protocol."""
    return hashlib.sha256(f"{PROBE_DATASET}/{task_id}".encode()).hexdigest()[:16]


def _probe_id(check: ProtocolCheck, index: int) -> str:
    return f"{check.kind}-{index}"


def _task(
    task_id: str,
    check: ProtocolCheck,
    instruction: str,
    environment: EnvironmentRequirements,
    *,
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
        protocol_check=check,
    )


def probe_tasks(*, grader_environment: EnvironmentRequirements, variants: int) -> tuple[TaskSpec, ...]:
    """``variants`` tasks per protocol check, each with its own nonce; ``grader_environment`` grades the answer file."""
    if variants < 1:
        raise ValueError("At least one variant per check is required")
    tasks = []
    for index in range(variants):
        tool_check = ToolExecutionCheck(tool=COMMAND_TOOL.name)
        tool_id = _probe_id(tool_check, index)
        url_check = UrlLoadingCheck(tool=FETCH_TOOL.name, url=PROBE_URL)
        url_id = _probe_id(url_check, index)
        tasks.extend(
            (
                _task(
                    _probe_id(RequiredFileCheck(), index),
                    RequiredFileCheck(),
                    f"Write {{nonce}} to {ANSWER_PATH}. Only the file is graded; an answer in chat does not count.",
                    TASK_ENVIRONMENT,
                    grader_environment=grader_environment,
                    answer_type=AnswerType.FILE,
                    output_paths=(ANSWER_PATH,),
                ),
                _task(
                    tool_id,
                    tool_check,
                    f"Run `cat /{NONCE_FILE}` with the {COMMAND_TOOL.name} tool and reply with only its "
                    "output. The value cannot be guessed.",
                    TASK_ENVIRONMENT,
                    interaction_tools=(COMMAND_TOOL,),
                    tool_bindings={COMMAND_TOOL.name: ShellToolBinding(command_parameter="command")},
                    resources=ResourceGroups(worker=(inline_resource(NONCE_FILE, probe_nonce(tool_id).encode()),)),
                ),
                _task(
                    _probe_id(ThinkingDisabledCheck(), index),
                    ThinkingDisabledCheck(),
                    "Reply with only this token: {nonce}",
                    EnvironmentRequirements(),
                ),
                _task(
                    url_id,
                    url_check,
                    f"Retrieve {PROBE_URL} with the {FETCH_TOOL.name} tool and reply with only the token it "
                    "contains. The token cannot be guessed.",
                    EnvironmentRequirements(
                        tool_providers={
                            FETCH_REQUIREMENT: fetch_requirement(
                                {PROBE_URL: HttpPage(status=200, body=probe_nonce(url_id))}
                            )
                        }
                    ),
                    interaction_tools=(FETCH_TOOL,),
                ),
            )
        )
    return tuple(tasks)
