# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal evaluation tasks for agent protocols, with a chosen number of variants per protocol.

Each answer is a nonce derived from the task id. A model gets it only by following the protocol
(writing the answer file, running the shell tool, or fetching the declared page), so the reward is
the protocol result.
"""

import hashlib
from enum import StrEnum

from taskcompendium.fetch import FETCH_TOOL, HttpPage, fetch_requirement
from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    FunctionDefinition,
    PlainText,
    ResourceGroups,
    ShellToolBinding,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import DEFAULT_WORKSPACE, ExactSpec


class ProtocolCheck(StrEnum):
    REQUIRED_FILE = "required_file"
    TOOL_EXECUTION = "tool_execution"
    URL_LOADING = "url_loading"


PROBE_DATASET = "protocol-probes"
PROBE_REVISION = "1"
ANSWER_PATH = f"{DEFAULT_WORKSPACE}/answer.txt"
NONCE_FILE = "probe/nonce"
PROBE_URL = "https://protocol-probe.invalid/token"
FETCH_REQUIREMENT = "web"
TASK_ENVIRONMENT = EnvironmentRequirements(
    command_semantics=CommandSemantics.SHELL_SIMULATOR,
    capabilities=("shell", "filesystem"),
    working_directory=DEFAULT_WORKSPACE,
)
"""The file and tool probes run in a shell simulator; the URL probe needs no machine."""
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


def probe_nonce(task_id: str) -> str:
    """The answer to the probe task ``task_id``; unguessable without following the protocol."""
    return hashlib.sha256(f"{PROBE_DATASET}/{task_id}".encode()).hexdigest()[:16]


def _task(
    task_id: str,
    instruction: str,
    environment: EnvironmentRequirements,
    *,
    grader_environment: EnvironmentRequirements | None = None,
    answer_type: AnswerType = AnswerType.TEXT,
    **fields,
) -> TaskSpec:
    """A probe task whose answer is ``task_id``'s nonce; ``fields`` sets its tools, outputs and resources."""
    expected = probe_nonce(task_id)
    spec = ExactSpec((expected,), output=ANSWER_PATH) if answer_type == AnswerType.FILE else ExactSpec((expected,))
    package = verifyit_package(spec, environment=grader_environment)
    return TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction.format(nonce=expected)),)),
        environment_requirements=environment,
        answer_type=answer_type,
        answer_format=PlainText(),
        grader=package.grader,
        source=Source(dataset=PROBE_DATASET, revision=PROBE_REVISION, row=task_id, importer_revision=PROBE_REVISION),
        **fields,
    )


def probe_tasks(*, grader_environment: EnvironmentRequirements, variants: int) -> tuple[TaskSpec, ...]:
    """``variants`` tasks per protocol check, each with its own nonce; ``grader_environment`` grades the answer file."""
    tasks = []
    for index in range(variants):
        tool_id = f"{ProtocolCheck.TOOL_EXECUTION}-{index}"
        url_id = f"{ProtocolCheck.URL_LOADING}-{index}"
        tasks.extend(
            (
                _task(
                    f"{ProtocolCheck.REQUIRED_FILE}-{index}",
                    f"Write {{nonce}} to {ANSWER_PATH}. Only the file is graded; an answer in chat does not count.",
                    TASK_ENVIRONMENT,
                    grader_environment=grader_environment,
                    answer_type=AnswerType.FILE,
                    output_paths=(ANSWER_PATH,),
                ),
                _task(
                    tool_id,
                    f"Run `cat /{NONCE_FILE}` with the {COMMAND_TOOL.name} tool and reply with only its "
                    "output. The value cannot be guessed.",
                    TASK_ENVIRONMENT,
                    interaction_tools=(COMMAND_TOOL,),
                    tool_bindings={COMMAND_TOOL.name: ShellToolBinding(command_parameter="command")},
                    resources=ResourceGroups(worker=(inline_resource(NONCE_FILE, probe_nonce(tool_id).encode()),)),
                ),
                _task(
                    url_id,
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
