# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check compatible task/protocol pairs and export native Harbor packages."""

import json
from pathlib import Path
from typing import Any

import msgspec
import tomlkit
from tasktrove_verify.spec import Mode

from taskcompendium.extraction import rendering_instruction, validate_extractor
from taskcompendium.grading import EXECUTABLE_MODES, source_verifier
from taskcompendium.grading_paths import submission_relative
from taskcompendium.models import (
    HARBOR_REVISION,
    AssistantFinal,
    Chat,
    ChatWithTools,
    ContainerRuntime,
    DockerEnvironment,
    ExecutionConfig,
    FileSubmission,
    FinalState,
    ImageOverlay,
    NoEnvironment,
    PlainText,
    Protocol,
    PythonRuntime,
    ResourceRole,
    ShellSimEnvironment,
    TaskSpecification,
)
from taskcompendium.resources import materialize
from taskcompendium.serialization import specification_hash, to_json

LOWERING_VERSION = "0.1"


def validate_lowering(specification: TaskSpecification, protocol: Protocol, execution: ExecutionConfig) -> None:
    """Reject incompatible interactions, environments, answer formats, and runtimes."""
    source_verifier(specification.verifier)
    if specification.verifier.mode in {Mode.JUNIT, Mode.GOTEST}:
        raise ValueError("This spike does not yet support isolated JUnit or Go execution")
    if specification.verifier.mode == Mode.STDIO and specification.verifier.parameters.get("special_judge"):
        raise ValueError("Special judges require a dedicated isolated execution adapter")
    semantic = specification.environment
    actual = execution.environment
    if not isinstance(semantic, NoEnvironment) and semantic != actual:
        raise ValueError("The selected environment must preserve the task's declared world")
    if isinstance(protocol.interaction, Chat):
        if not isinstance(actual, NoEnvironment) or not isinstance(protocol.submission, AssistantFinal):
            raise ValueError("Chat supports only assistant submissions without an agent filesystem")
        if execution.agent not in {"chat", "replay"}:
            raise ValueError("Plain chat requires a chat-compatible agent")
        if any(ResourceRole.AGENT in r.roles for r in specification.resources):
            raise ValueError("Filesystem inputs require ChatWithTools")
    elif isinstance(protocol.interaction, ChatWithTools):
        if not protocol.interaction.tools or isinstance(actual, NoEnvironment):
            raise ValueError("ChatWithTools requires tools backed by an environment")
        if execution.agent == "chat":
            raise ValueError("Plain chat agent cannot use tools")
    if execution.agent in {"terminus-2", "mini-swe-agent"} and not isinstance(actual, DockerEnvironment):
        raise ValueError("Installed terminal agents require a real Docker environment")
    validate_workspace_submission(specification, protocol)
    if isinstance(protocol.submission, FinalState):
        if specification.answer_requirements.kind != "final_state" or not protocol.submission.paths:
            raise ValueError("Final-state submission requires explicit state paths and task requirements")
        for path in protocol.submission.paths:
            submission_relative(
                path,
                actual.workdir if not isinstance(actual, NoEnvironment) else "/app",
                actual.additional_directories if not isinstance(actual, NoEnvironment) else (),
            )
    else:
        if specification.answer_requirements.kind == "final_state":
            raise ValueError("State-modification tasks require a final-state submission")
        validate_extractor(protocol.submission.extractor)
        if specification.answer_requirements.kind != "value" and not isinstance(
            protocol.submission.extractor, PlainText
        ):
            raise ValueError("Intrinsic literal/format requirements cannot be replaced by answer wrappers")
        if isinstance(protocol.submission, FileSubmission):
            submission_relative(
                protocol.submission.path,
                actual.workdir if not isinstance(actual, NoEnvironment) else "/app",
                actual.additional_directories if not isinstance(actual, NoEnvironment) else (),
            )
    if specification.verifier.mode in EXECUTABLE_MODES and not isinstance(
        specification.verifier_runtime, ContainerRuntime
    ):
        raise ValueError("Executable verifiers require an isolated container runtime")
    if isinstance(specification.verifier_runtime, PythonRuntime) and specification.verifier.mode == Mode.REASONING_GYM:
        raise ValueError("Reasoning-gym is outside the supported verifier subset")
    if isinstance(semantic, NoEnvironment) and not isinstance(actual, NoEnvironment) and actual.workdir != "/app":
        raise ValueError("Protocol-provided filesystems currently require /app as workdir")
    if execution.timeout <= 0:
        raise ValueError("Trial timeout must be positive")


def validate_workspace_submission(specification: TaskSpecification, protocol: Protocol) -> None:
    """Validate exclusions before either snapshot export or container launch."""
    runtime = specification.verifier_runtime
    submission = protocol.submission
    if isinstance(submission, FinalState) and submission.excluded_paths:
        if not isinstance(specification.environment, DockerEnvironment) or submission.paths != (".",):
            raise ValueError("Snapshot exclusions require Docker and a complete workspace submission")
    if not isinstance(runtime, ContainerRuntime) or not isinstance(runtime.workspace, ImageOverlay):
        return
    if not isinstance(submission, FinalState) or submission.paths != (".",):
        raise ValueError("Image overlay requires a complete workspace final-state submission")
    if not isinstance(specification.environment, DockerEnvironment):
        raise ValueError("Image overlay requires a Docker environment")
    if not set(runtime.workspace.preserved_directories).issubset(submission.excluded_paths):
        raise ValueError("Snapshot exclusions must cover every preserved image dependency directory")
    for resource in specification.resources:
        if resource.path.split("/")[0] in runtime.workspace.preserved_directories:
            raise ValueError("Task resources cannot replace preserved image dependency directories")


def render_instruction(specification: TaskSpecification, protocol: Protocol) -> str:
    """Render task and output requirements without exposing evaluation machinery."""
    submission = protocol.submission
    if isinstance(submission, FinalState):
        return specification.instructions
    suffix = rendering_instruction(submission.extractor)
    if isinstance(submission, FileSubmission):
        suffix += f" Write your submission to {submission.path}."
    return f"{specification.instructions.rstrip()}\n\n{suffix}\n"


def export_task(
    specification: TaskSpecification,
    protocol: Protocol,
    execution: ExecutionConfig,
    destination: Path,
    *,
    agent_kwargs: dict[str, Any] | None = None,
    agent_env: dict[str, str] | None = None,
    environment_kwargs: dict[str, Any] | None = None,
    verifier_kwargs: dict[str, Any] | None = None,
    model_name: str | None = None,
) -> Path:
    """Write one task and its native Harbor execution template into a new directory."""
    validate_lowering(specification, protocol, execution)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    environment_dir = destination / "environment"
    environment_dir.mkdir()
    if any(ResourceRole.AGENT in resource.roles for resource in specification.resources):
        materialize(specification, ResourceRole.AGENT, environment_dir / "inputs")
    (destination / "tests").mkdir()
    (destination / "tests/test.sh").write_text("#!/bin/sh\necho 'Use the TaskCompendium verifier adapter' >&2\nexit 1\n")
    specification_json = to_json(specification)
    (destination / "specification.json").write_bytes(specification_json)
    (destination / "protocol.json").write_bytes(msgspec.json.encode(protocol))
    (destination / "instruction.md").write_text(render_instruction(specification, protocol))
    task_config: dict[str, Any] = {
        "version": "1.0",
        "agent": {"timeout_sec": execution.timeout},
        "verifier": {"timeout_sec": execution.timeout},
        "environment": {"allow_internet": False},
    }
    environment = execution.environment
    environment_config: dict[str, Any]
    if isinstance(environment, NoEnvironment):
        environment_config = {"import_path": "taskcompendium.harbor.environments:NoToolEnvironment"}
    elif isinstance(environment, ShellSimEnvironment):
        task_config["environment"]["workdir"] = environment.workdir
        environment_config = {
            "import_path": "taskcompendium.harbor.environments:ShellSimEnvironment",
            "kwargs": {
                "limits": {"cpu": environment.max_steps, "output": environment.max_output_bytes},
            },
        }
    else:
        task_config["environment"].update(docker_image=environment.image, workdir=environment.workdir)
        environment_config = {"import_path": "taskcompendium.harbor.environments:TaskDockerEnvironment"}
    environment_config.setdefault("kwargs", {}).update(environment_kwargs or {})
    agents = {
        "replay": "taskcompendium.harbor.agents:ReplayAgent",
        "chat": "taskcompendium.harbor.agents:DirectChatAgent",
        "tool_chat": "taskcompendium.harbor.agents:ShellToolAgent",
        "mini-swe-agent": "taskcompendium.harbor.installed_agents:PreinstalledMiniSweAgent",
    }
    agent: dict[str, Any] = {"upload_agent_logs": False, "kwargs": agent_kwargs or {}, "env": agent_env or {}}
    if execution.agent in agents:
        agent["import_path"] = agents[execution.agent]
    else:
        agent["name"] = execution.agent
    if model_name is not None:
        agent["model_name"] = model_name
    config = {
        "environment": environment_config,
        "agent": agent,
        "verifier": {"import_path": "taskcompendium.harbor.verifier:SemanticVerifier", "kwargs": verifier_kwargs or {}},
    }
    (destination / "task.toml").write_text(tomlkit.dumps(task_config))
    (destination / "execution.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {
        "specification_sha256": specification_hash(specification),
        "protocol": msgspec.to_builtins(protocol),
        "lowering_version": LOWERING_VERSION,
        "harbor_revision": HARBOR_REVISION,
        "verifier_runtime": msgspec.to_builtins(specification.verifier_runtime),
        "execution": msgspec.to_builtins(execution),
        "source": msgspec.to_builtins(specification.metadata.source),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination
