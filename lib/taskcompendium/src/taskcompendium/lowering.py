# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check compatible task/protocol pairs and export native Harbor packages."""

import json
from pathlib import Path
from typing import Any

import msgspec
import tomlkit
from tasktrove_verify.spec import Mode

from taskcompendium.execution import (
    DockerEnvironment,
    HarborExecutionConfig,
    HarborLaunchConfig,
    HarborTaskBinding,
    NoEnvironment,
    ShellSimEnvironment,
    validate_launch,
    validate_requirements,
)
from taskcompendium.extraction import validate_extractor
from taskcompendium.grading import source_verifier
from taskcompendium.grading_paths import submission_relative
from taskcompendium.models import (
    EXECUTABLE_MODES,
    HARBOR_REVISION,
    AssistantFinal,
    Capability,
    CodeAnswerVerifier,
    ContainerRuntime,
    FileSubmission,
    FinalActionSubmission,
    FinalState,
    ImageOverlay,
    PlainText,
    PredictedActionVerifier,
    ProviderStateVerifier,
    Rendering,
    ResourceRole,
    TaskSpec,
    TaskSuccessPolicy,
    tasktrove_verifier,
    verifier_runtime,
)
from taskcompendium.rendering import render_instruction, result_tags
from taskcompendium.resources import materialize_resources
from taskcompendium.serialization import specification_hash, to_json

LOWERING_VERSION = "0.8"


def validate_lowering(
    specification: TaskSpec, protocol: Rendering, binding: HarborTaskBinding, step_index: int = 0
) -> None:
    """Reject incompatible interactions, environments, answer formats, and runtimes."""
    step = specification.steps[step_index]
    submission = protocol.submission
    if isinstance(step.verifier, PredictedActionVerifier):
        if not isinstance(submission, FinalActionSubmission):
            raise ValueError("Predicted-action verification requires final-action submission")
        if step.answer_requirements.kind != "text":
            raise ValueError("Final-action submission cannot replace intrinsic answer requirements")
    elif isinstance(submission, FinalActionSubmission):
        raise ValueError("Final-action submission requires predicted-action verification")
    elif isinstance(step.verifier, ProviderStateVerifier) and not isinstance(submission, AssistantFinal):
        raise ValueError("Provider-state verification requires a final assistant response")
    source = tasktrove_verifier(step.verifier)
    if source is not None:
        source_verifier(source)
    if source is not None and source.mode in EXECUTABLE_MODES and not isinstance(step.verifier, CodeAnswerVerifier):
        if not isinstance(submission, FinalState):
            raise ValueError("Executable workspace verification requires final-state submission")
    elif isinstance(submission, FinalState):
        raise ValueError("Final-state submission requires executable workspace verification")
    if source is not None and source.mode in {Mode.JUNIT, Mode.GOTEST}:
        raise ValueError("This spike does not yet support isolated JUnit or Go execution")
    if source is not None and source.mode == Mode.STDIO and source.parameters.get("special_judge"):
        raise ValueError("Special judges require a dedicated isolated execution adapter")
    semantic = specification.requirements
    actual = binding.environment
    if isinstance(step.verifier, ProviderStateVerifier) and step.verifier.interface not in semantic.action_interfaces:
        raise ValueError("Provider-state verifier must match a declared task action interface")
    validate_requirements(semantic, actual)
    if not binding.tools:
        if semantic.action_interfaces:
            raise ValueError("Required action interfaces need explicit tool bindings")
        if bool(semantic.capabilities) or not isinstance(protocol.submission, AssistantFinal | FinalActionSubmission):
            raise ValueError(
                "An empty tool list requires direct model submissions and no semantic environment requirement"
            )
        if any(ResourceRole.AGENT in r.roles for r in (*specification.resources, *step.resources)):
            raise ValueError("Filesystem inputs require explicit tool bindings")
    elif isinstance(actual, NoEnvironment):
        raise ValueError("Tool bindings require an environment")
    validate_workspace_submission(specification, protocol, step_index)
    if isinstance(protocol.submission, FinalState):
        if specification.steps[step_index].answer_requirements.kind != "final_state" or not protocol.submission.paths:
            raise ValueError("Final-state submission requires explicit state paths and task requirements")
        for path in protocol.submission.paths:
            submission_relative(
                path,
                actual.workdir if isinstance(actual, ShellSimEnvironment | DockerEnvironment) else "/app",
                actual.additional_directories if isinstance(actual, ShellSimEnvironment | DockerEnvironment) else (),
            )
    elif not isinstance(protocol.submission, FinalActionSubmission):
        if specification.steps[step_index].answer_requirements.kind == "final_state":
            raise ValueError("State-modification tasks require a final-state submission")
        validate_extractor(protocol.submission.extractor)
        if specification.steps[step_index].answer_requirements.kind != "text" and not isinstance(
            protocol.submission.extractor, PlainText
        ):
            raise ValueError("Intrinsic literal/format requirements cannot be replaced by answer wrappers")
        if isinstance(protocol.submission, FileSubmission):
            submission_relative(
                protocol.submission.path,
                actual.workdir if isinstance(actual, ShellSimEnvironment | DockerEnvironment) else "/app",
                actual.additional_directories if isinstance(actual, ShellSimEnvironment | DockerEnvironment) else (),
            )
    if (
        source is not None
        and source.mode in EXECUTABLE_MODES
        and not isinstance(verifier_runtime(step.verifier), ContainerRuntime)
    ):
        raise ValueError("Executable verifiers require an isolated container runtime")
    if source is not None and source.mode == Mode.REASONING_GYM:
        raise ValueError("Reasoning-gym is outside the supported verifier subset")
    if (
        not semantic.capabilities
        and isinstance(actual, ShellSimEnvironment | DockerEnvironment)
        and actual.workdir != "/app"
    ):
        raise ValueError("Rendering-provided filesystems currently require /app as workdir")


def validate_workspace_submission(specification: TaskSpec, protocol: Rendering, step_index: int = 0) -> None:
    """Validate exclusions before either snapshot export or container launch."""
    runtime = verifier_runtime(specification.steps[step_index].verifier)
    submission = protocol.submission
    if isinstance(submission, FinalState) and submission.excluded_paths:
        if specification.requirements.state.image is None or submission.paths != (".",):
            raise ValueError("Snapshot exclusions require Docker and a complete workspace submission")
    if not isinstance(runtime, ContainerRuntime) or not isinstance(runtime.workspace, ImageOverlay):
        return
    if not isinstance(submission, FinalState) or submission.paths != (".",):
        raise ValueError("Image overlay requires a complete workspace final-state submission")
    if specification.requirements.state.image is None:
        raise ValueError("Image overlay requires a Docker environment")
    if not set(runtime.workspace.preserved_directories).issubset(submission.excluded_paths):
        raise ValueError("Snapshot exclusions must cover every preserved image dependency directory")
    for resource in (*specification.resources, *specification.steps[step_index].resources):
        if resource.path.split("/")[0] in runtime.workspace.preserved_directories:
            raise ValueError("TaskSpec resources cannot replace preserved image dependency directories")


def resolve_harbor_execution(
    renderings: tuple[Rendering, ...],
    execution: HarborExecutionConfig,
    environment: dict[str, Any],
    *,
    agent_kwargs: dict[str, Any] | None = None,
    agent_env: dict[str, str] | None = None,
    verifier_kwargs: dict[str, Any] | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Resolve one explicit Harbor launch against a task-owned binding.

    This is intentionally separate from package export: model, agent, and
    timeouts are launch choices made by Harbor rather than semantic task data.
    """
    binding = execution.binding
    launch: HarborLaunchConfig = execution.launch
    validate_launch(binding, launch)
    if len(renderings) > 1 and launch.agent in {"terminus-2", "mini-swe-agent"}:
        raise ValueError("Native terminal agents do not yet retain conversation across ordered steps")
    agent_name = launch.agent
    agents = {
        "replay": "taskcompendium.harbor.agents:ReplayAgent",
        "chat": "taskcompendium.harbor.agents:DirectChatAgent",
        "tool_chat": "taskcompendium.harbor.agents:ShellToolAgent",
        "provider_chat": "taskcompendium.harbor.agents:ProviderToolAgent",
        "mini-swe-agent": "taskcompendium.harbor.installed_agents:PreinstalledMiniSweAgent",
    }
    agent: dict[str, Any] = {"upload_agent_logs": False, "kwargs": dict(agent_kwargs or {}), "env": agent_env or {}}
    if agent_name in {"chat", "replay"} and any(
        isinstance(rendering.submission, FinalActionSubmission) for rendering in renderings
    ):
        agent["import_path"] = (
            "taskcompendium.harbor.agents:ActionOutputAgent"
            if agent_name == "chat"
            else "taskcompendium.harbor.agents:ActionOutputReplayAgent"
        )
        agent["kwargs"]["output_contracts"] = [
            msgspec.to_builtins(rendering.submission)
            for rendering in renderings
            if isinstance(rendering.submission, FinalActionSubmission)
        ]
    elif agent_name in agents:
        agent["import_path"] = agents[agent_name]
    else:
        agent["name"] = agent_name
    if agent_name in {"tool_chat", "replay"} and binding.tools:
        agent["kwargs"]["tool_binding"] = msgspec.to_builtins(binding.tools[0])
    if model_name is not None:
        agent["model_name"] = model_name
    return {
        "environment": environment,
        "agent": agent,
        "verifier": {"import_path": "taskcompendium.harbor.verifier:SemanticVerifier", "kwargs": verifier_kwargs or {}},
    }


def lower_to_harbor(
    specification: TaskSpec,
    renderings: tuple[Rendering, ...],
    binding: HarborTaskBinding,
    destination: Path,
    *,
    reference_execution: HarborExecutionConfig | None = None,
    agent_kwargs: dict[str, Any] | None = None,
    agent_env: dict[str, str] | None = None,
    environment_kwargs: dict[str, Any] | None = None,
    verifier_kwargs: dict[str, Any] | None = None,
    model_name: str | None = None,
) -> Path:
    """Write one task-owned Harbor package without selecting a harness."""
    if agent_kwargs is not None and "tool_binding" in agent_kwargs:
        raise ValueError("Declare tools through the Harbor task binding, not agent kwargs")
    if reference_execution is None and any(
        value is not None for value in (agent_kwargs, agent_env, environment_kwargs, verifier_kwargs, model_name)
    ):
        raise ValueError("Reference execution options require reference_execution")
    if reference_execution is not None and reference_execution.binding != binding:
        raise ValueError("Reference execution must resolve this task binding")
    if len(renderings) != len(specification.steps):
        raise ValueError("Exactly one rendering per semantic step is required")
    final_actions = tuple(isinstance(rendering.submission, FinalActionSubmission) for rendering in renderings)
    if any(final_actions) and not all(final_actions):
        raise ValueError("Final-action Harbor runs require a final-action submission for every step")
    for index, rendering in enumerate(renderings):
        validate_lowering(specification, rendering, binding, index)
    capabilities = set(specification.requirements.capabilities)
    if any(isinstance(rendering.submission, (FileSubmission, FinalState)) for rendering in renderings):
        capabilities.add(Capability.FILESYSTEM)
    if any(ResourceRole.AGENT in resource.roles for resource in specification.resources) or any(
        ResourceRole.AGENT in resource.roles for step in specification.steps for resource in step.resources
    ):
        capabilities.add(Capability.FILESYSTEM)
    requirements = msgspec.structs.replace(specification.requirements, capabilities=tuple(sorted(capabilities)))
    validate_requirements(requirements, binding.environment)
    multiple = len(specification.steps) > 1
    if multiple and any(
        r.path == "setup.sh" and ResourceRole.AGENT in r.roles for step in specification.steps for r in step.resources
    ):
        raise ValueError("Harbor reserves step workdir/setup.sh for executable setup")
    if multiple and specification.success_policy == TaskSuccessPolicy.ALL_REQUIRED_STEPS:
        raise ValueError(
            "Pinned Harbor cannot express all-required-steps aggregation; choose an explicit supported policy"
        )
    exclusions = {r.submission.excluded_paths for r in renderings if isinstance(r.submission, FinalState)}
    if len(exclusions) > 1:
        raise ValueError("Pinned Docker adapter requires consistent snapshot exclusions across steps")
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    environment_dir = destination / "environment"
    environment_dir.mkdir()
    agent_resources = tuple(resource for resource in specification.resources if ResourceRole.AGENT in resource.roles)
    if agent_resources:
        materialize_resources(agent_resources, environment_dir / "inputs")
    (destination / "tests").mkdir()
    (destination / "tests/test.sh").write_text("#!/bin/sh\necho 'Use the TaskCompendium verifier adapter' >&2\nexit 1\n")
    specification_json = to_json(specification)
    (destination / "specification.json").write_bytes(specification_json)
    (destination / "renderings.json").write_bytes(msgspec.json.encode(renderings))
    (destination / "binding.json").write_bytes(msgspec.json.encode(binding))
    step_names = [f"step-{index + 1}" for index in range(len(renderings))]
    for index, step in enumerate(specification.steps):
        step_dir = destination / "steps" / step_names[index] if multiple else destination
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "instruction.md").write_text(render_instruction(specification, renderings[index], index))
        public_dir = step_dir / "workdir" if multiple else environment_dir / "inputs"
        agent_step_resources = tuple(resource for resource in step.resources if ResourceRole.AGENT in resource.roles)
        if agent_step_resources:
            materialize_resources(agent_step_resources, public_dir)
    task_config: dict[str, Any] = {
        "version": "1.0",
        "environment": {"allow_internet": False},
    }
    if multiple:
        task_config["steps"] = [{"name": name} for name in step_names]
        task_config["multi_step_reward_strategy"] = specification.success_policy.value
    environment = binding.environment
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
    elif isinstance(environment, DockerEnvironment):
        task_config["environment"].update(docker_image=environment.image, workdir=environment.workdir)
        environment_config = {"import_path": "taskcompendium.harbor.environments:TaskDockerEnvironment"}
    else:
        environment_config = {"import_path": environment.adapter, "kwargs": dict(environment.configuration)}
    environment_config.setdefault("kwargs", {}).update(environment_kwargs or {})
    (destination / "task.toml").write_text(tomlkit.dumps(task_config))
    if reference_execution is not None:
        config = resolve_harbor_execution(
            renderings,
            reference_execution,
            environment_config,
            agent_kwargs=agent_kwargs,
            agent_env=agent_env,
            verifier_kwargs=verifier_kwargs,
            model_name=model_name,
        )
        (destination / "reference-execution.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {
        "specification_sha256": specification_hash(specification),
        "renderings": msgspec.to_builtins(renderings),
        "step_names": step_names,
        "success_policy": specification.success_policy.value,
        "lowering_version": LOWERING_VERSION,
        "harbor_revision": HARBOR_REVISION,
        "verifier_runtimes": [msgspec.to_builtins(verifier_runtime(step.verifier)) for step in specification.steps],
        "binding": msgspec.to_builtins(binding),
        "coverage_tags": tuple(sorted((*specification.coverage_tags, *result_tags(renderings)))),
        "source": msgspec.to_builtins(specification.metadata.source),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination
