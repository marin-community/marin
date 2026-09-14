# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execution choices for the Harbor lowering, separate from semantic task data."""

import re
from typing import Any, Literal

import msgspec

from taskcompendium.models import ActionInterface, Capability, TaskRequirements, image_digest, validate_directories


class NoEnvironment(msgspec.Struct, frozen=True, tag_field="kind", tag="none", forbid_unknown_fields=True):
    pass


class ShellSimEnvironment(msgspec.Struct, frozen=True, tag_field="kind", tag="shellsim", forbid_unknown_fields=True):
    workdir: str = "/app"
    max_steps: int = 100_000
    max_output_bytes: int = 1_048_576
    setup_commands: tuple[str, ...] = ()
    additional_directories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_directories(self.workdir, self.additional_directories)
        if self.max_steps <= 0 or self.max_output_bytes <= 0:
            raise ValueError("ShellSim budgets must be positive")


class DockerEnvironment(msgspec.Struct, frozen=True, tag_field="kind", tag="docker", forbid_unknown_fields=True):
    image: str
    workdir: str = "/app"
    setup_commands: tuple[str, ...] = ()
    additional_directories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        image_digest(self.image)
        validate_directories(self.workdir, self.additional_directories)


class ProviderEnvironment(msgspec.Struct, frozen=True, tag_field="kind", tag="provider", forbid_unknown_fields=True):
    """A provider adapter with private configuration, including any mutable-state seed."""

    interface: ActionInterface
    adapter: str
    configuration: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.adapter:
            raise ValueError("Provider environments require an adapter")


EnvironmentConfig = NoEnvironment | ShellSimEnvironment | DockerEnvironment | ProviderEnvironment


class Chat(msgspec.Struct, frozen=True, tag_field="kind", tag="chat", forbid_unknown_fields=True):
    pass


class ShellToolBinding(msgspec.Struct, frozen=True, tag_field="kind", tag="shell", forbid_unknown_fields=True):
    name: str
    backend: Literal["shellsim", "docker"]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.name):
            raise ValueError("Shell tool name must be an identifier")


class HarnessToolBinding(msgspec.Struct, frozen=True, tag_field="kind", tag="harness", forbid_unknown_fields=True):
    """A terminal-oriented interface supplied by a compatible Harbor agent."""

    interface: Literal["terminal"]
    backend: Literal["shellsim", "docker"]


class ProviderToolBinding(msgspec.Struct, frozen=True, tag_field="kind", tag="provider", forbid_unknown_fields=True):
    """Bind one declared semantic action interface to its provider adapter."""

    interface: str

    def __post_init__(self) -> None:
        if not self.interface:
            raise ValueError("Provider bindings require an action-interface name")


class ChatWithTools(msgspec.Struct, frozen=True, tag_field="kind", tag="chat_with_tools", forbid_unknown_fields=True):
    tools: tuple[ShellToolBinding | HarnessToolBinding | ProviderToolBinding, ...]

    def __post_init__(self) -> None:
        if not self.tools:
            raise ValueError("ChatWithTools requires explicit tool bindings")


class HarborTaskBinding(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Task-owned Harbor requirements, independent of the selected harness."""

    environment: EnvironmentConfig
    interaction: Chat | ChatWithTools
    context: Literal["fresh", "conversation"] = "fresh"

    def __post_init__(self) -> None:
        if isinstance(self.interaction, Chat):
            return
        if isinstance(self.environment, NoEnvironment):
            raise ValueError("Tool bindings require an execution environment")
        if len(self.interaction.tools) != 1:
            raise ValueError("The supported Harbor adapters require exactly one tool binding")
        binding = self.interaction.tools[0]
        if isinstance(binding, ProviderToolBinding):
            if not isinstance(self.environment, ProviderEnvironment):
                raise ValueError("Provider tool bindings require a provider environment")
            if binding.interface != self.environment.interface.name:
                raise ValueError("Provider binding must match its provider interface")
            return
        if isinstance(self.environment, ProviderEnvironment):
            raise ValueError("Provider environments require provider tool bindings")
        backend = "shellsim" if isinstance(self.environment, ShellSimEnvironment) else "docker"
        if binding.backend != backend:
            raise ValueError("Tool backend does not match the execution environment")
        if isinstance(binding, ShellToolBinding):
            return
        elif isinstance(binding, HarnessToolBinding):
            return


class HarborLaunchConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Task-independent compatibility choice made when starting a rollout."""

    agent: Literal["replay", "chat", "tool_chat", "provider_chat", "terminus-2", "mini-swe-agent"]


def validate_launch(binding: HarborTaskBinding, launch: HarborLaunchConfig) -> None:
    """Check that one Harbor launch can satisfy a task-owned binding."""
    if isinstance(binding.interaction, Chat):
        if launch.agent not in {"chat", "replay"}:
            raise ValueError("Selected agent requires explicit tool bindings")
        return
    tool = binding.interaction.tools[0]
    if isinstance(tool, ProviderToolBinding):
        if launch.agent != "provider_chat":
            raise ValueError("Provider tool bindings require the provider_chat agent")
        return
    if isinstance(tool, ShellToolBinding):
        if launch.agent != "tool_chat":
            raise ValueError("Shell tool bindings require the tool_chat agent")
        return
    if launch.agent not in {"replay", "terminus-2", "mini-swe-agent"}:
        raise ValueError("Terminal tool bindings require a compatible terminal agent")
    if tool.backend == "shellsim" and launch.agent != "replay":
        raise ValueError("ShellSim terminal bindings are supported only by replay")


class HarborExecutionConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Resolved Harbor launch, retained only for explicit reference executions."""

    binding: HarborTaskBinding
    launch: HarborLaunchConfig

    def __post_init__(self) -> None:
        validate_launch(self.binding, self.launch)


def provided_capabilities(environment: EnvironmentConfig) -> frozenset[Capability]:
    """Capabilities guaranteed by the currently implemented environment providers."""
    if isinstance(environment, NoEnvironment):
        return frozenset()
    if isinstance(environment, ProviderEnvironment):
        return frozenset()
    capabilities = {Capability.FILESYSTEM, Capability.SHELL}
    if isinstance(environment, DockerEnvironment):
        capabilities.add(Capability.PROCESS)
    return frozenset(capabilities)


def validate_requirements(requirements: TaskRequirements, environment: EnvironmentConfig) -> None:
    """Match capabilities and pinned state without choosing a harness."""
    missing = set(requirements.capabilities) - provided_capabilities(environment)
    if missing:
        raise ValueError(f"Environment lacks required capabilities: {sorted(missing)}")
    state = requirements.state
    if isinstance(environment, NoEnvironment):
        if requirements.action_interfaces:
            raise ValueError("Action interfaces require a provider environment")
        return
    if isinstance(environment, ProviderEnvironment):
        if requirements.action_interfaces != (environment.interface,):
            raise ValueError("Provider environment must match the task's declared action interface")
        if state != type(state)():
            raise ValueError("Provider environments cannot materialize workspace state")
        return
    if requirements.action_interfaces:
        raise ValueError("Action interfaces require a provider environment")
    if state.image is not None and (not isinstance(environment, DockerEnvironment) or environment.image != state.image):
        raise ValueError("Environment must preserve the task's pinned initial state image")
    if (environment.workdir, environment.setup_commands, environment.additional_directories) != (
        state.workdir,
        state.setup_commands,
        state.additional_directories,
    ):
        raise ValueError("Environment must preserve the task's declared workspace state")


def environment_for_requirements(requirements: TaskRequirements) -> EnvironmentConfig:
    """Select a provider for the example execution matrix, outside task data."""
    state = requirements.state
    if requirements.action_interfaces:
        raise ValueError("Selecting a stateful action provider requires explicit adapter configuration")
    if not requirements.capabilities:
        return NoEnvironment()
    if state.image is not None:
        return DockerEnvironment(state.image, state.workdir, state.setup_commands, state.additional_directories)
    if Capability.PROCESS in requirements.capabilities:
        raise ValueError("Selecting a process provider requires an explicit toolchain image")
    return ShellSimEnvironment(
        workdir=state.workdir, setup_commands=state.setup_commands, additional_directories=state.additional_directories
    )
