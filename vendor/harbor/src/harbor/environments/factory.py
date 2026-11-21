from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

from harbor.environments.base import BaseEnvironment
from harbor.environments.capabilities import EnvironmentResourceCapabilities
from harbor.environments.resource_policies import validate_resource_capabilities
from harbor.host_capabilities import HostCapabilities
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig
from harbor.models.trial.config import ResourceMode
from harbor.models.trial.paths import TrialPaths


class _EnvEntry(NamedTuple):
    module: str
    class_name: str


# Optional Shellbox backends load only when selected.
_ENVIRONMENT_REGISTRY: dict[EnvironmentType, _EnvEntry] = {
    EnvironmentType.LOCAL: _EnvEntry("harbor.environments.local", "LocalEnvironment"),
    EnvironmentType.DAYTONA: _EnvEntry(
        "harbor.environments.daytona", "DaytonaEnvironment"
    ),
    EnvironmentType.QEMU: _EnvEntry("harbor.environments.qemu", "QemuEnvironment"),
    EnvironmentType.SHELLSIM: _EnvEntry(
        "harbor.environments.shellsim", "ShellSimEnvironment"
    ),
}


def _load_environment_class(env_type: EnvironmentType) -> type[BaseEnvironment]:
    entry = _ENVIRONMENT_REGISTRY[env_type]
    module = importlib.import_module(entry.module)
    return getattr(module, entry.class_name)


class EnvironmentFactory:
    @classmethod
    def create_environment(
        cls,
        type: EnvironmentType,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        logger: logging.Logger | None = None,
        host_capabilities: HostCapabilities = HostCapabilities(),
        guest_environment: Mapping[str, str] = {},
        **kwargs,
    ) -> BaseEnvironment:
        environment_class = _load_environment_class(type)

        return environment_class(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            logger=logger,
            host_capabilities=host_capabilities,
            guest_environment=guest_environment,
            **kwargs,
        )

    @classmethod
    def run_preflight(cls, type: EnvironmentType) -> None:
        """Run preflight checks for the selected Shellbox adapter."""
        _load_environment_class(type).preflight()

    @classmethod
    def resource_capabilities(
        cls, type: EnvironmentType
    ) -> EnvironmentResourceCapabilities | None:
        return _load_environment_class(type).resource_capabilities()

    @classmethod
    def validate_config(cls, config: TrialEnvironmentConfig) -> None:
        if not config.delete:
            raise ValueError(
                "Shellbox environments cannot be retained: --no-delete and environment.delete=false are unsupported"
            )
        resource_capabilities = cls.resource_capabilities(config.type)
        if resource_capabilities is None:
            return
        validate_resource_capabilities(
            environment_label=config.type.value,
            resource_capabilities=resource_capabilities,
            cpu_enforcement_policy=config.cpu_enforcement_policy,
            memory_enforcement_policy=config.memory_enforcement_policy,
        )

    @classmethod
    def create_environment_from_config(
        cls,
        config: TrialEnvironmentConfig,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        logger: logging.Logger | None = None,
        host_capabilities: HostCapabilities = HostCapabilities(),
        guest_environment: Mapping[str, str] = {},
        **kwargs,
    ) -> BaseEnvironment:
        """Create the selected Shellbox adapter with its resource policy."""
        cls.validate_config(config)
        env_constructor_kwargs = {
            "override_cpus": config.override_cpus,
            "override_memory_mb": config.override_memory_mb,
            "override_storage_mb": config.override_storage_mb,
            "override_gpus": config.override_gpus,
            "override_tpu": config.override_tpu,
            "persistent_env": config.env,
            **config.kwargs,
            **kwargs,
        }
        if config.cpu_enforcement_policy != ResourceMode.AUTO:
            env_constructor_kwargs["cpu_enforcement_policy"] = (
                config.cpu_enforcement_policy
            )
        if config.memory_enforcement_policy != ResourceMode.AUTO:
            env_constructor_kwargs["memory_enforcement_policy"] = (
                config.memory_enforcement_policy
            )

        return cls.create_environment(
            type=config.type,
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            logger=logger,
            host_capabilities=host_capabilities,
            guest_environment=guest_environment,
            **env_constructor_kwargs,
        )
