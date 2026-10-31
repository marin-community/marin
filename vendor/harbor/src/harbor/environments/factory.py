from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import NamedTuple

from harbor.environments.base import BaseEnvironment
from harbor.environments.capabilities import EnvironmentResourceCapabilities
from harbor.environments.resource_policies import validate_resource_capabilities
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
    EnvironmentType.DAYTONA: _EnvEntry("harbor.environments.daytona", "DaytonaEnvironment"),
    EnvironmentType.QEMU: _EnvEntry("harbor.environments.qemu", "QemuEnvironment"),
    EnvironmentType.SHELLSIM: _EnvEntry("harbor.environments.shellsim", "ShellSimEnvironment"),
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
            **kwargs,
        )

    @classmethod
    def run_preflight(cls, type: EnvironmentType) -> None:
        """Run preflight checks for the selected Shellbox adapter."""
        _load_environment_class(type).preflight()

    @classmethod
    def resource_capabilities(cls, type: EnvironmentType) -> EnvironmentResourceCapabilities | None:
        return _load_environment_class(type).resource_capabilities()

    @classmethod
    def validate_resource_policies(cls, config: TrialEnvironmentConfig) -> None:
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
        **kwargs,
    ) -> BaseEnvironment:
        """Create the selected Shellbox adapter with its resource policy."""
        env_constructor_kwargs = {
            "override_cpus": config.override_cpus,
            "override_memory_mb": config.override_memory_mb,
            "override_storage_mb": config.override_storage_mb,
            "override_gpus": config.override_gpus,
            "override_tpu": config.override_tpu,
            "persistent_env": config.env,
            "extra_docker_compose": config.extra_docker_compose,
            **config.kwargs,
            **kwargs,
        }
        if config.cpu_enforcement_policy != ResourceMode.AUTO:
            env_constructor_kwargs["cpu_enforcement_policy"] = config.cpu_enforcement_policy
        if config.memory_enforcement_policy != ResourceMode.AUTO:
            env_constructor_kwargs["memory_enforcement_policy"] = config.memory_enforcement_policy

        return cls.create_environment(
            type=config.type,
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            logger=logger,
            **env_constructor_kwargs,
        )
