"""Public entrypoint-neutral debug launch contract."""

from skyrl_train.env_vars import (
    EnvVarManager,
    EnvVarScope,
)


def debug_environment(config) -> dict[str, str]:
    return EnvVarManager.from_config(config).environment_for(EnvVarScope.RAY_WORKER)


def apply_debug_mode(config, *, environ=None) -> dict[str, str]:
    return EnvVarManager.from_config(config, environ=environ).apply_to_process(EnvVarScope.DRIVER, environ=environ)
