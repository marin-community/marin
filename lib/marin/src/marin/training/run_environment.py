# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib
import logging
import os
from collections.abc import Iterable, Mapping
from copy import deepcopy

from fray import GpuConfig, ResourceConfig, TpuConfig

logger = logging.getLogger(__name__)

VLLM_TARGET_DEVICE_ENV = "VLLM_TARGET_DEVICE"
VLLM_TPU_TARGET_DEVICE = "tpu"


def _cli_helpers_module():
    return importlib.import_module("levanter.infra.cli_helpers")


def resolve_required_env_vars(
    env_var_names: Iterable[str],
    *,
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve required env vars from a source mapping.

    Args:
        env_var_names: Names to resolve.
        source: Optional env mapping. Defaults to ``os.environ``.

    Returns:
        Mapping of env var names to values in first-seen order.

    Raises:
        ValueError: If any requested env var is missing.
    """
    env = os.environ if source is None else source
    resolved: dict[str, str] = {}
    for env_var_name in env_var_names:
        if env_var_name in resolved:
            continue
        value = env.get(env_var_name)
        if value is None:
            raise ValueError(f"Missing required environment variable: {env_var_name}")
        resolved[env_var_name] = value
    return resolved


def add_run_env_variables(env: dict[str, str], *, passthrough_env_vars: Iterable[str] = ()) -> dict[str, str]:
    """Add run metadata and required eval env vars to a process environment."""
    env = deepcopy(env)

    git_commit = env.get("GIT_COMMIT") or os.environ.get("GIT_COMMIT")
    if not git_commit:
        try:
            git_commit = _cli_helpers_module().get_git_commit()
        except Exception:
            git_commit = None

    if git_commit:
        env["GIT_COMMIT"] = git_commit
    else:
        logger.warning("Failed to find or infer git commit for logging.")

    if "HF_DATASETS_TRUST_REMOTE_CODE" not in env:
        env["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
    if "HF_ALLOW_CODE_EVAL" not in env:
        env["HF_ALLOW_CODE_EVAL"] = "1"
    if "TOKENIZERS_PARALLELISM" not in env:
        env["TOKENIZERS_PARALLELISM"] = "false"
    if "TPU_MIN_LOG_LEVEL" not in env:
        env["TPU_MIN_LOG_LEVEL"] = "2"
    if "TPU_STDERR_LOG_LEVEL" not in env:
        env["TPU_STDERR_LOG_LEVEL"] = "2"
    if "JAX_COMPILATION_CACHE_DIR" not in env and (val := os.environ.get("JAX_COMPILATION_CACHE_DIR")):
        env["JAX_COMPILATION_CACHE_DIR"] = val
    for env_var_name, value in resolve_required_env_vars(
        (name for name in passthrough_env_vars if name not in env),
        source=os.environ,
    ).items():
        env[env_var_name] = value

    return env


def extras_for_resources(resources: ResourceConfig) -> list[str]:
    """Return the uv extras for a resource device config."""
    device = resources.device
    if isinstance(device, TpuConfig):
        return ["tpu"]
    if isinstance(device, GpuConfig):
        return ["gpu"]
    return []


def dependency_groups_for_resources(
    resources: ResourceConfig,
    dependency_groups: list[str] | None,
) -> list[str]:
    """Return explicit dependency groups, or infer accelerator extras from resources."""
    if dependency_groups is not None:
        return dependency_groups
    return extras_for_resources(resources)


def env_vars_for_dependency_groups(
    resources: ResourceConfig,
    dependency_groups: list[str],
    env_vars: dict[str, str] | None,
) -> dict[str, str]:
    """Return environment variables required by the selected dependency groups."""
    env = dict(env_vars or {})
    if "vllm" in dependency_groups and isinstance(resources.device, TpuConfig):
        # vLLM source installs choose CUDA by default unless the TPU target is explicit.
        env.setdefault(VLLM_TARGET_DEVICE_ENV, VLLM_TPU_TARGET_DEVICE)
    return env
