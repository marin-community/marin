# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib
import logging
import os
from collections.abc import Iterable, Mapping
from copy import deepcopy

logger = logging.getLogger(__name__)


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
