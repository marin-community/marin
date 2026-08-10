# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Operator environment injection (`defaults.inject_env`).

Resolves the environment variable names a cluster declares against the
operator's shell at `iris cluster start` and folds them toward tasks. This lives
in its own module (importing only the config model) so the controller backends
can use it without the `config` → `backends.factory` import cycle.
"""

import os
from collections.abc import Sequence

from iris.cluster.config import IrisClusterConfig

# Secret holding the cluster default env (S3 storage auth + operator-injected
# vars), projected into task pods and the controller via envFrom on Kubernetes.
TASK_ENV_SECRET_NAME = "iris-task-env"


def projects_task_env_secret(config: IrisClusterConfig) -> bool:
    """Whether the cluster populates the iris-task-env Secret.

    True when S3 storage auth must be injected or operator env is declared. The
    K8s controller creates the Secret exactly when this holds, and task pods /
    the controller add the `envFrom` reference only then — so both decisions
    share this single predicate rather than each re-deriving it (a drift would
    leave envFrom dereferencing a missing Secret).
    """
    return config.storage.remote_state_dir.startswith("s3://") or bool(config.defaults.inject_env)


def collect_inject_env(names: Sequence[str]) -> dict[str, str]:
    """Resolve operator-injected env var names against the launching shell.

    Reads each name from ``os.environ``. Aborts (rather than silently dropping)
    if any are unset, so a misconfigured launch fails fast instead of starting a
    cluster whose tasks are missing credentials. Returns ``{name: value}``.
    """
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for name in names:
        value = os.environ.get(name)
        if value is None:
            missing.append(name)
        else:
            resolved[name] = value
    if missing:
        raise ValueError(
            "Cluster config defaults.inject_env requires environment variables that are unset in "
            f"this shell: {', '.join(missing)}. Export them before `iris cluster start`."
        )
    return resolved


def merge_injected_into_task_env(
    config: IrisClusterConfig,
    injected: dict[str, str],
) -> None:
    """Fold operator-injected values into task_env in place (GCP/RPC path).

    Injected values are *defaults*: existing literal ``task_env`` entries win, so
    a cluster config that pins e.g. MARIN_PREFIX is not overridden by the
    operator's shell. Mirrors the values into ``worker.task_env`` so they reach
    workers through the bootstrap config.
    """
    for name, value in injected.items():
        if name not in config.defaults.task_env:
            config.defaults.task_env[name] = value
        if name not in config.defaults.worker.task_env:
            config.defaults.worker.task_env[name] = value


def with_injected_task_env(config: IrisClusterConfig) -> IrisClusterConfig:
    """Return a copy of config with operator-injected env folded into task_env.

    Used by the VM bootstrap path (GCP/manual), where the shipped config is the
    only channel to workers. Resolution happens here, in the launcher's shell;
    the controller VM never sees the operator's environment. Returns the input
    unchanged when inject_env is empty.
    """
    if not config.defaults.inject_env:
        return config
    injected = collect_inject_env(config.defaults.inject_env)
    merged = config.model_copy(deep=True)
    merge_injected_into_task_env(merged, injected)
    return merged
