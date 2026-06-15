# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Operator environment injection (`defaults.inject_env`).

Resolves the environment variable names a cluster declares against the
operator's shell at `iris cluster start` and folds them toward tasks. This lives
in its own module (importing only the proto) so the controller backends can use
it without the `config` → `backends.factory` import cycle.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from iris.rpc import config_pb2

# Secret holding operator-injected env (defaults.inject_env), projected into task
# pods (and the controller) via envFrom on Kubernetes.
TASK_ENV_SECRET_NAME = "iris-task-env"


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
    config: config_pb2.IrisClusterConfig,
    injected: dict[str, str],
) -> None:
    """Fold operator-injected values into task_env in place (GCP/RPC path).

    Injected values are *defaults*: existing literal ``task_env`` entries win, so
    a cluster config that pins e.g. MARIN_PREFIX is not overridden by the
    operator's shell. Mirrors the values into ``worker.task_env`` so they reach
    workers through the bootstrap config (the same channel apply_defaults uses).
    """
    for name, value in injected.items():
        if name not in config.defaults.task_env:
            config.defaults.task_env[name] = value
        if name not in config.defaults.worker.task_env:
            config.defaults.worker.task_env[name] = value


def with_injected_task_env(config: config_pb2.IrisClusterConfig) -> config_pb2.IrisClusterConfig:
    """Return a copy of config with operator-injected env folded into task_env.

    Used by the VM bootstrap path (GCP/manual), where the shipped config is the
    only channel to workers. Resolution happens here, in the launcher's shell;
    the controller VM never sees the operator's environment. Returns the input
    unchanged when inject_env is empty.
    """
    if not config.defaults.inject_env:
        return config
    injected = collect_inject_env(config.defaults.inject_env)
    merged = config_pb2.IrisClusterConfig()
    merged.CopyFrom(config)
    merge_injected_into_task_env(merged, injected)
    return merged
