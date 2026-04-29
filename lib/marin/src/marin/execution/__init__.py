# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import Any

from .executor_step_status import (
    STATUS_DEP_FAILED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SUCCESS,
)

_EXECUTOR_EXPORTS = {
    "Executor",
    "ExecutorInfo",
    "ExecutorMainConfig",
    "executor_main",
    "resolve_executor_step",
}

_DAG_EXPORTS = {
    "THIS_OUTPUT_PATH",
    "ExecutorStep",
    "InputName",
    "OutputName",
    "VersionedValue",
    "ensure_versioned",
    "get_executor_step",
    "output_path_of",
    "this_output_path",
    "unwrap_versioned_value",
    "versioned",
}


def __getattr__(name: str) -> Any:
    if name in _EXECUTOR_EXPORTS:
        executor_module = importlib.import_module(".executor", __name__)
        return getattr(executor_module, name)
    if name in _DAG_EXPORTS:
        dag_module = importlib.import_module(".dag", __name__)
        return getattr(dag_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
