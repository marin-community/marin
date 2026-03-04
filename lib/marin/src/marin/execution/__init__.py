# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from .executor import (
    THIS_OUTPUT_PATH,
    Executor,
    ExecutorInfo,
    ExecutorMainConfig,
    ExecutorStep,
    InputName,
    OutputName,
    VersionedValue,
    ensure_versioned,
    executor_main,
    get_executor_step,
    output_path_of,
    resolve_executor_step,
    this_output_path,
    unwrap_versioned_value,
    versioned,
)
from rigging.status_file import (
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SUCCESS,
)

STATUS_DEP_FAILED = "DEP_FAILED"  # Dependency failed — executor-specific status
