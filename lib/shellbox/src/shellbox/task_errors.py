# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Categories for Shellbox machine failures and its provider errors."""

from types import MappingProxyType

from rigging.task_errors import SHARED_ERROR_CATEGORIES, ErrorCategory, compose_error_categories

DAYTONA_CONFLICT_ERROR = "DaytonaConflictError"
DAYTONA_CONNECTION_ERROR = "DaytonaConnectionError"
DAYTONA_DOCKER_SETUP_ERROR = "DaytonaDockerSetupError"
DAYTONA_OUTPUT_UNAVAILABLE = "DaytonaOutputUnavailable"
DAYTONA_PROCESS_ERROR = "DaytonaProcessError"
MACHINE_TERMINATED = "MachineTerminated"
SANDBOX_CAPACITY_ERROR = "SandboxCapacityError"


SHELLBOX_ERROR_CATEGORIES = MappingProxyType(
    {
        DAYTONA_CONFLICT_ERROR: ErrorCategory.INFRASTRUCTURE,
        DAYTONA_CONNECTION_ERROR: ErrorCategory.INFRASTRUCTURE,
        DAYTONA_DOCKER_SETUP_ERROR: ErrorCategory.INFRASTRUCTURE,
        "DaytonaError": ErrorCategory.INFRASTRUCTURE,
        DAYTONA_OUTPUT_UNAVAILABLE: ErrorCategory.INFRASTRUCTURE,
        DAYTONA_PROCESS_ERROR: ErrorCategory.INFRASTRUCTURE,
        "DaytonaSandboxStopError": ErrorCategory.INFRASTRUCTURE,
        SANDBOX_CAPACITY_ERROR: ErrorCategory.INFRASTRUCTURE,
        MACHINE_TERMINATED: ErrorCategory.INFRASTRUCTURE,
        "SnapshotQuotaExceeded": ErrorCategory.INFRASTRUCTURE,
    }
)

ERROR_CATEGORIES = compose_error_categories(SHARED_ERROR_CATEGORIES, SHELLBOX_ERROR_CATEGORIES)
