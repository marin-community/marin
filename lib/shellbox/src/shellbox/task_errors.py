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
MACHINE_UNAVAILABLE = "MachineUnavailable"
# The Daytona v2 machine factory raises its provisioning failures under these names and
# MachineUnavailable, not the SDK's generic class names (ConnectionError, RateLimitError), which
# other libraries share. The Daytona-prefixed names above remain for trial results persisted by
# the v0 SDK.
DAYTONA_CAPACITY_EXHAUSTED = "DaytonaCapacityExhausted"
DAYTONA_COMMAND_STOPPED = "DaytonaCommandStopped"
DAYTONA_PROVISIONING_ERROR = "DaytonaProvisioningError"


SHELLBOX_ERROR_CATEGORIES = MappingProxyType(
    {
        DAYTONA_CONFLICT_ERROR: ErrorCategory.INFRASTRUCTURE,
        DAYTONA_CONNECTION_ERROR: ErrorCategory.INFRASTRUCTURE,
        DAYTONA_DOCKER_SETUP_ERROR: ErrorCategory.INFRASTRUCTURE,
        "DaytonaError": ErrorCategory.INFRASTRUCTURE,
        DAYTONA_OUTPUT_UNAVAILABLE: ErrorCategory.INFRASTRUCTURE,
        DAYTONA_PROCESS_ERROR: ErrorCategory.INFRASTRUCTURE,
        "DaytonaSandboxStopError": ErrorCategory.INFRASTRUCTURE,
        MACHINE_TERMINATED: ErrorCategory.INFRASTRUCTURE,
        MACHINE_UNAVAILABLE: ErrorCategory.INFRASTRUCTURE,
        "SnapshotQuotaExceeded": ErrorCategory.INFRASTRUCTURE,
        DAYTONA_CAPACITY_EXHAUSTED: ErrorCategory.INFRASTRUCTURE,
        DAYTONA_COMMAND_STOPPED: ErrorCategory.INFRASTRUCTURE,
        DAYTONA_PROVISIONING_ERROR: ErrorCategory.INFRASTRUCTURE,
    }
)

ERROR_CATEGORIES = compose_error_categories(SHARED_ERROR_CATEGORIES, SHELLBOX_ERROR_CATEGORIES)
