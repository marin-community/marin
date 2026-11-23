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
DAYTONA_COMMAND_STOPPED = "DaytonaCommandStopped"
# Daytona v2 SDK errors have generic class names; the Daytona-prefixed names
# above remain for trial results persisted by the v0 SDK.
CONFLICT_ERROR = "ConflictError"
CONNECTION_ERROR = "ConnectionError"
SANDBOX_CREATE_CONNECTION_ERROR = "SandboxCreateConnectionError"
SANDBOX_CREATE_REQUEST_TIMEOUT_ERROR = "SandboxCreateRequestTimeoutError"
CLIENT_TIMEOUT_ERROR = "ClientTimeoutError"
SERVICE_UNAVAILABLE_ERROR = "ServiceUnavailableError"


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
        "SnapshotQuotaExceeded": ErrorCategory.INFRASTRUCTURE,
        DAYTONA_COMMAND_STOPPED: ErrorCategory.INFRASTRUCTURE,
        CONFLICT_ERROR: ErrorCategory.INFRASTRUCTURE,
        CONNECTION_ERROR: ErrorCategory.INFRASTRUCTURE,
        SANDBOX_CREATE_CONNECTION_ERROR: ErrorCategory.INFRASTRUCTURE,
        SANDBOX_CREATE_REQUEST_TIMEOUT_ERROR: ErrorCategory.INFRASTRUCTURE,
        "SandboxTerminalStateError": ErrorCategory.INFRASTRUCTURE,
        "LimitExceededError": ErrorCategory.INFRASTRUCTURE,
        # Also the name of the OpenAI SDK's 429 error; both are provider throttling.
        "RateLimitError": ErrorCategory.INFRASTRUCTURE,
        "OverloadedError": ErrorCategory.INFRASTRUCTURE,
        CLIENT_TIMEOUT_ERROR: ErrorCategory.INFRASTRUCTURE,
        "RequestTimeoutError": ErrorCategory.INFRASTRUCTURE,
        SERVICE_UNAVAILABLE_ERROR: ErrorCategory.INFRASTRUCTURE,
        "CursorExpiredError": ErrorCategory.INFRASTRUCTURE,
        "ProcessKillIncompleteError": ErrorCategory.INFRASTRUCTURE,
    }
)

ERROR_CATEGORIES = compose_error_categories(SHARED_ERROR_CATEGORIES, SHELLBOX_ERROR_CATEGORIES)
