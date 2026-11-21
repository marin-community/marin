# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Categories for Shellbox machine failures and its provider errors."""

from types import MappingProxyType

from rigging.task_errors import SHARED_ERROR_CATEGORIES, ErrorCategory, compose_error_categories

SHELLBOX_ERROR_CATEGORIES = MappingProxyType(
    {
        "DaytonaConflictError": ErrorCategory.INFRASTRUCTURE,
        "DaytonaConnectionError": ErrorCategory.INFRASTRUCTURE,
        "DaytonaDockerSetupError": ErrorCategory.INFRASTRUCTURE,
        "DaytonaError": ErrorCategory.INFRASTRUCTURE,
        "DaytonaOutputUnavailable": ErrorCategory.INFRASTRUCTURE,
        "DaytonaProcessError": ErrorCategory.INFRASTRUCTURE,
        "DaytonaSandboxStopError": ErrorCategory.INFRASTRUCTURE,
        "MachineTerminated": ErrorCategory.INFRASTRUCTURE,
        "SnapshotQuotaExceeded": ErrorCategory.INFRASTRUCTURE,
    }
)

ERROR_CATEGORIES = compose_error_categories(SHARED_ERROR_CATEGORIES, SHELLBOX_ERROR_CATEGORIES)
