# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Machine-readable terminal reasons for enforced task resource limits."""

from enum import StrEnum


SANDBOX_STORAGE_LIMIT_CAPABILITY = "sandbox-storage-limit"


class ResourceLimitReason(StrEnum):
    DISK = "RESOURCE_LIMIT_DISK"
