# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Daytona provisioning choices, available without the provider SDK."""

from enum import StrEnum


class DaytonaProvisioning(StrEnum):
    SNAPSHOT = "snapshot"
    DECLARATIVE = "declarative"
