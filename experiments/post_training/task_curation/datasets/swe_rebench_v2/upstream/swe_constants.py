# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class TestStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
