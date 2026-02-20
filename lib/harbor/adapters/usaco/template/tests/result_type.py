# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# this gets its own file to prevent circular imports
from enum import Enum


class ResultType(int, Enum):
    ACCEPTED = 1
    WRONG_ANSWER = 2
    TIME_LIMIT_EXCEEDED = 3
    MEMORY_LIMIT_EXCEEDED = 4
    COMPILATION_ERROR = 5
    RUNTIME_ERROR = 6
    UNKNOWN = 7
