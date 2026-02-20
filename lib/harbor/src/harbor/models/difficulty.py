# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    UNKNOWN = "unknown"
