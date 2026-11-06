# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private inputs for the canonical instruction benchmark checkers."""

from dataclasses import dataclass
from enum import StrEnum

from verifyit.spec import Constraint


class InstructionProfile(StrEnum):
    HARNESS = "harness"
    IFBENCH = "ifbench"


@dataclass(frozen=True)
class InstructionSpec:
    constraints: tuple[Constraint, ...]
    profile: InstructionProfile
    prompt: str
