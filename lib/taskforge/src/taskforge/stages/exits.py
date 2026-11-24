# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from taskforge.model.review import Decision, ValidationResult
from taskforge.model.stage import Exit


@dataclass(frozen=True, slots=True)
class Validated(Exit):
    result: ValidationResult


@dataclass(frozen=True, slots=True)
class Decided(Exit):
    decision: Decision
