# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage results consumed by coordinator transitions."""

from dataclasses import dataclass

from taskforge.model.proposal import TaskProposal
from taskforge.model.review import Decision, ValidationResult
from taskforge.model.stage import Exit


@dataclass(frozen=True, slots=True)
class Validated(Exit):
    result: ValidationResult


@dataclass(frozen=True, slots=True)
class Decided(Exit):
    decision: Decision


@dataclass(frozen=True, slots=True)
class Triaged(Exit):
    proposal: TaskProposal


@dataclass(frozen=True, slots=True)
class TriageRejected(Exit):
    reasons: tuple[str, ...]
