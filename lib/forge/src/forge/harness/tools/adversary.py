# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from forge.model.controls import Candidate


@dataclass(frozen=True, slots=True)
class AdversaryReport:
    findings: tuple[str, ...]
    candidates: tuple[Candidate, ...]
