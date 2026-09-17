# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curriculum-independent task semantics used by assignment layers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticKey:
    summary: str
    hardest_part: str
    required_operations: tuple[str, ...]
    subject_hint: str
    answer_form: str

    def macro_embedding_text(self) -> str:
        """Return subject-preserving text for broad inventory routing."""

        return f"Subject: {self.subject_hint}. Task: {self.summary} Central operation: {self.hardest_part}"

    def unit_embedding_text(self) -> str:
        """Return operation-focused text for assignment within a macro area."""

        return self.hardest_part
