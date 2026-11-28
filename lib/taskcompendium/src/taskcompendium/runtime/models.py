# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execution evidence kept separately from task definitions and review findings."""

from dataclasses import dataclass

from taskcompendium.models import ConversationTrace, GradingAttempt, StateSubmission, decode_json_value


@dataclass(frozen=True)
class RuntimeEvidence:
    files: dict[str, bytes]
    state_json: str


def grading_attempt(conversation: ConversationTrace, evidence: RuntimeEvidence) -> GradingAttempt:
    """Pair a conversation with the files and decoded state an environment captured."""
    return GradingAttempt(conversation, evidence.files, StateSubmission(decode_json_value(evidence.state_json)))
