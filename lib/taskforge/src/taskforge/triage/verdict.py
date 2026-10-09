# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The triage verdict on one proposal: decision, structural results, rubric samples, and reasons."""

from enum import StrEnum


class TriageDecision(StrEnum):
    ACCEPT = "accept"
    REPAIR = "repair"
    REJECT = "reject"
