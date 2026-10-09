# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The decision triage reaches on one proposal."""

from enum import StrEnum


class TriageDecision(StrEnum):
    ACCEPT = "accept"
    REPAIR = "repair"
    REJECT = "reject"
