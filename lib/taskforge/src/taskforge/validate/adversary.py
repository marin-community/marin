# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The adversary roles that key the calibration summary's per-role adversary statistics."""

from enum import StrEnum


class AdversaryRole(StrEnum):
    SHORTCUT = "shortcut"
    """Find a submission the task's verifier accepts that violates the spirit of the task or does not need the
    task's intended computation: format-only output, lenient matching, files the grader trusts, an answer leaked
    by the instruction or the environment, a defensible alternative reading, or (when the consumer says so) a
    route that skips the capability the task was generated for."""
