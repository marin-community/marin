# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage identities and the shared base for coordinator exits."""

from dataclasses import dataclass
from enum import StrEnum


class Stage(StrEnum):
    TRIAGE = "triage"
    AUTHOR = "author"
    VALIDATE = "validate"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class Exit:
    """Transition tables dispatch on concrete subclasses."""
