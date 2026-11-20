# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage identities and the shared base for coordinator exits."""

from dataclasses import dataclass
from enum import StrEnum


class Stage(StrEnum):
    AUTHOR = "author"
    VALIDATE = "validate"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class Exit:
    """What a stage returns; the coordinator matches on the concrete subclass."""
