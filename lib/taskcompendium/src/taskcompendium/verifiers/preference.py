# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private preference labels without an invented exact-answer reward."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from taskcompendium.grading import GradeResult, GradingAttempt, Outcome, Verifier
from taskcompendium.models import ConversationEvent


class PairwisePreference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["pairwise"] = "pairwise"
    chosen: tuple[ConversationEvent, ...]
    rejected: tuple[ConversationEvent, ...]


class BinaryPreference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["binary"] = "binary"
    response: tuple[ConversationEvent, ...]
    preferred: bool


class PreferenceEvidenceVerifier(Verifier):
    """Retain source comparisons; grading a new response needs a separately bound model."""

    evidence: Annotated[PairwisePreference | BinaryPreference, Field(discriminator="kind")]
    source_metadata: dict[str, JsonValue]

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        del attempt
        return GradeResult(
            Outcome.INFRA_ERROR,
            None,
            "Source preference labels are private comparison evidence; no reward model is bound for new responses",
        )
