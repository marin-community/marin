# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full-text verification for fuzzy-duplicate candidates."""

from dataclasses import dataclass
from enum import StrEnum

import dupekit
from pydantic import BaseModel, ConfigDict, Field


class VerificationRejection(StrEnum):
    """Reason that the verifier retained a candidate."""

    MEMBER_LONGER = "member_longer"
    CONTAINMENT = "containment_below_threshold"
    MEMBER_UNIQUE = "too_many_member_unique"
    UNDER_TOKENIZED = "under_tokenized_char_jaccard_below_threshold"
    SATURATED = "saturated_token_sequence_not_contained"


class FuzzyVerificationParams(BaseModel):
    """Thresholds for an information-preserving duplicate decision."""

    model_config = ConfigDict(frozen=True)

    rule_version: str = "whitespace_3gram_subset_v4"
    ngram_size: int = Field(default=3, ge=1)
    minimum_member_containment: float = Field(default=1.0, ge=0, le=1)
    maximum_member_unique_ngrams: int = Field(default=0, ge=0)
    maximum_chars_per_token: float = Field(default=10.0, gt=0)
    under_tokenized_char_ngram_size: int = Field(default=5, ge=1)
    under_tokenized_minimum_char_jaccard: float = Field(default=0.90, ge=0, le=1)
    # A document that draws on a tiny vocabulary repeats its n-grams until the
    # distinct set saturates. Below this ratio of distinct n-grams to n-gram
    # positions, only exact normalized token-sequence containment is accepted.
    minimum_distinct_ngram_ratio: float = Field(default=0.9, ge=0, le=1)


@dataclass(frozen=True)
class PreparedVerificationText:
    """Text and token n-grams that the verifier can use more than once."""

    text: str
    chars: int
    tokens: int
    ngrams: dupekit.TokenNgrams
    under_tokenized: bool
    saturated: bool


@dataclass(frozen=True)
class VerificationResult:
    """Exact scores and the decision for one candidate."""

    accepted: bool
    rejection: VerificationRejection | None
    member_chars: int
    representative_chars: int
    member_tokens: int
    representative_tokens: int
    member_ngrams: int
    representative_ngrams: int
    shared_ngrams: int
    member_unique_ngrams: int
    member_containment: float
    jaccard: float
    under_tokenized: bool
    saturated: bool
    normalized_token_sequence_contained: bool | None
    char_jaccard: float | None


def prepare_verification_text(text: str, params: FuzzyVerificationParams) -> PreparedVerificationText:
    """Prepare one text for direct verification."""
    ngrams = dupekit.TokenNgrams(text, params.ngram_size)
    positions = max(ngrams.token_count - params.ngram_size + 1, 1)
    chars = len(text)
    return PreparedVerificationText(
        text=text,
        chars=chars,
        saturated=len(ngrams) / positions < params.minimum_distinct_ngram_ratio,
        tokens=ngrams.token_count,
        ngrams=ngrams,
        under_tokenized=chars / max(ngrams.token_count, 1) > params.maximum_chars_per_token,
    )


def _char_ngrams(text: str, size: int) -> set[str]:
    normalized = text.casefold()
    if len(normalized) < size:
        return {normalized}
    return {normalized[index : index + size] for index in range(len(normalized) - size + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    shared = len(left & right)
    union = len(left) + len(right) - shared
    return shared / union if union else 1.0


def character_ngram_jaccard(left: str, right: str, ngram_size: int) -> float:
    """Calculate case-folded character n-gram Jaccard similarity."""
    if ngram_size < 1:
        raise ValueError("ngram_size must be at least 1")
    return _jaccard(_char_ngrams(left, ngram_size), _char_ngrams(right, ngram_size))


def normalized_token_sequence_is_contained(member: str, representative: str) -> bool:
    """Test token-sequence containment after case and whitespace normalization."""
    normalized_member = " ".join(member.casefold().split())
    if not normalized_member:
        return False
    normalized_representative = " ".join(representative.casefold().split())
    start = 0
    while True:
        index = normalized_representative.find(normalized_member, start)
        if index < 0:
            return False
        end = index + len(normalized_member)
        starts_at_token = index == 0 or normalized_representative[index - 1] == " "
        ends_at_token = end == len(normalized_representative) or normalized_representative[end] == " "
        if starts_at_token and ends_at_token:
            return True
        start = index + 1


def line_count_ratio(left: str, right: str) -> float:
    """Calculate the ratio between the smaller and larger line counts."""
    left_lines = left.count("\n") + 1
    right_lines = right.count("\n") + 1
    return min(left_lines, right_lines) / max(left_lines, right_lines)


def verify_prepared_candidate(
    member: PreparedVerificationText,
    representative: PreparedVerificationText,
    params: FuzzyVerificationParams,
) -> VerificationResult:
    """Verify one prepared member against one prepared representative."""
    shared = member.ngrams.intersection_size(representative.ngrams)
    union = len(member.ngrams) + len(representative.ngrams) - shared
    member_containment = shared / len(member.ngrams) if member.ngrams else 1.0
    jaccard = shared / union if union else 1.0
    member_unique = len(member.ngrams) - shared
    under_tokenized = member.under_tokenized or representative.under_tokenized
    saturated = member.saturated or representative.saturated

    rejection = None
    char_jaccard = None
    normalized_token_sequence_contained = None
    if member.chars > representative.chars:
        rejection = VerificationRejection.MEMBER_LONGER
    elif member_containment < params.minimum_member_containment:
        rejection = VerificationRejection.CONTAINMENT
    elif member_unique > params.maximum_member_unique_ngrams:
        rejection = VerificationRejection.MEMBER_UNIQUE
    elif saturated:
        normalized_token_sequence_contained = normalized_token_sequence_is_contained(
            member.text,
            representative.text,
        )
        if not normalized_token_sequence_contained:
            rejection = VerificationRejection.SATURATED
    elif under_tokenized:
        char_jaccard = character_ngram_jaccard(
            member.text,
            representative.text,
            params.under_tokenized_char_ngram_size,
        )
        if char_jaccard < params.under_tokenized_minimum_char_jaccard:
            rejection = VerificationRejection.UNDER_TOKENIZED

    return VerificationResult(
        accepted=rejection is None,
        rejection=rejection,
        member_chars=member.chars,
        representative_chars=representative.chars,
        member_tokens=member.tokens,
        representative_tokens=representative.tokens,
        member_ngrams=len(member.ngrams),
        representative_ngrams=len(representative.ngrams),
        shared_ngrams=shared,
        member_unique_ngrams=member_unique,
        member_containment=member_containment,
        jaccard=jaccard,
        under_tokenized=under_tokenized,
        saturated=saturated,
        normalized_token_sequence_contained=normalized_token_sequence_contained,
        char_jaccard=char_jaccard,
    )


def verify_candidate(
    member_text: str,
    representative_text: str,
    params: FuzzyVerificationParams,
) -> VerificationResult:
    """Verify one member against one retained representative."""
    return verify_prepared_candidate(
        prepare_verification_text(member_text, params),
        prepare_verification_text(representative_text, params),
        params,
    )
