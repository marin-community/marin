# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact verification for MinHash candidate pairs."""

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class VerificationRejection(StrEnum):
    """Reason a MinHash candidate was retained."""

    MEMBER_LONGER = "member_longer"
    CONTAINMENT = "containment_below_threshold"
    MEMBER_UNIQUE = "too_many_member_unique"
    UNDER_TOKENIZED = "under_tokenized_char_jaccard_below_threshold"


class FuzzyVerificationParams(BaseModel):
    """Thresholds for an information-preserving direct-candidate deletion.

    The defaults require every case-folded whitespace token 3-gram in the
    removed member to occur in a no-shorter retained canonical. Text with too
    few whitespace tokens also needs full-text character-5 Jaccard of at least
    0.90.
    """

    model_config = ConfigDict(frozen=True)

    rule_version: str = "whitespace_3gram_subset_v1"
    ngram_size: int = Field(default=3, ge=1)
    minimum_member_containment: float = Field(default=1.0, ge=0, le=1)
    maximum_member_unique_ngrams: int = Field(default=0, ge=0)
    maximum_chars_per_token: float = Field(default=10.0, gt=0)
    under_tokenized_char_ngram_size: int = Field(default=5, ge=1)
    under_tokenized_minimum_char_jaccard: float = Field(default=0.90, ge=0, le=1)


@dataclass(frozen=True)
class VerificationResult:
    """Exact scores and decision for one candidate member."""

    accepted: bool
    rejection: VerificationRejection | None
    member_chars: int
    canonical_chars: int
    member_tokens: int
    canonical_tokens: int
    member_ngrams: int
    canonical_ngrams: int
    shared_ngrams: int
    member_unique_ngrams: int
    member_containment: float
    jaccard: float
    under_tokenized: bool
    char_jaccard: float | None


def _token_ngrams(text: str, size: int) -> tuple[int, set[tuple[str, ...]]]:
    tokens = text.casefold().split()
    if not tokens:
        return 0, set()
    if len(tokens) < size:
        return len(tokens), {tuple(tokens)}
    return len(tokens), {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _char_ngrams(text: str, size: int) -> set[str]:
    normalized = text.casefold()
    if len(normalized) < size:
        return {normalized}
    return {normalized[index : index + size] for index in range(len(normalized) - size + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = len(left) + len(right) - len(left & right)
    return len(left & right) / union if union else 1.0


def verify_candidate(
    member_text: str,
    canonical_text: str,
    params: FuzzyVerificationParams,
) -> VerificationResult:
    """Verify that a direct LSH member is safely contained by its canonical."""
    member_tokens, member_ngrams = _token_ngrams(member_text, params.ngram_size)
    canonical_tokens, canonical_ngrams = _token_ngrams(canonical_text, params.ngram_size)
    shared = len(member_ngrams & canonical_ngrams)
    union = len(member_ngrams) + len(canonical_ngrams) - shared
    member_containment = shared / len(member_ngrams) if member_ngrams else 1.0
    jaccard = shared / union if union else 1.0
    member_unique = len(member_ngrams) - shared
    member_chars = len(member_text)
    canonical_chars = len(canonical_text)
    under_tokenized = (
        member_chars / max(member_tokens, 1) > params.maximum_chars_per_token
        or canonical_chars / max(canonical_tokens, 1) > params.maximum_chars_per_token
    )

    rejection = None
    char_jaccard = None
    if member_chars > canonical_chars:
        rejection = VerificationRejection.MEMBER_LONGER
    elif member_containment < params.minimum_member_containment:
        rejection = VerificationRejection.CONTAINMENT
    elif member_unique > params.maximum_member_unique_ngrams:
        rejection = VerificationRejection.MEMBER_UNIQUE
    elif under_tokenized:
        char_jaccard = _jaccard(
            _char_ngrams(member_text, params.under_tokenized_char_ngram_size),
            _char_ngrams(canonical_text, params.under_tokenized_char_ngram_size),
        )
        if char_jaccard < params.under_tokenized_minimum_char_jaccard:
            rejection = VerificationRejection.UNDER_TOKENIZED

    return VerificationResult(
        accepted=rejection is None,
        rejection=rejection,
        member_chars=member_chars,
        canonical_chars=canonical_chars,
        member_tokens=member_tokens,
        canonical_tokens=canonical_tokens,
        member_ngrams=len(member_ngrams),
        canonical_ngrams=len(canonical_ngrams),
        shared_ngrams=shared,
        member_unique_ngrams=member_unique,
        member_containment=member_containment,
        jaccard=jaccard,
        under_tokenized=under_tokenized,
        char_jaccard=char_jaccard,
    )
