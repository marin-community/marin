# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.processing.classification.deduplication.fuzzy_verification import (
    FuzzyVerificationParams,
    VerificationRejection,
    verify_candidate,
)


def test_shorter_contained_member_is_verified():
    member = "alpha beta gamma delta epsilon zeta eta theta"
    canonical = f"preface context {member} appendix context"

    result = verify_candidate(member, canonical, FuzzyVerificationParams())

    assert result.accepted
    assert result.member_containment == 1.0
    assert result.member_unique_ngrams == 0


def test_richer_member_is_retained():
    canonical = "alpha beta gamma delta epsilon zeta"
    member = f"{canonical} unique appendix with additional facts"

    result = verify_candidate(member, canonical, FuzzyVerificationParams())

    assert not result.accepted
    assert result.rejection == VerificationRejection.MEMBER_LONGER


def test_default_subset_rule_rejects_templated_mutation():
    canonical_tokens = [f"shared{index}" for index in range(2_000)]
    member_tokens = canonical_tokens.copy()
    for index in (50, 150, 250):
        member_tokens[index] = f"change{index}"
    canonical = " ".join(canonical_tokens)
    member = " ".join(member_tokens)

    result = verify_candidate(member, canonical, FuzzyVerificationParams())

    assert not result.accepted
    assert result.member_containment >= 0.99
    assert result.member_unique_ngrams > 4
    assert result.rejection == VerificationRejection.CONTAINMENT


def test_inserted_section_does_not_relax_exact_subset_rule():
    member = "alpha beta gamma delta epsilon zeta eta theta"
    canonical = "alpha beta gamma inserted quoted section delta epsilon zeta eta theta"

    result = verify_candidate(member, canonical, FuzzyVerificationParams())

    assert not result.accepted
    assert result.member_unique_ngrams > 0
    assert result.rejection == VerificationRejection.CONTAINMENT


def test_configurable_novelty_cap_rejects_templated_mutation():
    canonical_tokens = [f"shared{index}" for index in range(2_000)]
    member_tokens = canonical_tokens.copy()
    for index in (50, 150, 250):
        member_tokens[index] = f"change{index}"
    params = FuzzyVerificationParams(
        minimum_member_containment=0.99,
        maximum_member_unique_ngrams=4,
    )

    result = verify_candidate(
        " ".join(member_tokens),
        " ".join(canonical_tokens),
        params,
    )

    assert result.member_containment >= params.minimum_member_containment
    assert result.member_unique_ngrams > params.maximum_member_unique_ngrams
    assert result.rejection == VerificationRejection.MEMBER_UNIQUE


def test_character_guard_rejects_non_whitespace_false_positive():
    shared_footer = "共通の著作権表示とナビゲーション"
    member = f"これは天文学についての異なる長い記事です。{shared_footer}"
    canonical = f"これは料理と地域製品についての無関係な説明です。{shared_footer}"
    permissive = FuzzyVerificationParams(
        minimum_member_containment=0,
        maximum_member_unique_ngrams=10,
    )

    result = verify_candidate(member, canonical, permissive)

    assert result.under_tokenized
    assert result.char_jaccard is not None
    assert result.char_jaccard < permissive.under_tokenized_minimum_char_jaccard
    assert result.rejection == VerificationRejection.UNDER_TOKENIZED


def test_character_guard_keeps_exact_non_whitespace_copy():
    text = "空白を含まない同一の日本語文書です。" * 20

    result = verify_candidate(text, text, FuzzyVerificationParams())

    assert result.accepted
    assert result.under_tokenized
    assert result.char_jaccard == 1.0
