# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from dupekit import TokenNgrams, TokenNgramSignature


def _token_ngrams(text: str, size: int) -> frozenset[tuple[str, ...]]:
    tokens = text.lower().split()
    if not tokens:
        return frozenset()
    if len(tokens) < size:
        return frozenset((tuple(tokens),))
    return frozenset(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))


@pytest.mark.parametrize(
    ("left", "right", "size"),
    [
        ("", "", 3),
        ("one", "one", 3),
        ("one", "two", 3),
        ("a b a b", "a b c", 2),
        ("Straße 東 🙂", "straße 東 different", 2),
        (" ".join(["repeated"] * 100), " ".join(["repeated"] * 50), 5),
    ],
)
def test_token_ngrams_matches_exact_set_cardinality(left: str, right: str, size: int):
    expected_left = _token_ngrams(left, size)
    expected_right = _token_ngrams(right, size)
    actual_left = TokenNgrams(left, size)
    actual_right = TokenNgrams(right, size)

    assert len(actual_left) == len(expected_left)
    assert len(actual_right) == len(expected_right)
    assert actual_left.token_count == len(left.lower().split())
    assert actual_right.token_count == len(right.lower().split())
    assert actual_left.intersection_size(actual_right) == len(expected_left & expected_right)
    assert actual_right.intersection_size(actual_left) == len(expected_right & expected_left)


def test_token_ngrams_rejects_zero_size():
    with pytest.raises(ValueError):
        TokenNgrams("token", 0)


def test_token_ngrams_with_different_sizes_do_not_intersect():
    assert TokenNgrams("one", 2).intersection_size(TokenNgrams("one", 3)) == 0


@pytest.mark.parametrize(
    ("member", "representative", "size"),
    [
        ("one two three", "zero one two three four", 2),
        ("ONE\tTWO   THREE", "zero one two three four", 2),
        ("a b a b", "zero a b a b c", 2),
        ("", "anything", 3),
    ],
)
def test_token_ngram_signature_never_rejects_an_exact_subset(member: str, representative: str, size: int):
    member_ngrams = TokenNgrams(member, size)
    representative_ngrams = TokenNgrams(representative, size)
    assert member_ngrams.intersection_size(representative_ngrams) == len(member_ngrams)

    direct_signature = TokenNgramSignature(member, size)
    representative_signature = TokenNgramSignature(representative, size)
    assert direct_signature.token_count == member_ngrams.token_count
    assert direct_signature.may_be_subset_of(representative_signature)
    assert member_ngrams.signature().may_be_subset_of(representative_signature)


def test_token_ngram_signature_rejects_a_missing_ngram():
    member = TokenNgramSignature("one two missing", 2)
    representative = TokenNgramSignature("one two present", 2)

    assert not member.may_be_subset_of(representative)


def test_token_ngram_signature_rejects_zero_size():
    with pytest.raises(ValueError):
        TokenNgramSignature("token", 0)
