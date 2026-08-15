# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from dupekit import TokenNgramFingerprintSignature, TokenNgrams, TokenNgramSignature


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


@pytest.mark.parametrize(
    "constructor",
    [TokenNgrams, TokenNgramSignature, TokenNgramFingerprintSignature],
)
def test_token_ngram_types_reject_zero_size(constructor):
    with pytest.raises(ValueError):
        constructor("token", 0)


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


@pytest.mark.parametrize("signature_type", [TokenNgramSignature, TokenNgramFingerprintSignature])
def test_token_ngram_signatures_reject_a_missing_ngram(signature_type):
    member = signature_type("one two missing", 2)
    representative = signature_type("one two present", 2)

    assert not member.may_be_subset_of(representative)


def test_fingerprint_signature_preserves_containment_metadata():
    member = TokenNgramFingerprintSignature("one\ntwo three", 2)
    representative = TokenNgramFingerprintSignature("zero one\ntwo three four", 2)

    assert member.chars == len("one\ntwo three")
    assert member.lines == 2
    assert member.token_count == 3
    assert member.ngram_count == 2
    assert member.may_be_subset_of(representative)

    restored = TokenNgramFingerprintSignature.from_bytes(member.to_bytes())
    assert restored.chars == member.chars
    assert restored.lines == member.lines
    assert restored.token_count == member.token_count
    assert restored.ngram_count == member.ngram_count
    assert restored.may_be_subset_of(representative)
