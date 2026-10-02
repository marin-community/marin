# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

r"""Mode exact: the output compared to ``expected`` as strings, after normalization.

Two candidates are tried, and either one matching scores 1: the content of the last ``\boxed{}``
when the output has one, and the whole output.

A single expected entry must equal the candidate. Several expected entries make the candidate a
list: it is split on newlines and commas, empty items are dropped, and the items must match the
expected entries in order when ``ordered``, otherwise as a multiset.

``ignore_case`` casefolds both sides. ``ignore_whitespace`` collapses every run of whitespace to a
single space. Outer whitespace is stripped by default; set ``strip_outer_whitespace=False``
with ``ignore_whitespace=False`` for literal boundary comparison.

``substring=True`` requires one nonempty normalized reference contained in the candidate.
It is an explicit benchmark contract; equality remains the default.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence, Set
from pathlib import Path
from typing import Literal, NamedTuple

from tasktrove_verify.grade import InvalidTask, Reward, empty_output_policy, read_output, scored
from tasktrove_verify.modes.extract import extract_boxed
from tasktrove_verify.spec import EmptyOutputPolicy, ExactSpec

ITEM_SEPARATOR = re.compile(r"[\n,]")
MAX_DETAIL_CHARS = 400
MAX_COLLECTION_ITEMS = 10_000
MAX_COLLECTION_CHARS = 1_000_000


def _normalize(text: str, spec: ExactSpec) -> str:
    value = text.strip() if spec.strip_outer_whitespace else text
    if spec.ignore_whitespace:
        value = re.sub(r"\s+", " ", value)
    return value.casefold() if spec.ignore_case else value


def _items(text: str, spec: ExactSpec) -> list[str]:
    items = (_normalize(part, spec) for part in ITEM_SEPARATOR.split(text))
    return [item for item in items if item]


def _matches(candidate: str, spec: ExactSpec) -> bool:
    if spec.substring:
        return _normalize(spec.expected[0], spec) in _normalize(candidate, spec)
    if len(spec.expected) == 1:
        return _normalize(candidate, spec) == _normalize(spec.expected[0], spec)
    expected = [_normalize(entry, spec) for entry in spec.expected]
    items = _items(candidate, spec)
    if spec.ordered:
        return items == expected
    return Counter(items) == Counter(expected)


def _validate_spec(spec: ExactSpec) -> None:
    if (
        not isinstance(spec.expected, (tuple, list))
        or not spec.expected
        or any(not isinstance(value, str) for value in spec.expected)
    ):
        raise InvalidTask("exact expects at least one expected string")
    if any(
        type(value) is not bool
        for value in (
            spec.ignore_case,
            spec.ignore_whitespace,
            spec.ordered,
            spec.strip_outer_whitespace,
            spec.substring,
        )
    ):
        raise InvalidTask("exact normalization flags must be booleans")
    if spec.substring and (len(spec.expected) != 1 or not _normalize(spec.expected[0], spec)):
        raise InvalidTask("exact substring expects one nonempty normalized reference")


def grade_exact_candidate(spec: ExactSpec, candidate: str) -> Reward:
    """Score answer content after the caller extracts it from its submission format."""
    _validate_spec(spec)
    policy = empty_output_policy(spec)
    if not isinstance(candidate, str):
        raise InvalidTask("exact candidate must be text")
    if policy is EmptyOutputPolicy.ZERO and not candidate.strip():
        return scored(0.0, reason="empty_output")
    return scored(
        float(_matches(candidate, spec)),
        extracted=candidate.strip()[:MAX_DETAIL_CHARS],
        expected=list(spec.expected),
    )


def _collection_counts(items: object) -> Counter[str]:
    if not isinstance(items, (Sequence, Set)) or isinstance(items, (str, bytes)):
        raise ValueError("items must be a sequence or set of strings")
    if len(items) > MAX_COLLECTION_ITEMS:
        raise ValueError("too many items")
    counts: Counter[str] = Counter()
    chars = 0
    for index, item in enumerate(items):
        if index >= MAX_COLLECTION_ITEMS:
            raise ValueError("too many items")
        if not isinstance(item, str):
            raise ValueError("items must be strings")
        chars += len(item)
        if chars > MAX_COLLECTION_CHARS:
            raise ValueError("item text exceeds limit")
        counts[item] += 1
    return counts


class _CollectionOverlap(NamedTuple):
    overlap: int
    reference_count: int
    candidate_count: int


def _collection_overlap(
    reference: Sequence[str] | Set[str],
    candidate: Sequence[str] | Set[str],
    multiplicity: Literal["set", "multiset"],
    empty_reference: Literal["zero", "invalid"],
) -> _CollectionOverlap | Reward:
    if multiplicity not in ("set", "multiset") or empty_reference not in ("zero", "invalid"):
        raise InvalidTask("invalid collection policy")
    try:
        references = _collection_counts(reference)
    except ValueError as error:
        raise InvalidTask(f"invalid collection reference: {error}") from error
    if not references and empty_reference == "invalid":
        raise InvalidTask("collection reference must not be empty")
    try:
        candidates = _collection_counts(candidate)
    except ValueError as error:
        return scored(0.0, reason="invalid_candidate", error=str(error))
    if multiplicity == "set":
        references, candidates = Counter(references.keys()), Counter(candidates.keys())
    return _CollectionOverlap(sum((references & candidates).values()), references.total(), candidates.total())


def grade_collection_f1(
    reference: Sequence[str] | Set[str],
    candidate: Sequence[str] | Set[str],
    *,
    multiplicity: Literal["set", "multiset"],
    empty_reference: Literal["zero", "invalid"],
    round_digits: int | None,
) -> Reward:
    """Grade literal prepared items; callers own tokenization, never overlap counts.

    Set scoring ignores repeats; multiset scoring consumes each reference occurrence.
    Empty candidate collections score zero. Rounding uses binary64 scaling and
    ties-to-even, as in fixed-decimal array scoring, without an optional dependency.
    """
    if round_digits is not None and (type(round_digits) is not int or not 0 <= round_digits <= 6):
        raise InvalidTask("collection F1 rounding must be an integer from zero to six or None")
    counts = _collection_overlap(reference, candidate, multiplicity, empty_reference)
    if isinstance(counts, Reward):
        return counts
    overlap, reference_count, candidate_count = counts
    reward = 2 * overlap / (reference_count + candidate_count) if reference_count and candidate_count else 0.0
    if round_digits is not None:
        scale = 10**round_digits
        reward = round(reward * scale) / scale
    return scored(reward, overlap=overlap, reference_count=reference_count, candidate_count=candidate_count)


def grade_collection_subset(
    reference: Sequence[str] | Set[str],
    candidate: Sequence[str] | Set[str],
    *,
    item_credit: float,
) -> Reward:
    """Award full set equality or per-item credit for a strict subset.

    Repeated items count once. Any extra candidate item forfeits all credit;
    an empty candidate scores zero. References must contain nonempty items and
    the configured strict-subset credit must remain in the unit reward domain.
    """
    if type(item_credit) not in (int, float) or not 0 <= item_credit <= 1 or not math.isfinite(item_credit):
        raise InvalidTask("subset item credit must be finite and bounded")
    try:
        references = _collection_counts(reference)
    except ValueError as error:
        raise InvalidTask(f"invalid collection reference: {error}") from error
    if not references or any(not item for item in references):
        raise InvalidTask("subset reference must contain nonempty items")
    if item_credit * (len(references) - 1) > 1:
        raise InvalidTask("strict-subset credit exceeds the unit reward domain")
    counts = _collection_overlap(reference, candidate, "set", "invalid")
    if isinstance(counts, Reward):
        return counts
    overlap, reference_count, candidate_count = counts
    reward = 0.0
    if candidate_count and overlap == candidate_count:
        reward = 1.0 if candidate_count == reference_count else item_credit * candidate_count
    return scored(reward, overlap=overlap, reference_count=reference_count, candidate_count=candidate_count)


def grade_collection_precision_interval(
    reference: Sequence[str] | Set[str],
    candidate: Sequence[str] | Set[str],
    *,
    minimum_percent: float,
    maximum_percent: float,
    multiplicity: Literal["set", "multiset"],
    empty_reference: Literal["zero", "invalid"],
) -> Reward:
    """Accept prepared-item precision within inclusive percentage bounds.

    Empty or malformed candidates always score zero, including intervals containing
    zero. A valid nonempty disjoint candidate has zero precision and can be accepted.
    """
    bounds = (minimum_percent, maximum_percent)
    if (
        any(type(bound) not in (int, float) or not -math.inf < bound < math.inf for bound in bounds)
        or minimum_percent > maximum_percent
    ):
        raise InvalidTask("precision interval requires finite ordered bounds")
    counts = _collection_overlap(reference, candidate, multiplicity, empty_reference)
    if isinstance(counts, Reward):
        return counts
    if not counts.candidate_count:
        return scored(0.0, reason="empty_output")
    precision_percent = counts.overlap / counts.candidate_count * 100
    return scored(
        float(minimum_percent <= precision_percent <= maximum_percent),
        precision_percent=precision_percent,
        overlap=counts.overlap,
        reference_count=counts.reference_count,
        candidate_count=counts.candidate_count,
    )


def grade(spec: ExactSpec, tests_dir: Path, workspace: Path) -> Reward:
    _validate_spec(spec)

    text = read_output(spec, workspace)
    if text is None:
        return scored(0.0, reason="no_output")
    boxed = extract_boxed(text)
    candidates = [boxed, text] if boxed is not None else [text]
    result = grade_exact_candidate(spec, candidates[0])
    if result.reward or len(candidates) == 1:
        return result
    fallback = grade_exact_candidate(spec, candidates[1])
    return scored(fallback.reward, **result.detail)
