# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete ordered-pair review of fuzzy-duplicate candidate clusters.

The production verifier compares each cluster member with at most two
representatives, so its output cannot say whether a retained document failed
the rule or was never compared with the document that would have removed it.
This module scores every ordered pair of a cluster, replays the bounded
production algorithm on the same pairs, and measures the text a rejected
candidate would lose. Together those separate three causes of a retained
candidate:

- ``rule_blocked``: the rule rejects every ordered pair of the cluster.
- ``budget_blocked``: the rule accepts a pair the comparison budget never made.
- ``no_similar``: no pair comes close, so the candidate is an LSH false
  positive that no threshold change would reach.
"""

import difflib
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from marin.processing.classification.deduplication.fuzzy_verification import (
    FuzzyVerificationParams,
    character_ngram_jaccard,
    line_count_ratio,
    normalized_token_sequence_is_contained,
    prepare_verification_text,
    verify_prepared_candidate,
)
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    ANCHOR_SCAN_CHARS,
    ANCHOR_SCAN_RECORDS,
    REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
    LocalRepresentativeParams,
)

# A candidate whose best ordered pair falls below both bounds shares almost no
# text with the rest of its cluster. Connected components link transitively, so
# such a member is reachable only through other documents.
NO_SIMILAR_MAX_CONTAINMENT = 0.8
NO_SIMILAR_MAX_JACCARD = 0.7

BOILERPLATE_PATTERNS = (
    re.compile(r"^\s*$"),
    re.compile(r"^\s*(#|//|/\*|\*|<!--|--)"),
    re.compile(r"^\s*(import|from|using|require|include|package|use)\b"),
    re.compile(r"^\s*[})\];,]+\s*$"),
    re.compile(r"^\s*(copyright|licen[cs]e|all rights reserved)", re.IGNORECASE),
)


@dataclass(frozen=True)
class ClusterDocument:
    """One candidate cluster member with the text the verifier compared."""

    source_name: str
    id: str
    text: str
    buckets: tuple[str, ...]
    is_cluster_canonical: bool
    dropped: bool
    """True when the run being reviewed wrote a duplicate marker for it."""


@dataclass(frozen=True)
class PairScore:
    """One ordered pair: is ``member`` removable given ``representative``."""

    member_index: int
    representative_index: int
    accepted: bool
    rejection: str | None
    member_containment: float
    jaccard: float
    char_jaccard: float | None
    token_sequence_contained: bool
    saturated: bool
    under_tokenized: bool
    member_chars: int
    representative_chars: int
    member_unique_ngrams: int
    line_ratio: float
    novel_tokens: int
    """Distinct case-folded tokens of the member that the representative lacks.

    A member that adds no token adds no new word to the corpus. Cutting a block
    out of a document leaves the remaining text a strict subset by token, but
    it creates new n-grams where the two sides join, so the n-gram subset rule
    rejects it. This count separates those seam artifacts from real additions.
    """


@dataclass(frozen=True)
class Novelty:
    """How much of a member's text is absent from its representative."""

    novel_lines: int
    member_lines: int
    novel_line_ratio: float
    novel_boilerplate_lines: int
    novel_substantive_lines: int
    novel_tokens: int
    member_tokens: int
    novel_token_ratio: float
    novel_characters: int


@dataclass
class ClusterReview:
    """One cluster, its complete pair matrix, and every rule's removal set."""

    cluster_id: str
    documents: list[ClusterDocument]
    pairs: list[PairScore]
    bounded: set[int]
    removable: dict[str, set[int]] = field(default_factory=dict)


def score_pairs(documents: list[ClusterDocument], params: FuzzyVerificationParams) -> list[PairScore]:
    """Score every ordered pair of one cluster."""
    prepared = [prepare_verification_text(document.text, params) for document in documents]
    token_sets = [frozenset(document.text.casefold().split()) for document in documents]
    scores: list[PairScore] = []
    for member_index, member in enumerate(prepared):
        for representative_index, representative in enumerate(prepared):
            if member_index == representative_index:
                continue
            result = verify_prepared_candidate(member, representative, params)
            # The verifier calculates the two guard scores only when its own
            # control flow reaches them. The review needs them on every pair.
            char_jaccard = result.char_jaccard
            if char_jaccard is None and result.under_tokenized:
                char_jaccard = character_ngram_jaccard(
                    member.text, representative.text, params.under_tokenized_char_ngram_size
                )
            contained = result.normalized_token_sequence_contained
            if contained is None:
                contained = normalized_token_sequence_is_contained(member.text, representative.text)
            scores.append(
                PairScore(
                    member_index=member_index,
                    representative_index=representative_index,
                    accepted=result.accepted,
                    rejection=result.rejection.value if result.rejection else None,
                    member_containment=result.member_containment,
                    jaccard=result.jaccard,
                    char_jaccard=char_jaccard,
                    token_sequence_contained=contained,
                    saturated=result.saturated,
                    under_tokenized=result.under_tokenized,
                    member_chars=result.member_chars,
                    representative_chars=result.representative_chars,
                    member_unique_ngrams=result.member_unique_ngrams,
                    line_ratio=line_count_ratio(member.text, representative.text),
                    novel_tokens=len(token_sets[member_index] - token_sets[representative_index]),
                )
            )
    return scores


def bounded_replay(
    documents: list[ClusterDocument],
    pairs: list[PairScore],
    local_params: LocalRepresentativeParams = REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
) -> set[int]:
    """Replay the production verifier, including its comparison budget.

    Reproduces ``verify_fuzzy_dups``: the shuffle orders records by content ID,
    the anchor is the longest document of a bounded head, and each member gets
    at most ``maximum_comparisons_per_document`` comparisons -- the anchor,
    then LSH-bucket nominees under the near-equality local gate.
    """
    pair_by_key = {(pair.member_index, pair.representative_index): pair for pair in pairs}
    order = sorted(range(len(documents)), key=lambda index: (len(documents[index].id), documents[index].id, index))

    head: list[int] = []
    buffered = 0
    for index in order:
        head.append(index)
        buffered += len(documents[index].text)
        if len(head) >= ANCHOR_SCAN_RECORDS or buffered >= ANCHOR_SCAN_CHARS:
            break
    anchor = max(head, key=lambda index: (len(documents[index].text), documents[index].id))

    retained = [anchor]
    representative_chars = 0
    removed: set[int] = set()
    seen_ids = {documents[anchor].id}

    for index in order:
        if index == anchor or documents[index].id in seen_ids:
            # An equal content ID is byte-identical text; global exact dedup owns it.
            continue
        seen_ids.add(documents[index].id)

        primary = pair_by_key.get((index, anchor))
        matched = primary is not None and primary.accepted
        if not matched:
            member_buckets = set(documents[index].buckets)
            shared_counts = {
                position: shared
                for position, representative_index in enumerate(retained)
                if position > 0 and (shared := len(member_buckets & set(documents[representative_index].buckets)))
            }
            nominees = sorted(shared_counts, key=lambda position: (-shared_counts[position], position))
            for position in nominees[: local_params.maximum_comparisons_per_document - 1]:
                representative_index = retained[position]
                pair = pair_by_key.get((index, representative_index))
                if pair is None or not pair.accepted:
                    continue
                if documents[index].text.casefold().split() != documents[representative_index].text.casefold().split():
                    continue
                if pair.line_ratio < local_params.minimum_local_line_count_ratio:
                    continue
                matched = True
                break

        if matched:
            removed.add(index)
            continue
        chars = len(documents[index].text)
        if (
            len(retained) < local_params.maximum_representatives_per_cluster
            and chars <= local_params.maximum_local_representative_chars
            and representative_chars + chars <= local_params.maximum_local_representative_chars_per_cluster
        ):
            retained.append(index)
            representative_chars += chars
    return removed


def _subset_accepted(pair: PairScore) -> bool:
    return pair.member_containment >= 1.0 and pair.member_unique_ngrams == 0


def rule_production(pair: PairScore, params: FuzzyVerificationParams) -> bool:
    return pair.accepted


def rule_no_length_gate(pair: PairScore, params: FuzzyVerificationParams) -> bool:
    """Production rule without the member-not-longer requirement."""
    if not _subset_accepted(pair):
        return False
    if pair.saturated:
        return pair.token_sequence_contained
    if pair.under_tokenized:
        return (pair.char_jaccard or 0.0) >= params.under_tokenized_minimum_char_jaccard
    return True


def rule_no_saturation_gate(pair: PairScore, params: FuzzyVerificationParams) -> bool:
    """Production rule with the saturation guard removed only.

    A saturated pair keeps no extra guard, which makes this a strict relaxation.
    Applying the character guard instead would reject accepted pairs.
    """
    if pair.member_chars > pair.representative_chars or not _subset_accepted(pair):
        return False
    if pair.saturated:
        return True
    if pair.under_tokenized:
        return (pair.char_jaccard or 0.0) >= params.under_tokenized_minimum_char_jaccard
    return True


def rule_no_under_tokenized_gate(pair: PairScore, params: FuzzyVerificationParams) -> bool:
    """Production rule with the under-tokenized character guard removed only."""
    if pair.member_chars > pair.representative_chars or not _subset_accepted(pair):
        return False
    if pair.saturated:
        return pair.token_sequence_contained
    return True


def rule_subset_only(pair: PairScore, params: FuzzyVerificationParams) -> bool:
    """Pure 3-gram subset containment, with both extra guards removed."""
    return pair.member_chars <= pair.representative_chars and _subset_accepted(pair)


def seam_tolerant_rule(maximum_unique_ngrams: int) -> Callable[[PairScore, FuzzyVerificationParams], bool]:
    """Production rule, widened to accept a member that adds no new token.

    Cutting a block out of a document leaves a strict subset by token but joins
    two pieces that were apart, and the join makes n-grams the representative
    never held. The subset rule counts those as new information and keeps the
    document. Requiring an empty novel-token set keeps the information-
    preserving guarantee while it tolerates a bounded number of seams.
    """

    def rule(pair: PairScore, params: FuzzyVerificationParams) -> bool:
        if pair.member_chars > pair.representative_chars:
            return False
        if _subset_accepted(pair):
            return rule_production(pair, params) or rule_subset_only(pair, params)
        return pair.novel_tokens == 0 and pair.member_unique_ngrams <= maximum_unique_ngrams

    return rule


def containment_rule(threshold: float) -> Callable[[PairScore, FuzzyVerificationParams], bool]:
    def rule(pair: PairScore, params: FuzzyVerificationParams) -> bool:
        return pair.member_chars <= pair.representative_chars and pair.member_containment >= threshold

    return rule


def jaccard_rule(threshold: float) -> Callable[[PairScore, FuzzyVerificationParams], bool]:
    def rule(pair: PairScore, params: FuzzyVerificationParams) -> bool:
        return pair.jaccard >= threshold

    return rule


RULES: dict[str, Callable[[PairScore, FuzzyVerificationParams], bool]] = {
    "production": rule_production,
    "no_length_gate": rule_no_length_gate,
    "no_saturation_gate": rule_no_saturation_gate,
    "no_under_tokenized_gate": rule_no_under_tokenized_gate,
    "subset_only": rule_subset_only,
    "seam_tolerant_4": seam_tolerant_rule(4),
    "seam_tolerant_16": seam_tolerant_rule(16),
    "seam_tolerant_unbounded": seam_tolerant_rule(1 << 30),
    "containment_0.99": containment_rule(0.99),
    "containment_0.95": containment_rule(0.95),
    "containment_0.90": containment_rule(0.90),
    "containment_0.80": containment_rule(0.80),
    "jaccard_0.90": jaccard_rule(0.90),
    "jaccard_0.80": jaccard_rule(0.80),
    "jaccard_0.70": jaccard_rule(0.70),
}


def greedy_removals(documents: list[ClusterDocument], pairs: list[PairScore], accepts: Iterable[PairScore]) -> set[int]:
    """Documents removable under a rule, keeping one survivor per closed set.

    An accepted pair is a directed edge from the member to its representative.
    Removing every document that has an outgoing edge can delete a whole
    mutually-removable set, so the pass walks documents from the shortest text
    and removes one only while a representative of it still survives.
    """
    edges: dict[int, set[int]] = defaultdict(set)
    for pair in accepts:
        edges[pair.member_index].add(pair.representative_index)
    removed: set[int] = set()
    for index in sorted(range(len(documents)), key=lambda i: (len(documents[i].text), i)):
        if edges[index] - removed:
            removed.add(index)
    return removed


def review_cluster(
    cluster_id: str,
    documents: list[ClusterDocument],
    params: FuzzyVerificationParams,
    local_params: LocalRepresentativeParams = REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
) -> ClusterReview:
    """Score one cluster under every rule and replay the production budget."""
    pairs = score_pairs(documents, params)
    review = ClusterReview(
        cluster_id=cluster_id,
        documents=documents,
        pairs=pairs,
        bounded=bounded_replay(documents, pairs, local_params),
    )
    for name, rule in RULES.items():
        review.removable[name] = greedy_removals(documents, pairs, [pair for pair in pairs if rule(pair, params)])
    return review


def classify_document(review: ClusterReview, index: int) -> str:
    """Explain what the bounded production run did with document ``index``."""
    if index in review.bounded:
        return "removed"
    if index in review.removable["production"]:
        return "budget_blocked"
    member_pairs = [pair for pair in review.pairs if pair.member_index == index]
    best_containment = max((pair.member_containment for pair in member_pairs), default=0.0)
    best_jaccard = max((pair.jaccard for pair in member_pairs), default=0.0)
    if best_containment < NO_SIMILAR_MAX_CONTAINMENT and best_jaccard < NO_SIMILAR_MAX_JACCARD:
        return "no_similar"
    return "rule_blocked"


def is_boilerplate_line(line: str) -> bool:
    """True for a line that carries no document-specific information."""
    return any(pattern.match(line) for pattern in BOILERPLATE_PATTERNS)


def measure_novelty(member: str, representative: str) -> tuple[Novelty, list[str]]:
    """Return the novelty score of ``member`` and the lines only it holds.

    The verification rule answers whether a member is a subset. It does not
    answer how much text a removal loses, which is what a threshold change must
    be judged against.
    """
    member_lines = member.splitlines()
    representative_lines = {line.strip() for line in representative.splitlines()}
    novel = [line for line in member_lines if line.strip() not in representative_lines]
    substantive = [line for line in novel if not is_boilerplate_line(line)]

    member_tokens = member.casefold().split()
    representative_tokens = set(representative.casefold().split())
    novel_tokens = [token for token in member_tokens if token not in representative_tokens]

    return (
        Novelty(
            novel_lines=len(novel),
            member_lines=len(member_lines),
            novel_line_ratio=len(novel) / len(member_lines) if member_lines else 0.0,
            novel_boilerplate_lines=len(novel) - len(substantive),
            novel_substantive_lines=len(substantive),
            novel_tokens=len(novel_tokens),
            member_tokens=len(member_tokens),
            novel_token_ratio=len(novel_tokens) / len(member_tokens) if member_tokens else 0.0,
            novel_characters=sum(len(line) for line in novel),
        ),
        novel,
    )


def best_partner(review: ClusterReview, index: int) -> PairScore | None:
    """The pair that comes closest to removing document ``index``.

    Ranks by member containment, then by Jaccard, over the pairs that satisfy
    the rule's length direction. A relaxed containment threshold would use this
    representative.
    """
    candidates = [
        pair
        for pair in review.pairs
        if pair.member_index == index and pair.member_chars <= pair.representative_chars
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda pair: (pair.member_containment, pair.jaccard))


def unified_diff(member: str, representative: str, limit: int) -> str:
    """A bounded unified diff of the representative against the member."""
    lines = []
    for position, line in enumerate(
        difflib.unified_diff(
            representative.splitlines(),
            member.splitlines(),
            fromfile="representative",
            tofile="member",
            lineterm="",
            n=1,
        )
    ):
        if position >= limit:
            lines.append(f"... diff truncated at {limit} lines")
            break
        lines.append(line)
    return "\n".join(lines)
