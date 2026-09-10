# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Find duplicates inside a materialized fuzzy-duplicate candidate cluster.

The shipped verifier compares each cluster member with at most two
representatives, so roughly half the removals its own rule authorizes are never
found. Given the cluster text grouped in one place, the whole cluster can be
solved instead of sampled.

A document is a duplicate when a *longer* surviving document already holds its
content. Containment of word n-grams measures that directly, and the direction
matters: a symmetric score such as Jaccard calls two documents duplicates when
each holds text the other lacks, and removing either loses information.

Cluster sizes are heavily skewed. Small clusters take an exact all-pairs scan.
Large ones generate candidate pairs from a rare-n-gram inverted index, which
needs no extra input and keeps the work near-linear.

The solver is complete only within the cluster group it receives. The
materializer can partition very large connected components with a MinHash key;
that bound can separate containment pairs with low Jaccard similarity.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import dupekit
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

_EMPTY = np.empty(0, dtype=np.uint64)


class ClusterDedupParams(BaseModel):
    """Thresholds and work bounds for one cluster."""

    model_config = ConfigDict(frozen=True)

    ngram_size: int = Field(default=3, ge=1)
    minimum_containment: float = Field(default=0.60, ge=0, le=1)
    """Share of the member's n-grams the representative must already hold.

    Calibrated against 312 blind-labeled candidate pairs, stratified over five
    content types. A near-duplicate survives light editing -- a changed word, a
    curly quote, reflowed markup -- but every such edit destroys the n-grams
    that span it, so a threshold near 1.0 rejects genuine duplicates. Verified
    duplicates sit at 0.62 to 0.81, while pairs that connected components merged
    by transitive closure sit near 0. At 0.60 the rule keeps 87.7% of true
    duplicates against 11.3% for a 1.0 threshold, and 1.5% of what it accepts is
    a genuinely different document.
    """

    exact_scan_maximum: int = Field(default=256, ge=2)
    """Below this member count a cluster takes an exhaustive ordered-pair scan."""

    probe_ngrams: int = Field(default=32, ge=1)
    """How many of a member's rarest n-grams probe the inverted index."""

    maximum_posting_length: int = Field(default=512, ge=1)
    """N-grams held by more documents than this are boilerplate and are skipped."""

    maximum_candidates: int = Field(default=32, ge=1)
    """Maximum representatives checked for one member in the index path."""


@dataclass(frozen=True)
class ClusterDocument:
    id: str
    text: str


@dataclass(frozen=True)
class PreparedDocument:
    """N-gram hashes and the sizes the rule compares."""

    index: int
    chars: int
    ngrams: np.ndarray
    token_hashes: np.ndarray
    text: str


@dataclass(frozen=True)
class Removal:
    """One duplicate decision."""

    member_index: int
    representative_index: int
    containment: float
    jaccard: float
    novel_tokens: int
    comparisons: int


def ngram_hashes(text: str, ngram_size: int) -> np.ndarray:
    """Sorted unique 64-bit hashes of the case-folded word n-grams.

    A 64-bit collision is unlikely for the bounded clusters this solver reads,
    so the hash array stands in for the n-gram set.
    """
    tokens = text.casefold().split()
    if not tokens:
        return _EMPTY
    if len(tokens) < ngram_size:
        shingles = [" ".join(tokens).encode("utf-8", "surrogatepass")]
    else:
        shingles = [
            " ".join(tokens[start : start + ngram_size]).encode("utf-8", "surrogatepass")
            for start in range(len(tokens) - ngram_size + 1)
        ]
    return np.unique(np.asarray(dupekit.hash_xxh3_64_batch(shingles), dtype=np.uint64))


def prepare(documents: Sequence[ClusterDocument], params: ClusterDedupParams) -> list[PreparedDocument]:
    prepared = []
    for index, document in enumerate(documents):
        prepared.append(
            PreparedDocument(
                index=index,
                chars=len(document.text),
                ngrams=ngram_hashes(document.text, params.ngram_size),
                token_hashes=ngram_hashes(document.text, 1),
                text=document.text,
            )
        )
    return prepared


def novel_token_count(
    member: PreparedDocument,
    representative: PreparedDocument,
) -> int:
    """Words of the member that the representative does not hold."""
    return int(member.token_hashes.size) - _overlap(member.token_hashes, representative.token_hashes)


def _overlap(left: np.ndarray, right: np.ndarray) -> int:
    """Size of the intersection of two sorted unique hash arrays."""
    if left.size == 0 or right.size == 0:
        return 0
    if left.size > right.size:
        left, right = right, left
    position = np.searchsorted(right, left)
    position[position >= right.size] = right.size - 1
    return int(np.count_nonzero(right[position] == left))


@dataclass(frozen=True)
class _NgramIndex:
    """Inverted index from n-gram hash to the documents that hold it."""

    values: np.ndarray
    counts: np.ndarray
    starts: np.ndarray
    owners: np.ndarray


def _build_index(prepared: list[PreparedDocument]) -> _NgramIndex:
    sizes = np.fromiter((document.ngrams.size for document in prepared), dtype=np.int64, count=len(prepared))
    values = np.concatenate([document.ngrams for document in prepared]) if sizes.sum() else _EMPTY
    owners = np.repeat(np.arange(len(prepared), dtype=np.int32), sizes)
    order = np.argsort(values, kind="stable")
    values = values[order]
    owners = owners[order]
    if values.size == 0:
        return _NgramIndex(values=values, counts=_EMPTY.astype(np.int64), starts=_EMPTY.astype(np.int64), owners=owners)
    # ``np.unique`` would sort this array a second time, which dominated the
    # stage on a cluster holding a hundred million n-grams. The array is
    # already sorted, so the run boundaries are one comparison per element.
    boundary = np.empty(values.size, dtype=bool)
    boundary[0] = True
    np.not_equal(values[1:], values[:-1], out=boundary[1:])
    starts = np.flatnonzero(boundary)
    counts = np.diff(np.append(starts, values.size))
    return _NgramIndex(values=values[starts], counts=counts, starts=starts, owners=owners)


def _index_candidates(
    member: PreparedDocument,
    index: _NgramIndex,
    rank: np.ndarray,
    params: ClusterDedupParams,
) -> np.ndarray:
    """Documents that share the member's rarest n-grams, best first.

    Probing with the rarest n-grams is what keeps this near-linear. A member
    whose containment clears the threshold shares almost every n-gram it has,
    so each rare probe finds the representative with high probability, while a
    common n-gram would return most of the cluster and carry no signal.
    """
    if member.ngrams.size == 0:
        return np.empty(0, dtype=np.int32)
    position = np.searchsorted(index.values, member.ngrams)
    position[position >= index.values.size] = index.values.size - 1
    present = index.values[position] == member.ngrams
    position = position[present]
    if position.size == 0:
        return np.empty(0, dtype=np.int32)

    counts = index.counts[position]
    usable = counts <= params.maximum_posting_length
    position, counts = position[usable], counts[usable]
    if position.size == 0:
        return np.empty(0, dtype=np.int32)
    if position.size > params.probe_ngrams:
        rarest = np.argpartition(counts, params.probe_ngrams)[: params.probe_ngrams]
        position, counts = position[rarest], counts[rarest]

    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int32)
    offsets = np.repeat(index.starts[position], counts)
    within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    owners = index.owners[offsets + within]

    # Tally the probed documents only. Counting into an array the size of the
    # cluster would cost one allocation per member, which is quadratic in the
    # member count and stalls on a cluster of 100,000.
    distinct, shared = np.unique(owners, return_counts=True)
    # Only a document that ranks ahead of the member -- longer, or equal and
    # earlier -- can be its representative.
    ahead = rank[distinct] < rank[member.index]
    distinct, shared = distinct[ahead], shared[ahead]
    if distinct.size == 0:
        return np.empty(0, dtype=np.int32)
    if distinct.size > params.maximum_candidates:
        strongest = np.lexsort((rank[distinct], -shared))[: params.maximum_candidates]
        distinct = distinct[strongest]
    ranked = distinct[np.argsort(rank[distinct])]
    return ranked.astype(np.int32)


def _candidate_pairs(
    prepared: list[PreparedDocument], order: list[int], params: ClusterDedupParams
) -> dict[int, list[int]]:
    """Map each member to the representatives worth an exact comparison.

    A representative must be at least as long as the member, which is the
    direction the rule accepts, so only earlier-ranked documents ever appear.
    """
    if len(prepared) <= params.exact_scan_maximum:
        rank = {index: position for position, index in enumerate(order)}
        return {index: [other for other in order if rank[other] < rank[index]] for index in order}

    rank = np.empty(len(prepared), dtype=np.int64)
    for position, index in enumerate(order):
        rank[index] = position
    ngram_index = _build_index(prepared)
    return {index: _index_candidates(prepared[index], ngram_index, rank, params).tolist() for index in order}


def find_duplicates(
    documents: Sequence[ClusterDocument],
    params: ClusterDedupParams,
) -> list[Removal]:
    """Solve one cluster: every member that a longer survivor already holds.

    Walks members from the longest, so a representative is always decided
    before the documents it can absorb, and a removed document never becomes
    the representative of another.
    """
    prepared = prepare(documents, params)
    order = sorted(range(len(prepared)), key=lambda index: (-prepared[index].chars, documents[index].id, index))
    candidates = _candidate_pairs(prepared, order, params)

    removed: set[int] = set()
    removals: list[Removal] = []
    for member in order:
        comparisons = 0
        member_prepared = prepared[member]
        if member_prepared.ngrams.size == 0:
            continue
        for representative in candidates[member]:
            if representative in removed:
                continue
            other = prepared[representative]
            comparisons += 1
            shared = _overlap(member_prepared.ngrams, other.ngrams)
            containment = shared / member_prepared.ngrams.size
            if containment < params.minimum_containment:
                continue
            novel_tokens = novel_token_count(member_prepared, other)
            union = member_prepared.ngrams.size + other.ngrams.size - shared
            removed.add(member)
            removals.append(
                Removal(
                    member_index=member,
                    representative_index=representative,
                    containment=containment,
                    jaccard=shared / union if union else 1.0,
                    novel_tokens=novel_tokens,
                    comparisons=comparisons,
                )
            )
            break
    return removals
