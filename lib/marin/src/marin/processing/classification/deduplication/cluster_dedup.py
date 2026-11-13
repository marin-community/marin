# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the hero duplicate rule used by v11-c075-restored.

Reference run: https://github.com/marin-community/marin/pull/8405

The solver processes documents in descending character count. Input order
resolves equal lengths in that processing order. A surviving representative must meet the
directional word n-gram containment threshold to remove a member.

Small clusters compare each member with earlier survivors. Large clusters use
a rare-n-gram index with limits on posting lists and candidate counts. These
limits can miss duplicates within a cluster. The materializer can also split
large components with a MinHash key, which can separate containment pairs.
The production index includes member-only probes and applies its candidate cap
before it excludes removed documents. These limits are part of its output rule.
Common postings remain eligible when every posting exceeds the limit. Indexed
candidate ties use NumPy selection order, which can change across NumPy versions.
"""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import dupekit
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

_EMPTY = np.empty(0, dtype=np.uint64)


class ClusterDedupParams(BaseModel):
    """Thresholds and work bounds for one cluster."""

    model_config = ConfigDict(frozen=True)

    rule_version: Literal["containment_ngram3_v2"] = "containment_ngram3_v2"
    ngram_size: int = Field(default=3, ge=1)
    minimum_containment: float = Field(default=0.75, ge=0, le=1)
    """Minimum fraction of member n-grams that a representative must contain."""

    exact_scan_maximum: int = Field(default=256, ge=2)
    """Clusters with at most this many members use an exact scan."""

    probe_ngrams: int = Field(default=32, ge=1)
    """How many of a member's rarest n-grams probe the inverted index."""

    maximum_posting_length: int = Field(default=512, ge=1)
    """Preferred maximum posting size. If all exceed it, use all postings."""

    maximum_candidates: int = Field(default=32, ge=1)
    """Candidate limit before the solver excludes removed representatives."""


@dataclass(frozen=True)
class _PreparedDocument:
    index: int
    chars: int
    ngrams: np.ndarray


@dataclass(frozen=True)
class Removal:
    member_index: int
    representative_index: int
    containment: float
    jaccard: float
    novel_tokens: int
    comparisons: int
    """Comparisons for this member through its first accepted representative."""


@dataclass(frozen=True)
class CandidateDuplicate:
    """One possible duplicate pair with its prepared n-gram hashes."""

    member_index: int
    representative_index: int
    member_ngrams: np.ndarray
    representative_ngrams: np.ndarray


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


def _prepare(documents: Sequence[str], params: ClusterDedupParams) -> list[_PreparedDocument]:
    prepared = []
    for index, document in enumerate(documents):
        prepared.append(
            _PreparedDocument(
                index=index,
                chars=len(document),
                ngrams=ngram_hashes(document, params.ngram_size),
            )
        )
    return prepared


def _novel_token_count(
    member_index: int,
    representative_index: int,
    documents: Sequence[str],
    cache: dict[int, frozenset[str]],
) -> int:
    """Words of the member that the representative does not hold."""
    for index in (member_index, representative_index):
        if index not in cache:
            cache[index] = frozenset(documents[index].casefold().split())
    return len(cache[member_index] - cache[representative_index])


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


def _build_index(prepared: list[_PreparedDocument]) -> _NgramIndex:
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
    member: _PreparedDocument,
    index: _NgramIndex,
    rank: np.ndarray,
    params: ClusterDedupParams,
) -> np.ndarray:
    """Select candidates by shared probe count, as in the production rule."""
    if member.ngrams.size == 0:
        return np.empty(0, dtype=np.int32)
    position = np.searchsorted(index.values, member.ngrams)

    counts = index.counts[position]
    usable = counts <= params.maximum_posting_length
    if np.any(usable):
        position, counts = position[usable], counts[usable]
    if position.size > params.probe_ngrams:
        rarest = np.argpartition(counts, params.probe_ngrams)[: params.probe_ngrams]
        position, counts = position[rarest], counts[rarest]

    total = int(counts.sum())
    offsets = np.repeat(index.starts[position], counts)
    within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    owners = index.owners[offsets + within]

    # Count only the probed documents to prevent a cluster-sized allocation
    # for every member.
    distinct, shared = np.unique(owners, return_counts=True)
    ahead = rank[distinct] < rank[member.index]
    distinct, shared = distinct[ahead], shared[ahead]
    if distinct.size == 0:
        return np.empty(0, dtype=np.int32)
    if distinct.size > params.maximum_candidates:
        strongest = np.argpartition(-shared, params.maximum_candidates - 1)[: params.maximum_candidates]
        distinct, shared = distinct[strongest], shared[strongest]
    return distinct[np.argsort(-shared)].astype(np.int32)


def find_candidate_duplicates(
    documents: Sequence[str],
    params: ClusterDedupParams,
) -> Iterator[tuple[CandidateDuplicate, ...]]:
    """Find possible representative pairs for each member.

    Representatives have at least as many characters as their members.
    Input order resolves equal-length processing ties. Indexed candidates use
    shared probe counts, including NumPy tie order, to order representatives.

    Args:
        documents: Text for each cluster member.
        params: Duplicate-rule thresholds and work bounds.

    Yields:
        One tuple for each member that has possible representatives. Each tuple
        has the representatives in comparison order.
    """
    prepared = _prepare(documents, params)
    order = sorted(range(len(prepared)), key=lambda index: (-prepared[index].chars, index))
    rank = np.empty(len(prepared), dtype=np.int64)
    rank[order] = np.arange(len(order))
    ngram_index = _build_index(prepared) if len(prepared) > params.exact_scan_maximum else None

    for position, member in enumerate(order):
        member_prepared = prepared[member]
        if member_prepared.ngrams.size == 0:
            continue
        candidates = (
            order[:position]
            if ngram_index is None
            else _index_candidates(member_prepared, ngram_index, rank, params).tolist()
        )
        candidate_duplicates: list[CandidateDuplicate] = []
        for representative in candidates:
            other = prepared[representative]
            candidate_duplicates.append(
                CandidateDuplicate(
                    member_index=member,
                    representative_index=representative,
                    member_ngrams=member_prepared.ngrams,
                    representative_ngrams=other.ngrams,
                )
            )
        if candidate_duplicates:
            yield tuple(candidate_duplicates)


def resolve_duplicate_clusters(
    documents: Sequence[str],
    candidate_groups: Iterable[Sequence[CandidateDuplicate]],
    params: ClusterDedupParams,
) -> list[Removal]:
    """Select cluster representatives from scored candidate groups.

    Args:
        documents: Text for each cluster member.
        candidate_groups: Groups in member processing order. Each group holds
            one member's candidates in representative comparison order.
        params: Duplicate-rule thresholds and work bounds.

    Returns:
        The removed members and their selected representatives.
    """
    removed = np.zeros(len(documents), dtype=bool)
    removals: list[Removal] = []
    token_cache: dict[int, frozenset[str]] = {}
    order = sorted(range(len(documents)), key=lambda index: (-len(documents[index]), index))
    rank = np.empty(len(documents), dtype=np.int64)
    rank[order] = np.arange(len(order))
    previous_member_rank = -1
    for candidates in candidate_groups:
        if not candidates:
            raise ValueError("Candidate groups must not be empty")
        member_index = candidates[0].member_index
        member_rank = rank[member_index]
        if member_rank <= previous_member_rank:
            raise ValueError("Candidate groups must follow member processing order")
        previous_member_rank = member_rank

        comparisons = 0
        for candidate in candidates:
            if candidate.member_index != member_index:
                raise ValueError("A candidate group must contain one member")
            if rank[candidate.representative_index] >= member_rank:
                raise ValueError("A representative must occur before its member")
            if removed[candidate.representative_index]:
                continue
            comparisons += 1
            shared = _overlap(candidate.member_ngrams, candidate.representative_ngrams)
            containment = shared / candidate.member_ngrams.size
            if containment < params.minimum_containment:
                continue
            union = candidate.member_ngrams.size + candidate.representative_ngrams.size - shared
            removed[candidate.member_index] = True
            removals.append(
                Removal(
                    member_index=candidate.member_index,
                    representative_index=candidate.representative_index,
                    containment=containment,
                    jaccard=shared / union if union else 1.0,
                    novel_tokens=_novel_token_count(
                        candidate.member_index,
                        candidate.representative_index,
                        documents,
                        token_cache,
                    ),
                    comparisons=comparisons,
                )
            )
            break
    return removals
