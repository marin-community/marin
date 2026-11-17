# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Character n-gram spelling features (``spelling_ngram_source``).

Each token's spelling is normalised as in ``spelling_bytes`` (leading spaces stripped, lowercased), wrapped in
word-boundary markers, and split into its set of character 1- to ``MAX_NGRAM``-grams. Only n-grams that occur in at
least ``MIN_NGRAM_TOKENS`` tokens of the vocabulary get a table row; the rest are dropped. Bytes that are not whole
UTF-8 characters (a token holding part of a multi-byte character) each count as their own character.
"""

import collections
import dataclasses
import functools

import numpy as np

from experiments.grug.fast_track.spelling_bytes import normalised_spelling, stored_surfaces

MAX_NGRAM = 4
MIN_NGRAM_TOKENS = 2
# Word-boundary markers; characters are their (non-negative) code points.
_WORD_START = -1
_WORD_END = -2

Ngram = tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class NgramTables:
    """Per-token n-gram rows, for a vocabulary of ``V`` tokens with at most ``K`` kept n-grams each."""

    ngram_ids: np.ndarray  # [V, K] int32, the token's kept n-gram rows (0 in unused slots)
    ngram_present: np.ndarray  # [V, K] float32, 1 where the slot holds an n-gram
    num_ngrams: int


def spelling_ngrams(surface: bytes) -> frozenset[Ngram]:
    """The character 1- to ``MAX_NGRAM``-grams of ``surface``'s normalised, boundary-marked spelling; empty for a
    token whose normalised spelling is empty."""
    spelling, _, _ = normalised_spelling(surface)
    if not spelling:
        return frozenset()
    chars = [_WORD_START, *map(ord, spelling.decode("utf-8", errors="surrogateescape")), _WORD_END]
    return frozenset(
        tuple(chars[start : start + n]) for n in range(1, MAX_NGRAM + 1) for start in range(len(chars) - n + 1)
    )


def ngram_tables_from_surfaces(surfaces: list[bytes]) -> NgramTables:
    per_token = [spelling_ngrams(surface) for surface in surfaces]
    counts = collections.Counter(ngram for ngrams in per_token for ngram in ngrams)
    kept = sorted(ngram for ngram, count in counts.items() if count >= MIN_NGRAM_TOKENS)
    row_of = {ngram: row for row, ngram in enumerate(kept)}
    rows = [sorted(row_of[ngram] for ngram in ngrams if ngram in row_of) for ngrams in per_token]
    width = max((len(r) for r in rows), default=0)
    ngram_ids = np.zeros((len(surfaces), width), np.int32)
    ngram_present = np.zeros((len(surfaces), width), np.float32)
    for i, r in enumerate(rows):
        ngram_ids[i, : len(r)] = r
        ngram_present[i, : len(r)] = 1
    return NgramTables(ngram_ids, ngram_present, len(kept))


@functools.cache
def ngram_tables(vocab_size: int) -> NgramTables:
    """The tables of the first ``vocab_size`` tokens of ``spelling_bytes.TOKEN_BYTES_FILE``."""
    surfaces = stored_surfaces()
    if vocab_size > len(surfaces):
        raise ValueError(f"spelling_ngram_source: vocab_size {vocab_size} exceeds the {len(surfaces)} stored tokens")
    return ngram_tables_from_surfaces(surfaces[:vocab_size])
