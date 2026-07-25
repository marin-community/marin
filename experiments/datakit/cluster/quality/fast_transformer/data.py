# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize and pack oracle-scored text for the fast-transformer.

Tokenizes text with a HuggingFace tokenizer, builds a compact vocabulary from the
training split (mirroring fasttext's ``minCount`` pruning so every embedding row is
actually trained and the table stays small), and packs into dense padded arrays.
"""

import functools
import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=8)
def load_tokenizer(tokenizer_name: str):
    """Load a HuggingFace tokenizer, memoized per process.

    ``AutoTokenizer.from_pretrained`` re-runs the slow→fast conversion of the
    250K-vocab tokenizer on every call; scoring calls this once per batch, so
    without the cache a many-batch shard reloads the tokenizer hundreds of times.
    """
    return AutoTokenizer.from_pretrained(tokenizer_name)


# Reserved compact ids. Real tokens are remapped to dense ids starting at 2.
PAD_ID = 0
UNK_ID = 1
NUM_RESERVED = 2


@dataclass(frozen=True)
class PackedSplit:
    """Dense padded token ids + regression targets for one split."""

    ids: np.ndarray  # [N, T] int32, PAD_ID padded on the right
    scores: np.ndarray  # [N] float32, normalized quality in [0, 1]

    @property
    def n(self) -> int:
        return int(self.ids.shape[0])


@dataclass(frozen=True)
class PackedData:
    train: PackedSplit
    eval: PackedSplit
    vocab_size: int  # compact vocab size, including PAD + UNK
    tokenizer_name: str
    max_tokens: int


def _encode(tokenizer, texts: list[str], max_tokens: int) -> list[list[int]]:
    """Tokenize *texts* (no special tokens), truncating to ``max_tokens``."""
    # Pre-truncate by characters to bound tokenizer work; ~8 chars/token is a
    # safe over-estimate so we never starve the max_tokens budget.
    char_cap = max_tokens * 8
    capped = [t[:char_cap] for t in texts]
    encoded = tokenizer(
        capped,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )["input_ids"]
    return encoded


def _build_vocab(train_ids: list[list[int]], min_count: int, max_vocab: int | None = None) -> dict[int, int]:
    """Map raw token ids seen >= ``min_count`` times to dense ids.

    ``max_vocab`` caps the table to the most frequent tokens (everything else maps
    to UNK). The cap matters for the NTP softmax, whose cost scales with vocab.
    """
    counts: Counter[int] = Counter()
    for row in train_ids:
        counts.update(row)
    frequent = [(tok, c) for tok, c in counts.items() if c >= min_count]
    if max_vocab is not None and len(frequent) > max_vocab:
        frequent = sorted(frequent, key=lambda tc: -tc[1])[:max_vocab]
    kept = sorted(tok for tok, _ in frequent)
    remap = {tok: i + NUM_RESERVED for i, tok in enumerate(kept)}
    logger.info(
        "vocab: %d raw tokens -> %d kept (min_count=%d, max_vocab=%s)", len(counts), len(kept), min_count, max_vocab
    )
    return remap


def _pack(raw_ids: list[list[int]], remap: dict[int, int], scores: np.ndarray, max_tokens: int) -> PackedSplit:
    n = len(raw_ids)
    ids = np.full((n, max_tokens), PAD_ID, dtype=np.int32)
    for i, row in enumerate(raw_ids):
        mapped = [remap.get(t, UNK_ID) for t in row[:max_tokens]]
        ids[i, : len(mapped)] = mapped
    return PackedSplit(ids=ids, scores=scores)


def encode_texts(tokenizer_name: str, texts: list[str], max_tokens: int) -> list[list[int]]:
    """Tokenize raw in-memory texts (no parquet read), truncating to ``max_tokens``."""
    return _encode(load_tokenizer(tokenizer_name), texts, max_tokens)


def build_remap(raw_ids: list[list[int]], min_count: int, max_vocab: int | None = None) -> dict[int, int]:
    return _build_vocab(raw_ids, min_count, max_vocab)


def pack(raw_ids: list[list[int]], remap: dict[int, int], scores: np.ndarray, max_tokens: int) -> PackedSplit:
    return _pack(raw_ids, remap, scores, max_tokens)
