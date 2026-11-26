# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Encode, window and pack oracle-scored text for the fast-transformer.

Token ids come from the datakit tokenize encoding path
(:func:`marin.processing.tokenize._core.text_preprocessor`), so the ids the
trainer sees for a label text are byte-identical to the ids the tokenize stage
writes for the same text (BOS/EOS included). The stage scores those stored ids
directly; this module builds a compact vocabulary from the training split
(mirroring fasttext's ``minCount`` pruning so every embedding row is actually
trained and the table stays small), slices begin/middle/end token windows, and
packs into dense padded arrays.
"""

import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.tokenizers import load_tokenizer
from marin.processing.tokenize._core import text_preprocessor

logger = logging.getLogger(__name__)


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


def encode_texts(tokenizer_name: str, texts: list[str]) -> list[list[int]]:
    """Encode in-memory texts exactly as the datakit tokenize stage does (untruncated).

    ``tokenizer_name`` is a hub name or a local tokenizer dir.
    """
    proc = text_preprocessor(TextLmDatasetFormat(), load_tokenizer(tokenizer_name))
    return [r["input_ids"] for r in proc([{"text": t} for t in texts])]


def bme_windows(ids: np.ndarray, max_tokens: int) -> list[np.ndarray]:
    """Begin/middle/end ``max_tokens`` windows of one document's token ids.

    A doc of at most ``max_tokens`` ids is a single window. Longer docs give three
    windows of exactly ``max_tokens`` (``max_tokens`` is even). Windows are copies
    so a caller can drop the Arrow batch the ids were read from.
    """
    n = len(ids)
    if n <= max_tokens:
        return [ids[:n].copy()]
    m = n // 2
    half = max_tokens // 2
    return [ids[:max_tokens].copy(), ids[m - half : m + half].copy(), ids[-max_tokens:].copy()]


def remap_table(remap: dict[int, int]) -> np.ndarray:
    """Dense lookup table for ``remap``; unknown ids map to ``UNK_ID``.

    The last row is a guaranteed ``UNK_ID`` sentinel, so callers clamp with
    ``table[np.minimum(raw, len(table) - 1)]`` and any raw id above the largest
    known one lands there.
    """
    size = max(remap) + 2
    table = np.full(size, UNK_ID, dtype=np.int32)
    table[list(remap)] = list(remap.values())
    return table


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


def build_remap(raw_ids: list[list[int]], min_count: int, max_vocab: int | None = None) -> dict[int, int]:
    return _build_vocab(raw_ids, min_count, max_vocab)


def pack(raw_ids: list[list[int]], remap: dict[int, int], scores: np.ndarray, max_tokens: int) -> PackedSplit:
    return _pack(raw_ids, remap, scores, max_tokens)
