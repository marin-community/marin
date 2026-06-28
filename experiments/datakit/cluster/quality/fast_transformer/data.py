# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize and pack the LLM-scored quality parquets for the fast-transformer.

Reads the same scored parquets the fasttext baseline uses
(:mod:`experiments.datakit.cluster.quality.v0.score` output), tokenizes the
text with a HuggingFace tokenizer, builds a compact vocabulary from the
training split (mirroring fasttext's ``minCount`` pruning so every embedding
row is actually trained and the table stays small), and packs everything into
dense padded arrays ready for JAX.

The packed arrays are cached to ``cache_dir`` keyed by the tokenizer, sequence
length, and ``min_count`` so an architecture sweep tokenizes once.
"""

import hashlib
import json
import logging
import math
import os
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem import url_to_fs
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Reserved compact ids. Real tokens are remapped to dense ids starting at 2.
PAD_ID = 0
UNK_ID = 1
NUM_RESERVED = 2


@dataclass(frozen=True)
class PackedSplit:
    """Dense padded token ids + regression targets for one split."""

    ids: np.ndarray  # [N, T] int32, PAD_ID padded on the right
    lengths: np.ndarray  # [N] int32, real (pre-pad) token count, clipped to T
    scores: np.ndarray  # [N] float32, normalized quality in [0, 1]
    sources: list[str]

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


def _read_scored(path: str) -> tuple[list[str], np.ndarray, list[str]]:
    """Return (texts, normalized_scores, sources), filtered like the baseline.

    Drops rows the fasttext pipeline also drops: refused/invalid oracle scores
    (``score_raw < 0``), missing/NaN normalized scores, and empty text.
    """
    fs, resolved = url_to_fs(path)
    with fs.open(resolved, "rb") as fh:
        table = pq.read_table(fh)
    cols = {n: table.column(n).to_pylist() for n in ("source", "text", "score_raw", "score_normalized")}
    texts: list[str] = []
    scores: list[float] = []
    sources: list[str] = []
    for src, text, raw, norm in zip(
        cols["source"], cols["text"], cols["score_raw"], cols["score_normalized"], strict=True
    ):
        if raw is None or int(raw) < 0:
            continue
        if norm is None or (isinstance(norm, float) and math.isnan(norm)):
            continue
        if not text:
            continue
        texts.append(str(text))
        scores.append(float(norm))
        sources.append(str(src))
    return texts, np.asarray(scores, dtype=np.float32), sources


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


def _build_vocab(train_ids: list[list[int]], min_count: int) -> dict[int, int]:
    """Map raw token ids seen >= ``min_count`` times in train to dense ids."""
    counts: Counter[int] = Counter()
    for row in train_ids:
        counts.update(row)
    kept = sorted(tok for tok, c in counts.items() if c >= min_count)
    remap = {tok: i + NUM_RESERVED for i, tok in enumerate(kept)}
    logger.info("vocab: %d raw tokens -> %d kept (min_count=%d)", len(counts), len(kept), min_count)
    return remap


def _pack(
    raw_ids: list[list[int]], remap: dict[int, int], scores: np.ndarray, sources: list[str], max_tokens: int
) -> PackedSplit:
    n = len(raw_ids)
    ids = np.full((n, max_tokens), PAD_ID, dtype=np.int32)
    lengths = np.zeros(n, dtype=np.int32)
    for i, row in enumerate(raw_ids):
        mapped = [remap.get(t, UNK_ID) for t in row[:max_tokens]]
        ids[i, : len(mapped)] = mapped
        lengths[i] = len(mapped)
    return PackedSplit(ids=ids, lengths=lengths, scores=scores, sources=sources)


def _cache_key(train_path: str, eval_path: str, tokenizer_name: str, max_tokens: int, min_count: int) -> str:
    blob = json.dumps([train_path, eval_path, tokenizer_name, max_tokens, min_count], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def encode_corpus(
    tokenizer_name: str, parquet_path: str, max_tokens: int
) -> tuple[list[list[int]], np.ndarray, list[str]]:
    """Read + tokenize one oracle-schema parquet (no vocab remap yet)."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    texts, scores, sources = _read_scored(parquet_path)
    return _encode(tokenizer, texts, max_tokens), scores, sources


def build_remap(raw_ids: list[list[int]], min_count: int) -> dict[int, int]:
    return _build_vocab(raw_ids, min_count)


def pack(
    raw_ids: list[list[int]], remap: dict[int, int], scores: np.ndarray, sources: list[str], max_tokens: int
) -> PackedSplit:
    return _pack(raw_ids, remap, scores, sources, max_tokens)


def load_packed(
    *,
    train_path: str,
    eval_path: str,
    tokenizer_name: str,
    max_tokens: int,
    min_count: int,
    cache_dir: str,
) -> PackedData:
    """Tokenize + pack both splits, caching the dense arrays under ``cache_dir``."""
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(train_path, eval_path, tokenizer_name, max_tokens, min_count)
    cache_path = os.path.join(cache_dir, f"packed-{key}.npz")
    if os.path.exists(cache_path):
        logger.info("loading packed cache %s", cache_path)
        z = np.load(cache_path, allow_pickle=True)
        return PackedData(
            train=PackedSplit(z["tr_ids"], z["tr_len"], z["tr_score"], list(z["tr_src"])),
            eval=PackedSplit(z["ev_ids"], z["ev_len"], z["ev_score"], list(z["ev_src"])),
            vocab_size=int(z["vocab_size"]),
            tokenizer_name=tokenizer_name,
            max_tokens=max_tokens,
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tr_texts, tr_scores, tr_src = _read_scored(train_path)
    ev_texts, ev_scores, ev_src = _read_scored(eval_path)
    logger.info("read %d train / %d eval scored rows", len(tr_texts), len(ev_texts))

    tr_raw = _encode(tokenizer, tr_texts, max_tokens)
    ev_raw = _encode(tokenizer, ev_texts, max_tokens)
    remap = _build_vocab(tr_raw, min_count)
    vocab_size = len(remap) + NUM_RESERVED

    train = _pack(tr_raw, remap, tr_scores, tr_src, max_tokens)
    eval_split = _pack(ev_raw, remap, ev_scores, ev_src, max_tokens)
    logger.info(
        "packed: train ids %s (median len %d), eval ids %s, vocab %d",
        train.ids.shape,
        int(np.median(train.lengths)),
        eval_split.ids.shape,
        vocab_size,
    )

    np.savez_compressed(
        cache_path,
        tr_ids=train.ids,
        tr_len=train.lengths,
        tr_score=train.scores,
        tr_src=np.asarray(train.sources, dtype=object),
        ev_ids=eval_split.ids,
        ev_len=eval_split.lengths,
        ev_score=eval_split.scores,
        ev_src=np.asarray(eval_split.sources, dtype=object),
        vocab_size=vocab_size,
    )
    return PackedData(
        train=train,
        eval=eval_split,
        vocab_size=vocab_size,
        tokenizer_name=tokenizer_name,
        max_tokens=max_tokens,
    )
