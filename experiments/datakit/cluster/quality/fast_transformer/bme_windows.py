# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cut the begin/middle/end 512-token grading windows for the bme labeling scheme.

The scale-up labels windows rather than whole-document prefixes: a document over
``LONG_DOC_TOKENS`` gemma tokens contributes three disjoint 512-token windows
(first, middle, last), each its own training example; anything shorter
contributes a single begin window covering its first 512 tokens (the whole
document when it is under one window). The exact window text and its token
offsets travel with every grade, so nothing downstream re-derives what the
grader saw — the same principle as the PDF oracle sample's segment columns.

Tokenization is the gigatoken backend of the Gemma-3 tokenizer (parity-proven
BPE, ~7-8x faster than HF). Callers must gate a run on
:func:`check_gigatoken_parity` over a sample of their own corpus before trusting
the fast path, mirroring the fusion arm's per-run gate.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np

from experiments.datakit.cluster.quality.fast_transformer.data import load_gigatoken, load_tokenizer

logger = logging.getLogger(__name__)

GEMMA_TOKENIZER = "unsloth/gemma-3-270m-it"
WINDOW_TOKENS = 512
# At or under three windows the slices overlap, so only the begin window is graded.
LONG_DOC_TOKENS = 3 * WINDOW_TOKENS
WINDOW_POSITIONS = ("begin", "middle", "end")
PARITY_SAMPLE = 256


@dataclass(frozen=True)
class Window:
    """One grading window: its position, token offsets in the document, and text."""

    position: str  # begin / middle / end
    token_start: int
    token_end: int
    text: str


def encode_documents(texts: list[str], batch_size: int = 512) -> list[list[int]]:
    """Full-document gemma token ids (no truncation, no special tokens)."""
    tokenizer = load_gigatoken(GEMMA_TOKENIZER)
    ids: list[list[int]] = []
    for start in range(0, len(texts), batch_size):
        ids.extend(tokenizer(texts[start : start + batch_size], add_special_tokens=False)["input_ids"])
        if start and start % (batch_size * 50) == 0:
            logger.info("bme_windows: tokenized %d/%d documents", start, len(texts))
    return ids


def check_gigatoken_parity(texts: list[str], seed: int = 0) -> None:
    """Fail loudly unless gigatoken reproduces the HF gemma ids exactly on a sample."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(texts), size=min(PARITY_SAMPLE, len(texts)), replace=False)
    docs = [texts[i] for i in idx]
    hf = load_tokenizer(GEMMA_TOKENIZER)
    fast = load_gigatoken(GEMMA_TOKENIZER)
    t0 = time.time()
    hf_ids = hf(docs, add_special_tokens=False)["input_ids"]
    t1 = time.time()
    fast_ids = fast(docs, add_special_tokens=False)["input_ids"]
    t2 = time.time()
    mismatched = sum(1 for a, b in zip(hf_ids, fast_ids, strict=True) if list(a) != list(b))
    if mismatched:
        raise ValueError(f"gigatoken diverges from HF gemma on {mismatched}/{len(docs)} sampled documents")
    logger.info(
        "bme_windows parity: %d documents identical (hf %.2fs, gigatoken %.2fs, %.1fx)",
        len(docs),
        t1 - t0,
        t2 - t1,
        (t1 - t0) / max(t2 - t1, 1e-9),
    )


def doc_windows(token_ids: list[int]) -> list[Window]:
    """The grading windows of one document, with exact token offsets.

    Decoding is pinned off the cleanup pass for the same reason as the PDF oracle
    sample: the decoded window is both what the grader sees and what a scorer is
    later trained on, so it has to stay the document's own text.
    """
    tokenizer = load_tokenizer(GEMMA_TOKENIZER)

    def cut(position: str, start: int, end: int) -> Window:
        text = tokenizer.decode(token_ids[start:end], clean_up_tokenization_spaces=False)
        return Window(position=position, token_start=start, token_end=end, text=text)

    n = len(token_ids)
    if n <= LONG_DOC_TOKENS:
        return [cut("begin", 0, min(n, WINDOW_TOKENS))]
    middle = (n - WINDOW_TOKENS) // 2
    return [
        cut("begin", 0, WINDOW_TOKENS),
        cut("middle", middle, middle + WINDOW_TOKENS),
        cut("end", n - WINDOW_TOKENS, n),
    ]
