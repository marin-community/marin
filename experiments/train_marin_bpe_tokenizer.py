# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train a byte-level BPE tokenizer with Llama-3 rules, then derive smaller vocabularies by
truncating the ordered merge list to a prefix.

Only the largest vocabulary is trained; 64k/32k/16k are the first ``K`` merges of the 128k
merge sequence (``K = target_vocab - 256 base bytes - num_specials``), so the vocabularies are
strictly nested (16k subset of 32k subset of 64k subset of 128k). This is valid because BPE
applies its merges greedily in order, so a prefix of the merge list is itself a complete BPE.

Llama-3 rules: the cl100k/Llama-3 pre-tokenizer split regex (contraction + word + number +
punctuation + whitespace classes, digits grouped in runs of 1-3), byte-level alphabet (no UNK),
and the Llama-3 special tokens.

    python -m experiments.train_marin_bpe_tokenizer --selftest         # local, no data
    python -m experiments.train_marin_bpe_tokenizer --corpus-glob '...' --out-dir ...
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator, Sequence

from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers.models import BPE

logger = logging.getLogger(__name__)

# cl100k / Llama-3 pre-tokenizer split. Digits are grouped in runs of up to 3 ("llama3 rules for
# digits"); the case-insensitive contraction set and the punctuation/whitespace classes match
# Llama-3's tokenizer.
LLAMA3_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# Core Llama-3 / Marin special tokens (kept small so they do not dominate the 16k budget). Marin
# renames two reserved slots to the think tokens (see experiments/marin_tokenizer.py).
SPECIAL_TOKENS: tuple[str, ...] = (
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|start_think|>",
    "<|end_think|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|pad|>",
)
BASE_ALPHABET_SIZE = 256  # byte-level: all 256 byte characters are always in the vocab
VOCAB_SIZES: tuple[int, ...] = (16384, 32768, 65536, 131072)  # 16k / 32k / 64k / 128k (powers of two)


def build_untrained() -> Tokenizer:
    """A byte-level BPE tokenizer with the Llama-3 split pre-tokenizer and no UNK."""
    tokenizer = Tokenizer(BPE(byte_fallback=False, unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(pattern=Regex(LLAMA3_SPLIT_PATTERN), behavior="isolated", invert=False),
            # Split already applied the regex, so ByteLevel just maps bytes to its alphabet.
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def train(corpus: Iterator[str], vocab_size: int, special_tokens: Sequence[str] = SPECIAL_TOKENS) -> Tokenizer:
    """Train a byte-level BPE at ``vocab_size`` on ``corpus``."""
    tokenizer = build_untrained()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        # Force all 256 byte characters into the base vocab so the tokenizer never emits UNK.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=list(special_tokens),
        show_progress=False,
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    return tokenizer


def truncate(tokenizer_json: dict, target_vocab: int, num_specials: int) -> dict:
    """Return a tokenizer.json dict with the vocab truncated to ``target_vocab``.

    The BPE trainer assigns ids as [specials, 256 byte chars, then one per merge in order], so the
    first ``target_vocab`` ids are exactly the specials + bytes + the first ``K`` merges. Keeping
    ids < target_vocab and the first ``K`` merges yields a valid nested BPE.
    """
    model = tokenizer_json["model"]
    keep_merges = target_vocab - BASE_ALPHABET_SIZE - num_specials
    if keep_merges < 0:
        raise ValueError(f"target_vocab {target_vocab} too small for {num_specials} specials + 256 bytes")
    new = json.loads(json.dumps(tokenizer_json))  # deep copy
    new["model"]["vocab"] = {tok: i for tok, i in model["vocab"].items() if i < target_vocab}
    new["model"]["merges"] = model["merges"][:keep_merges]
    # added_tokens carry the specials; all have ids < target_vocab, keep those that survive.
    new["added_tokens"] = [t for t in tokenizer_json.get("added_tokens", []) if t["id"] < target_vocab]
    if len(new["model"]["vocab"]) != target_vocab:
        raise ValueError(f"expected {target_vocab} vocab entries, got {len(new['model']['vocab'])}")
    Tokenizer.from_str(json.dumps(new))  # validate it parses/builds
    return new


def _selftest() -> None:
    logging.basicConfig(level=logging.INFO)
    base = [
        "The year 2026 had 365 days and the price was $1234.56, wasn't it?",
        "def add(x, y):\n    return x + y  # add two numbers 12 34 567 8901\n",
        "naïve café — hello, world!  Numbers: 007 42 100000 and emoji 🎯📉.",
    ]
    # Enrich so BPE reaches a comfortable vocab: many distinct words/numbers create many merges.
    varied = [
        f"record {i} value={i * 7} label obj_{i % 97} tag <cat_{i % 13}> payload {'na' * (i % 11)} end.\n"
        for i in range(3000)
    ]
    corpus = base * 300 + varied
    big = train(iter(corpus), vocab_size=2000)
    js = json.loads(big.to_str())
    n_special = len(SPECIAL_TOKENS)
    achieved = len(js["model"]["vocab"])
    logger.info("trained vocab achieved=%d", achieved)
    # digit grouping (llama3 \p{N}{1,3}): "100000" pretokenizes into runs of <=3 digits
    pre = big.pre_tokenizer.pre_tokenize_str("100000")
    logger.info("pretok '100000' -> %s", [p[0] for p in pre])
    assert all(len(p[0].replace("Ġ", "")) <= 3 for p in pre), "digits must group in runs <= 3"
    # special tokens got the low ids, bytes next
    for i, s in enumerate(SPECIAL_TOKENS):
        assert js["model"]["vocab"][s] == i, f"special {s} expected id {i}"
    # truncate to nested sizes below the achieved vocab and check subset + roundtrip
    prev_merges = None
    for tv in (300, 500, achieved):
        t = truncate(js, tv, n_special) if tv < achieved else js
        merges = t["model"]["merges"]
        if prev_merges is not None:
            assert merges[: len(prev_merges)] == prev_merges, "merges must be a growing prefix (nested)"
        prev_merges = merges
        tok = Tokenizer.from_str(json.dumps(t))
        text = "hello world 2026 café 🎯"
        assert tok.decode(tok.encode(text).ids) == text, f"roundtrip failed at vocab {tv}"
        logger.info("vocab=%d merges=%d roundtrip OK", len(t["model"]["vocab"]), len(merges))
    logger.info("SELFTEST PASSED")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    raise SystemExit("data-driven training path not wired yet; use --selftest")


if __name__ == "__main__":
    main()
