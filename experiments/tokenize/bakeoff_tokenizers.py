# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Registry of tokenizer arms for the grug-moe FLOP-equivalent bake-off.

Each arm is a name + a loadable tokenizer reference (a HuggingFace id or a path
the cluster workers can read) + its vocab size (the grug model's embedding/LM-head
dimension, taken as ``len(tokenizer.get_vocab())``) + the design axis it exercises.

Phase 2 arms are off-the-shelf tokenizers loaded straight from the Hub; later
phases add tokenizers we build (derived vocab sizes, SuperBPE, number-aware,
Unigram-LM), which register here once their artifacts exist. See
``.agents/projects/20260703_tokenizer_flop_equivalent_bakeoff.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Axis(StrEnum):
    """The design axis an arm exercises (§3 of the protocol)."""

    BASELINE = "baseline"  # off-the-shelf vocab families (axis A)
    DERIVED_VOCAB = "derived_vocab"  # rank-truncated Marin 32k/64k (axis A)
    TRAINED_BPE = "trained_bpe"  # plain BPE trained on the grug-moe mix, not borrowed (axis A)
    PRETOK = "pretok"  # number-aware / capcode (axis B)
    SUPERBPE = "superbpe"  # superword merges (axis C)
    NGRAM = "ngram"  # Over-Tokenized n-gram input embeddings (axis D)
    BYTE = "byte"  # byte-level floor (axis E)
    UNIGRAM = "unigram"  # Unigram-LM vs BPE (axis F)


@dataclass(frozen=True)
class TokenizerArm:
    """One tokenizer under test.

    ``ref`` is what :func:`levanter.tokenizers.load_tokenizer` /
    ``marin.experiment.data.tokenized(tokenizer=...)`` receive — a HuggingFace id
    or a readable path. ``vocab_size`` is the grug model's embedding dimension and
    must equal the tokenizer's real vocab (asserted at build time).
    """

    name: str
    ref: str
    vocab_size: int
    axis: Axis
    note: str = ""


# Off-the-shelf baseline tokenizers, vocab sizes measured via len(get_vocab()). marin-128k is
# the incumbent (Llama-3 vocab + Marin chat template); the rest span the vocab-size axis from
# gpt-neox's 50k to gemma-3's 262k so serving cost can be read against quality across scales.
BASELINE_ARMS: tuple[TokenizerArm, ...] = (
    TokenizerArm("marin-128k", "marin-community/marin-tokenizer", 128_256, Axis.BASELINE, "incumbent (Llama-3 vocab)"),
    TokenizerArm("gpt-neox-50k", "EleutherAI/gpt-neox-20b", 50_277, Axis.BASELINE, "small vocab reference"),
    TokenizerArm("qwen3-152k", "Qwen/Qwen3-8B", 151_669, Axis.BASELINE, "digit-splitting, strong code"),
    TokenizerArm("gpt-oss-200k", "openai/gpt-oss-20b", 200_019, Axis.BASELINE, "o200k_base"),
    TokenizerArm("gemma3-262k", "google/gemma-3-4b-pt", 262_145, Axis.BASELINE, "largest vocab, multilingual"),
)

# Phase 3 SuperBPE (axis C): pretrained superword tokenizers from "SuperBPE: Space Travel
# for Language Models" (arXiv 2503.13423). The marin tokenize pipeline loads these through
# levanter.load_tokenizer, which reads tokenizer.json directly (tokenizers.Tokenizer.from_file)
# and therefore honors the Sequence pretokenizer that lets BPE merges bridge whitespace,
# yielding the superword fertility win (measured ~-21% tokens/byte vs Llama-3 on a mixed
# English+code sample; the paper reports up to -33% at 200k vocab). Loading these same refs
# through transformers.AutoTokenizer.from_pretrained instead gives subword-only output: their
# tokenizer_config sets tokenizer_class=GPT2Tokenizer, and GPT2TokenizerFast overwrites the
# pretokenizer with a whitespace-splitting ByteLevel so the superword tokens never fire.
SUPERBPE_ARMS: tuple[TokenizerArm, ...] = (
    TokenizerArm(
        "superbpe-128k",
        "alisawuffles/superbpe-tokenizer-128k",
        128_001,
        Axis.SUPERBPE,
        "English superword BPE, Llama-3-comparable vocab, ~-21% tok/byte",
    ),
    TokenizerArm(
        "superbpe-180k",
        "allenai/superbpe-experimental_v0.1.0",
        180_021,
        Axis.SUPERBPE,
        "experimental superword BPE, larger vocab",
    ),
)

# Track C (issue #6796): tokenizers trained from scratch on the grug-moe data mix (English web
# + code + math; see corpus.py/train_tokenizers.py), rather than borrowed off-the-shelf. Refs
# resolve through the `mirror://tokenizers/trained/<name>/...` cache that
# push_trained_tokenizers.py populates (see that module for why a bare ref, not a raw s3:// path).
# Vocab sizes are each spec's requested size + 1 (the added `<|endoftext|>` special token);
# every config in the sweep reached its full requested vocab (see
# experiments/tokenize/results/trained_tokenizers_manifest.json for per-arm training time).
TRAINED_BPE_ARMS: tuple[TokenizerArm, ...] = (
    TokenizerArm(
        "trained-bpe-64k", "trained/trained-bpe-64k", 64_001, Axis.TRAINED_BPE, "plain BPE, trained on our mix"
    ),
    TokenizerArm(
        "trained-bpe-96k", "trained/trained-bpe-96k", 96_001, Axis.TRAINED_BPE, "plain BPE, trained on our mix"
    ),
    TokenizerArm(
        "trained-bpe-128k", "trained/trained-bpe-128k", 128_001, Axis.TRAINED_BPE, "plain BPE, trained on our mix"
    ),
)

# Track C SuperBPE: our own two-stage superword BPE (superbpe_trainer.py, a from-scratch
# reimplementation of arXiv:2503.13423 — see that module's docstring), trained on the same mix,
# at a (vocab, transition-point t) sweep plus a small-vocab pair. `note` records t/vocab.
TRAINED_SUPERBPE_ARMS: tuple[TokenizerArm, ...] = (
    TokenizerArm(
        "trained-superbpe-64k-t32k",
        "trained/trained-superbpe-64k-t32k",
        64_001,
        Axis.SUPERBPE,
        "trained SuperBPE, t/vocab=32k/64k",
    ),
    TokenizerArm(
        "trained-superbpe-80k-t40k",
        "trained/trained-superbpe-80k-t40k",
        80_001,
        Axis.SUPERBPE,
        "trained SuperBPE, t/vocab=40k/80k",
    ),
    TokenizerArm(
        "trained-superbpe-96k-t38k",
        "trained/trained-superbpe-96k-t38k",
        96_001,
        Axis.SUPERBPE,
        "trained SuperBPE, t/vocab=38k/96k",
    ),
    TokenizerArm(
        "trained-superbpe-96k-t77k",
        "trained/trained-superbpe-96k-t77k",
        96_001,
        Axis.SUPERBPE,
        "trained SuperBPE, t/vocab=77k/96k",
    ),
    TokenizerArm(
        "trained-superbpe-128k-t51k",
        "trained/trained-superbpe-128k-t51k",
        128_001,
        Axis.SUPERBPE,
        "trained SuperBPE, t/vocab=51k/128k",
    ),
    TokenizerArm(
        "trained-superbpe-128k-t102k",
        "trained/trained-superbpe-128k-t102k",
        128_001,
        Axis.SUPERBPE,
        "trained SuperBPE, t/vocab=102k/128k",
    ),
    TokenizerArm(
        "trained-superbpe-160k-t64k",
        "trained/trained-superbpe-160k-t64k",
        160_001,
        Axis.SUPERBPE,
        "trained SuperBPE, t/vocab=64k/160k",
    ),
    TokenizerArm(
        "trained-superbpe-160k-t128k",
        "trained/trained-superbpe-160k-t128k",
        160_001,
        Axis.SUPERBPE,
        "trained SuperBPE, t/vocab=128k/160k",
    ),
)

# Registered arms. Extended in later phases as built tokenizers land (their refs will be
# HF ids under marin-community/ or S3 paths under the cw-rno2a prefix).
ALL_ARMS: tuple[TokenizerArm, ...] = BASELINE_ARMS + SUPERBPE_ARMS + TRAINED_BPE_ARMS + TRAINED_SUPERBPE_ARMS

# Vocab sizes to add to marin.processing.tokenize.data_configs._KNOWN_VOCAB_SIZES so
# dry-runs/fingerprints don't hit the Hub. Kept here next to the arm definitions.
KNOWN_VOCAB_SIZES: dict[str, int] = {arm.ref: arm.vocab_size for arm in ALL_ARMS}


def arm_by_name(name: str) -> TokenizerArm:
    for arm in ALL_ARMS:
        if arm.name == name:
            return arm
    raise KeyError(f"unknown tokenizer arm {name!r}; known: {[a.name for a in ALL_ARMS]}")
