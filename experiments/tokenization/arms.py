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

# Track C: tokenizers trained from scratch on the grug-moe data mix (English web
# + code + math; see corpus.py/train_tokenizer.py), rather than borrowed off-the-shelf. Refs
# resolve through the `mirror://tokenizers/trained/<name>/...` cache that
# train_tokenizer.py's push_to_mirror populates (see that module for why a bare ref, not a raw s3:// path).
# Vocab sizes are each spec's requested size + 1 (the added `<|endoftext|>` special token);
# every config in the sweep reached its full requested vocab (see
# experiments/tokenization/results/trained_tokenizers_manifest.json for per-arm training time).
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

# Track C SuperBPE: our own two-stage superword BPE (superbpe.py, a from-scratch
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
        "trained-superbpe-48k-t24k",
        "trained/trained-superbpe-48k-t24k",
        48_001,
        Axis.SUPERBPE,
        "trained SuperBPE, t/vocab=24k/48k",
    ),
    TokenizerArm(
        "trained-superbpe-40k-t20k",
        "trained/trained-superbpe-40k-t20k",
        40_001,
        Axis.SUPERBPE,
        "trained SuperBPE, t/vocab=20k/40k",
    ),
    TokenizerArm(
        "trained-superbpe-32k-t16k",
        "trained/trained-superbpe-32k-t16k",
        32_001,
        Axis.SUPERBPE,
        "trained SuperBPE, t/vocab=16k/32k",
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

# Soak arms: 64k & 128k SuperBPE (t/T=0.5) retrained on the representative grug mixture, with
# the two pretokenizer variants under test — individual-digit encoding (math) and the Llama-3
# production word regex — scored at 10B-total/500M-active over a 24h run (see EXPERIMENT_LOG
# EXP-011). The base 64k arm doubles as the n-gram carrier (BAKEOFF_NGRAM toggles the model-side
# hashed n-gram embedding at launch, no separate tokenizer).
SOAK_ARMS: tuple[TokenizerArm, ...] = (
    TokenizerArm("soak-superbpe-64k", "trained/soak-superbpe-64k", 64_001, Axis.SUPERBPE, "soak base, t/T=0.5"),
    TokenizerArm("soak-superbpe-128k", "trained/soak-superbpe-128k", 128_001, Axis.SUPERBPE, "soak base, t/T=0.5"),
    TokenizerArm(
        "soak-superbpe-64k-digits",
        "trained/soak-superbpe-64k-digits",
        64_001,
        Axis.PRETOK,
        "soak, individual-digit encoding",
    ),
    TokenizerArm(
        "soak-superbpe-128k-digits",
        "trained/soak-superbpe-128k-digits",
        128_001,
        Axis.PRETOK,
        "soak, individual-digit encoding",
    ),
    TokenizerArm(
        "soak-superbpe-64k-llama",
        "trained/soak-superbpe-64k-llama",
        64_001,
        Axis.PRETOK,
        "soak, Llama-3 word regex",
    ),
    TokenizerArm(
        "soak-superbpe-128k-llama",
        "trained/soak-superbpe-128k-llama",
        128_001,
        Axis.PRETOK,
        "soak, Llama-3 word regex",
    ),
)

# Fixed soak arms: the base 64k/128k SuperBPE, the individual-digit-pretok, and the Llama-3-word-
# regex-pretok SOAK_ARMS above, retrained after commit 11bd2f4e9c fixed the stage-2 corpus
# sampling bug (the soak-* arms above were trained on an English-only stage-2 sample; see
# train_tokenizer.py). The -llama variants hold the stage-1 pretokenizer equal to the
# marin-128k baseline, so a SuperBPE win over the baseline can be attributed to superwords rather
# than the pretokenizer regex. Must match train_tokenizer.py's `_FIXED_SOAK_BASE_NAMES`. Refs
# follow the same `trained/<name>` convention train_tokenizer.arm_ref produces.
_FIXED_SOAK_BASE_NAMES: tuple[str, ...] = (
    "soak-superbpe-64k",
    "soak-superbpe-128k",
    "soak-superbpe-64k-digits",
    "soak-superbpe-128k-digits",
    "soak-superbpe-64k-llama",
    "soak-superbpe-128k-llama",
)

SOAK_FIXED_ARMS: tuple[TokenizerArm, ...] = tuple(
    TokenizerArm(f"{arm.name}-fixed", f"trained/{arm.name}-fixed", arm.vocab_size, arm.axis, f"{arm.note}, stage-2 fix")
    for arm in SOAK_ARMS
    if arm.name in _FIXED_SOAK_BASE_NAMES
)

# Registered arms. Extended in later phases as built tokenizers land (their refs will be
# HF ids under marin-community/ or S3 paths under the cw-rno2a prefix).
ALL_ARMS: tuple[TokenizerArm, ...] = (
    BASELINE_ARMS + SUPERBPE_ARMS + TRAINED_BPE_ARMS + TRAINED_SUPERBPE_ARMS + SOAK_ARMS + SOAK_FIXED_ARMS
)

# Vocab sizes to add to marin.processing.tokenize.data_configs._KNOWN_VOCAB_SIZES so
# dry-runs/fingerprints don't hit the Hub. Kept here next to the arm definitions.
KNOWN_VOCAB_SIZES: dict[str, int] = {arm.ref: arm.vocab_size for arm in ALL_ARMS}


def arm_by_name(name: str) -> TokenizerArm:
    for arm in ALL_ARMS:
        if arm.name == name:
            return arm
    raise KeyError(f"unknown tokenizer arm {name!r}; known: {[a.name for a in ALL_ARMS]}")
