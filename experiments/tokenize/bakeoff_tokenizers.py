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


# Phase 2 baselines: off-the-shelf, all confirmed loadable with vocab sizes measured
# via len(get_vocab()). marin-128k is the incumbent baseline (Llama-3 vocab + Marin
# chat template). o200k (gpt-oss) and gemma-3 were the two arms that edged out Llama-3
# in the prior sweep (#5821) after the byte-accounting fix.
BASELINE_ARMS: tuple[TokenizerArm, ...] = (
    TokenizerArm("marin-128k", "marin-community/marin-tokenizer", 128_256, Axis.BASELINE, "incumbent (Llama-3 vocab)"),
    TokenizerArm("gpt-neox-50k", "EleutherAI/gpt-neox-20b", 50_277, Axis.BASELINE, "small vocab reference"),
    TokenizerArm("qwen3-152k", "Qwen/Qwen3-8B", 151_669, Axis.BASELINE, "digit-splitting, strong code"),
    TokenizerArm("gpt-oss-200k", "openai/gpt-oss-20b", 200_019, Axis.BASELINE, "o200k_base"),
    TokenizerArm("gemma3-262k", "google/gemma-3-4b-pt", 262_145, Axis.BASELINE, "largest vocab, multilingual"),
)

# Registered arms. Extended in later phases as built tokenizers land (their refs will be
# HF ids under marin-community/ or S3 paths under the cw-rno2a prefix).
ALL_ARMS: tuple[TokenizerArm, ...] = BASELINE_ARMS

# Vocab sizes to add to marin.processing.tokenize.data_configs._KNOWN_VOCAB_SIZES so
# dry-runs/fingerprints don't hit the Hub. Kept here next to the arm definitions.
KNOWN_VOCAB_SIZES: dict[str, int] = {arm.ref: arm.vocab_size for arm in ALL_ARMS}


def arm_by_name(name: str) -> TokenizerArm:
    for arm in ALL_ARMS:
        if arm.name == name:
            return arm
    raise KeyError(f"unknown tokenizer arm {name!r}; known: {[a.name for a in ALL_ARMS]}")
