# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Build and push a custom WordLevel tokenizer for the protein-docs document
formats.

Covers all three document types in ``timodonnell/protein-docs``:
* ``contacts-and-distances-v1`` (original; ``train_protein_1b.py`` uses this)
* ``deterministic-positives-only``
* ``random-3-bins``

The original contacts-and-distances-v1 vocabulary is preserved verbatim at
indices 0..2839 — token IDs do not shift — so models trained against earlier
revisions of this tokenizer (e.g. ``protein-contacts-1b-3.5e-4-distance-masked-7d355e``
at vocab_size 2840) keep working on their original document type. New tokens
needed by the additional document types are appended at the end.

The reference implementation for the contacts-and-distances-v1 vocab is:
  timodonnell/LlamaFold-experiments/experiments/exp6_contact_prediction/src/data.py

Usage:
    python experiments/protein/create_protein_tokenizer.py
    python experiments/protein/create_protein_tokenizer.py --push timodonnell/protein-docs-tokenizer
"""

import argparse

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

# -- Vocabulary components (order must match reference) --

CONTROL_TOKENS = [
    "<contacts-and-distances-v1>",
    "<begin_sequence>",
    "<begin_statements>",
    "<end>",
]

CONTACT_TYPES = [
    "<long-range-contact>",
    "<medium-range-contact>",
    "<short-range-contact>",
]

DISTANCE_MARKER = ["<distance>"]

# 64 bins at 0.5 A resolution
DISTANCE_BINS = [f"<d{v/2:.1f}>" for v in range(1, 65)]

PLDDT_BINS = [
    "<plddt_lt70>",
    "<plddt_70_75>",
    "<plddt_75_80>",
    "<plddt_80_85>",
    "<plddt_85_90>",
    "<plddt_90_95>",
    "<plddt_95_100>",
]

AMINO_ACIDS = [
    "<ALA>",
    "<ARG>",
    "<ASN>",
    "<ASP>",
    "<CYS>",
    "<GLN>",
    "<GLU>",
    "<GLY>",
    "<HIS>",
    "<ILE>",
    "<LEU>",
    "<LYS>",
    "<MET>",
    "<PHE>",
    "<PRO>",
    "<SER>",
    "<THR>",
    "<TRP>",
    "<TYR>",
    "<VAL>",
]

ATOMS = [
    "<C>",
    "<CA>",
    "<CB>",
    "<CD>",
    "<CD1>",
    "<CD2>",
    "<CE>",
    "<CE1>",
    "<CE2>",
    "<CE3>",
    "<CG>",
    "<CG1>",
    "<CG2>",
    "<CH2>",
    "<CZ>",
    "<CZ2>",
    "<CZ3>",
    "<N>",
    "<ND1>",
    "<ND2>",
    "<NE>",
    "<NE1>",
    "<NE2>",
    "<NH1>",
    "<NH2>",
    "<NZ>",
    "<O>",
    "<OD1>",
    "<OD2>",
    "<OE1>",
    "<OE2>",
    "<OG>",
    "<OG1>",
    "<OH>",
    "<SD>",
    "<SG>",
    "<OXT>",
]

# Position tokens <p0> through <p2700>
MAX_POSITION = 2700
POSITION_TOKENS = [f"<p{i}>" for i in range(MAX_POSITION + 1)]

UNK_TOKEN = ["<UNK>"]

# -- Extension tokens for the additional document types --
#
# Appended after UNK_TOKEN so existing token IDs (0..2839) remain unchanged;
# checkpoints trained at vocab_size=2840 are still compatible with this
# tokenizer for the contacts-and-distances-v1 document type.

# Document-type control tokens (peer of "<contacts-and-distances-v1>")
EXTRA_DOCUMENT_TYPES = [
    "<deterministic-positives-only>",
    "<random-3-bins>",
]

# Contact-block delimiters used by the new document types.
EXTRA_STATEMENT_MARKERS = [
    "<begin_contacts>",
    "<end_contacts>",
]

# Coarse 3-bin distance categories used by the random-3-bins document type
# (in addition to the 64-bin DISTANCE_BINS used by contacts-and-distances-v1).
RANDOM_3_BINS_DISTANCE_BINS = [
    "<bin_lt4>",
    "<bin_4_12>",
    "<bin_gt12>",
]

# Modifier tokens used by the random-3-bins document type to mark whether
# a particular contact came from the "correction" or "non-correction" stream.
RANDOM_3_BINS_MODIFIERS = [
    "<correction>",
    "<non-correction>",
]


def get_all_domain_tokens(*, all_doc_types: bool = False) -> list[str]:
    """Return the complete domain vocabulary in canonical order.

    Order is load-bearing: appending new groups at the end preserves all
    existing token IDs from the original contacts-and-distances-v1 vocab.

    Args:
        all_doc_types: when True, include extra tokens needed for
            ``deterministic-positives-only`` and ``random-3-bins`` document
            types. Use the legacy 2840-token vocab (``False``, default) for
            tokenizer artifacts intended to be loadable against checkpoints
            trained at vocab_size=2840 (e.g. the existing protein-docs
            distance-masked runs); use the extended 2849-token vocab
            (``True``) for new training runs that mix all three document
            types.
    """
    base = (
        CONTROL_TOKENS
        + CONTACT_TYPES
        + DISTANCE_MARKER
        + DISTANCE_BINS
        + PLDDT_BINS
        + AMINO_ACIDS
        + ATOMS
        + POSITION_TOKENS
        + UNK_TOKEN
    )
    if not all_doc_types:
        return base
    return base + (
        EXTRA_DOCUMENT_TYPES + EXTRA_STATEMENT_MARKERS + RANDOM_3_BINS_DISTANCE_BINS + RANDOM_3_BINS_MODIFIERS
    )


def create_protein_tokenizer(*, all_doc_types: bool = False) -> PreTrainedTokenizerFast:
    """Build a WordLevel tokenizer for the protein-docs format(s).

    Defaults to the legacy contacts-and-distances-v1 vocab (2840 tokens) for
    backward compatibility with checkpoints trained at vocab_size=2840. Pass
    ``all_doc_types=True`` for the extended vocab covering all three document
    types in ``timodonnell/protein-docs`` (2849 tokens).
    """
    domain_tokens = get_all_domain_tokens(all_doc_types=all_doc_types)

    # Full vocab: <pad>, <eos>, then all domain tokens
    all_tokens = ["<pad>", "<eos>", *domain_tokens]
    vocab = {token: idx for idx, token in enumerate(all_tokens)}

    tokenizer_model = WordLevel(vocab=vocab, unk_token="<UNK>")
    tokenizer = Tokenizer(tokenizer_model)
    tokenizer.pre_tokenizer = WhitespaceSplit()

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<UNK>",
        pad_token="<pad>",
        eos_token="<eos>",
    )


def main():
    parser = argparse.ArgumentParser(description="Create protein-docs tokenizer")
    parser.add_argument(
        "--push",
        type=str,
        default=None,
        help="HuggingFace Hub repo to push to (e.g. timodonnell/protein-docs-tokenizer)",
    )
    parser.add_argument(
        "--save-local",
        type=str,
        default=None,
        help="Local directory to save the tokenizer",
    )
    parser.add_argument(
        "--all-doc-types",
        action="store_true",
        help=(
            "Include extra tokens needed for deterministic-positives-only and "
            "random-3-bins document types (vocab 2849 vs legacy 2840). Use this "
            "for tokenizers feeding new training runs that mix all document "
            "types; omit for backward-compat artifacts loadable against "
            "checkpoints trained at vocab_size=2840."
        ),
    )
    args = parser.parse_args()

    tokenizer = create_protein_tokenizer(all_doc_types=args.all_doc_types)
    vocab_size = len(tokenizer)
    print(f"Created tokenizer with {vocab_size} tokens")

    # Sanity checks
    sample = "<contacts-and-distances-v1> <begin_sequence> <MET> <LYS> <end>"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    print(f"Sample encode: {sample!r}")
    print(f"  -> token ids: {encoded}")
    print(f"  -> decoded:   {decoded!r}")

    if args.save_local:
        tokenizer.save_pretrained(args.save_local)
        print(f"Saved to {args.save_local}")

    if args.push:
        tokenizer.push_to_hub(args.push)
        print(f"Pushed to https://huggingface.co/{args.push}")

    if not args.save_local and not args.push:
        print("Use --push or --save-local to persist the tokenizer.")


if __name__ == "__main__":
    main()
