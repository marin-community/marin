# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Build and push a custom WordLevel tokenizer for the protein-docs
contacts-and-distances-v1 document format.

The vocabulary mirrors the reference implementation at:
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


def get_all_domain_tokens() -> list[str]:
    """Return the complete domain vocabulary in canonical order."""
    return (
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


def create_protein_tokenizer() -> PreTrainedTokenizerFast:
    """Build a WordLevel tokenizer for the contacts-and-distances-v1 format."""
    domain_tokens = get_all_domain_tokens()

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
    args = parser.parse_args()

    tokenizer = create_protein_tokenizer()
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
