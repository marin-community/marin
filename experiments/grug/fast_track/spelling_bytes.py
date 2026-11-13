# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Spelling-normalised byte embeddings (``spelling_byte_embeds``), after Spelling Bee Embeddings (arXiv:2601.18030).

Each token's input embedding row adds a learned sum over the first ``BYTE_POSITIONS`` UTF-8 bytes of its surface
string, one 256-row table per byte position. The bytes are read after stripping leading spaces and lowercasing, so
`` The``, ``the`` and ``The`` share every byte row; a leading-space row and one of ``SpellingCase``'s rows keep what
the normalisation removes. The model adds the sum to the vocabulary table before the token gather.

``TOKEN_BYTES_FILE`` holds the vocabulary: line ``i`` is token ``i``'s surface bytes in hex, empty for special ids.
It was written from the hero 16k BPE tokenizer by running this module.
"""

import dataclasses
import enum
import functools
import pathlib
import sys

import numpy as np
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from transformers.convert_slow_tokenizer import bytes_to_unicode

BYTE_POSITIONS = 8
TOKEN_BYTES_FILE = pathlib.Path(__file__).with_name("hero_bpe_v16384_token_bytes.txt")
TOKENIZER = "hero-bpe-v16384"


class SpellingCase(enum.IntEnum):
    LOWER = 0
    CAPITALISED = 1
    ALL_CAPS = 2
    MIXED = 3


@dataclasses.dataclass(frozen=True)
class SpellingTables:
    """Per-token inputs of the byte rows, for a vocabulary of ``V`` tokens."""

    byte_ids: np.ndarray  # [V, BYTE_POSITIONS] int32, the normalised byte at each position (0 past the end)
    byte_present: np.ndarray  # [V, BYTE_POSITIONS] float32, 1 where the position holds a byte
    leading_space: np.ndarray  # [V] float32
    case: np.ndarray  # [V] int32 SpellingCase


def text_case(text: str) -> SpellingCase:
    """Case of ``text``'s cased characters; a single capital letter is ``CAPITALISED``."""
    cased = [c for c in text if c.isupper() or c.islower()]
    upper = [c.isupper() for c in cased]
    if not any(upper):
        return SpellingCase.LOWER
    if all(upper) and len(cased) > 1:
        return SpellingCase.ALL_CAPS
    if upper[0] and not any(upper[1:]):
        return SpellingCase.CAPITALISED
    return SpellingCase.MIXED


def normalised_spelling(surface: bytes) -> tuple[bytes, bool, SpellingCase]:
    """Surface bytes with leading spaces stripped and lowercased, whether there was a leading space, and the case.
    A token that is not whole UTF-8 (part of a multi-byte character) is lowercased and cased on its ASCII bytes."""
    stripped = surface.lstrip(b" ")
    leading_space = len(stripped) < len(surface)
    try:
        text = stripped.decode("utf-8")
    except UnicodeDecodeError:
        return stripped.lower(), leading_space, text_case(stripped.decode("ascii", errors="ignore"))
    return text.lower().encode("utf-8"), leading_space, text_case(text)


def spelling_tables_from_surfaces(surfaces: list[bytes]) -> SpellingTables:
    vocab = len(surfaces)
    byte_ids = np.zeros((vocab, BYTE_POSITIONS), np.int32)
    byte_present = np.zeros((vocab, BYTE_POSITIONS), np.float32)
    leading_space = np.zeros((vocab,), np.float32)
    case = np.zeros((vocab,), np.int32)
    for i, surface in enumerate(surfaces):
        spelling, space, case[i] = normalised_spelling(surface)
        head = list(spelling[:BYTE_POSITIONS])
        byte_ids[i, : len(head)] = head
        byte_present[i, : len(head)] = 1
        leading_space[i] = space
    return SpellingTables(byte_ids, byte_present, leading_space, case)


def stored_surfaces() -> list[bytes]:
    return [bytes.fromhex(line) for line in TOKEN_BYTES_FILE.read_text().splitlines()]


@functools.cache
def spelling_tables(vocab_size: int) -> SpellingTables:
    """The tables of the first ``vocab_size`` tokens of ``TOKEN_BYTES_FILE``."""
    surfaces = stored_surfaces()
    if vocab_size > len(surfaces):
        raise ValueError(f"spelling_byte_embeds: vocab_size {vocab_size} exceeds the {len(surfaces)} stored tokens")
    return spelling_tables_from_surfaces(surfaces[:vocab_size])


def token_surfaces(tokenizer: MarinTokenizer, vocab_size: int) -> list[bytes]:
    """The surface bytes of a byte-level BPE tokenizer's first ``vocab_size`` ids; special ids are empty."""
    byte_of = {char: byte for byte, char in bytes_to_unicode().items()}
    special = set(tokenizer.all_special_ids)
    pieces = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    return [b"" if i in special else bytes(byte_of[c] for c in piece) for i, piece in enumerate(pieces)]


def check_token_bytes(tokenizer: MarinTokenizer, vocab_size: int) -> None:
    """Raise unless ``TOKEN_BYTES_FILE`` holds ``tokenizer``'s first ``vocab_size`` token surfaces."""
    if token_surfaces(tokenizer, vocab_size) != stored_surfaces()[:vocab_size]:
        raise ValueError(f"spelling_byte_embeds: {TOKEN_BYTES_FILE.name} does not match {tokenizer.name_or_path}")


if __name__ == "__main__":
    tokenizer = load_tokenizer(sys.argv[1] if len(sys.argv) > 1 else TOKENIZER)
    surfaces = token_surfaces(tokenizer, len(tokenizer))
    TOKEN_BYTES_FILE.write_text("".join(f"{surface.hex()}\n" for surface in surfaces))
