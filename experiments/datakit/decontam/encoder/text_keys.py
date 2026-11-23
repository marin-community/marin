# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Text streams and window hashes: the keys of the stage-1 lookup.

A text is compared as its stream: lowercase letters and digits, after Unicode
compatibility folding (NFKC) and with LaTeX commands removed (their arguments
stay). Spacing, punctuation, Markdown and markup do not change the stream, so a
copy keeps the stream of its source whatever its typesetting.

A record can hold a copy inside a JSON string, or inside a JSON string in a
JSON string. Its escapes then break the stream: ``\\n`` followed by a word
reads as a LaTeX command and removes the word. Stage 1 therefore reads a record
three ways: as it is, with its JSON escapes read once, and read twice.
"""

import re
import unicodedata
from enum import StrEnum

import numpy as np

# Width of a key window, in stream characters.
WINDOW = 20
_TEX_COMMAND = re.compile(r"\\[a-zA-Z]+")
_NOT_STREAM = re.compile(r"[\W_]+")
_MULTIPLIER = np.uint64(0x9E3779B97F4A7C15)
_LETTERS = "abcdefghijklmnopqrstuvwxyz"
# Levels of JSON escaping that a record reading undoes.
ESCAPE_DEPTH = 2
# One JSON escape. An escaped backslash is one unit, so a JSON-escaped LaTeX command (two
# backslashes, then "theta") reads as the command, not as a tab.
_JSON_ESCAPE = re.compile(r"\\(\\|[\"/ntr]|u[0-9a-fA-F]{4})")
_ESCAPED_CHARACTER = {"\\": "\\", '"': '"', "/": "/"}


def text_stream(text: str) -> str:
    """The lowercase letters and digits of *text*, after NFKC and without LaTeX commands."""
    folded = unicodedata.normalize("NFKC", text or "")
    return _NOT_STREAM.sub("", _TEX_COMMAND.sub(" ", folded).lower())


def item_stream(question: str, choices: list[str], width: int = WINDOW) -> str:
    """The stream that an item's keys come from: its question, or, for a question shorter than a
    *width*-character window, the question followed by each choice with its letter (a, b, ...)."""
    stream = text_stream(question)
    if len(stream) < width and choices:
        stream += "".join(_LETTERS[i % len(_LETTERS)] + text_stream(c) for i, c in enumerate(choices))
    return stream


class KeyStream(StrEnum):
    """A stream that stage 1 takes keys from; an item can have keys from several."""

    QUESTION = "question"
    """The question, or a short question with its lettered choices (``item_stream``)."""
    UNLETTERED = "unlettered"
    """A short question with its choices and no letters, for copies that do not letter the choices."""
    PASSAGE = "passage"
    """The item's passage, so a copy is found when its question alone is common."""


def key_streams(question: str, choices: list[str], passage: str, width: int = WINDOW) -> dict[KeyStream, str]:
    """The streams of one item that stage 1 keys, by kind."""
    streams = {KeyStream.QUESTION: item_stream(question, choices, width)}
    if len(text_stream(question)) < width and choices:
        streams[KeyStream.UNLETTERED] = text_stream(question) + "".join(text_stream(c) for c in choices)
    if passage and len(text_stream(passage)) >= width:
        streams[KeyStream.PASSAGE] = text_stream(passage)
    return streams


def json_unescaped(text: str) -> str:
    """*text* with one level of JSON string escapes read; ``\\n``, ``\\t`` and ``\\r`` become spaces."""

    def read(match: re.Match) -> str:
        escape = match.group(1)
        return chr(int(escape[1:], 16)) if escape[0] == "u" else _ESCAPED_CHARACTER.get(escape, " ")

    decoded = _JSON_ESCAPE.sub(read, text)
    # An escaped surrogate pair decodes to two code points; the round trip joins them.
    return decoded.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


def readings(text: str) -> list[str]:
    """*text*, then *text* with its JSON escapes read once and twice, while it has a backslash."""
    found = [text]
    while len(found) <= ESCAPE_DEPTH and "\\" in found[-1]:
        found.append(json_unescaped(found[-1]))
    return found


def window_hashes(stream: str, width: int = WINDOW) -> np.ndarray:
    """A polynomial hash, modulo 2**64, of every *width*-character window of *stream*."""
    codes = np.frombuffer(stream.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
    count = len(codes) - width + 1
    if count <= 0:
        return np.empty(0, dtype=np.uint64)
    hashes = np.zeros(count, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for offset in range(width):
            hashes = hashes * _MULTIPLIER + codes[offset : offset + count]
    return hashes


def hit_indices(stream: str, sorted_hashes: np.ndarray, width: int = WINDOW) -> np.ndarray:
    """Indices into *sorted_hashes* of the *width*-character windows that *stream* holds."""
    hashes = window_hashes(stream, width)
    if not len(hashes) or not len(sorted_hashes):
        return np.empty(0, dtype=np.int64)
    slot = np.minimum(np.searchsorted(sorted_hashes, hashes), len(sorted_hashes) - 1)
    return np.unique(slot[sorted_hashes[slot] == hashes])


def record_hit_indices(text: str, sorted_hashes: np.ndarray, width: int = WINDOW) -> np.ndarray:
    """Indices into *sorted_hashes* of the windows that any reading of the record *text* holds."""
    hits = [hit_indices(text_stream(reading), sorted_hashes, width) for reading in readings(text)]
    return np.unique(np.concatenate(hits))
