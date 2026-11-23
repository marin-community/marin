# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rarest-key candidates for eval decontamination: text streams, item keys, and the candidates of a record.

A text is compared as its stream: lowercase letters and digits, after Unicode compatibility folding
(NFKC) and with LaTeX commands removed (their arguments stay). Spacing, punctuation, Markdown and markup
do not change the stream, so a copy keeps the stream of its source whatever its typesetting.

A record can hold a copy inside a JSON string, or inside a JSON string in a JSON string. Its escapes
then break the stream: ``\\n`` followed by a word reads as a LaTeX command and removes the word. A record
is therefore read three ways (``readings``): as it is, with its JSON escapes read once, and read twice.

Each eval item has up to three key streams (``key_streams``): its question; for a question shorter
than a key, the question and its choices without letters; and its passage. A stream's key is its
rarest ``WINDOW``-character window in a reference sample (``keys_table``); a stream other than the
question keeps its key only when no reference record holds it. A record that holds an item's key is a
candidate for that item; ``record_pairs`` returns one candidate per item with the record text around the
hit (``excerpt``), for the stage-2 classifier.
"""

import bisect
import itertools
import re
import unicodedata
from enum import StrEnum
from typing import NamedTuple

import numpy as np
import pyarrow as pa

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
WINDOWS_PER_ITEM = 16
MAX_DOC_CHARS = 5_000_000
# A key table rejects most window hashes with one lookup in a table of 2**FILTER_BITS flags (64 MB),
# indexed by the top bits of the hash, before the binary search; this cuts the lookup time about 14x.
FILTER_BITS = 26
_FILTER_SHIFT = np.uint64(64 - FILTER_BITS)
KEY_COLUMNS = (
    "id",
    "kind",
    "first_hash",
    "first_count",
    "key1_hash",
    "key1_count",
    "key1_offset",
    "key2_hash",
    "key2_count",
)
# A chunk of a reading (``Reading``) starts at the first safe character after this many characters, and a
# stretch without one is split after MAX_CHUNK_CHARS. A safe character starts no NFKC composition with the
# character before it and ends any LaTeX command (see ``Reading``).
CHUNK_CHARS = 4096
MAX_CHUNK_CHARS = 65536
_SAFE = re.compile(r"[\s0-9!-/:-@\[\]^-`{-~\u4e00-\u9fff\uac00-\ud7a3]")
BEFORE = 1500
AFTER = 3000
LOCATOR_COLUMNS = ("item", "kind", "eval", "key1_hash", "key", "offset", "question", "question_chars")


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


def record_window_hashes(text: str, width: int = WINDOW) -> np.ndarray:
    """The hashes of every *width*-character window of every reading of the record *text*."""
    return np.concatenate([window_hashes(text_stream(reading), width) for reading in readings(text)])


def candidate_windows(streams: list[str], width: int = WINDOW) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hashes of the candidate windows, the item of each, and its offset in the item stream."""
    hashes, owners, offsets = [], [], []
    for number, stream in enumerate(streams):
        if len(stream) < width:
            continue
        last = len(stream) - width
        starts = sorted({round(k * last / (WINDOWS_PER_ITEM - 1)) for k in range(WINDOWS_PER_ITEM)}) if last else [0]
        window = window_hashes(stream, width)
        hashes.append(window[starts])
        owners.append(np.full(len(starts), number, dtype=np.int32))
        offsets.append(np.array(starts, dtype=np.int32))
    return np.concatenate(hashes), np.concatenate(owners), np.concatenate(offsets)


class Windows(NamedTuple):
    """The candidate windows of all key streams: hash, stream number, and offset in the stream; and the item id
    and stream kind of each stream."""

    hashes: np.ndarray
    owners: np.ndarray
    offsets: np.ndarray
    ids: np.ndarray
    kinds: np.ndarray


class KeyTable:
    """Keys of one kind: the sorted unique hashes, and the items behind each hash."""

    def __init__(self, hashes: np.ndarray, items: np.ndarray) -> None:
        self.unique, inverse = np.unique(hashes, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        self.items = items[order]
        self.starts = np.searchsorted(inverse[order], np.arange(len(self.unique) + 1))
        self.present = np.zeros(1 << FILTER_BITS, dtype=bool)
        self.present[self.unique >> _FILTER_SHIFT] = True

    def found(self, hashes: np.ndarray) -> np.ndarray:
        """Indices into ``unique`` of the keys among *hashes* (window hashes of a record's readings)."""
        hashes = hashes[self.present[hashes >> _FILTER_SHIFT]]
        if not len(hashes):
            return np.empty(0, dtype=np.int64)
        slot = np.minimum(np.searchsorted(self.unique, hashes), len(self.unique) - 1)
        return np.unique(slot[self.unique[slot] == hashes])

    def items_of(self, found: np.ndarray) -> list:
        return [item for index in found.tolist() for item in self.items[self.starts[index] : self.starts[index + 1]]]


def keys_table(windows: Windows, window_counts: np.ndarray, width: int = WINDOW) -> pa.Table:
    """One row per key stream: its first window, its rarest window (key 1), and the rarest window that does
    not overlap key 1 (key 2). Hash columns are uint64; 0 means "no such key", and its count is -1.

    A stream other than the question keeps its key only when its rarest window occurs in no reference
    record: an extra key that common text holds (a generic stem taken for a passage) would only add load.
    """
    hashes, owners, offsets, ids, kinds = windows
    columns: dict[str, list] = {name: [] for name in KEY_COLUMNS}
    order = np.lexsort((offsets, window_counts, owners))
    keyed: set[int] = set()
    start = 0
    while start < len(order):
        owner = int(owners[order[start]])
        end = start
        while end < len(order) and owners[order[end]] == owner:
            end += 1
        group = order[start:end]
        first = group[0]
        if str(kinds[owner]) != KeyStream.QUESTION and window_counts[first] > 0:
            start = end
            continue
        second = next((k for k in group[1:] if abs(int(offsets[k]) - int(offsets[first])) >= width), None)
        lowest = group[int(np.argmin(offsets[group]))]
        keyed.add(owner)
        values = (
            str(ids[owner]),
            str(kinds[owner]),
            int(hashes[lowest]),
            int(window_counts[lowest]),
            int(hashes[first]),
            int(window_counts[first]),
            int(offsets[first]),
            int(hashes[second]) if second is not None else 0,
            int(window_counts[second]) if second is not None else -1,
        )
        for name, value in zip(columns, values, strict=True):
            columns[name].append(value)
        start = end
    for owner in range(len(ids)):
        if owner not in keyed:
            for name, value in zip(columns, (str(ids[owner]), str(kinds[owner]), 0, -1, 0, -1, -1, 0, -1), strict=True):
                columns[name].append(value)
    return pa.table(
        {
            name: pa.array(np.array(values, dtype=np.uint64)) if name.endswith("_hash") else pa.array(values)
            for name, values in columns.items()
        }
    )


class Reading:
    """One reading of a record (``readings``): its text, stream and window hashes, with two lookups that cost
    one pass over the record in all, whatever the number of hits.

    ``first`` finds the first place of a stream piece through a sorted index of the window hashes.
    ``text_offset`` maps a stream position to a text offset. A stream is additive at a safe character
    (``_SAFE``: whitespace, ASCII digits and punctuation other than the backslash, CJK ideographs, Hangul
    syllables): no LaTeX command, NFKC composition or case rule crosses one. So the reading keeps the stream
    length before each chunk start (a safe character about every ``CHUNK_CHARS`` characters) and searches
    one chunk. A stretch of more than ``MAX_CHUNK_CHARS`` characters without a safe character is split
    anyway, and an offset inside it is estimated in proportion.
    """

    def __init__(self, text: str, width: int = WINDOW) -> None:
        self.text = text
        self.stream = text_stream(text)
        self.width = width
        self.hashes = window_hashes(self.stream, width)
        self._order: np.ndarray | None = None
        self._sorted: np.ndarray | None = None
        self._starts: list[int] = []
        self._before: list[int] = []
        self._exact: list[bool] = []

    def first(self, piece: str) -> int:
        """The first position of the stream *piece* in the stream, as ``str.find`` gives it."""
        if len(piece) < self.width:
            return self.stream.find(piece)
        if self._order is None or self._sorted is None:
            self._order = np.argsort(self.hashes, kind="stable")
            self._sorted = self.hashes[self._order]
        head = window_hashes(piece[: self.width], self.width)[0]
        low, high = np.searchsorted(self._sorted, head, side="left"), np.searchsorted(self._sorted, head, side="right")
        for position in np.sort(self._order[low:high]).tolist():
            if self.stream.startswith(piece, position):
                return position
        return -1

    def _index(self) -> None:
        starts, exact, position = [0], [True], CHUNK_CHARS
        while position < len(self.text):
            match = _SAFE.search(self.text, position, position + MAX_CHUNK_CHARS)
            boundary = match.start() if match else position + MAX_CHUNK_CHARS
            if boundary >= len(self.text):
                break
            # The chunk that ends at a forced split is not additive at its end; offsets inside it are estimates.
            exact[-1] = exact[-1] and match is not None
            starts.append(boundary)
            exact.append(True)
            position = boundary + CHUNK_CHARS
        before = [0]
        for start, end in itertools.pairwise(starts):
            before.append(before[-1] + len(text_stream(self.text[start:end])))
        self._starts, self._before, self._exact = starts, before, exact

    def text_offset(self, position: int) -> int:
        """The smallest prefix length of the text whose stream is longer than *position* (an estimate inside a
        stretch without safe characters; see the class)."""
        if not self._starts:
            self._index()
        chunk = bisect.bisect_right(self._before, position) - 1
        start, base = self._starts[chunk], self._before[chunk]
        end = self._starts[chunk + 1] if chunk + 1 < len(self._starts) else len(self.text)
        if not self._exact[chunk]:
            length = self._before[chunk + 1] - base
            return min(end, start + (position - base + 1) * (end - start) // max(length, 1))
        low, high = start, end
        while low < high:
            middle = (low + high) // 2
            if base + len(text_stream(self.text[start:middle])) > position:
                high = middle
            else:
                low = middle + 1
        return low


def record_readings(text: str, width: int = WINDOW) -> list[Reading]:
    """The readings of the record *text* (``readings``) with their streams and window hashes."""
    return [Reading(reading, width) for reading in readings(text)]


def _located(readings_of: list[Reading], question: str, key: str, offset: int) -> tuple[Reading, int]:
    """The first reading with the question or key, and the estimated item position in its stream (-1 if absent)."""
    for reading in readings_of:
        # A question shorter than a window is no place marker: it can occur anywhere. Its key holds the choices.
        at = reading.first(question) if len(question) >= WINDOW else -1
        if at < 0 and key:
            # The record holds the key (stage 1 found it); the item starts that far before it.
            hit = reading.first(key)
            at = max(0, hit - offset) if hit >= 0 else -1
        if at >= 0:
            return reading, at
    return readings_of[0], -1


def excerpt(
    readings_of: list[Reading], question_chars: int, question: str, key: str, offset: int
) -> tuple[str, str, int]:
    """The record text around the item (``BEFORE`` characters before it, ``AFTER`` after its question), the
    reading's stream, and the item's position in that stream (-1 when only an estimate exists).

    *readings_of* are the record's readings (``record_readings``), *question* the item's question stream and
    *question_chars* the length of its question text, *key* the matched key window and *offset* its position
    in the keyed stream.
    """
    reading, at = _located(readings_of, question, key, offset)
    start = reading.text_offset(max(at, 0)) - 1 if at >= 0 else 0
    return reading.text[max(0, start - BEFORE) : start + question_chars + AFTER], reading.stream, at


def record_pairs(record_id: str, text: str, table: KeyTable, locators: pa.Table, width: int = WINDOW) -> list[dict]:
    """One pair per item whose key-1 window a reading of *text* holds; an item keyed by several streams
    takes the first locator row found."""
    streams = record_readings(text, width)
    found = table.found(np.concatenate([reading.hashes for reading in streams]))
    pairs: dict[str, dict] = {}
    for row in table.items_of(found):
        located = {name: locators.column(name)[int(row)].as_py() for name in LOCATOR_COLUMNS if name != "key1_hash"}
        if located["item"] in pairs:
            continue
        right, _, at = excerpt(
            streams, located["question_chars"], located["question"], located["key"], located["offset"]
        )
        pairs[located["item"]] = {
            "record": record_id,
            "item": located["item"],
            "kind": located["kind"],
            "eval": located["eval"],
            "right": right,
            "record_chars": len(text),
            "located": at >= 0,
        }
    return list(pairs.values())
