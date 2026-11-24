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
import functools
import itertools
import os
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
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
# A chunk of a reading (``Reading``) starts at the first safe character (``_safe_pattern``) after this many
# characters. A chunk longer than LONG_CHUNK_CHARS of plain ASCII text maps positions by counting.
CHUNK_CHARS = 4096
LONG_CHUNK_CHARS = 4 * CHUNK_CHARS
# Places of a question's first window that ``Reading.first`` checks; past them, ``_located`` uses the key.
FIRST_PLACES = 64
_ALNUM = np.zeros(256, dtype=np.int32)
_ALNUM[np.frombuffer(b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", np.uint8)] = 1
BEFORE = 1500
AFTER = 3000
# Pieces of a record over MAX_DOC_CHARS overlap by this much, so a key hit and its excerpt lie in one piece.
PIECE_OVERLAP = 2 * (BEFORE + AFTER)
# A piece edge moves back by at most this much so that it does not cut a LaTeX command or an escape.
EDGE_SHIFT = 16
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


def template_bounds(streams: list[str], groups: list[str], width: int = WINDOW) -> list[tuple[int, int]]:
    """For each stream, the part outside the prefix and suffix that every stream of its group shares (an eval's
    prompt template: "Question: ... Answer:", a fixed instruction). A copy of an item always holds the item's own
    part, whatever template it was written with; a stream whose own part is shorter than a window keeps all of it.
    """
    members: dict[str, list[int]] = {}
    for number, group in enumerate(groups):
        members.setdefault(group, []).append(number)
    bounds = [(0, len(stream)) for stream in streams]
    for numbers in members.values():
        if len(numbers) < 2:
            continue
        group_streams = [streams[n] for n in numbers]
        prefix = len(os.path.commonprefix(group_streams))
        suffix = len(os.path.commonprefix([stream[::-1] for stream in group_streams]))
        for n in numbers:
            low, high = prefix, len(streams[n]) - suffix
            if high - low >= width:
                bounds[n] = (low, high)
    return bounds


def candidate_windows(
    streams: list[str], width: int = WINDOW, bounds: list[tuple[int, int]] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hashes of the candidate windows, the item of each, and its offset in the item stream. With *bounds*, a
    stream's windows lie inside its (start, end) bound (see ``template_bounds``)."""
    hashes, owners, offsets = [], [], []
    for number, stream in enumerate(streams):
        if len(stream) < width:
            continue
        low, high = bounds[number] if bounds is not None else (0, len(stream))
        if not (0 <= low and high - low >= width and high <= len(stream)):
            raise ValueError(f"bound {(low, high)} of a {len(stream)}-character stream must hold a window")
        last = high - width - low
        starts = (
            sorted({low + round(k * last / (WINDOWS_PER_ITEM - 1)) for k in range(WINDOWS_PER_ITEM)}) if last else [low]
        )
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


def keys_table(
    windows: Windows, window_counts: np.ndarray, width: int = WINDOW, tie: np.ndarray | None = None
) -> pa.Table:
    """One row per key stream: its first window, its rarest window (key 1), and the rarest window that does
    not overlap key 1 (key 2). Hash columns are uint64; 0 means "no such key", and its count is -1. Among equally
    rare windows the lowest *tie* value (one per candidate window) wins, then the earliest.

    A stream other than the question keeps its key only when its rarest window occurs in no reference
    record: an extra key that common text holds (a generic stem taken for a passage) would only add load.
    """
    hashes, owners, offsets, ids, kinds = windows
    columns: dict[str, list] = {name: [] for name in KEY_COLUMNS}
    order = np.lexsort((offsets, window_counts, owners) if tie is None else (offsets, tie, window_counts, owners))
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


@functools.cache
def _safe_pattern() -> re.Pattern:
    """A pattern of the safe characters: the stream length of a text is additive before each of them.

    A safe character is a starter that no character composes with, whose NFKC form starts with such a starter,
    and whose NFKC form does not start with an ASCII letter or a backslash (which could continue or start a
    LaTeX command across the split). Lowercasing keeps the length of each character, so it never breaks
    additivity. Computed over the Basic Multilingual Plane, plus the CJK ideograph planes.
    """
    # Hangul composes by algorithm, not by the decomposition table: a leading consonant takes a vowel jamo,
    # and a syllable takes a trailing consonant jamo.
    seconds = {*range(0x1161, 0x1176), *range(0x11A8, 0x11C3)}
    for code in range(0x110000):
        parts = unicodedata.decomposition(chr(code)).split()
        if len(parts) == 2 and not parts[0].startswith("<"):
            seconds.add(int(parts[1], 16))

    def safe(code: int) -> bool:
        char = chr(code)
        folded = unicodedata.normalize("NFKC", char)
        if not folded or unicodedata.combining(char) or code in seconds:
            return False
        head = folded[0]
        return not unicodedata.combining(head) and ord(head) not in seconds and not re.match(r"[A-Za-z\\]", head)

    codes = [code for code in range(0x10000) if not 0xD800 <= code <= 0xDFFF and safe(code)]
    ranges, start = [], codes[0]
    for previous, code in itertools.pairwise([*codes, -1]):
        if code != previous + 1:
            ranges.append(f"\\U{start:08x}-\\U{previous:08x}" if previous > start else f"\\U{start:08x}")
            start = code
    return re.compile("[" + "".join(ranges) + "\\U00020000-\\U0003134f]")


class Reading:
    """One reading of a record (``readings``): its text, stream and window hashes, with the lookups that
    ``excerpt`` needs.

    ``first`` finds the first place of a stream piece through a sorted index of the window hashes; it checks
    at most ``FIRST_PLACES`` places where the piece's first window occurs. ``text_offset`` maps a stream
    position to a text offset. A stream's length is additive before a safe character (``_safe_pattern``:
    whitespace, digits, punctuation other than the backslash, and the letters of most scripts other than
    ASCII letters): no LaTeX command or NFKC composition crosses one. So the reading keeps the exact stream
    length before each chunk start (the first safe character after ``CHUNK_CHARS`` characters) and searches
    one chunk; a long chunk of plain ASCII text is mapped by counting its letters and digits.

    Limit: the cost is not linear for every record. In a stretch of length L > ``LONG_CHUNK_CHARS`` with no
    safe character whose body is not plain ASCII without a backslash (fullwidth letters, combining marks, or
    LaTeX commands with nothing between them), H lookups cost O(H * L * log L).
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
        self._counts: dict[int, np.ndarray] = {}

    def first(self, piece: str) -> int:
        """The first position of the stream *piece* in the stream (``str.find``), among the first
        ``FIRST_PLACES`` places of its first window; -1 if none of them holds the piece."""
        if len(piece) < self.width:
            return self.stream.find(piece)
        if self._order is None or self._sorted is None:
            # A stable sort keeps the places of equal hashes in stream order.
            self._order = np.argsort(self.hashes, kind="stable")
            self._sorted = self.hashes[self._order]
        head = window_hashes(piece[: self.width], self.width)[0]
        low = int(np.searchsorted(self._sorted, head, side="left"))
        high = min(int(np.searchsorted(self._sorted, head, side="right")), low + FIRST_PLACES)
        for position in self._order[low:high].tolist():
            if self.stream.startswith(piece, position):
                return position
        return -1

    def _index(self) -> None:
        starts, position, safe = [0], CHUNK_CHARS, _safe_pattern()

        def split_before_long_stretch(end: int) -> None:
            # A long stretch without a safe character follows: start a chunk at the last safe character before
            # it, so that the stretch is one chunk of its own.
            if end - starts[-1] > LONG_CHUNK_CHARS:
                last = max((m.start() for m in safe.finditer(self.text, starts[-1] + 1, position)), default=None)
                if last is not None:
                    starts.append(last)

        while (match := safe.search(self.text, position)) is not None:
            split_before_long_stretch(match.start())
            starts.append(match.start())
            position = match.start() + CHUNK_CHARS
        split_before_long_stretch(len(self.text))
        before = [0]
        for start, end in itertools.pairwise(starts):
            before.append(before[-1] + len(text_stream(self.text[start:end])))
        self._starts, self._before = starts, before

    def text_offset(self, position: int) -> int:
        """The smallest prefix length of the text whose stream is longer than *position*."""
        if not self._starts:
            self._index()
        chunk = bisect.bisect_right(self._before, position) - 1
        start, base = self._starts[chunk], self._before[chunk]
        low, high = start, self._starts[chunk + 1] if chunk + 1 < len(self._starts) else len(self.text)
        if high - start > LONG_CHUNK_CHARS and chunk not in self._counts:
            head, rest = self.text[start], self.text[start + 1 : high]
            if rest.isascii() and "\\" not in rest and "\\" not in unicodedata.normalize("NFKC", head):
                # A safe character, then plain ASCII without a backslash: NFKC, lowercasing and LaTeX removal
                # leave the rest alone, so its stream is its letters and digits.
                counts = np.cumsum(_ALNUM[np.frombuffer(rest.encode("ascii"), np.uint8)])
                self._counts[chunk] = np.concatenate([[0], counts]) + len(text_stream(head))
        counts = self._counts.get(chunk)
        if counts is not None:
            index = int(np.searchsorted(counts, position - base, side="right"))
            return start + index + 1 if index < len(counts) else high
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
        at = reading.first(question) if len(question) >= reading.width else -1
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
    reading's stream, and the item's position in that stream: where its question is, else an estimate from
    the key, or -1 when neither is found (the excerpt then starts at the record's start).

    *readings_of* are the record's readings (``record_readings``), *question* the item's question stream and
    *question_chars* the length of its question text, *key* the matched key window and *offset* its position
    in the keyed stream.
    """
    reading, at = _located(readings_of, question, key, offset)
    start = reading.text_offset(max(at, 0)) - 1 if at >= 0 else 0
    return reading.text[max(0, start - BEFORE) : start + question_chars + AFTER], reading.stream, at


@dataclass(frozen=True)
class Candidate:
    """A stage-1 candidate: a record that holds an item's key, with the record text around the item."""

    record: str
    item: str
    kind: str
    eval: str
    right: str
    record_chars: int
    located: bool


def pieces(text: str, size: int = MAX_DOC_CHARS, overlap: int = PIECE_OVERLAP, width: int = WINDOW) -> Iterator[str]:
    """Pieces of *text* of at most *size* characters. Consecutive pieces share at least *overlap* characters, so
    every stretch that long lies inside one piece, and the shared part widens (up to half a piece) until its text
    stream holds *width* characters, so a key window whose letters are spread over a long run of spaces or
    punctuation also lies inside one piece. A piece edge does not cut a backslash sequence of up to
    ``EDGE_SHIFT`` characters (a LaTeX command or escape). Each piece starts more than *overlap* characters after
    the one before. Escapes are read before cutting (``long_record_pairs``), so a piece is scanned as it is."""
    if size <= 2 * overlap + 4 * EDGE_SHIFT:
        raise ValueError(f"piece size {size} must exceed twice the overlap {overlap} and four edge shifts")
    start = 0
    while True:
        end = len(text) if start + size >= len(text) else _edge(text, start + size)
        yield text[start:end]
        if end == len(text):
            return
        shared = overlap
        while True:
            start_next = _edge(text, end - shared)
            if len(text_stream(text[start_next:end])) >= width or shared >= size // 2:
                break
            shared = min(2 * shared, size // 2)
        start = start_next


def _edge(text: str, position: int) -> int:
    """*position*, moved back before a LaTeX command or escape that it would cut: to the start of the last
    backslash run within ``EDGE_SHIFT`` characters before it."""
    backslash = text.rfind("\\", max(0, position - EDGE_SHIFT), position)
    if backslash < 0:
        return position
    while backslash > max(0, position - EDGE_SHIFT) and text[backslash - 1] == "\\":
        backslash -= 1
    return backslash


def long_record_pairs(
    record_id: str,
    text: str,
    table: KeyTable,
    locators: pa.Table,
    width: int = WINDOW,
    size: int = MAX_DOC_CHARS,
    overlap: int = PIECE_OVERLAP,
) -> list[Candidate]:
    """``record_pairs`` for a long record: each of its ``readings`` (JSON escapes read on the whole record, as for a
    short record) is cut into ``pieces``, and each piece is scanned as that reading. Every distinct excerpt is a
    pair, so a later exact copy is scored even when an earlier piece holds an edited one; a hit near a piece's edge
    also gives a pair with its context cut, which the record's best score makes harmless."""
    found: dict[tuple[str, str], Candidate] = {}
    for reading in readings(text):
        for piece in pieces(reading, size, overlap, width):
            for pair in _pairs(record_id, [Reading(piece, width)], len(text), table, locators):
                found.setdefault((pair.item, pair.right), pair)
    return list(found.values())


def record_pairs(record_id: str, text: str, table: KeyTable, locators: pa.Table, width: int = WINDOW) -> list[Candidate]:
    """One candidate per item whose key-1 window a reading of *text* holds; an item keyed by several streams
    takes the first locator row found."""
    return _pairs(record_id, record_readings(text, width), len(text), table, locators)


def _pairs(
    record_id: str, readings_of: list[Reading], record_chars: int, table: KeyTable, locators: pa.Table
) -> list[Candidate]:
    found = table.found(np.concatenate([reading.hashes for reading in readings_of]))
    pairs: dict[str, Candidate] = {}
    for row in table.items_of(found):
        located = {name: locators.column(name)[int(row)].as_py() for name in LOCATOR_COLUMNS if name != "key1_hash"}
        if located["item"] in pairs:
            continue
        right, _, at = excerpt(
            readings_of, located["question_chars"], located["question"], located["key"], located["offset"]
        )
        pairs[located["item"]] = Candidate(
            record=record_id,
            item=located["item"],
            kind=located["kind"],
            eval=located["eval"],
            right=right,
            record_chars=record_chars,
            located=at >= 0,
        )
    return list(pairs.values())
