# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Flag machine output and data dumps in long pretraining documents.

Long-context training wants documents whose later tokens depend on distant
earlier ones. Length-selected web and code corpora are instead full of text
that is incompressible (binary, base64), trivially repetitive (a column of
zeros) or structured data whose only long-range pattern is its schema. The
quality classifier does not separate these, so this module does it from the
text alone.

A document is judged from windows of its text, so a caller holding a
multi-million-token row can decode a few windows instead of the whole row.
Each window is cut into chunks, each chunk gets at most one
:class:`JunkCategory`, and the document is junk when enough chunks are.
"""

import re
import zlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

CHUNK_CHARS = 4096
# Shorter chunks carry too little text for the ratio features to be stable.
MIN_CHUNK_CHARS = 512
# A document is junk when at least this share of its chunks is junk of any kind,
# or when this smaller share is machine junk: a notebook that is one third
# base64 images is not worth its other two thirds.
JUNK_CHUNK_SHARE = 0.6
MACHINE_JUNK_CHUNK_SHARE = 0.3


class JunkCategory(StrEnum):
    BINARY = "binary"
    ENCODED_BLOB = "encoded_blob"
    REPETITION = "repetition"
    LOCK_FILE = "lock_file"
    MINIFIED_CODE = "minified_code"
    NUMERIC_TABLE = "numeric_table"
    STRUCTURED_DATA = "structured_data"
    ITEM_LIST = "item_list"


# Nothing here is worth a long-context training position.
MACHINE_JUNK = frozenset(
    {
        JunkCategory.BINARY,
        JunkCategory.ENCODED_BLOB,
        JunkCategory.REPETITION,
        JunkCategory.LOCK_FILE,
        JunkCategory.MINIFIED_CODE,
    }
)
# Real but shallow structure: worth a small capped share, not the half of the tokens it holds unfiltered.
DATA_DUMPS = frozenset({JunkCategory.NUMERIC_TABLE, JunkCategory.STRUCTURED_DATA, JunkCategory.ITEM_LIST})

BINARY_CHAR_SHARE = 0.03
ENCODED_CHAR_SHARE = 0.4
# zlib output over input. Prose sits near 0.4 and minified code near 0.25.
REPETITION_COMPRESSION_RATIO = 0.08
REPEATED_LINE_SHARE = 0.7
# Records that compress well without repeating verbatim: the same line shape
# with different values, or one long line of fields.
TEMPLATED_COMPRESSION_RATIO = 0.2
TEMPLATED_COMPRESSED_LINE_SHARE = 0.4
# Formatted source code is repetitive too; lines ending a statement or block mark it.
CODE_LINE_SHARE = 0.3
MIN_LINES = 12
MIN_RECORD_LINES = 4
LOCK_MARKERS_PER_CHUNK = 6
# Minified and bundled code: long lines dense with syntax, or source held in escaped strings.
MINIFIED_MEAN_LINE_CHARS = 300
MINIFIED_SYNTAX_SHARE = 0.06
MINIFIED_SPACE_SHARE = 0.12
ESCAPED_NEWLINES_PER_CHAR = 0.004
NUMERIC_DIGIT_SHARE = 0.4
NUMERIC_LETTER_SHARE = 0.25
JSON_KEYS_PER_CHAR = 0.01
MARKUP_CHAR_SHARE = 0.4
DELIMITED_LINE_SHARE = 0.7
# Three separators make a row for most delimiters; a tab inside a line is deliberate, so one is enough.
MIN_DELIMITERS = {",": 3, ";": 3, "|": 3, "\t": 1}
URL_CHAR_SHARE = 0.3
# Prose has a comma every 80 characters or so; a record has a separator every 12 or fewer.
SEPARATOR_SHARE = 0.06
KEY_VALUE_LINE_SHARE = 0.6
LOG_LINE_SHARE = 0.5
SQL_LINE_SHARE = 0.2
TEMPLATED_LINE_SHARE = 0.8
# Word lists and label files: many short lines of a few words and almost no syntax.
ITEM_MEAN_LINE_CHARS = 28
ITEM_LINE_SHARE = 0.85
ITEM_LINE_WORDS = 6
ITEM_SYNTAX_SHARE = 0.03
# Words are counted by spaces, which says nothing about scripts written without them.
ITEM_NON_ASCII_SHARE = 0.3

_BINARY_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f�␀]")
_ENCODED_RUNS = re.compile(
    r"[A-Za-z0-9+/_-]{120,}={0,2}"  # base64 and other unbroken alphanumeric payloads
    r"|(?:(?:0x)?[0-9a-fA-F]{2}[\s,:]{0,2}){64,}"  # hex dumps and byte arrays
    r"|(?:[A-Za-z0-9+/]{1,12}[,;]+){40,}"  # source-map VLQ mappings
    r"|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\[0-3][0-7]{2}"  # escaped bytes and code points
    r"|\b[0-9a-fA-F]{16,}\b|\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b"  # hashes and UUIDs
)
_LOCK_MARKER = re.compile(r'integrity|"resolved"|resolved "|checksum|\bh1:|sha(?:1|256|512)[-:=]|content-hash')
_HASH = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
_CODE_SYNTAX = re.compile(r"[;:{}()=]")
_ESCAPED_NEWLINE = re.compile(r"\\[nr]")
_JSON_KEY = re.compile(r'"[^"\n]{1,60}"\s*:')
_MARKUP = re.compile(r"<[^<>]{1,400}>")
# Tabs and semicolons count only between fields, so indentation and statement ends do not.
# Pipes are left to the per-line rule: a report's markdown tables are not a data dump.
_SEPARATOR = re.compile(r",|(?<=\S)\t(?=\S)|;(?=\S)")
_CODE_LINE_END = re.compile(r"[;{}]\s*$")
# A short key and at most three words of value. Statements ending in ``;`` or ``{`` are code, not records.
_KEY_VALUE_LINE = re.compile(r"^\s*(?:- )?[\w .()\[\]/-]{1,40}?\s*[:=]\s*(?:\S+(?: \S+){0,2})?(?<![;{,])\s*$")
_LOG_LINE = re.compile(
    r"^\[?(?:\d{4}[-/]\d\d[-/]\d\d[ T]\d\d:\d\d|\d\d:\d\d:\d\d[.,]\d+|\w{3} +\d+ \d\d:\d\d:\d\d"
    r"|(?:DEBUG|INFO|WARN|WARNING|ERROR|TRACE)\b)"
)
_SQL_LINE = re.compile(r"^\s*(?:INSERT INTO|\(\d+\s*,)", re.I)
_DIGITS = re.compile(r"\d+")
_URL = re.compile(r"https?://[^\s\"'<>),|]+")


@dataclass(frozen=True)
class ChunkFeatures:
    """Measurements of one chunk of text. Shares are of the chunk's characters unless stated."""

    binary_share: float
    encoded_share: float
    compression_ratio: float
    lines: int
    mean_line_chars: float
    repeated_line_share: float
    templated_line_share: float
    lock_markers: int
    syntax_share: float
    space_share: float
    code_line_share: float
    escaped_newlines_per_char: float
    digit_share: float  # of non-space characters
    letter_share: float  # of non-space characters
    separator_share: float  # of non-space characters
    json_keys_per_char: float
    markup_share: float
    url_share: float
    non_ascii_share: float
    delimited_line_share: float
    key_value_line_share: float
    log_line_share: float
    sql_line_share: float
    item_line_share: float


@dataclass(frozen=True)
class JunkVerdict:
    """What a document's chunks were judged to be.

    Attributes:
        category: The most common junk category when the document is junk, else ``None``.
        junk_share: Share of judged chunks with any junk category.
        chunks: Number of chunks judged.
        counts: Chunks per category.
    """

    category: JunkCategory | None
    junk_share: float
    chunks: int
    counts: dict[JunkCategory, int]


def _matched_share(pattern: re.Pattern[str], text: str) -> float:
    return sum(len(match.group(0)) for match in pattern.finditer(text)) / len(text)


def _delimited_line_share(lines: list[str]) -> float:
    """Largest share of lines that agree on how many of one delimiter they hold, indentation aside."""
    best = 0
    for delimiter, minimum in MIN_DELIMITERS.items():
        counts = Counter(line.strip().count(delimiter) for line in lines)
        best = max(best, max((n for fields, n in counts.items() if fields >= minimum), default=0))
    return best / len(lines)


def chunk_features(chunk: str) -> ChunkFeatures:
    size = len(chunk)
    lines = [line for line in chunk.split("\n") if line.strip()]
    count = max(1, len(lines))
    visible = max(1, sum(1 for char in chunk if not char.isspace()))
    encoded = chunk.encode("utf-8", "replace")
    return ChunkFeatures(
        binary_share=_matched_share(_BINARY_CHARS, chunk),
        encoded_share=_matched_share(_ENCODED_RUNS, chunk),
        compression_ratio=len(zlib.compress(encoded, 6)) / len(encoded),
        lines=len(lines),
        mean_line_chars=size / count,
        repeated_line_share=1 - len(set(lines)) / count,
        templated_line_share=1 - len({_DIGITS.sub("0", line) for line in lines}) / count,
        lock_markers=min(len(_LOCK_MARKER.findall(chunk)), len(_HASH.findall(chunk))),
        syntax_share=len(_CODE_SYNTAX.findall(chunk)) / size,
        space_share=(size - visible) / size,
        code_line_share=sum(1 for line in lines if _CODE_LINE_END.search(line)) / count,
        escaped_newlines_per_char=len(_ESCAPED_NEWLINE.findall(chunk)) / size,
        digit_share=sum(1 for char in chunk if char.isdigit()) / visible,
        letter_share=sum(1 for char in chunk if char.isalpha()) / visible,
        separator_share=len(_SEPARATOR.findall(chunk)) / visible,
        json_keys_per_char=len(_JSON_KEY.findall(chunk)) / size,
        markup_share=_matched_share(_MARKUP, chunk),
        url_share=_matched_share(_URL, chunk),
        non_ascii_share=sum(1 for char in chunk if ord(char) > 127) / size,
        delimited_line_share=_delimited_line_share(lines) if lines else 0.0,
        key_value_line_share=sum(1 for line in lines if _KEY_VALUE_LINE.match(line)) / count,
        log_line_share=sum(1 for line in lines if _LOG_LINE.match(line)) / count,
        sql_line_share=sum(1 for line in lines if _SQL_LINE.match(line)) / count,
        item_line_share=sum(1 for line in lines if len(line.split()) <= ITEM_LINE_WORDS) / count,
    )


def features_junk_category(f: ChunkFeatures) -> JunkCategory | None:
    """Apply the rules in order of how little the text is worth, so the worst label wins."""
    enough_lines = f.lines >= MIN_LINES
    if f.binary_share > BINARY_CHAR_SHARE:
        return JunkCategory.BINARY
    if f.encoded_share > ENCODED_CHAR_SHARE:
        return JunkCategory.ENCODED_BLOB
    if f.compression_ratio < REPETITION_COMPRESSION_RATIO:
        return JunkCategory.REPETITION
    if enough_lines and f.repeated_line_share > REPEATED_LINE_SHARE:
        return JunkCategory.REPETITION
    if f.lock_markers >= LOCK_MARKERS_PER_CHUNK:
        return JunkCategory.LOCK_FILE
    if f.digit_share > NUMERIC_DIGIT_SHARE and f.letter_share < NUMERIC_LETTER_SHARE:
        return JunkCategory.NUMERIC_TABLE
    if f.json_keys_per_char > JSON_KEYS_PER_CHAR:
        return JunkCategory.STRUCTURED_DATA
    if f.mean_line_chars > MINIFIED_MEAN_LINE_CHARS and (
        (f.syntax_share > MINIFIED_SYNTAX_SHARE and f.space_share < MINIFIED_SPACE_SHARE)
        or f.escaped_newlines_per_char > ESCAPED_NEWLINES_PER_CHAR
    ):
        return JunkCategory.MINIFIED_CODE
    if f.markup_share > MARKUP_CHAR_SHARE:
        return JunkCategory.STRUCTURED_DATA
    if f.separator_share > SEPARATOR_SHARE or f.url_share > URL_CHAR_SHARE:
        return JunkCategory.STRUCTURED_DATA
    if f.compression_ratio < TEMPLATED_COMPRESSION_RATIO and (
        f.mean_line_chars > MINIFIED_MEAN_LINE_CHARS
        or (
            enough_lines
            and f.templated_line_share > TEMPLATED_COMPRESSED_LINE_SHARE
            and f.code_line_share < CODE_LINE_SHARE
        )
    ):
        return JunkCategory.STRUCTURED_DATA
    if f.lines >= MIN_RECORD_LINES and (f.log_line_share > LOG_LINE_SHARE or f.sql_line_share > SQL_LINE_SHARE):
        return JunkCategory.STRUCTURED_DATA
    if not enough_lines:
        return None
    if (
        f.delimited_line_share > DELIMITED_LINE_SHARE
        or f.key_value_line_share > KEY_VALUE_LINE_SHARE
        or f.templated_line_share > TEMPLATED_LINE_SHARE
    ):
        return JunkCategory.STRUCTURED_DATA
    if (
        f.mean_line_chars < ITEM_MEAN_LINE_CHARS
        and f.item_line_share > ITEM_LINE_SHARE
        and f.syntax_share < ITEM_SYNTAX_SHARE
        and f.non_ascii_share < ITEM_NON_ASCII_SHARE
    ):
        return JunkCategory.ITEM_LIST
    return None


def chunks(windows: Iterable[str]) -> Iterable[str]:
    """Cut windows of a document into the chunks the filter judges."""
    for window in windows:
        for start in range(0, len(window), CHUNK_CHARS):
            chunk = window[start : start + CHUNK_CHARS]
            if len(chunk) >= MIN_CHUNK_CHARS:
                yield chunk


def junk_verdict(windows: Iterable[str]) -> JunkVerdict:
    """Judge a document from windows of its text.

    Args:
        windows: The whole document as one string, or any sample of its text.
            Sample the interior as well as the ends: a data dump often opens
            with a readable header.
    """
    counts: Counter[JunkCategory] = Counter()
    judged = 0
    for chunk in chunks(windows):
        judged += 1
        category = features_junk_category(chunk_features(chunk))
        if category is not None:
            counts[category] += 1
    if not judged:
        return JunkVerdict(None, 0.0, 0, {})
    junk_share = sum(counts.values()) / judged
    machine = Counter({category: n for category, n in counts.items() if category in MACHINE_JUNK})
    if machine.total() / judged >= MACHINE_JUNK_CHUNK_SHARE:
        return JunkVerdict(machine.most_common(1)[0][0], junk_share, judged, dict(counts))
    category = counts.most_common(1)[0][0] if junk_share >= JUNK_CHUNK_SHARE else None
    return JunkVerdict(category, junk_share, judged, dict(counts))
