# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic structural measurements of bounded document views."""

import math
import re
import zlib
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields

import numpy as np

FEATURES_VERSION = 1
CHUNK_CHARS = 4096
MIN_CHUNK_CHARS = 512
MIN_DELIMITERS = {",": 3, ";": 3, "|": 3, "\t": 1}
SINGLE_TAB_LINE_CHARS = 80
ITEM_LINE_WORDS = 6

_BINARY_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f�␀]")
_ENCODED_RUNS = re.compile(
    r"[A-Za-z0-9+/_-]{120,}={0,2}"  # base64 and other unbroken alphanumeric payloads
    r"|(?:[A-Za-z0-9+/]{56,}={0,2}(?:\\n)?\s+){3,}"  # the same, wrapped at a fixed line width
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
# Pipe-separated structure is measured separately by the per-line feature.
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
    """Measurements of one text chunk.

    Line shares use nonempty lines. Digit, letter and separator shares use
    non-space characters; other shares use all characters. Counts and rates
    are named separately.
    """

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


def _matched_share(pattern: re.Pattern[str], text: str) -> float:
    return sum(len(match.group(0)) for match in pattern.finditer(text)) / len(text)


def _delimited_line_share(lines: list[str]) -> float:
    """Largest share of lines that agree on how many of one delimiter they hold, indentation aside."""
    best = 0
    stripped = [line.strip() for line in lines]
    for delimiter, minimum in MIN_DELIMITERS.items():
        counts = Counter(
            (
                0
                if delimiter == "\t" and len(line) > SINGLE_TAB_LINE_CHARS and line.count("\t") == 1
                else line.count(delimiter)
            )
            for line in stripped
        )
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


def chunks(windows: Iterable[str]) -> Iterable[str]:
    """Cut windows of a document into the chunks the filter judges."""
    for window in windows:
        for start in range(0, len(window), CHUNK_CHARS):
            chunk = window[start : start + CHUNK_CHARS]
            if len(chunk) >= MIN_CHUNK_CHARS:
                yield chunk


TRAJECTORY = re.compile(
    r"This trajectory |# Repository Context|<tool_call>|<\|(?:im_start|start_header_id)\|>"
    r'|"(?:gold_patch|model_patch|instance_id|test_patch)"'
)
BENCHMARK = re.compile(r"\b(?:SWE-bench|HumanEval|Terminal-Bench|pass@\d+|benchmark_results)\b", re.I)
QA = re.compile(
    r"(?:^|\n)\s*(?:Question|Answer|Problem|Solution|Q|A|Pregunta|Respuesta|Frage|Antwort|質問|答え|问题|答案)\s*[:\uff1a]",
    re.I,
)
HTML = re.compile(r"<(?:html|head|body|script|style|div|table|tr|td|th|span|!DOCTYPE)\b", re.I)
STRUCTURED = re.compile(r"\b(?:INSERT INTO|CREATE TABLE|sourceMappingURL|lockfileVersion|dependencies|integrity)\b")
SPECIAL = re.compile(r"<\|(?:begin_of_text|end_of_text)\|>")
EXTRA_NAMES = (
    "char_entropy",
    "class_entropy",
    "line_p90",
    "line_max",
    "word_mean",
    "trajectory_rate",
    "benchmark_rate",
    "qa_rate",
    "html_rate",
    "structure_rate",
    "replacement_share",
    "ascii_share",
)
CHUNK_NAMES = tuple(field.name for field in fields(ChunkFeatures)) + EXTRA_NAMES
FEATURE_NAMES = (
    "log_tokens",
    "log_visible_chars",
    *(f"{stat}_{name}" for stat in ("mean", "max", "min", "std") for name in CHUNK_NAMES),
)


def entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    return -sum((count / total) * math.log2(count / total) for count in counts if count) if total else 0.0


def measurements(text: str) -> list[float]:
    base = list(asdict(chunk_features(text)).values())
    size = len(text)
    lines = [len(line) for line in text.splitlines()]
    words = text.split()
    classes = Counter("L" if c.isalpha() else "D" if c.isdigit() else "S" if c.isspace() else "P" for c in text)
    extra = [
        entropy(list(Counter(text).values())),
        entropy(list(classes.values())),
        np.percentile(lines, 90) if lines else 0,
        max(lines, default=0),
        np.mean([len(word) for word in words]) if words else 0,
        *(len(pattern.findall(text)) * 1000 / size for pattern in (TRAJECTORY, BENCHMARK, QA, HTML, STRUCTURED)),
        text.count("\ufffd") / size,
        sum(ord(c) < 128 for c in text) / size,
    ]
    return base + extra


def document_features(windows: Sequence[str], tokens: int) -> np.ndarray:
    """Measure a document from views selected with the model's window policy."""
    cleaned = [SPECIAL.sub("", window) for window in windows]
    rows = [measurements(chunk) for chunk in chunks(cleaned)]
    # Include short documents instead of manufacturing all-zero structural signals.
    if not rows:
        text = "\n".join(cleaned)
        rows = [measurements(text)] if text else [[0.0] * len(CHUNK_NAMES)]
    values = np.asarray(rows, dtype=np.float64)
    features = np.concatenate(
        (
            [np.log1p(tokens), np.log1p(sum(map(len, cleaned)))],
            values.mean(axis=0),
            values.max(axis=0),
            values.min(axis=0),
            values.std(axis=0),
        )
    )
    assert len(features) == len(FEATURE_NAMES)
    return features
