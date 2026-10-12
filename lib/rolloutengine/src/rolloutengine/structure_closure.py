# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect JSON prefixes without treating unfinished generation as invalid JSON."""

import json
import re
from dataclasses import dataclass
from enum import StrEnum


class PrefixStatus(StrEnum):
    COMPLETE = "complete"
    EXTENDABLE = "extendable"
    INVALID = "invalid"


@dataclass(frozen=True)
class ClosureState:
    status: PrefixStatus
    open_structures: tuple[str, ...]
    closing_suffix: str | None
    error_offset: int | None = None
    reason: str | None = None


class _PrefixEnd(Exception):
    def __init__(self, suffix: str | None):
        self.suffix = suffix


class _InvalidPrefix(Exception):
    def __init__(self, offset: int, reason: str):
        self.offset = offset
        self.reason = reason


_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_NUMBER_PREFIX = re.compile(r"-?(?:(?:0|[1-9][0-9]*)(?:\.[0-9]*)?|(?:0|[1-9][0-9]*)(?:\.[0-9]+)?[eE][+-]?[0-9]*)?")
_WHITESPACE = " \t\r\n"
_HEX = frozenset("0123456789abcdefABCDEF")
_CLOSE = {"{": "}", "[": "]", '"': '"'}


class _JsonPrefix:
    def __init__(self, text: str):
        self.text = text
        self.position = 0
        self.open: list[str] = []

    def skip_space(self):
        while self.position < len(self.text) and self.text[self.position] in _WHITESPACE:
            self.position += 1

    def suffix(self) -> str:
        return "".join(_CLOSE[value] for value in reversed(self.open))

    def fail(self, reason: str):
        raise _InvalidPrefix(self.position, reason)

    def value(self):
        self.skip_space()
        if self.position == len(self.text):
            raise _PrefixEnd(None)
        char = self.text[self.position]
        if char == '"':
            self.string(key=False)
        elif char == "{":
            self.object()
        elif char == "[":
            self.array()
        elif char in "tfn":
            literal = {"t": "true", "f": "false", "n": "null"}[char]
            for expected in literal:
                if self.position == len(self.text):
                    raise _PrefixEnd(None)
                if self.text[self.position] != expected:
                    self.fail("Invalid JSON literal")
                self.position += 1
        elif char in "-0123456789":
            self.number()
        else:
            self.fail("Expected a JSON value")

    def number(self):
        start = self.position
        while self.position < len(self.text):
            char = self.text[self.position]
            if char in _WHITESPACE + ",]}":
                break
            candidate = self.text[start : self.position + 1]
            if not _NUMBER_PREFIX.fullmatch(candidate):
                self.fail("Invalid JSON number")
            self.position += 1
        number = self.text[start : self.position]
        if _NUMBER.fullmatch(number):
            return
        if self.position == len(self.text):
            raise _PrefixEnd(None)
        self.fail("Unfinished JSON number")

    def string(self, *, key: bool) -> str:
        start = self.position
        self.position += 1
        self.open.append('"')
        while self.position < len(self.text):
            char = self.text[self.position]
            if char == '"':
                self.position += 1
                self.open.pop()
                return json.loads(self.text[start : self.position])
            if ord(char) < 32:
                self.fail("Unescaped control character in JSON string")
            self.position += 1
            if char != "\\":
                continue
            if self.position == len(self.text):
                raise _PrefixEnd(None)
            escape = self.text[self.position]
            if escape not in '"\\/bfnrtu':
                self.fail("Invalid JSON escape")
            self.position += 1
            if escape == "u":
                for _ in range(4):
                    if self.position == len(self.text):
                        raise _PrefixEnd(None)
                    if self.text[self.position] not in _HEX:
                        self.fail("Invalid JSON unicode escape")
                    self.position += 1
        raise _PrefixEnd(None if key else self.suffix())

    def object(self):
        self.open.append("{")
        self.position += 1
        self.skip_space()
        if self.position == len(self.text):
            raise _PrefixEnd(self.suffix())
        if self.text[self.position] == "}":
            self.position += 1
            self.open.pop()
            return
        keys = set()
        while True:
            if self.position == len(self.text):
                raise _PrefixEnd(None)
            if self.text[self.position] != '"':
                self.fail("Expected an object key")
            key = self.string(key=True)
            if key in keys:
                self.position -= 1
                self.fail("Duplicate JSON key")
            keys.add(key)
            self.skip_space()
            if self.position == len(self.text):
                raise _PrefixEnd(None)
            if self.text[self.position] != ":":
                self.fail("Expected colon after object key")
            self.position += 1
            self.value()
            if self.separator("}"):
                return

    def array(self):
        self.open.append("[")
        self.position += 1
        self.skip_space()
        if self.position == len(self.text):
            raise _PrefixEnd(self.suffix())
        if self.text[self.position] == "]":
            self.position += 1
            self.open.pop()
            return
        while True:
            self.value()
            if self.separator("]"):
                return

    def separator(self, closer: str) -> bool:
        self.skip_space()
        if self.position == len(self.text):
            raise _PrefixEnd(self.suffix())
        char = self.text[self.position]
        if char == closer:
            self.position += 1
            self.open.pop()
            return True
        if char != ",":
            self.fail("Expected comma or matching closer")
        self.position += 1
        self.skip_space()
        return False


def json_closure_state(prefix: str) -> ClosureState:
    """Classify a JSON prefix and provide a syntax-only closing suffix if possible.

    An extendable prefix is not a generation failure until the completion
    boundary. A suffix never fills a key, value, literal, number, or escape.
    Closing a string or empty container may still truncate intended content:
    callers must separately verify the semantic boundary before using a suffix
    as supervision. Duplicate object keys are rejected. Offsets index decoded
    Python characters, not model tokens or UTF-8 bytes.
    """
    parser = _JsonPrefix(prefix)
    try:
        parser.value()
        parser.skip_space()
        if parser.position != len(prefix):
            parser.fail("Content after the complete JSON value")
    except _PrefixEnd as error:
        return ClosureState(PrefixStatus.EXTENDABLE, tuple(parser.open), error.suffix)
    except _InvalidPrefix as error:
        return ClosureState(PrefixStatus.INVALID, tuple(parser.open), None, error.offset, error.reason)
    return ClosureState(PrefixStatus.COMPLETE, (), "")


def delimited_json_closure_state(prefix: str, *, opener: str, closer: str) -> ClosureState:
    """Inspect one explicitly delimited JSON region, including its opening marker.

    Markers inside JSON strings are literal content. The JSON payload has no
    schema checks here. Trailing non-whitespace content after the close marker
    is outside this single-region contract and is rejected.
    """
    if not opener or not closer or opener == closer:
        raise ValueError("Require distinct nonempty JSON delimiters")
    if opener.startswith(prefix) and prefix != opener:
        return ClosureState(PrefixStatus.EXTENDABLE, (), None)
    if not prefix.startswith(opener):
        return ClosureState(PrefixStatus.INVALID, (), None, 0, "Expected opening marker")
    parser = _JsonPrefix(prefix[len(opener) :])
    try:
        parser.value()
        parser.skip_space()
    except _PrefixEnd as error:
        suffix = None if error.suffix is None else error.suffix + closer
        return ClosureState(PrefixStatus.EXTENDABLE, (opener, *parser.open), suffix)
    except _InvalidPrefix as error:
        return ClosureState(PrefixStatus.INVALID, (opener, *parser.open), None, len(opener) + error.offset, error.reason)
    offset = len(opener) + parser.position
    remainder = prefix[offset:]
    if closer.startswith(remainder) and remainder != closer:
        return ClosureState(PrefixStatus.EXTENDABLE, (opener,), closer[len(remainder) :])
    if remainder.startswith(closer):
        tail = remainder[len(closer) :]
        if not tail.strip(_WHITESPACE):
            return ClosureState(PrefixStatus.COMPLETE, (), "")
        offset += len(closer) + len(tail) - len(tail.lstrip(_WHITESPACE))
        return ClosureState(PrefixStatus.INVALID, (), None, offset, "Content after closing marker")
    matching = 0
    while matching < min(len(remainder), len(closer)) and remainder[matching] == closer[matching]:
        matching += 1
    return ClosureState(PrefixStatus.INVALID, (opener,), None, offset + matching, "Expected closing marker")


def paired_marker_closure_state(prefix: str, *, opener: str, closer: str) -> ClosureState:
    """Inspect one non-nesting marker pair around opaque content.

    Use only where these exact markers are structural, such as model reasoning
    special tokens. This is not a parser for prose quoting the marker spelling.
    Content completion requires separate evidence before suffix supervision.
    """
    if not opener or not closer or opener == closer:
        raise ValueError("Require distinct nonempty markers")
    if opener.startswith(prefix) and prefix != opener:
        return ClosureState(PrefixStatus.EXTENDABLE, (), None)
    if not prefix.startswith(opener):
        return ClosureState(PrefixStatus.INVALID, (), None, 0, "Expected opening marker")
    body = prefix[len(opener) :]
    close_at = body.find(closer)
    repeat_at = body.find(opener)
    if repeat_at >= 0 and (close_at < 0 or repeat_at < close_at):
        return ClosureState(
            PrefixStatus.INVALID, (opener,), None, len(opener) + repeat_at + len(opener) - 1, "Repeated opening marker"
        )
    if close_at >= 0:
        tail = body[close_at + len(closer) :]
        if tail.strip(_WHITESPACE):
            offset = len(opener) + close_at + len(closer) + len(tail) - len(tail.lstrip(_WHITESPACE))
            return ClosureState(PrefixStatus.INVALID, (), None, offset, "Content after closing marker")
        return ClosureState(PrefixStatus.COMPLETE, (), "")
    for length in range(min(len(body), len(closer) - 1), 0, -1):
        if body.endswith(closer[:length]):
            return ClosureState(PrefixStatus.EXTENDABLE, (opener,), closer[length:])
    return ClosureState(PrefixStatus.EXTENDABLE, (opener,), closer)


def required_code_fence_state(prefix: str) -> ClosureState:
    """Inspect a single Markdown code block whose task explicitly requires closure.

    Support backtick and tilde fences, at most three opening/closing indent
    spaces, and closing runs at least as long as the opener. Body punctuation
    is opaque. Ordinary Markdown permits an unclosed fence at EOF; this stricter
    contract must therefore be declared in the task, not inferred globally.
    """
    lines = prefix.split("\n")
    opening = re.fullmatch(r" {0,3}(`{3,}|~{3,})([^\n]*)", lines[0])
    if opening is None:
        if re.fullmatch(r" {0,3}(?:`{0,2}|~{0,2})", prefix):
            return ClosureState(PrefixStatus.EXTENDABLE, (), None)
        return ClosureState(PrefixStatus.INVALID, (), None, 0, "Expected code fence")
    fence, info = opening.groups()
    if fence[0] == "`" and "`" in info:
        return ClosureState(PrefixStatus.INVALID, (), None, lines[0].find("`", opening.end(1)), "Backtick in fence info")
    closer_pattern = re.compile(r" {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}[ \t]*")
    offset = len(lines[0]) + 1
    for index, line in enumerate(lines[1:], 1):
        if closer_pattern.fullmatch(line):
            tail = "\n".join(lines[index + 1 :])
            if tail.strip(_WHITESPACE):
                return ClosureState(PrefixStatus.INVALID, (), None, offset + len(line) + 1, "Content after code fence")
            return ClosureState(PrefixStatus.COMPLETE, (), "")
        offset += len(line) + 1
    last = lines[-1]
    if len(lines) > 1 and re.fullmatch(r" {0,3}" + re.escape(fence[0]) + "{0," + str(len(fence) - 1) + "}", last):
        suffix = fence[len(last.lstrip(" ")) :]
    else:
        suffix = "\n" + fence
    return ClosureState(PrefixStatus.EXTENDABLE, (fence,), suffix)
