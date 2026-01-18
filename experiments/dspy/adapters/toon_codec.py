# SPDX-License-Identifier: MIT
"""TOON (Token-Oriented Object Notation) encoder and decoder.

A compact, human-readable serialization format optimized for LLM contexts.
Achieves 30-60% token reduction vs JSON while maintaining readability.

This module provides the core `encode()` and `decode()` functions for
converting between Python values and TOON format strings.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal, TypedDict, TypeGuard

# =============================================================================
JsonPrimitive = str | int | float | bool | None
JsonObject = dict[str, Any]
JsonArray = list[Any]
JsonValue = JsonPrimitive | JsonArray | JsonObject
Delimiter = str
Depth = int


class EncodeOptions(TypedDict, total=False):
    """Options for TOON encoding."""

    indent: int
    delimiter: Delimiter
    lengthMarker: Literal["#"] | Literal[False]


@dataclass
class ResolvedEncodeOptions:
    """Resolved encoding options with defaults."""

    indent: int = 2
    delimiter: str = ","
    lengthMarker: str | Literal[False] = False


@dataclass
class DecodeOptions:
    """Options for TOON decoding."""

    indent: int = 2
    strict: bool = True

# =============================================================================
COMMA: Delimiter = ","
COLON = ":"
SPACE = " "
PIPE: Delimiter = "|"
TAB: Delimiter = "\t"

OPEN_BRACKET = "["
CLOSE_BRACKET = "]"
OPEN_BRACE = "{"
CLOSE_BRACE = "}"

NULL_LITERAL = "null"
TRUE_LITERAL = "true"
FALSE_LITERAL = "false"

BACKSLASH = "\\"
DOUBLE_QUOTE = '"'
NEWLINE = "\n"
CARRIAGE_RETURN = "\r"

LIST_ITEM_MARKER = "-"
LIST_ITEM_PREFIX = "- "

DELIMITERS: dict[str, Delimiter] = {"comma": COMMA, "tab": TAB, "pipe": PIPE}
DEFAULT_DELIMITER: Delimiter = COMMA

# Regex patterns
NUMERIC_REGEX = r"^-?\d+(?:\.\d+)?(?:e[+-]?\d+)?$"
OCTAL_REGEX = r"^0\d+$"
VALID_KEY_REGEX = r"^[A-Z_][\w.]*$"

# =============================================================================
class ToonDecodeError(Exception):
    """TOON decoding error."""

    pass

def escape_string(value: str) -> str:
    """Escape special characters in a string for encoding."""
    return (
        value.replace(BACKSLASH, BACKSLASH + BACKSLASH)
        .replace(DOUBLE_QUOTE, BACKSLASH + DOUBLE_QUOTE)
        .replace(NEWLINE, BACKSLASH + "n")
        .replace(CARRIAGE_RETURN, BACKSLASH + "r")
        .replace(TAB, BACKSLASH + "t")
    )


def unescape_string(value: str) -> str:
    """Unescape a string by processing escape sequences."""
    result = ""
    i = 0
    while i < len(value):
        if value[i] == BACKSLASH:
            if i + 1 >= len(value):
                raise ValueError("Invalid escape sequence: backslash at end of string")
            next_char = value[i + 1]
            if next_char == "n":
                result += NEWLINE
            elif next_char == "t":
                result += TAB
            elif next_char == "r":
                result += CARRIAGE_RETURN
            elif next_char == BACKSLASH:
                result += BACKSLASH
            elif next_char == DOUBLE_QUOTE:
                result += DOUBLE_QUOTE
            else:
                raise ValueError(f"Invalid escape sequence: \\{next_char}")
            i += 2
            continue
        result += value[i]
        i += 1
    return result


def is_boolean_or_null_literal(value: str) -> bool:
    """Check if value is a boolean or null literal."""
    return value.lower() in (TRUE_LITERAL, FALSE_LITERAL, NULL_LITERAL)


def is_numeric_like(value: str) -> bool:
    """Check if a string looks like a number."""
    return bool(re.match(NUMERIC_REGEX, value, re.IGNORECASE) or re.match(OCTAL_REGEX, value))


def is_numeric_literal(token: str) -> bool:
    """Check if token is a valid numeric literal."""
    if not token:
        return False
    try:
        float(token)
        return True
    except ValueError:
        return False


def is_valid_unquoted_key(key: str) -> bool:
    """Check if a key can be used without quotes."""
    if not key:
        return False
    return bool(re.match(VALID_KEY_REGEX, key, re.IGNORECASE))


def is_safe_unquoted(value: str, delimiter: str = COMMA) -> bool:
    """Determine if a string value can be safely encoded without quotes."""
    if not value:
        return False
    if value != value.strip():
        return False
    if is_boolean_or_null_literal(value) or is_numeric_like(value):
        return False
    if ":" in value or '"' in value or "\\" in value:
        return False
    if re.search(r"[\[\]{}]", value):
        return False
    if re.search(r"[\n\r\t]", value):
        return False
    if delimiter in value:
        return False
    if value.startswith(LIST_ITEM_MARKER):
        return False
    return True


def find_unquoted_char(content: str, char: str, start: int = 0) -> int:
    """Find the index of a character outside of quoted sections."""
    in_quotes = False
    i = start
    while i < len(content):
        if content[i] == BACKSLASH and i + 1 < len(content) and in_quotes:
            i += 2
            continue
        if content[i] == DOUBLE_QUOTE:
            in_quotes = not in_quotes
            i += 1
            continue
        if content[i] == char and not in_quotes:
            return i
        i += 1
    return -1


def find_first_unquoted(line: str, chars: list[str]) -> tuple[int, str | None]:
    """Find the first occurrence of any char from chars outside quotes."""
    in_quotes = False
    i = 0
    while i < len(line):
        if line[i] == BACKSLASH and i + 1 < len(line) and in_quotes:
            i += 2
            continue
        if line[i] == DOUBLE_QUOTE:
            in_quotes = not in_quotes
            i += 1
            continue
        if not in_quotes and line[i] in chars:
            return (i, line[i])
        i += 1
    return (-1, None)


def parse_delimited_values(content: str, delimiter: str) -> list[str]:
    """Parse delimited values respecting quoted strings."""
    if not content.strip():
        return []
    values = []
    current = ""
    in_quotes = False
    i = 0
    while i < len(content):
        c = content[i]
        if c == BACKSLASH and i + 1 < len(content) and in_quotes:
            current += c + content[i + 1]
            i += 2
            continue
        if c == DOUBLE_QUOTE:
            in_quotes = not in_quotes
            current += c
            i += 1
            continue
        if c == delimiter and not in_quotes:
            values.append(current.strip())
            current = ""
            i += 1
            continue
        current += c
        i += 1
    if current.strip():
        values.append(current.strip())
    return values

# =============================================================================
def normalize_value(value: Any) -> JsonValue:
    """Normalize Python value to JSON-compatible type."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or value != value:
            return None
        if value == 0.0 and math.copysign(1.0, value) == -1.0:
            return 0
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            return None
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        try:
            return [normalize_value(item) for item in sorted(value)]
        except TypeError:
            return [normalize_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, Mapping):
        return {str(k): normalize_value(v) for k, v in value.items()}
    if callable(value):
        return None
    return None


def is_json_primitive(value: Any) -> TypeGuard[JsonPrimitive]:
    """Check if value is a JSON primitive type."""
    return value is None or isinstance(value, (str, int, float, bool))


def is_json_array(value: Any) -> TypeGuard[JsonArray]:
    """Check if value is a JSON array."""
    return isinstance(value, list)


def is_json_object(value: Any) -> TypeGuard[JsonObject]:
    """Check if value is a JSON object."""
    return isinstance(value, dict)


def is_array_of_primitives(value: JsonArray) -> bool:
    """Check if array contains only primitive values."""
    return not value or all(is_json_primitive(item) for item in value)


def is_array_of_arrays(value: JsonArray) -> bool:
    """Check if array contains only arrays."""
    return not value or all(is_json_array(item) for item in value)


def is_array_of_objects(value: JsonArray) -> bool:
    """Check if array contains only objects."""
    return not value or all(is_json_object(item) for item in value)

class LineWriter:
    """Manages indented text output with optimized indent caching."""

    def __init__(self, indent_size: int) -> None:
        self._lines: list[str] = []
        normalized_indent = indent_size if indent_size > 0 else 1
        self._indentation_string = " " * normalized_indent
        self._indent_cache: dict[int, str] = {0: ""}
        self._indent_size = indent_size

    def push(self, depth: Depth, content: str) -> None:
        if depth not in self._indent_cache:
            if self._indent_size == 0:
                self._indent_cache[depth] = " " * depth
            else:
                self._indent_cache[depth] = self._indentation_string * depth
        indent = self._indent_cache[depth]
        self._lines.append(indent + content)

    def to_string(self) -> str:
        return "\n".join(self._lines)


def encode_primitive(value: JsonPrimitive, delimiter: str = COMMA) -> str:
    """Encode a primitive value."""
    if value is None:
        return NULL_LITERAL
    if isinstance(value, bool):
        return TRUE_LITERAL if value else FALSE_LITERAL
    if isinstance(value, (int, float)):
        if isinstance(value, int):
            return str(value)
        formatted = str(value)
        if "e" in formatted or "E" in formatted:
            dec = Decimal(str(value))
            formatted = format(dec, "f")
        return formatted
    if isinstance(value, str):
        if is_safe_unquoted(value, delimiter):
            return value
        return f"{DOUBLE_QUOTE}{escape_string(value)}{DOUBLE_QUOTE}"
    return str(value)


def encode_key(key: str) -> str:
    """Encode an object key."""
    if is_valid_unquoted_key(key):
        return key
    return f"{DOUBLE_QUOTE}{escape_string(key)}{DOUBLE_QUOTE}"


def format_header(
    key: str | None,
    length: int,
    fields: list[str] | None,
    delimiter: Delimiter,
    length_marker: str | Literal[False] | None,
) -> str:
    """Format array/table header."""
    marker_prefix = length_marker if length_marker else ""
    fields_str = ""
    if fields:
        encoded_fields = [encode_key(field) for field in fields]
        fields_str = f"{OPEN_BRACE}{delimiter.join(encoded_fields)}{CLOSE_BRACE}"
    if delimiter != COMMA:
        length_str = f"{OPEN_BRACKET}{marker_prefix}{length}{delimiter}{CLOSE_BRACKET}"
    else:
        length_str = f"{OPEN_BRACKET}{marker_prefix}{length}{CLOSE_BRACKET}"
    if key:
        return f"{encode_key(key)}{length_str}{fields_str}{COLON}"
    return f"{length_str}{fields_str}{COLON}"


def detect_tabular_header(arr: list[JsonObject], delimiter: str) -> list[str] | None:
    """Detect if array can use tabular format and return header keys."""
    if not arr:
        return None
    first_keys = list(arr[0].keys())
    first_keys_set = set(first_keys)
    for obj in arr:
        if set(obj.keys()) != first_keys_set:
            return None
        if not all(is_json_primitive(value) for value in obj.values()):
            return None
    return first_keys


def _encode_value(
    value: JsonValue,
    options: ResolvedEncodeOptions,
    writer: LineWriter,
    depth: Depth = 0,
) -> None:
    """Encode a value to TOON format."""
    if is_json_primitive(value):
        writer.push(depth, encode_primitive(value, options.delimiter))
    elif is_json_array(value):
        _encode_array(value, options, writer, depth, None)
    elif is_json_object(value):
        _encode_object(value, options, writer, depth, None)


def _encode_object(
    obj: JsonObject,
    options: ResolvedEncodeOptions,
    writer: LineWriter,
    depth: Depth,
    key: str | None,
) -> None:
    """Encode an object to TOON format."""
    if key:
        writer.push(depth, f"{encode_key(key)}:")
    for obj_key, obj_value in obj.items():
        _encode_key_value_pair(obj_key, obj_value, options, writer, depth if not key else depth + 1)


def _encode_key_value_pair(
    key: str,
    value: JsonValue,
    options: ResolvedEncodeOptions,
    writer: LineWriter,
    depth: Depth,
) -> None:
    """Encode a key-value pair."""
    if is_json_primitive(value):
        primitive_str = encode_primitive(value, options.delimiter)
        writer.push(depth, f"{encode_key(key)}: {primitive_str}")
    elif is_json_array(value):
        _encode_array(value, options, writer, depth, key)
    elif is_json_object(value):
        _encode_object(value, options, writer, depth, key)


def _encode_array(
    arr: JsonArray,
    options: ResolvedEncodeOptions,
    writer: LineWriter,
    depth: Depth,
    key: str | None,
) -> None:
    """Encode an array to TOON format."""
    if not arr:
        header = format_header(key, 0, None, options.delimiter, options.lengthMarker)
        writer.push(depth, header)
        return
    if is_array_of_primitives(arr):
        _encode_inline_primitive_array(arr, options, writer, depth, key)
    elif is_array_of_objects(arr):
        tabular_header = detect_tabular_header(arr, options.delimiter)
        if tabular_header:
            _encode_tabular_array(arr, tabular_header, options, writer, depth, key)
        else:
            _encode_mixed_array(arr, options, writer, depth, key)
    else:
        _encode_mixed_array(arr, options, writer, depth, key)


def _encode_inline_primitive_array(
    arr: JsonArray,
    options: ResolvedEncodeOptions,
    writer: LineWriter,
    depth: Depth,
    key: str | None,
) -> None:
    """Encode an array of primitives inline."""
    encoded_values = [encode_primitive(item, options.delimiter) for item in arr]
    joined = options.delimiter.join(encoded_values)
    header = format_header(key, len(arr), None, options.delimiter, options.lengthMarker)
    writer.push(depth, f"{header} {joined}")


def _encode_tabular_array(
    arr: list[JsonObject],
    fields: list[str],
    options: ResolvedEncodeOptions,
    writer: LineWriter,
    depth: Depth,
    key: str | None,
) -> None:
    """Encode array of uniform objects in tabular format."""
    header = format_header(key, len(arr), fields, options.delimiter, options.lengthMarker)
    writer.push(depth, header)
    for obj in arr:
        row_values = [encode_primitive(obj[field], options.delimiter) for field in fields]
        row = options.delimiter.join(row_values)
        writer.push(depth + 1, row)


def _encode_mixed_array(
    arr: JsonArray,
    options: ResolvedEncodeOptions,
    writer: LineWriter,
    depth: Depth,
    key: str | None,
) -> None:
    """Encode mixed array as list items."""
    header = format_header(key, len(arr), None, options.delimiter, options.lengthMarker)
    writer.push(depth, header)
    for item in arr:
        if is_json_primitive(item):
            writer.push(depth + 1, f"{LIST_ITEM_PREFIX}{encode_primitive(item, options.delimiter)}")
        elif is_json_object(item):
            _encode_object_as_list_item(item, options, writer, depth + 1)
        elif is_json_array(item):
            if is_array_of_primitives(item):
                encoded = [encode_primitive(v, options.delimiter) for v in item]
                joined = options.delimiter.join(encoded)
                h = format_header(None, len(item), None, options.delimiter, options.lengthMarker)
                line = f"{LIST_ITEM_PREFIX}{h}"
                if joined:
                    line += f" {joined}"
                writer.push(depth + 1, line)
            else:
                _encode_array(item, options, writer, depth + 1, None)


def _encode_object_as_list_item(
    obj: JsonObject,
    options: ResolvedEncodeOptions,
    writer: LineWriter,
    depth: Depth,
) -> None:
    """Encode object as a list item."""
    keys = list(obj.items())
    if not keys:
        writer.push(depth, LIST_ITEM_PREFIX.rstrip())
        return
    first_key, first_value = keys[0]
    if is_json_primitive(first_value):
        encoded_val = encode_primitive(first_value, options.delimiter)
        writer.push(depth, f"{LIST_ITEM_PREFIX}{encode_key(first_key)}: {encoded_val}")
    else:
        writer.push(depth, LIST_ITEM_PREFIX.rstrip())
        _encode_key_value_pair(first_key, first_value, options, writer, depth + 1)
    for k, v in keys[1:]:
        _encode_key_value_pair(k, v, options, writer, depth + 1)


def encode(value: Any, options: EncodeOptions | None = None) -> str:
    """Encode a value into TOON format.

    Args:
        value: The value to encode (must be JSON-serializable)
        options: Optional encoding options

    Returns:
        TOON-formatted string

    Example:
        >>> encode({"name": "Alice", "age": 30})
        'name: Alice\\nage: 30'

        >>> encode([{"id": 1, "name": "A"}, {"id": 2, "name": "B"}])
        '[2,]{id,name}:\\n  1,A\\n  2,B'
    """
    normalized = normalize_value(value)
    resolved = ResolvedEncodeOptions()
    if options:
        resolved.indent = options.get("indent", 2)
        resolved.delimiter = options.get("delimiter", DEFAULT_DELIMITER)
        resolved.lengthMarker = options.get("lengthMarker", False)
        if resolved.delimiter in DELIMITERS:
            resolved.delimiter = DELIMITERS[resolved.delimiter]
    writer = LineWriter(resolved.indent)
    _encode_value(normalized, resolved, writer, 0)
    return writer.to_string()

# =============================================================================
@dataclass
class ParsedLine:
    """Represents a parsed line with indentation info."""

    raw: str
    depth: int
    indent: int
    content: str
    line_num: int

    @property
    def is_blank(self) -> bool:
        return not self.content.strip()


def _to_parsed_lines(input_str: str, indent_size: int, strict: bool) -> tuple[list[ParsedLine], dict[int, int]]:
    """Parse input string into ParsedLine objects."""
    lines = input_str.split("\n")
    parsed = []
    blank_lines_info: dict[int, int] = {}
    for i, raw in enumerate(lines):
        if not raw.strip():
            parsed.append(ParsedLine(raw=raw, depth=0, indent=0, content="", line_num=i + 1))
            continue
        indent = len(raw) - len(raw.lstrip())
        if indent_size > 0:
            depth = indent // indent_size
        else:
            depth = indent
        content = raw.lstrip()
        parsed.append(ParsedLine(raw=raw, depth=depth, indent=indent, content=content, line_num=i + 1))
    return parsed, blank_lines_info


def _parse_key(key_str: str) -> str:
    """Parse a key (quoted or unquoted)."""
    key_str = key_str.strip()
    if key_str.startswith(DOUBLE_QUOTE):
        if not key_str.endswith(DOUBLE_QUOTE) or len(key_str) < 2:
            raise ToonDecodeError("Unterminated quoted key")
        try:
            return unescape_string(key_str[1:-1])
        except ValueError as e:
            raise ToonDecodeError(str(e)) from e
    return key_str


def _parse_primitive(token: str) -> JsonValue:
    """Parse a primitive token."""
    token = token.strip()
    if token.startswith(DOUBLE_QUOTE):
        if not token.endswith(DOUBLE_QUOTE) or len(token) < 2:
            raise ToonDecodeError("Unterminated string: missing closing quote")
        try:
            return unescape_string(token[1:-1])
        except ValueError as e:
            raise ToonDecodeError(str(e)) from e
    if is_boolean_or_null_literal(token):
        if token == TRUE_LITERAL:
            return True
        if token == FALSE_LITERAL:
            return False
        return None
    if token and is_numeric_literal(token):
        try:
            if "." not in token and "e" not in token.lower():
                return int(token)
            return float(token)
        except ValueError:
            pass
    return token


def _split_key_value(line: str) -> tuple[str, str]:
    """Split a line into key and value at first unquoted colon."""
    colon_idx = find_unquoted_char(line, COLON)
    if colon_idx == -1:
        raise ToonDecodeError("Missing colon after key")
    key = line[:colon_idx].strip()
    value = line[colon_idx + 1 :].strip()
    return (key, value)


def _parse_header(
    line: str,
) -> tuple[str | None, int, str, list[str] | None] | None:
    """Parse an array header."""
    line = line.strip()
    bracket_start = find_unquoted_char(line, OPEN_BRACKET)
    if bracket_start == -1:
        return None
    key = None
    if bracket_start > 0:
        key_part = line[:bracket_start].strip()
        key = _parse_key(key_part) if key_part else None
    bracket_end = find_unquoted_char(line, CLOSE_BRACKET, bracket_start)
    if bracket_end == -1:
        return None
    bracket_content = line[bracket_start + 1 : bracket_end]
    if bracket_content.startswith("#"):
        bracket_content = bracket_content[1:]
    delimiter = COMMA
    length_str = bracket_content
    if bracket_content.endswith(TAB):
        delimiter = TAB
        length_str = bracket_content[:-1]
    elif bracket_content.endswith(PIPE):
        delimiter = PIPE
        length_str = bracket_content[:-1]
    elif bracket_content.endswith(COMMA):
        delimiter = COMMA
        length_str = bracket_content[:-1]
    try:
        length = int(length_str)
    except ValueError:
        return None
    fields = None
    after_bracket = line[bracket_end + 1 :].strip()
    if after_bracket.startswith(OPEN_BRACE):
        brace_end = find_unquoted_char(after_bracket, CLOSE_BRACE)
        if brace_end == -1:
            raise ToonDecodeError("Unterminated fields segment")
        fields_content = after_bracket[1:brace_end]
        field_tokens = parse_delimited_values(fields_content, delimiter)
        fields = [_parse_key(f.strip()) for f in field_tokens]
        after_bracket = after_bracket[brace_end + 1 :].strip()
    if not after_bracket.startswith(COLON):
        return None
    return (key, length, delimiter, fields)


def _is_row_line(line: str, delimiter: str) -> bool:
    """Check if a line is a tabular row (not a key-value line)."""
    pos, char = find_first_unquoted(line, [delimiter, COLON])
    if pos == -1:
        return True
    return char == delimiter


def _decode_inline_array(content: str, delimiter: str, expected_length: int, strict: bool) -> list[Any]:
    """Decode an inline primitive array."""
    if not content and expected_length == 0:
        return []
    tokens = parse_delimited_values(content, delimiter)
    values = [_parse_primitive(token) for token in tokens]
    if strict and len(values) != expected_length:
        raise ToonDecodeError(f"Expected {expected_length} values, but got {len(values)}")
    return values


def _decode_tabular_array(
    lines: list[ParsedLine],
    start_idx: int,
    header_depth: int,
    fields: list[str],
    delimiter: str,
    expected_length: int,
    strict: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Decode a tabular array."""
    result = []
    i = start_idx
    row_depth = header_depth + 1
    while i < len(lines):
        line = lines[i]
        if line.is_blank:
            if strict:
                if line.depth >= row_depth:
                    raise ToonDecodeError("Blank lines not allowed inside arrays")
                else:
                    break
            else:
                i += 1
                continue
        if line.depth < row_depth:
            break
        if line.depth > row_depth:
            break
        content = line.content
        if _is_row_line(content, delimiter):
            tokens = parse_delimited_values(content, delimiter)
            values = [_parse_primitive(token) for token in tokens]
            if strict and len(values) != len(fields):
                raise ToonDecodeError(f"Expected {len(fields)} values in row, but got {len(values)}")
            obj = {fields[j]: values[j] for j in range(min(len(fields), len(values)))}
            result.append(obj)
            i += 1
        else:
            break
    if strict and len(result) != expected_length:
        raise ToonDecodeError(f"Expected {expected_length} rows, but got {len(result)}")
    return result, i


def _decode_list_array(
    lines: list[ParsedLine],
    start_idx: int,
    header_depth: int,
    delimiter: str,
    expected_length: int,
    strict: bool,
) -> tuple[list[Any], int]:
    """Decode a list-format array (mixed/non-uniform)."""
    result: list[Any] = []
    i = start_idx
    item_depth = header_depth + 1
    while i < len(lines):
        line = lines[i]
        if line.is_blank:
            if strict:
                if line.depth >= item_depth:
                    raise ToonDecodeError("Blank lines not allowed inside arrays")
                else:
                    break
            else:
                i += 1
                continue
        if line.depth < item_depth:
            break
        content = line.content
        if not content.startswith(LIST_ITEM_PREFIX):
            break
        item_content = content[len(LIST_ITEM_PREFIX) :].strip()
        item_header = _parse_header(item_content)
        if item_header is not None:
            key, length, item_delim, fields = item_header
            if key is None:
                colon_idx = item_content.find(COLON)
                if colon_idx != -1:
                    inline_part = item_content[colon_idx + 1 :].strip()
                    if inline_part or length == 0:
                        item_val = _decode_inline_array(inline_part, item_delim, length, strict)
                        result.append(item_val)
                        i += 1
                        continue
        try:
            key_str, value_str = _split_key_value(item_content)
            obj_item: dict[str, Any] = {}
            key = _parse_key(key_str)
            if not value_str:
                nested = _decode_object(lines, i + 1, line.depth + 1, strict)
                obj_item[key] = nested
                i += 1
                while i < len(lines) and lines[i].depth > line.depth + 1:
                    i += 1
            else:
                obj_item[key] = _parse_primitive(value_str)
                i += 1
            while i < len(lines) and lines[i].depth == line.depth + 1:
                field_line = lines[i]
                if field_line.is_blank:
                    i += 1
                    continue
                try:
                    field_key_str, field_value_str = _split_key_value(field_line.content)
                    field_key = _parse_key(field_key_str)
                    if not field_value_str:
                        obj_item[field_key] = _decode_object(lines, i + 1, field_line.depth, strict)
                        i += 1
                        while i < len(lines) and lines[i].depth > field_line.depth:
                            i += 1
                    else:
                        obj_item[field_key] = _parse_primitive(field_value_str)
                        i += 1
                except ToonDecodeError:
                    break
            result.append(obj_item)
        except ToonDecodeError:
            if not item_content:
                result.append({})
            else:
                result.append(_parse_primitive(item_content))
            i += 1
    if strict and len(result) != expected_length:
        raise ToonDecodeError(f"Expected {expected_length} items, but got {len(result)}")
    return result, i


def _decode_array_from_header(
    lines: list[ParsedLine],
    header_idx: int,
    header_depth: int,
    header_info: tuple[str | None, int, str, list[str] | None],
    strict: bool,
) -> tuple[list[Any], int]:
    """Decode array starting from a header line."""
    key, length, delimiter, fields = header_info
    header_line = lines[header_idx].content
    try:
        _, inline_content = _split_key_value(header_line)
    except ToonDecodeError:
        inline_content = ""
    if inline_content or (not fields and length == 0):
        return (_decode_inline_array(inline_content, delimiter, length, strict), header_idx + 1)
    if fields is not None:
        return _decode_tabular_array(lines, header_idx + 1, header_depth, fields, delimiter, length, strict)
    else:
        return _decode_list_array(lines, header_idx + 1, header_depth, delimiter, length, strict)


def _decode_object(lines: list[ParsedLine], start_idx: int, parent_depth: int, strict: bool) -> dict[str, Any]:
    """Decode an object starting at given line index."""
    result: dict[str, Any] = {}
    i = start_idx
    expected_depth = parent_depth if start_idx == 0 else parent_depth + 1
    while i < len(lines):
        line = lines[i]
        if line.is_blank:
            i += 1
            continue
        if line.depth < expected_depth:
            break
        if line.depth > expected_depth:
            i += 1
            continue
        content = line.content
        header_info = _parse_header(content)
        if header_info is not None:
            key, length, delimiter, fields = header_info
            if key is not None:
                array_val, next_i = _decode_array_from_header(lines, i, line.depth, header_info, strict)
                result[key] = array_val
                i = next_i
                continue
        try:
            key_str, value_str = _split_key_value(content)
        except ToonDecodeError:
            if strict:
                raise
            i += 1
            continue
        key = _parse_key(key_str)
        if not value_str:
            result[key] = _decode_object(lines, i + 1, line.depth, strict)
            i += 1
            while i < len(lines) and lines[i].depth > line.depth:
                i += 1
        else:
            result[key] = _parse_primitive(value_str)
            i += 1
    return result


def decode(input_str: str, options: DecodeOptions | None = None) -> JsonValue:
    """Decode a TOON-formatted string to a Python value.

    Args:
        input_str: TOON-formatted string
        options: Optional decoding options

    Returns:
        Decoded Python value

    Raises:
        ToonDecodeError: If input is malformed

    Example:
        >>> decode("name: Alice\\nage: 30")
        {'name': 'Alice', 'age': 30}

        >>> decode("[2,]{id,name}:\\n  1,A\\n  2,B")
        [{'id': 1, 'name': 'A'}, {'id': 2, 'name': 'B'}]
    """
    if options is None:
        options = DecodeOptions()
    indent_size = options.indent
    strict = options.strict
    parsed_lines, _ = _to_parsed_lines(input_str, indent_size, strict)
    lines = [
        ParsedLine(
            raw=line.raw,
            depth=line.depth,
            indent=line.indent,
            content=line.content.strip(),
            line_num=line.line_num,
        )
        for line in parsed_lines
    ]
    non_blank_lines = [ln for ln in lines if not ln.is_blank]
    if not non_blank_lines:
        return {}
    first_line = non_blank_lines[0]
    header_info = _parse_header(first_line.content)
    if header_info is not None and header_info[0] is None:
        arr, _ = _decode_array_from_header(lines, 0, 0, header_info, strict)
        return arr
    if len(non_blank_lines) == 1:
        line_content = first_line.content
        try:
            _split_key_value(line_content)
        except ToonDecodeError:
            if header_info is None:
                return _parse_primitive(line_content)
    return _decode_object(lines, 0, 0, strict)
