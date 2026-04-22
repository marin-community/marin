# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic text renderers for binary / network / security artifacts.

These renderers are pure functions: given the same input, they produce
byte-for-byte identical output. That matters for PPL eval slices, where any
nondeterminism would leak into tokenizer bucketing and produce spurious gap
signal.

The public surface:

* :func:`render_hex_dump` — xxd-style offset/hex/ASCII rendering of raw bytes.
* :func:`render_zeek_tsv_value` — serialize a single Zeek field value.
* :func:`render_zeek_tsv_record` — serialize a single Zeek log record as a TSV line.
* :func:`render_zeek_tsv_log` — serialize a whole Zeek log block with canonical
  ``#separator`` / ``#fields`` / ``#types`` headers and ``#close`` trailer.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

# Zeek canonical log-format constants. Zeek writes its logs with these defaults;
# re-serializing with the same values round-trips field-for-field.
DEFAULT_ZEEK_SEPARATOR = "\t"
DEFAULT_ZEEK_SET_SEPARATOR = ","
DEFAULT_ZEEK_EMPTY_FIELD = "(empty)"
DEFAULT_ZEEK_UNSET_FIELD = "-"

# xxd-style hex dump defaults. 16 bytes/line matches `xxd` and `hexdump -C`,
# which keeps output visually compatible with tooling the model may have seen.
DEFAULT_HEX_BYTES_PER_LINE = 16
DEFAULT_HEX_OFFSET_WIDTH = 8

# ASCII printable range used by hex dumps (space through tilde, inclusive).
_HEX_ASCII_MIN = 0x20
_HEX_ASCII_MAX = 0x7E


def render_hex_dump(
    data: bytes,
    *,
    bytes_per_line: int = DEFAULT_HEX_BYTES_PER_LINE,
    offset_width: int = DEFAULT_HEX_OFFSET_WIDTH,
    include_ascii: bool = True,
    uppercase: bool = False,
    offset_start: int = 0,
) -> str:
    """Render ``data`` as an xxd-style hex dump.

    Each line looks like::

        00000000  48 65 6c 6c 6f 20 57 6f  72 6c 64 0a              |Hello World.   |

    The dump ends with a single trailing newline.

    Args:
        data: Raw bytes to render. Empty ``data`` returns the empty string.
        bytes_per_line: Number of bytes per line. Must be positive and even so
            the two half-groups align.
        offset_width: Zero-padded hex width for the leading offset column.
        include_ascii: Whether to append the ``|...|`` printable-ASCII gutter.
        uppercase: Render hex digits in uppercase.
        offset_start: Logical offset of ``data[0]`` (lets callers resume a
            multi-chunk dump at a known offset).

    Returns:
        The rendered hex dump as a string. Empty string when ``data`` is empty.
    """
    if bytes_per_line <= 0:
        raise ValueError(f"bytes_per_line must be positive, got {bytes_per_line}")
    if bytes_per_line % 2 != 0:
        raise ValueError(f"bytes_per_line must be even, got {bytes_per_line}")
    if offset_width < 1:
        raise ValueError(f"offset_width must be >= 1, got {offset_width}")
    if offset_start < 0:
        raise ValueError(f"offset_start must be non-negative, got {offset_start}")

    if not data:
        return ""

    hex_format = "{:02X}" if uppercase else "{:02x}"
    offset_format = f"{{:0{offset_width}X}}" if uppercase else f"{{:0{offset_width}x}}"
    half = bytes_per_line // 2
    hex_column_width = bytes_per_line * 3  # "xx " per byte
    # Extra single space between the two half-groups (standard xxd layout).

    lines: list[str] = []
    for line_start in range(0, len(data), bytes_per_line):
        chunk = data[line_start : line_start + bytes_per_line]
        offset = offset_format.format(offset_start + line_start)

        left = " ".join(hex_format.format(b) for b in chunk[:half])
        right = " ".join(hex_format.format(b) for b in chunk[half:])
        if right:
            hex_column = f"{left}  {right}" if left else f"  {right}"
        else:
            hex_column = left

        # Pad the hex column to a fixed width so the ASCII gutter aligns even
        # on short trailing lines.
        hex_column = hex_column.ljust(hex_column_width)

        if include_ascii:
            ascii_column = "".join(chr(b) if _HEX_ASCII_MIN <= b <= _HEX_ASCII_MAX else "." for b in chunk)
            # Pad the ASCII gutter to match a full line so columns line up.
            ascii_column = ascii_column.ljust(bytes_per_line)
            lines.append(f"{offset}  {hex_column} |{ascii_column}|")
        else:
            lines.append(f"{offset}  {hex_column.rstrip()}")

    return "\n".join(lines) + "\n"


def render_zeek_tsv_value(
    value: Any,
    *,
    set_separator: str = DEFAULT_ZEEK_SET_SEPARATOR,
    empty_field: str = DEFAULT_ZEEK_EMPTY_FIELD,
    unset_field: str = DEFAULT_ZEEK_UNSET_FIELD,
) -> str:
    """Serialize a single Zeek field ``value`` back to its canonical TSV form.

    * ``None`` → ``unset_field`` (default ``"-"``)
    * Empty containers / empty strings → ``empty_field`` (default ``"(empty)"``)
    * Lists / tuples / sets / frozensets → items joined with ``set_separator``.
    * ``bool`` → ``"T"`` / ``"F"`` (Zeek-native boolean serialization).
    * Everything else → ``str(value)``.

    Any Zeek delimiter characters appearing inside rendered values are left in
    place; the caller is responsible for choosing delimiters that do not clash
    with the data. This matches Zeek's own behavior.
    """
    if value is None:
        return unset_field

    if isinstance(value, bool):
        return "T" if value else "F"

    if isinstance(value, (list, tuple, set, frozenset)):
        items = [render_zeek_tsv_value(v, set_separator=set_separator, empty_field=empty_field, unset_field=unset_field) for v in value]
        if not items:
            return empty_field
        return set_separator.join(items)

    if isinstance(value, str):
        if value == "":
            return empty_field
        return value

    return str(value)


def render_zeek_tsv_record(
    record: Mapping[str, Any],
    fields: Sequence[str],
    *,
    separator: str = DEFAULT_ZEEK_SEPARATOR,
    set_separator: str = DEFAULT_ZEEK_SET_SEPARATOR,
    empty_field: str = DEFAULT_ZEEK_EMPTY_FIELD,
    unset_field: str = DEFAULT_ZEEK_UNSET_FIELD,
) -> str:
    """Serialize one Zeek log record as a single TSV line.

    Fields missing from ``record`` are written as ``unset_field``. No trailing
    newline is appended; the caller joins records with ``\\n``.
    """
    if not fields:
        raise ValueError("fields must be a non-empty sequence of field names")

    rendered = [
        render_zeek_tsv_value(
            record.get(field),
            set_separator=set_separator,
            empty_field=empty_field,
            unset_field=unset_field,
        )
        for field in fields
    ]
    return separator.join(rendered)


def render_zeek_tsv_log(
    records: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    zeek_path: str,
    types: Sequence[str] | None = None,
    open_time: str | None = None,
    close_time: str | None = None,
    separator: str = DEFAULT_ZEEK_SEPARATOR,
    set_separator: str = DEFAULT_ZEEK_SET_SEPARATOR,
    empty_field: str = DEFAULT_ZEEK_EMPTY_FIELD,
    unset_field: str = DEFAULT_ZEEK_UNSET_FIELD,
) -> str:
    """Serialize a collection of Zeek log records back to their canonical TSV form.

    The output mirrors Zeek's on-disk log format::

        #separator \\x09
        #set_separator  ,
        #empty_field    (empty)
        #unset_field    -
        #path   conn
        #open   2024-01-01-00-00-00
        #fields ts  uid id.orig_h   ...
        #types  time    string  addr    ...
        <tab-separated record>
        ...
        #close  2024-01-01-01-00-00

    Args:
        records: Iterable of records (dict-like) to serialize.
        fields: Ordered Zeek field names. Must be non-empty.
        zeek_path: The Zeek log path (e.g. ``"conn"``, ``"dns"``, ``"http"``).
        types: Optional Zeek type strings matching ``fields`` one-for-one.
        open_time / close_time: Optional ``#open`` / ``#close`` markers. When
            omitted the header/trailer lines are not emitted.
        separator / set_separator / empty_field / unset_field: Zeek delimiters.

    Returns:
        The serialized log as a single string ending in a newline.
    """
    if not fields:
        raise ValueError("fields must be a non-empty sequence of field names")
    if types is not None and len(types) != len(fields):
        raise ValueError(f"types length ({len(types)}) must match fields length ({len(fields)})")

    separator_hex = _escape_separator_for_header(separator)
    header_lines: list[str] = [
        f"#separator {separator_hex}",
        f"#set_separator{separator}{set_separator}",
        f"#empty_field{separator}{empty_field}",
        f"#unset_field{separator}{unset_field}",
        f"#path{separator}{zeek_path}",
    ]
    if open_time is not None:
        header_lines.append(f"#open{separator}{open_time}")
    header_lines.append(f"#fields{separator}{separator.join(fields)}")
    if types is not None:
        header_lines.append(f"#types{separator}{separator.join(types)}")

    body_lines = [
        render_zeek_tsv_record(
            record,
            fields,
            separator=separator,
            set_separator=set_separator,
            empty_field=empty_field,
            unset_field=unset_field,
        )
        for record in records
    ]

    trailer_lines = [f"#close{separator}{close_time}"] if close_time is not None else []

    return "\n".join([*header_lines, *body_lines, *trailer_lines]) + "\n"


def _escape_separator_for_header(separator: str) -> str:
    """Render the Zeek ``#separator`` header value.

    Zeek prints the separator as ``\\x09`` (literal backslash + hex byte) in the
    header rather than as the raw tab character.
    """
    if len(separator) != 1:
        raise ValueError(f"separator must be a single character, got {separator!r}")
    return f"\\x{ord(separator):02x}"
