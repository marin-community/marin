# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Human-facing rendering: sizes, listings, and file previews.

Marin's object stores are full of JSON and JSONL — task records, metrics, manifests —
so a preview that dumps raw bytes wastes the trip. :func:`file_lines` renders tabular
JSON as a table and falls back to text, then to a byte count for binary data. The
plain-line renderers are shared with the curses TUI, which cannot use ``rich``.
"""

import json
from datetime import datetime

from rich.console import Console
from rich.table import Table

from rigging.fsutil.compression import uncompressed_name
from rigging.fsutil.listing import Entry

_SIZE_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")

# Longest single-cell value rendered from a JSON object before truncation.
_MAX_CELL = 120


def format_size(size: int | None) -> str:
    """A byte count in the largest unit that keeps it under 1024, or ``-`` for ``None``."""
    if size is None:
        return "-"
    value = float(size)
    for unit in _SIZE_UNITS:
        if value < 1024 or unit == _SIZE_UNITS[-1]:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} {_SIZE_UNITS[-1]}"


def format_time(when: datetime | None) -> str:
    return "-" if when is None else when.strftime("%Y-%m-%d %H:%M")


def print_entries(console: Console, entries: list[Entry], *, long: bool) -> None:
    """Print a listing: names only, or a size/modified table under ``long``."""
    if not long:
        for entry in entries:
            console.print(f"{entry.name}/" if entry.is_dir else entry.name, highlight=False)
        return

    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("size", justify="right")
    table.add_column("modified")
    table.add_column("name")
    for entry in entries:
        name = f"{entry.name}/" if entry.is_dir else entry.name
        table.add_row(format_size(entry.size), format_time(entry.mtime), name)
    console.print(table)


def file_lines(name: str, raw: bytes) -> list[str]:
    """Render *raw* as display lines, using *name*'s extension to pick a JSON reader.

    A tabular ``.json`` or ``.jsonl`` file renders as a table. The renderer ignores one
    supported compression suffix. Other files render as text or a byte count.
    """
    name = uncompressed_name(name)
    if name.endswith((".json", ".jsonl")):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return [f"[binary file, {len(raw)} bytes]"]
        return _json_lines(name, text)

    try:
        return raw.decode("utf-8").splitlines() or ["(empty file)"]
    except UnicodeDecodeError:
        return [f"[binary file, {len(raw)} bytes]"]


def _json_lines(name: str, text: str) -> list[str]:
    if name.endswith(".jsonl"):
        records = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                return text.splitlines()
        if not records:
            return ["(empty file)"]
        return _table_lines(records)

    try:
        return _table_lines(json.loads(text))
    except json.JSONDecodeError:
        return text.splitlines()


def _table_lines(data: object) -> list[str]:
    """Render parsed JSON as an aligned table when it is tabular, else as indented JSON."""
    if isinstance(data, list) and data and all(isinstance(row, dict) for row in data):
        headers = list({key: None for row in data for key in row})
        rows = [[_cell(row.get(header)) for header in headers] for row in data]
        return _aligned(headers, rows)
    if isinstance(data, dict):
        return _aligned(["key", "value"], [[key, _cell(value)] for key, value in data.items()])
    return json.dumps(data, indent=2, default=str).splitlines()


def _cell(value: object) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= _MAX_CELL else text[: _MAX_CELL - 3] + "..."


def _aligned(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [
        max(len(header), *(len(row[i]) for row in rows)) if rows else len(header) for i, header in enumerate(headers)
    ]
    lines = [_row(headers, widths), _row(["-" * width for width in widths], widths)]
    lines.extend(_row(row, widths) for row in rows)
    return lines


def _row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True)).rstrip()
