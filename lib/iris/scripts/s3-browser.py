#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TUI browser for the CoreWeave S3 bucket.

Connects to the CoreWeave object storage endpoint and lets you
interactively explore bucket contents. Press Enter on a file to
view its contents; JSON files are rendered as tables via tabulate.

Usage:
    export CW_KEY_ID=<your-key-id>
    export CW_KEY_SECRET=<your-key-secret>
    uv run python lib/iris/scripts/s3-browser.py [prefix]
"""

import curses
import json
import os
import sys
from dataclasses import dataclass

import s3fs
from tabulate import tabulate

# CoreWeave Object Storage uses virtual-hosted-style addressing:
# bucket-specific URLs look like https://<bucket>.cwobject.com.
# s3fs needs the BASE endpoint so it can prepend the bucket name itself.
ENDPOINT = "https://cwobject.com"
BUCKET = "marin-us-west-04a"

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB — refuse to download files larger than this


@dataclass
class Entry:
    name: str
    display: str
    size: int | None  # None for directories
    is_dir: bool


def connect() -> s3fs.S3FileSystem:
    key_id = os.environ.get("CW_KEY_ID")
    key_secret = os.environ.get("CW_KEY_SECRET")
    if not key_id or not key_secret:
        print("ERROR: CW_KEY_ID and CW_KEY_SECRET must be set", file=sys.stderr)
        sys.exit(1)

    return s3fs.S3FileSystem(
        key=key_id,
        secret=key_secret,
        endpoint_url=ENDPOINT,
        config_kwargs={"s3": {"addressing_style": "virtual"}},
    )


def list_prefix(fs: s3fs.S3FileSystem, prefix: str) -> list[Entry]:
    """List entries under a prefix, returning directories and files."""
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    try:
        items = fs.ls(prefix, detail=True)
    except Exception as e:
        return [Entry(name="", display=f"[error: {e}]", size=None, is_dir=False)]

    entries: list[Entry] = []
    for item in items:
        key = item["name"]
        if key.rstrip("/") == prefix.rstrip("/"):
            continue

        is_dir = item["type"] == "directory"
        rel = key[len(prefix) :].rstrip("/")
        if not rel:
            continue

        if is_dir:
            entries.append(Entry(name=key, display=f"{rel}/", size=None, is_dir=True))
        else:
            entries.append(Entry(name=key, display=rel, size=item.get("size", 0), is_dir=False))

    entries.sort(key=lambda e: (not e.is_dir, e.display.lower()))
    return entries


def format_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:>7.1f} {unit}" if isinstance(size, float) else f"{size:>7} {unit}"
        size = size / 1024  # type: ignore[assignment]
    return f"{size:>7.1f} PB"


def fetch_file(fs: s3fs.S3FileSystem, path: str, max_bytes: int = MAX_FILE_SIZE) -> bytes:
    """Download a file from S3, up to max_bytes."""
    with fs.open(path, "rb") as f:
        return f.read(max_bytes)


def render_json_table(data: object) -> str:
    """Render JSON data as a tabulate table if it has tabular structure.

    Handles: list-of-dicts, single dict, and list-of-lists.
    Returns tabulate output, or falls back to pretty-printed JSON.
    """
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        return tabulate(data, headers="keys", tablefmt="rounded_outline")
    if isinstance(data, dict):
        rows = [[k, _truncate(str(v), 120)] for k, v in data.items()]
        return tabulate(rows, headers=["Key", "Value"], tablefmt="rounded_outline")
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        return tabulate(data, tablefmt="rounded_outline")
    # Not tabular — fall back to pretty JSON
    return json.dumps(data, indent=2)


def _truncate(s: str, maxlen: int) -> str:
    return s if len(s) <= maxlen else s[: maxlen - 3] + "..."


def render_file_content(path: str, raw: bytes) -> list[str]:
    """Convert raw file bytes into display lines. JSON files get table formatting."""
    filename = path.rsplit("/", 1)[-1] if "/" in path else path

    if filename.endswith((".json", ".jsonl")):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return ["[binary file — cannot display]"]

        # JSONL: each line is a separate JSON object
        if filename.endswith(".jsonl"):
            lines_raw = [ln for ln in text.strip().splitlines() if ln.strip()]
            if not lines_raw:
                return ["(empty file)"]
            try:
                records = [json.loads(ln) for ln in lines_raw]
                if isinstance(records[0], dict):
                    table = tabulate(records, headers="keys", tablefmt="rounded_outline")
                    return table.splitlines()
            except json.JSONDecodeError:
                pass
            return text.splitlines()

        # Regular JSON
        try:
            data = json.loads(text)
            return render_json_table(data).splitlines()
        except json.JSONDecodeError:
            return text.splitlines()

    # Non-JSON: try text, fall back to hex summary
    try:
        text = raw.decode("utf-8")
        return text.splitlines() or ["(empty file)"]
    except UnicodeDecodeError:
        return [f"[binary file, {len(raw)} bytes]"]


def show_file_viewer(stdscr: curses.window, path: str, lines: list[str]) -> None:
    """Scrollable read-only file viewer inside the curses TUI."""
    curses.curs_set(0)
    scroll = 0
    filename = path.rsplit("/", 1)[-1] if "/" in path else path

    while True:
        height, width = stdscr.getmaxyx()
        header_lines = 2
        footer_lines = 1
        visible = height - header_lines - footer_lines

        stdscr.clear()

        # Header
        stdscr.addnstr(0, 0, filename, width - 1, curses.color_pair(2) | curses.A_BOLD)
        line_info = f"{len(lines)} lines"
        stdscr.addnstr(1, 0, line_info, width - 1, curses.A_DIM)

        # Content
        for i in range(min(visible, len(lines) - scroll)):
            row = header_lines + i
            line = lines[scroll + i]
            stdscr.addnstr(row, 0, line[: width - 1], width - 1)

        # Footer
        footer_y = height - 1
        pos_text = (
            f" Line {scroll + 1}-{min(scroll + visible, len(lines))}/{len(lines)}  [q/Backspace] close  [j/k] scroll"
        )
        stdscr.addnstr(footer_y, 0, pos_text[: width - 1], width - 1, curses.A_DIM)

        stdscr.refresh()

        key = stdscr.getch()
        if key in (ord("q"), 27, curses.KEY_BACKSPACE, 127, ord("h"), curses.KEY_LEFT):
            break
        elif key in (curses.KEY_UP, ord("k")):
            scroll = max(0, scroll - 1)
        elif key in (curses.KEY_DOWN, ord("j")):
            scroll = min(max(0, len(lines) - visible), scroll + 1)
        elif key == curses.KEY_PPAGE:
            scroll = max(0, scroll - visible)
        elif key == curses.KEY_NPAGE:
            scroll = min(max(0, len(lines) - visible), scroll + visible)
        elif key == ord("g"):
            scroll = 0
        elif key == ord("G"):
            scroll = max(0, len(lines) - visible)


def run_tui(stdscr: curses.window, fs: s3fs.S3FileSystem, initial_prefix: str) -> None:
    curses.curs_set(0)
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)  # directories
    curses.init_pair(2, curses.COLOR_YELLOW, -1)  # status bar
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_WHITE)  # selected row

    prefix = initial_prefix
    cursor = 0
    scroll = 0
    entries: list[Entry] = []
    status = "Loading..."
    need_reload = True
    history: list[tuple[str, int, int]] = []

    while True:
        if need_reload:
            status = f"Loading s3://{prefix} ..."
            stdscr.clear()
            stdscr.addstr(0, 0, status, curses.color_pair(2))
            stdscr.refresh()

            entries = list_prefix(fs, prefix)
            cursor = 0
            scroll = 0
            need_reload = False
            status = f"s3://{prefix}  ({len(entries)} items)"

        height, width = stdscr.getmaxyx()
        header_lines = 2
        footer_lines = 2
        visible = height - header_lines - footer_lines

        if len(entries) > 0:
            cursor = max(0, min(cursor, len(entries) - 1))
            if cursor < scroll:
                scroll = cursor
            if cursor >= scroll + visible:
                scroll = cursor - visible + 1
        else:
            cursor = 0
            scroll = 0

        stdscr.clear()

        # Header
        path_display = f"s3://{prefix}" if prefix else f"s3://{BUCKET}/"
        stdscr.addnstr(0, 0, path_display, width - 1, curses.color_pair(2) | curses.A_BOLD)

        # Column header
        col_header = f"  {'Name':<{max(20, width - 20)}}{'Size':>12}"
        stdscr.addnstr(1, 0, col_header[: width - 1], width - 1, curses.A_DIM)

        # Entries
        if not entries:
            stdscr.addnstr(header_lines, 2, "(empty)", width - 3)
        else:
            for i in range(min(visible, len(entries) - scroll)):
                idx = scroll + i
                entry = entries[idx]
                row = header_lines + i

                if entry.is_dir:
                    size_str = "      DIR"
                    style = curses.color_pair(1)
                elif entry.size is not None:
                    size_str = format_size(entry.size)
                    style = 0
                else:
                    size_str = ""
                    style = 0

                name_width = max(20, width - 14)
                display_name = entry.display[:name_width]
                line = f"  {display_name:<{name_width}}{size_str:>12}"

                if idx == cursor:
                    style = curses.color_pair(3) | curses.A_BOLD
                    stdscr.addnstr(row, 0, line.ljust(width - 1), width - 1, style)
                else:
                    stdscr.addnstr(row, 0, line[: width - 1], width - 1, style)

        # Footer / help
        footer_y = height - 2
        position = f" {cursor + 1}/{len(entries)} " if entries else ""
        help_text = " [Enter] open  [Backspace/h] back  [q] quit  [r] refresh"
        stdscr.addnstr(footer_y, 0, help_text[: width - 1], width - 1, curses.A_DIM)
        stdscr.addnstr(footer_y + 1, 0, position[: width - 1], width - 1, curses.color_pair(2))

        stdscr.refresh()

        key = stdscr.getch()

        if key == ord("q") or key == 27:
            break
        elif key == curses.KEY_UP or key == ord("k"):
            cursor = max(0, cursor - 1)
        elif key == curses.KEY_DOWN or key == ord("j"):
            cursor = min(len(entries) - 1, cursor + 1) if entries else 0
        elif key == curses.KEY_PPAGE:
            cursor = max(0, cursor - visible)
        elif key == curses.KEY_NPAGE:
            cursor = min(len(entries) - 1, cursor + visible) if entries else 0
        elif key == ord("g"):
            cursor = 0
        elif key == ord("G"):
            cursor = len(entries) - 1 if entries else 0
        elif key == ord("r"):
            need_reload = True
        elif key in (curses.KEY_ENTER, 10, 13):
            if entries:
                entry = entries[cursor]
                if entry.is_dir:
                    history.append((prefix, cursor, scroll))
                    prefix = entry.name
                    need_reload = True
                else:
                    # View file contents
                    if entry.size is not None and entry.size > MAX_FILE_SIZE:
                        status = (
                            f"File too large ({format_size(entry.size).strip()}) — max {MAX_FILE_SIZE // (1024*1024)} MB"
                        )
                    else:
                        stdscr.clear()
                        stdscr.addnstr(0, 0, f"Downloading {entry.display} ...", width - 1, curses.color_pair(2))
                        stdscr.refresh()
                        try:
                            raw = fetch_file(fs, entry.name)
                            lines = render_file_content(entry.name, raw)
                            show_file_viewer(stdscr, entry.name, lines)
                        except Exception as e:
                            status = f"Error reading file: {e}"
        elif key in (curses.KEY_BACKSPACE, 127, ord("h"), curses.KEY_LEFT):
            if history:
                prefix, cursor, scroll = history.pop()
                need_reload = True
            elif "/" in prefix.rstrip("/"):
                history.append((prefix, cursor, scroll))
                prefix = prefix.rstrip("/").rsplit("/", 1)[0]
                need_reload = True


def main() -> None:
    fs = connect()

    initial_prefix = BUCKET
    if len(sys.argv) > 1:
        arg = sys.argv[1].removeprefix("s3://")
        if not arg.startswith(BUCKET):
            initial_prefix = f"{BUCKET}/{arg}"
        else:
            initial_prefix = arg

    print(f"Connecting to {ENDPOINT} ...")
    print(f"Bucket: {BUCKET}")

    try:
        items = fs.ls(initial_prefix, detail=False)
    except Exception as e:
        print(f"ERROR: Could not list bucket: {e}", file=sys.stderr)
        sys.exit(1)
    if not items:
        print(f"WARNING: No objects found under s3://{initial_prefix}", file=sys.stderr)

    curses.wrapper(lambda stdscr: run_tui(stdscr, fs, initial_prefix))


if __name__ == "__main__":
    main()
