# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The interactive ``fsutil browse`` screen.

A curses browser over the whole bucket tree: the root lists every declared bucket, and
descending crosses backends without ceremony because each listing routes itself. Reads
are bounded by the same preview cap the CLI uses, so opening a file is always cheap.
"""

import curses
from dataclasses import dataclass, field

from rigging.filesystem.buckets import MissingCredentials
from rigging.fsutil.listing import ROOT, Entry, list_entries, parent_url, read_decompressed_preview, total_size
from rigging.fsutil.render import file_lines, format_size, format_time

_HELP = "[enter] open  [backspace] up  [/] filter  [s] sort  [d] size  [y] print URL  [q] quit"
_VIEWER_HELP = "[q] close  [j/k] scroll  [g/G] top/bottom"

# Sort orders cycled by `s`, in cycle order.
_SORTS = ("name", "size", "modified")


@dataclass
class Screen:
    """Everything the browser shows: where it is, what it found, and how it is sorted."""

    url: str
    entries: list[Entry] = field(default_factory=list)
    cursor: int = 0
    scroll: int = 0
    filter_text: str = ""
    sort: str = _SORTS[0]
    status: str = ""

    def visible(self) -> list[Entry]:
        entries = [e for e in self.entries if self.filter_text.lower() in e.name.lower()]
        if self.sort == "size":
            entries.sort(key=lambda e: (not e.is_dir, -(e.size or 0)))
        elif self.sort == "modified":
            entries.sort(key=lambda e: (not e.is_dir, e.mtime is None, e.mtime), reverse=True)
        return entries

    def selected(self) -> Entry | None:
        entries = self.visible()
        return entries[self.cursor] if 0 <= self.cursor < len(entries) else None


def run(url: str = ROOT) -> None:
    """Open the browser at *url* and block until the user quits."""
    curses.wrapper(lambda stdscr: _loop(stdscr, url))


def _loop(stdscr: "curses.window", url: str) -> None:
    curses.curs_set(0)
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)

    screen = Screen(url=url)
    _reload(screen)

    while True:
        _draw(stdscr, screen)
        key = stdscr.getch()
        if key in (ord("q"), 27):
            return
        _handle(stdscr, screen, key)


def _reload(screen: Screen) -> None:
    """Re-list the current location, reporting a failure in the status line."""
    screen.cursor = 0
    screen.scroll = 0
    screen.filter_text = ""
    try:
        screen.entries = list_entries(screen.url)
        screen.status = f"{len(screen.entries)} entries"
    except MissingCredentials as e:
        screen.entries = []
        screen.status = str(e)
    except Exception as e:  # every backend raises its own listing errors; show, don't crash
        screen.entries = []
        screen.status = f"error: {e}"


def _handle(stdscr: "curses.window", screen: Screen, key: int) -> None:
    entries = screen.visible()

    if key in (curses.KEY_DOWN, ord("j")):
        screen.cursor = min(len(entries) - 1, screen.cursor + 1) if entries else 0
    elif key in (curses.KEY_UP, ord("k")):
        screen.cursor = max(0, screen.cursor - 1)
    elif key == curses.KEY_NPAGE:
        screen.cursor = min(len(entries) - 1, screen.cursor + 10) if entries else 0
    elif key == curses.KEY_PPAGE:
        screen.cursor = max(0, screen.cursor - 10)
    elif key == ord("g"):
        screen.cursor = 0
    elif key == ord("G"):
        screen.cursor = max(0, len(entries) - 1)
    elif key in (curses.KEY_ENTER, 10, 13, curses.KEY_RIGHT, ord("l")):
        _open(stdscr, screen)
    elif key in (curses.KEY_BACKSPACE, 127, 8, curses.KEY_LEFT, ord("h")):
        screen.url = parent_url(screen.url)
        _reload(screen)
    elif key == ord("r"):
        _reload(screen)
    elif key == ord("/"):
        screen.filter_text = _prompt(stdscr, "filter: ")
        screen.cursor = 0
    elif key == ord("s"):
        screen.sort = _SORTS[(_SORTS.index(screen.sort) + 1) % len(_SORTS)]
        screen.cursor = 0
    elif key == ord("d"):
        _measure(stdscr, screen)
    elif key == ord("y"):
        selected = screen.selected()
        if selected:
            screen.status = selected.url


def _open(stdscr: "curses.window", screen: Screen) -> None:
    selected = screen.selected()
    if selected is None:
        return
    if selected.is_dir:
        screen.url = selected.url
        _reload(screen)
        return

    try:
        preview = read_decompressed_preview(selected.url)
        lines = file_lines(selected.name, preview.data)
    except Exception as e:
        screen.status = f"error: {e}"
        return
    truncated_bytes = len(preview.data) if preview.truncated else None
    _view(stdscr, selected.name, lines, truncated_bytes)


def _measure(stdscr: "curses.window", screen: Screen) -> None:
    """Size the highlighted prefix, which means walking it — so say so while it runs."""
    selected = screen.selected()
    if selected is None:
        return
    screen.status = f"sizing {selected.name} ..."
    _draw(stdscr, screen)
    try:
        size, count = total_size(selected.url)
        screen.status = f"{selected.name}: {format_size(size)} in {count} objects"
    except Exception as e:
        screen.status = f"error: {e}"


def _draw(stdscr: "curses.window", screen: Screen) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    entries = screen.visible()
    body_height = max(1, height - 3)

    if screen.cursor < screen.scroll:
        screen.scroll = screen.cursor
    elif screen.cursor >= screen.scroll + body_height:
        screen.scroll = screen.cursor - body_height + 1

    header = screen.url or "buckets"
    stdscr.addnstr(0, 0, header, width - 1, curses.color_pair(1) | curses.A_BOLD)
    subtitle = f"sort: {screen.sort}" + (f"  filter: {screen.filter_text}" if screen.filter_text else "")
    stdscr.addnstr(1, 0, subtitle, width - 1, curses.A_DIM)

    for row, entry in enumerate(entries[screen.scroll : screen.scroll + body_height]):
        index = screen.scroll + row
        name = f"{entry.name}/" if entry.is_dir else entry.name
        line = f"{format_size(entry.size):>10}  {format_time(entry.mtime):>16}  {name}"
        attr = curses.A_REVERSE if index == screen.cursor else (curses.color_pair(2) if entry.is_dir else 0)
        stdscr.addnstr(2 + row, 0, line, width - 1, attr)

    status = screen.status or _HELP
    attr = curses.color_pair(3) if status.startswith("error") else curses.A_DIM
    stdscr.addnstr(height - 1, 0, status[: width - 1], width - 1, attr)
    stdscr.refresh()


def _view(stdscr: "curses.window", name: str, lines: list[str], truncated_bytes: int | None) -> None:
    """Scrollable read-only pager for one file's rendered lines."""
    scroll = 0
    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        body_height = max(1, height - 3)

        stdscr.addnstr(0, 0, name, width - 1, curses.color_pair(1) | curses.A_BOLD)
        subtitle = f"{len(lines)} lines"
        if truncated_bytes is not None:
            subtitle += f"  preview truncated at {format_size(truncated_bytes)}"
        stdscr.addnstr(1, 0, subtitle, width - 1, curses.A_DIM)
        for row, line in enumerate(lines[scroll : scroll + body_height]):
            stdscr.addnstr(2 + row, 0, line, width - 1)
        stdscr.addnstr(height - 1, 0, _VIEWER_HELP[: width - 1], width - 1, curses.A_DIM)
        stdscr.refresh()

        key = stdscr.getch()
        max_scroll = max(0, len(lines) - body_height)
        if key in (ord("q"), 27, curses.KEY_BACKSPACE, 127, 8, curses.KEY_LEFT, ord("h")):
            return
        elif key in (curses.KEY_DOWN, ord("j")):
            scroll = min(max_scroll, scroll + 1)
        elif key in (curses.KEY_UP, ord("k")):
            scroll = max(0, scroll - 1)
        elif key == curses.KEY_NPAGE:
            scroll = min(max_scroll, scroll + body_height)
        elif key == curses.KEY_PPAGE:
            scroll = max(0, scroll - body_height)
        elif key == ord("g"):
            scroll = 0
        elif key == ord("G"):
            scroll = max_scroll


def _prompt(stdscr: "curses.window", label: str) -> str:
    """Read a line from the status row with echo on, restoring the screen state after."""
    height, width = stdscr.getmaxyx()
    curses.echo()
    curses.curs_set(1)
    stdscr.addnstr(height - 1, 0, " " * (width - 1), width - 1)
    stdscr.addnstr(height - 1, 0, label, width - 1)
    stdscr.refresh()
    try:
        return stdscr.getstr(height - 1, len(label), width - len(label) - 2).decode("utf-8")
    finally:
        curses.noecho()
        curses.curs_set(0)
