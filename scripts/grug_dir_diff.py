# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate an HTML side-by-side diff report for two code directories."""

from __future__ import annotations

import argparse
import difflib
import html
import os
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_EXTENSIONS: tuple[str, ...] = (".py", ".pyi", ".js", ".jsx", ".ts", ".tsx")
DEFAULT_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "node_modules",
        "__pycache__",
    }
)
STATUS_ORDER: dict[str, int] = {
    "changed": 0,
    "added": 1,
    "removed": 2,
    "unchanged": 3,
}


@dataclass(frozen=True)
class DiffEntry:
    """Summary record for one compared relative file path."""

    rel_path: str
    status: str
    left_path: Path | None
    right_path: Path | None
    added_lines: int
    deleted_lines: int


def parse_extensions(raw_extensions: str) -> tuple[str, ...]:
    """Normalize a comma-separated extension list into lowercase dot-prefixed entries."""
    normalized: list[str] = []
    for token in raw_extensions.split(","):
        extension = token.strip().lower()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        if extension not in normalized:
            normalized.append(extension)

    if not normalized:
        msg = "At least one extension must be provided when --all-files is not set"
        raise ValueError(msg)

    return tuple(normalized)


def collect_files(
    root: Path,
    *,
    extensions: tuple[str, ...],
    include_all_files: bool,
    ignored_dirs: frozenset[str] = DEFAULT_IGNORED_DIRS,
) -> dict[str, Path]:
    """Collect relative file paths from ``root`` honoring extension and ignore filters."""
    files: dict[str, Path] = {}

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [dirname for dirname in dirnames if dirname not in ignored_dirs]

        for filename in filenames:
            file_path = Path(dirpath) / filename
            if not include_all_files and file_path.suffix.lower() not in extensions:
                continue

            rel_path = file_path.relative_to(root).as_posix()
            files[rel_path] = file_path

    return files


def read_text_lines(path: Path) -> list[str]:
    """Read text content from ``path`` as UTF-8 with replacement for invalid bytes."""
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def line_change_counts(left_lines: list[str], right_lines: list[str]) -> tuple[int, int]:
    """Count added/deleted lines using sequence-level opcodes."""
    added = 0
    deleted = 0
    matcher = difflib.SequenceMatcher(a=left_lines, b=right_lines)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            deleted += i2 - i1
            added += j2 - j1
        elif tag == "delete":
            deleted += i2 - i1
        elif tag == "insert":
            added += j2 - j1
    return added, deleted


def render_report_page(
    *,
    entries: list[DiffEntry],
    left_dir: Path,
    right_dir: Path,
    output_file: Path,
    extensions: tuple[str, ...],
    include_all_files: bool,
    show_unchanged: bool,
    context_lines: int,
) -> None:
    """Render one-page report with summary, file table, and inline diffs."""
    changed = sum(1 for entry in entries if entry.status == "changed")
    added = sum(1 for entry in entries if entry.status == "added")
    removed = sum(1 for entry in entries if entry.status == "removed")
    unchanged = sum(1 for entry in entries if entry.status == "unchanged")
    rendered_entries = [entry for entry in entries if show_unchanged or entry.status != "unchanged"]

    anchor_ids: dict[str, str] = {}
    for index, entry in enumerate(rendered_entries, start=1):
        anchor_ids[entry.rel_path] = f"file-{index}"

    rows: list[str] = []
    for entry in entries:
        anchor_id = anchor_ids.get(entry.rel_path)
        if anchor_id is None:
            file_cell = html.escape(entry.rel_path)
        else:
            file_cell = f'<a href="#{html.escape(anchor_id)}">{html.escape(entry.rel_path)}</a>'

        rows.append(
            "<tr>"
            f"<td>{file_cell}</td>"
            f'<td><span class="status {html.escape(entry.status)}">{html.escape(entry.status)}</span></td>'
            f'<td class="num">+{entry.added_lines}</td>'
            f'<td class="num">-{entry.deleted_lines}</td>'
            "</tr>"
        )

    extensions_label = "all files" if include_all_files else ", ".join(extensions)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    sections: list[str] = []
    for entry in rendered_entries:
        left_lines = read_text_lines(entry.left_path) if entry.left_path else []
        right_lines = read_text_lines(entry.right_path) if entry.right_path else []
        table_html = difflib.HtmlDiff(tabsize=4, wrapcolumn=120).make_table(
            left_lines,
            right_lines,
            fromdesc=html.escape(str(left_dir / entry.rel_path)),
            todesc=html.escape(str(right_dir / entry.rel_path)),
            context=True,
            numlines=context_lines,
        )
        anchor_id = anchor_ids[entry.rel_path]
        sections.append(
            '<section class="diff-section" '
            f'id="{html.escape(anchor_id)}">'
            '<div class="diff-header">'
            f"<h2>{html.escape(entry.rel_path)}</h2>"
            f'<span class="status {html.escape(entry.status)}">{html.escape(entry.status)}</span>'
            "</div>"
            f"{table_html}"
            '<p><a href="#top">Back to top</a></p>'
            "</section>"
        )

    index_html = f"""<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Directory Diff Report</title>
    <style>
      :root {{
        --bg: #f6f8fb;
        --surface: #ffffff;
        --ink: #111827;
        --muted: #6b7280;
        --line: #e5e7eb;
        --changed: #b45309;
        --added: #166534;
        --removed: #b91c1c;
        --unchanged: #4b5563;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        background: var(--bg);
        color: var(--ink);
        font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }}
      main {{ max-width: 1180px; margin: 0 auto; padding: 24px 20px 40px; }}
      .panel {{
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 14px 16px;
      }}
      .summary {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin-top: 12px;
      }}
      .metric {{
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 8px 10px;
        background: #fafafa;
      }}
      .metric .label {{ color: var(--muted); font-size: 12px; }}
      .metric .value {{ font-size: 22px; font-weight: 700; }}
      .filters {{ color: var(--muted); font-size: 14px; margin-top: 10px; }}
      table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 16px;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
      }}
      th, td {{
        text-align: left;
        border-bottom: 1px solid var(--line);
        padding: 10px 12px;
      }}
      th {{
        background: #f9fafb;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-size: 12px;
      }}
      td.num {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      }}
      a {{ color: #0059b8; text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}
      .status {{
        display: inline-block;
        border-radius: 999px;
        padding: 2px 8px;
        font-size: 11px;
        text-transform: uppercase;
        font-weight: 600;
      }}
      .status.changed {{ color: var(--changed); background: #ffedd5; }}
      .status.added {{ color: var(--added); background: #dcfce7; }}
      .status.removed {{ color: var(--removed); background: #fee2e2; }}
      .status.unchanged {{ color: var(--unchanged); background: #e5e7eb; }}
      .diff-section {{
        margin-top: 18px;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 12px;
      }}
      .diff-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
      }}
      .diff-header h2 {{
        margin: 6px 0 10px;
        font-size: 18px;
      }}
      table.diff {{
        width: 100%;
        border-collapse: collapse;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        font-size: 12px;
        background: #fff;
      }}
      .diff th, .diff td {{ padding: 2px 6px; vertical-align: top; }}
      .diff_next {{ white-space: nowrap; }}
      .diff_add {{ background: #e8f5e9; }}
      .diff_sub {{ background: #ffebee; }}
      .diff_chg {{ background: #fff8e1; }}
      .diff_header {{ background: #f3f4f6; }}
    </style>
  </head>
  <body>
    <main id=\"top\">
      <section class=\"panel\">
        <h1>Directory Diff Report</h1>
        <div><strong>Left:</strong> {html.escape(str(left_dir.resolve()))}</div>
        <div><strong>Right:</strong> {html.escape(str(right_dir.resolve()))}</div>
        <div class=\"filters\"><strong>Filter:</strong> {html.escape(extensions_label)}</div>
        <div class=\"filters\"><strong>Generated:</strong> {now_utc}</div>
        <div class=\"filters\"><strong>Inline Diffs:</strong> {len(rendered_entries)} files</div>
        <div class=\"summary\">
          <div class=\"metric\"><div class=\"label\">Changed</div><div class=\"value\">{changed}</div></div>
          <div class=\"metric\"><div class=\"label\">Added</div><div class=\"value\">{added}</div></div>
          <div class=\"metric\"><div class=\"label\">Removed</div><div class=\"value\">{removed}</div></div>
          <div class=\"metric\"><div class=\"label\">Unchanged</div><div class=\"value\">{unchanged}</div></div>
        </div>
      </section>
      <table>
        <thead>
          <tr>
            <th>File</th>
            <th>Status</th>
            <th>Lines Added</th>
            <th>Lines Deleted</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
      {''.join(sections)}
    </main>
  </body>
</html>
"""

    output_file.write_text(index_html, encoding="utf-8")


def build_directory_diff_report(
    *,
    left_dir: Path,
    right_dir: Path,
    output_dir: Path,
    extensions: tuple[str, ...],
    include_all_files: bool,
    show_unchanged: bool,
    context_lines: int,
) -> tuple[Path, list[DiffEntry]]:
    """Build the full one-page HTML report and return the index path with all entries."""
    left_files = collect_files(
        left_dir,
        extensions=extensions,
        include_all_files=include_all_files,
    )
    right_files = collect_files(
        right_dir,
        extensions=extensions,
        include_all_files=include_all_files,
    )

    all_paths = sorted(set(left_files) | set(right_files))
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[DiffEntry] = []
    for rel_path in all_paths:
        left_path = left_files.get(rel_path)
        right_path = right_files.get(rel_path)

        if left_path is None:
            right_lines = read_text_lines(right_path)
            status = "added"
            added_lines = len(right_lines)
            deleted_lines = 0
        elif right_path is None:
            left_lines = read_text_lines(left_path)
            status = "removed"
            added_lines = 0
            deleted_lines = len(left_lines)
        else:
            left_lines = read_text_lines(left_path)
            right_lines = read_text_lines(right_path)
            if left_lines == right_lines:
                status = "unchanged"
                added_lines = 0
                deleted_lines = 0
            else:
                status = "changed"
                added_lines, deleted_lines = line_change_counts(left_lines, right_lines)

        entry = DiffEntry(
            rel_path=rel_path,
            status=status,
            left_path=left_path,
            right_path=right_path,
            added_lines=added_lines,
            deleted_lines=deleted_lines,
        )
        entries.append(entry)

    entries.sort(key=lambda entry: (STATUS_ORDER[entry.status], entry.rel_path))
    index_file = output_dir / "index.html"
    render_report_page(
        entries=entries,
        left_dir=left_dir,
        right_dir=right_dir,
        output_file=index_file,
        extensions=extensions,
        include_all_files=include_all_files,
        show_unchanged=show_unchanged,
        context_lines=context_lines,
    )

    return index_file, entries


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a browsable HTML side-by-side diff for two directories. "
            "Defaults to Python/JS/TS file extensions."
        )
    )
    parser.add_argument("left", type=Path, help="Left directory")
    parser.add_argument("right", type=Path, help="Right directory")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(".grug-diff-report"),
        help="Output report directory (default: .grug-diff-report)",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help=("Comma-separated file extensions to include, e.g. '.py,.pyi'. " "Ignored if --all-files is set."),
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Include all files (not just filtered extensions).",
    )
    parser.add_argument(
        "--show-unchanged",
        action="store_true",
        help="Generate per-file pages for unchanged files as well.",
    )
    parser.add_argument(
        "--context-lines",
        type=int,
        default=3,
        help="Context lines shown around changes in side-by-side view.",
    )
    parser.add_argument(
        "--open",
        dest="open",
        action="store_true",
        default=True,
        help="Open the generated index page in your default browser (default: enabled).",
    )
    parser.add_argument(
        "--no-open",
        dest="open",
        action="store_false",
        help="Do not open the generated report in a browser.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()

    if not args.left.is_dir():
        raise NotADirectoryError(f"Left path is not a directory: {args.left}")
    if not args.right.is_dir():
        raise NotADirectoryError(f"Right path is not a directory: {args.right}")

    if args.context_lines < 0:
        raise ValueError("--context-lines must be >= 0")

    extensions = parse_extensions(args.extensions) if not args.all_files else tuple()

    index_path, entries = build_directory_diff_report(
        left_dir=args.left,
        right_dir=args.right,
        output_dir=args.out,
        extensions=extensions,
        include_all_files=args.all_files,
        show_unchanged=args.show_unchanged,
        context_lines=args.context_lines,
    )

    changed = sum(1 for entry in entries if entry.status == "changed")
    added = sum(1 for entry in entries if entry.status == "added")
    removed = sum(1 for entry in entries if entry.status == "removed")
    unchanged = sum(1 for entry in entries if entry.status == "unchanged")

    print(f"Wrote report: {index_path.resolve()}")
    print(f"Changed: {changed}  Added: {added}  Removed: {removed}  Unchanged: {unchanged}")

    if args.open:
        webbrowser.open(index_path.resolve().as_uri(), new=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
