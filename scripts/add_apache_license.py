#!/usr/bin/env python
# Copyright 2024 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility for adding Apache license headers to source files.

The default copyright holder is ``The Marin Authors``. Override with
``--holder`` if your project uses a different name.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

DEFAULT_HOLDER = "The Marin Authors"

APACHE_LICENSE_TEMPLATE = """{p} Copyright {year} {holder}
{p}
{p} Licensed under the Apache License, Version 2.0 (the \"License\");
{p} you may not use this file except in compliance with the License.
{p} You may obtain a copy of the License at
{p}
{p}     http://www.apache.org/licenses/LICENSE-2.0
{p}
{p} Unless required by applicable law or agreed to in writing, software
{p} distributed under the License is distributed on an \"AS IS\" BASIS,
{p} WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
{p} See the License for the specific language governing permissions and
{p} limitations under the License.
"""

SKIP_DIRS = {".git", ".venv", "build", "dist"}

# Mapping from file extension to comment prefix.
COMMENT_PREFIXES = {
    ".py": "#",
    ".sh": "#",
    ".bash": "#",
    ".rs": "//",
    ".c": "//",
    ".h": "//",
    ".cpp": "//",
    ".hpp": "//",
}


def find_license_line(text: str, prefix: str, holder: str) -> int | None:
    """Return the line number of an existing license header, if present."""
    head = text.splitlines()[:20]
    for i, line in enumerate(head):
        if line.startswith(f"{prefix} Copyright") and holder in line:
            return i
    return None


def add_or_update_license(path: Path, year: int, holder: str, prefix: str) -> bool:
    """Ensure ``path`` contains an up-to-date Apache license header.

    Args:
        path: File path to update.
        year: Year to use in the license.
        holder: Copyright holder string.
        prefix: Comment prefix for the file type.

    Returns:
        True if the file was modified, False otherwise.
    """

    original = path.read_text()
    lines = original.splitlines()
    line_no = find_license_line(original, prefix, holder)
    if line_no is not None:
        if str(year) not in lines[line_no]:
            lines[line_no] = f"{prefix} Copyright {year} {holder}"
            newline = "\n" if original.endswith("\n") else ""
            path.write_text("\n".join(lines) + newline)
            return True
        return False

    header = APACHE_LICENSE_TEMPLATE.format(p=prefix, year=year, holder=holder)
    path.write_text(f"{header}\n{original}")
    return True


def iter_source_files(root: Path, extension: str):
    """Yield source files under root matching extension."""
    for file in root.rglob(f"*{extension}"):
        if file.is_file() and not any(part in SKIP_DIRS for part in file.parts):
            yield file


def main() -> None:
    parser = argparse.ArgumentParser(description="Add Apache license headers to source files.")
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="Root directory to search.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=datetime.now().year,
        help="Copyright year to use in the header.",
    )
    parser.add_argument(
        "--holder",
        default=DEFAULT_HOLDER,
        help="Copyright holder to use in the header.",
    )
    parser.add_argument(
        "--ext",
        action="append",
        default=[".py"],
        help="File extensions to process.",
    )
    args = parser.parse_args()

    updated = 0
    for ext in args.ext:
        prefix = COMMENT_PREFIXES.get(ext, "#")
        for file in iter_source_files(args.root, ext):
            if add_or_update_license(file, args.year, args.holder, prefix):
                updated += 1
    print(f"Updated {updated} files.")


if __name__ == "__main__":
    main()
