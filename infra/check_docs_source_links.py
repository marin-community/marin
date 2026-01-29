# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate GitHub source links in docs point to paths that exist locally."""

import re
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
ROOT_DIR = DOCS_DIR.parent

LINK_RE = re.compile(r"\]\(([^)]+)\)")
GITHUB_RE = re.compile(r"https://github\.com/marin-community/marin/(blob|tree)/[^/]+/(?P<path>.+)")


def _normalize_url(url: str) -> str:
    url = url.strip()
    if url.startswith("<") and url.endswith(">"):
        url = url[1:-1]
    if "#" in url:
        url = url.split("#", 1)[0]
    if "?" in url:
        url = url.split("?", 1)[0]
    return url


def _check_docs() -> list[str]:
    errors: list[str] = []
    if not DOCS_DIR.exists():
        return errors

    for md_path in DOCS_DIR.rglob("*.md"):
        text = md_path.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(text):
            url = _normalize_url(match.group(1))
            gh_match = GITHUB_RE.match(url)
            if not gh_match:
                continue

            rel_path = gh_match.group("path")
            local_path = ROOT_DIR / rel_path

            if "blob" in url and not local_path.is_file():
                errors.append(f"{md_path.relative_to(ROOT_DIR)}: {url}")
            elif "tree" in url and not local_path.exists():
                errors.append(f"{md_path.relative_to(ROOT_DIR)}: {url}")

    return errors


def main() -> int:
    errors = _check_docs()
    if not errors:
        print("Docs source links: OK")
        return 0

    print("Docs source links: broken")
    for entry in errors:
        print(entry)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
