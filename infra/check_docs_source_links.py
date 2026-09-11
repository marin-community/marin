# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate GitHub source links in docs point to paths that exist locally.

Also reports (advisory-only) broken relative links in markdown outside docs/,
which mkdocs --strict does not cover: lib/*/docs, OPS.md, AGENTS.md, READMEs.
"""

import re
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
ROOT_DIR = DOCS_DIR.parent

LINK_RE = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
GITHUB_RE = re.compile(r"https://github\.com/marin-community/marin/(blob|tree)/(?P<ref>[^/]+)/(?P<path>.+)")
# A commit-pinned ref (full or abbreviated SHA) names an immutable snapshot, so its
# target may legitimately no longer exist at HEAD; only branch/tag links must resolve.
SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")

# projects/ and logbooks/ under any .agents dir are historical snapshots: links
# there describe the tree as of writing and rot by design. snapshots/ holds test
# fixture data. Top-level docs/ is covered by mkdocs --strict.
SKIP_PART_NAMES = {"node_modules", "dist", "build", "target", "projects", "logbooks", "snapshots"}
KEEP_HIDDEN_DIRS = {".agents", ".github", ".claude"}
LINE_SUFFIX_RE = re.compile(r":(\d+)(-\d+)?$")


def _normalize_url(url: str) -> str:
    url = url.strip()
    if url.startswith("<") and url.endswith(">"):
        url = url[1:-1]
    if "#" in url:
        url = url.split("#", 1)[0]
    if "?" in url:
        url = url.split("?", 1)[0]
    return url


def _iter_links(md_path: Path):
    """Yield (raw, normalized) markdown link targets in one file."""
    text = md_path.read_text(encoding="utf-8")
    for match in LINK_RE.finditer(text):
        yield match.group(1), _normalize_url(match.group(1))


def _check_docs() -> list[str]:
    errors: list[str] = []
    if not DOCS_DIR.exists():
        return errors

    for md_path in DOCS_DIR.rglob("*.md"):
        for _, url in _iter_links(md_path):
            gh_match = GITHUB_RE.match(url)
            if not gh_match:
                continue

            if SHA_RE.match(gh_match.group("ref")):
                continue

            rel_path = gh_match.group("path")
            local_path = ROOT_DIR / rel_path

            if "blob" in url and not local_path.is_file():
                errors.append(f"{md_path.relative_to(ROOT_DIR)}: {url}")
            elif "tree" in url and not local_path.exists():
                errors.append(f"{md_path.relative_to(ROOT_DIR)}: {url}")

    return errors


def _iter_relative_check_files() -> list[Path]:
    files = []
    for md_path in ROOT_DIR.rglob("*.md"):
        rel = md_path.relative_to(ROOT_DIR)
        if rel.parts[0] == "docs":
            continue
        parts = set(rel.parts[:-1])
        if parts & SKIP_PART_NAMES:
            continue
        if any(p.startswith(".") and p not in KEEP_HIDDEN_DIRS for p in parts):
            continue
        files.append(md_path)
    return files


def _check_relative_links() -> list[str]:
    findings: list[str] = []
    for md_path in _iter_relative_check_files():
        for raw, url in _iter_links(md_path):
            if not url or "://" in url or url.startswith(("mailto:", "#", "{")):
                continue
            if GITHUB_RE.match(url):
                continue
            url = LINE_SUFFIX_RE.sub("", url)
            base = ROOT_DIR if url.startswith("/") else md_path.parent
            target = (base / url.lstrip("/")).resolve()
            if not target.exists():
                findings.append(f"{md_path.relative_to(ROOT_DIR)}: {raw}")
    return findings


def main() -> int:
    status = 0
    errors = _check_docs()
    if not errors:
        print("Docs source links: OK")
    else:
        print("Docs source links: broken")
        for entry in errors:
            print(entry)
        status = 1

    advisories = _check_relative_links()
    if advisories:
        print(f"\nAdvisory: {len(advisories)} broken relative links outside docs/ (not gating):")
        for entry in advisories:
            print(entry)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
