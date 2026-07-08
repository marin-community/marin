# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared, dependency-free helpers for the code-search evaluation.

Kept to the standard library so the engine adapters — which run in isolated ``uv``
environments with their own heavy deps (sentence-transformers, bm25s, ...) — can
import it over ``PYTHONPATH`` without pulling in the marin package. Defines the
corpus (which files are indexable), a uniform line-window chunker, and the on-disk
JSON contracts the pipeline and adapters exchange.

Contracts (one JSON object per line):

- queries file: ``{"query_id": str, "query": str, "gold_files": [str], ...}``
- hits file:    ``{"query_id": str, "hits": [{"file": str, "start_line": int,
  "end_line": int, "score": float}]}`` — ``file`` is repo-relative, hits ranked best-first.

Snippets and token counts are derived by the scorer from ``(file, start_line,
end_line)`` against the repo, so answer size is measured the same way for every
engine regardless of how it chunks.
"""

import argparse
import json
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

CHARS_PER_TOK = 4.0
DEFAULT_K = 10  # engines return top-K; pass@k for k<=K is computed by truncation
CHUNK_LINES = 40  # line-window chunk size for the indexing engines
CHUNK_OVERLAP = 10
MAX_FILE_BYTES = 1_000_000  # skip files larger than this (generated/vendored blobs)

# The searchable corpus: source + docs, excluding vendored/generated trees and binaries.
INCLUDE_EXTS = frozenset(
    {
        ".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".vue", ".go", ".java", ".c", ".h",
        ".cc", ".cpp", ".hpp", ".sh", ".sql", ".proto", ".toml", ".yaml", ".yml",
        ".md", ".rst", ".txt", ".cfg", ".ini",
    }
)  # fmt: skip
EXCLUDE_DIRS = frozenset(
    {
        ".git", "node_modules", "dist", "build", "__pycache__", ".venv", "venv",
        ".mypy_cache", ".ruff_cache", ".pytest_cache", "target", ".worktrees",
        ".claude-worktrees", "vendor", ".next", "coverage", "htmlcov", ".idea",
    }
)  # fmt: skip


@dataclass(frozen=True)
class Hit:
    file: str  # repo-relative path
    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    score: float

    def to_json(self) -> dict:
        return {"file": self.file, "start_line": self.start_line, "end_line": self.end_line, "score": self.score}


def iter_source_files(repo_root: str) -> Iterator[str]:
    """Yield repo-relative paths of indexable source/doc files."""
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() not in INCLUDE_EXTS:
                continue
            full = os.path.join(dirpath, fn)
            try:
                if os.path.getsize(full) > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield os.path.relpath(full, repo_root)


def read_text(repo_root: str, relpath: str) -> str:
    with open(os.path.join(repo_root, relpath), encoding="utf-8", errors="replace") as fh:
        return fh.read()


def line_windows(n_lines: int, size: int = CHUNK_LINES, overlap: int = CHUNK_OVERLAP) -> list[tuple[int, int]]:
    """1-indexed inclusive (start, end) line ranges tiling ``n_lines`` with overlap."""
    if n_lines <= 0:
        return []
    step = max(size - overlap, 1)
    out = []
    start = 1
    while start <= n_lines:
        out.append((start, min(start + size - 1, n_lines)))
        if start + size - 1 >= n_lines:
            break
        start += step
    return out


def chunk_file(repo_root: str, relpath: str) -> list[dict]:
    """Split a file into overlapping line-window chunks: ``{file, start_line, end_line, text}``."""
    lines = read_text(repo_root, relpath).splitlines()
    chunks = []
    for start, end in line_windows(len(lines)):
        text = "\n".join(lines[start - 1 : end])
        if text.strip():
            chunks.append({"file": relpath, "start_line": start, "end_line": end, "text": text})
    return chunks


def snippet(repo_root: str, relpath: str, start_line: int, end_line: int) -> str:
    """The text of ``relpath`` lines [start_line, end_line], empty if unreadable."""
    try:
        lines = read_text(repo_root, relpath).splitlines()
    except OSError:
        return ""
    return "\n".join(lines[max(start_line - 1, 0) : end_line])


def est_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOK)


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _dir_size(path: str) -> int:
    total = 0
    for dp, _dn, fns in os.walk(path):
        for fn in fns:
            try:
                total += os.path.getsize(os.path.join(dp, fn))
            except OSError:
                pass
    return total


# An engine adapter supplies these two callables.
BuildIndex = Callable[[str, str], None]  # (repo_root, index_dir) -> None
QueryIndex = Callable[[str, str, list[dict], int], list[dict]]  # (repo, index, queries, k) -> [{query_id, hits}]


def run_engine_cli(build_index: BuildIndex, query_index: QueryIndex) -> None:
    """Standard ``build`` / ``query`` CLI shared by every engine adapter.

    ``build`` times index construction and records index size to
    ``<index_dir>/build_meta.json``; ``query`` reads the benchmark and writes ranked
    hits. Centralizing this keeps timing and size measurement identical across engines.
    """
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--repo", required=True)
    b.add_argument("--index", required=True)
    q = sub.add_parser("query")
    q.add_argument("--repo", required=True)
    q.add_argument("--index", required=True)
    q.add_argument("--queries", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--k", type=int, default=DEFAULT_K)
    args = ap.parse_args()

    if args.cmd == "build":
        os.makedirs(args.index, exist_ok=True)
        t0 = time.time()
        build_index(args.repo, args.index)
        meta = {"build_seconds": round(time.time() - t0, 2), "index_bytes": _dir_size(args.index)}
        with open(os.path.join(args.index, "build_meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        return

    queries = read_jsonl(args.queries)
    t0 = time.time()
    results = query_index(args.repo, args.index, queries, args.k)
    latency = (time.time() - t0) / max(len(queries), 1)
    for r in results:
        r.setdefault("query_latency", round(latency, 4))
    write_jsonl(args.out, results)
