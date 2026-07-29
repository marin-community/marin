# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Incremental semantic and exact-text search over tracked repository files."""

import hashlib
import math
import sqlite3
import struct
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import search_config
from search_result import SearchResult

MAX_FILE_BYTES = 256 * 1024
CHUNK_CHARACTERS = 6_000
CHUNK_OVERLAP_LINES = 8
EMBEDDING_BATCH_SIZE = 32
INDEX_VERSION = f"1:{search_config.EMBED_MODEL}:{CHUNK_CHARACTERS}"
REPOSITORY_CACHE_DIRECTORY = "repositories"
DAEMON_SOCKET_NAME = "search.sock"

INDEXED_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cfg",
        ".cpp",
        ".css",
        ".go",
        ".h",
        ".html",
        ".ini",
        ".java",
        ".js",
        ".json",
        ".md",
        ".proto",
        ".py",
        ".rs",
        ".rst",
        ".sh",
        ".sql",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".vue",
        ".yaml",
        ".yml",
    }
)
INDEXED_NAMES = frozenset({"AGENTS.md", "Dockerfile", "LICENSE", "Makefile"})
EXCLUDED_PARTS = frozenset(
    {
        ".cache",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "checkpoints",
        "dist",
        "node_modules",
        "target",
        "third_party",
        "vendor",
        "wandb",
    }
)
SECRET_PARTS = frozenset({".secrets", "secret", "secrets"})
SECRET_NAMES = frozenset(
    {
        ".env",
        "credentials.json",
        "service-account.json",
        "service_account.json",
    }
)
SECRET_SUFFIXES = frozenset({".key", ".p12", ".pem", ".pfx"})
GENERATED_NAMES = frozenset(
    {
        "package-lock.json",
        "pnpm-lock.yaml",
        "uv.lock",
        "yarn.lock",
    }
)

EmbeddingProvider = Callable[[str, list[str]], list[list[float]]]


@dataclass(frozen=True)
class TextChunk:
    index: int
    start_line: int
    text: str


@dataclass(frozen=True)
class IndexedFile:
    path: str
    digest: str
    title: str
    chunks: tuple[TextChunk, ...]


@dataclass
class RankedChunk:
    path: str
    title: str
    chunk_index: int
    start_line: int
    text: str
    score: float
    distance: float | None
    lexical_score: float | None


def repository_root(path: Path | None = None) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def tracked_paths(root: Path) -> list[PurePosixPath]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [PurePosixPath(value.decode()) for value in result.stdout.split(b"\0") if value]


def eligible_path(path: PurePosixPath) -> bool:
    """Whether a tracked path is safe and useful enough to enter the file index."""
    lowered_parts = tuple(part.lower() for part in path.parts)
    if any(part in EXCLUDED_PARTS or part in SECRET_PARTS for part in lowered_parts):
        return False
    if lowered_parts[:2] == ("config", "external"):
        return False
    name = path.name
    lowered_name = name.lower()
    if lowered_name in SECRET_NAMES or lowered_name in GENERATED_NAMES:
        return False
    if path.suffix.lower() in SECRET_SUFFIXES:
        return False
    if lowered_name.endswith((".min.js", ".min.css", ".generated.py", ".generated.ts")):
        return False
    return name in INDEXED_NAMES or path.suffix.lower() in INDEXED_SUFFIXES


def read_indexed_file(root: Path, path: PurePosixPath) -> IndexedFile | None:
    absolute = root.joinpath(*path.parts)
    try:
        if not absolute.is_file() or absolute.stat().st_size > MAX_FILE_BYTES:
            return None
        data = absolute.read_bytes()
    except OSError:
        return None
    if b"\0" in data:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    digest = hashlib.sha256(data).hexdigest()
    title = file_title(path, text)
    return IndexedFile(str(path), digest, title, tuple(text_chunks(text)))


def file_title(path: PurePosixPath, text: str) -> str:
    if path.suffix.lower() in {".md", ".rst"}:
        for line in text.splitlines()[:80]:
            heading = line.lstrip("#").strip() if line.startswith("#") else ""
            if heading:
                return heading
    return path.name


def text_chunks(text: str) -> Iterable[TextChunk]:
    lines = text.splitlines()
    if not lines:
        yield TextChunk(0, 1, "")
        return
    start = 0
    index = 0
    while start < len(lines):
        length = 0
        end = start
        while end < len(lines) and (length < CHUNK_CHARACTERS or end == start):
            length += len(lines[end]) + 1
            end += 1
        yield TextChunk(index, start + 1, "\n".join(lines[start:end]))
        if end == len(lines):
            return
        start = max(start + 1, end - CHUNK_OVERLAP_LINES)
        index += 1


def cache_path(root: Path, cache_root: Path) -> Path:
    key = hashlib.sha256(str(root).encode()).hexdigest()[:16]
    return cache_root / REPOSITORY_CACHE_DIRECTORY / f"{key}.sqlite3"


def connect_index(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            path TEXT NOT NULL,
            digest TEXT NOT NULL,
            title TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            start_line INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            PRIMARY KEY (path, chunk_index)
        )
        """
    )
    version = conn.execute("SELECT value FROM metadata WHERE key = 'version'").fetchone()
    if version != (INDEX_VERSION,):
        conn.execute("DELETE FROM chunks")
        conn.execute(
            "INSERT INTO metadata(key, value) VALUES ('version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (INDEX_VERSION,),
        )
        conn.commit()
    return conn


def embedding_bytes(values: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def decode_embedding(value: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(value) // 4}f", value)


def batches(values: Sequence[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def refresh_index(root: Path, conn: sqlite3.Connection, embed: EmbeddingProvider) -> int:
    """Refresh changed tracked files and return the number of embedded chunks."""
    current: dict[str, IndexedFile] = {}
    for path in tracked_paths(root):
        if not eligible_path(path):
            continue
        indexed = read_indexed_file(root, path)
        if indexed is not None:
            current[indexed.path] = indexed

    stored = dict(conn.execute("SELECT path, min(digest) FROM chunks GROUP BY path"))
    changed = [entry for path, entry in current.items() if stored.get(path) != entry.digest]
    texts = [f"{entry.path}\n{entry.title}\n\n{chunk.text}" for entry in changed for chunk in entry.chunks]
    vectors = [vector for batch in batches(texts, EMBEDDING_BATCH_SIZE) for vector in embed("passage", batch)]
    if len(vectors) != len(texts):
        raise ValueError(f"embedding provider returned {len(vectors)} vectors for {len(texts)} passages")

    current_paths = set(current)
    with conn:
        if current_paths:
            placeholders = ", ".join("?" for _ in current_paths)
            conn.execute(f"DELETE FROM chunks WHERE path NOT IN ({placeholders})", tuple(sorted(current_paths)))
        else:
            conn.execute("DELETE FROM chunks")
        offset = 0
        for entry in changed:
            conn.execute("DELETE FROM chunks WHERE path = ?", (entry.path,))
            for chunk in entry.chunks:
                conn.execute(
                    """
                    INSERT INTO chunks(
                        path, digest, title, chunk_index, start_line, text, embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.path,
                        entry.digest,
                        entry.title,
                        chunk.index,
                        chunk.start_line,
                        chunk.text,
                        embedding_bytes(vectors[offset]),
                    ),
                )
                offset += 1
    return len(texts)


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"embedding dimensions differ: {len(left)} != {len(right)}")
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(sum(value * value for value in right))
    if denominator == 0:
        return 1.0
    return 1.0 - sum(a * b for a, b in zip(left, right, strict=True)) / denominator


def best_line(text: str, query: str, start_line: int) -> tuple[int, str]:
    lines = text.splitlines()
    lowered = query.casefold()
    for offset, line in enumerate(lines):
        if lowered in line.casefold():
            return start_line + offset, line.strip()
    for offset, line in enumerate(lines):
        if line.strip():
            return start_line + offset, line.strip()
    return start_line, ""


def search_files(
    query: str,
    limit: int,
    embed: EmbeddingProvider,
    *,
    root: Path | None = None,
    index_path: Path | None = None,
) -> list[SearchResult]:
    """Return one hybrid-ranked result per tracked repository file."""
    resolved_root = repository_root(root)
    resolved_index_path = index_path or cache_path(resolved_root, Path.home() / ".cache" / "echo")
    with connect_index(resolved_index_path) as conn:
        refresh_index(resolved_root, conn, embed)
    query_vectors = embed("query", [query])
    if len(query_vectors) != 1:
        raise ValueError("embedding provider must return one query vector")
    return search_index(query, query_vectors[0], limit, resolved_index_path)


def search_index(
    query: str,
    query_vector: Sequence[float],
    limit: int,
    index_path: Path,
) -> list[SearchResult]:
    """Search a warm index without refreshing it."""
    with connect_index(index_path) as conn:
        rows = [
            (str(row[0]), str(row[1]), int(row[2]), int(row[3]), str(row[4]), bytes(row[5]))
            for row in conn.execute("SELECT path, title, chunk_index, start_line, text, embedding FROM chunks")
        ]

    semantic = sorted(
        ((cosine_distance(query_vector, decode_embedding(row[5])), row) for row in rows),
        key=lambda item: (item[0], item[1][0], item[1][2]),
    )[: search_config.candidate_limit(limit)]
    lowered_query = query.casefold()
    lexical = sorted(
        (
            (
                (3.0 if lowered_query in row[0].casefold() else 0.0) + row[4].casefold().count(lowered_query),
                row,
            )
            for row in rows
            if lowered_query in row[0].casefold() or lowered_query in row[4].casefold()
        ),
        key=lambda item: (-item[0], item[1][0], item[1][2]),
    )[: search_config.candidate_limit(limit)]

    candidates: dict[tuple[str, int], RankedChunk] = {}
    for rank, (distance, row) in enumerate(semantic, 1):
        if distance > search_config.MAX_SEMANTIC_DISTANCE:
            continue
        candidates[(row[0], row[2])] = RankedChunk(
            path=row[0],
            title=row[1],
            chunk_index=row[2],
            start_line=row[3],
            text=row[4],
            score=1.0 / (search_config.RRF_K + rank),
            distance=distance,
            lexical_score=None,
        )
    for rank, (lexical_score, row) in enumerate(lexical, 1):
        candidate = candidates.setdefault(
            (row[0], row[2]),
            RankedChunk(
                path=row[0],
                title=row[1],
                chunk_index=row[2],
                start_line=row[3],
                text=row[4],
                score=0.0,
                distance=None,
                lexical_score=None,
            ),
        )
        candidate.score += search_config.LEXICAL_WEIGHT / (search_config.RRF_K + rank)
        candidate.lexical_score = lexical_score

    by_path: dict[str, RankedChunk] = {}
    for candidate in candidates.values():
        previous = by_path.get(candidate.path)
        if previous is None or candidate.score > previous.score:
            by_path[candidate.path] = candidate

    results: list[SearchResult] = []
    for path, candidate in by_path.items():
        line, matching_line = best_line(candidate.text, query, candidate.start_line)
        results.append(
            SearchResult(
                id=f"file:{path}",
                domain="file",
                title=candidate.title,
                subtitle=f"{path}:{line}",
                url=path,
                snippet=matching_line[:240],
                score=candidate.score,
                distance=candidate.distance,
                lexical_score=candidate.lexical_score,
            )
        )
    return sorted(results, key=lambda result: (-result.score, result.url))[:limit]
