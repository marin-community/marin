# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Safety filtering, chunking, and embedding for repository files."""

import hashlib
import struct
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

MAX_FILE_BYTES = 256 * 1024
CHUNK_CHARACTERS = 1_800
CHUNK_OVERLAP_LINES = 4
EMBEDDING_BATCH_SIZE = 32

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

EmbeddingProvider = Callable[[list[str]], Iterable[Sequence[float]]]


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


@dataclass(frozen=True)
class EmbeddedChunk:
    path: str
    digest: str
    title: str
    chunk_index: int
    start_line: int
    text: str
    embedding: bytes


def repository_path(value: str) -> PurePosixPath | None:
    """Return a safe repository-relative path."""
    if "\\" in value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        return None
    return path


def eligible_path(path: PurePosixPath) -> bool:
    """Whether a tracked path is safe and useful enough to enter the index."""
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


def indexed_file(path: PurePosixPath, data: bytes) -> IndexedFile | None:
    """Decode and chunk one eligible repository blob."""
    if not eligible_path(path) or len(data) > MAX_FILE_BYTES or b"\0" in data:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    digest = hashlib.sha256(data).hexdigest()
    return IndexedFile(str(path), digest, file_title(path, text), tuple(text_chunks(text)))


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


def embed_files(files: Iterable[IndexedFile], embed: EmbeddingProvider) -> list[EmbeddedChunk]:
    """Embed repository files in bounded batches."""
    pending: list[tuple[IndexedFile, TextChunk]] = []
    embedded: list[EmbeddedChunk] = []
    for file in files:
        for chunk in file.chunks:
            pending.append((file, chunk))
            if len(pending) == EMBEDDING_BATCH_SIZE:
                embedded.extend(embed_batch(pending, embed))
                pending.clear()
    if pending:
        embedded.extend(embed_batch(pending, embed))
    return embedded


def embed_batch(
    pending: Sequence[tuple[IndexedFile, TextChunk]],
    embed: EmbeddingProvider,
) -> list[EmbeddedChunk]:
    passages = [f"{file.path}\n{file.title}\n\n{chunk.text}" for file, chunk in pending]
    vectors = list(embed(passages))
    if len(vectors) != len(pending):
        raise ValueError(f"embedding provider returned {len(vectors)} vectors for {len(pending)} passages")
    return [
        EmbeddedChunk(
            path=file.path,
            digest=file.digest,
            title=file.title,
            chunk_index=chunk.index,
            start_line=chunk.start_line,
            text=chunk.text,
            embedding=struct.pack(f"<{len(vector)}f", *vector),
        )
        for (file, chunk), vector in zip(pending, vectors, strict=True)
    ]


def decode_embedding(value: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(value) // 4}f", value)
