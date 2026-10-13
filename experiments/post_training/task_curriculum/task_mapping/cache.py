# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent content-addressed cache for embeddings."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from pathlib import Path

import numpy as np
from rigging.cache import combined_content_hash


def embedding_cache_key(text: str, model: str) -> str:
    return combined_content_hash((text, model))


class EmbeddingCache(AbstractContextManager["EmbeddingCache"]):
    """SQLite-backed cache used by local and single-node curriculum runs."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(cache_key TEXT PRIMARY KEY, dimensions INTEGER NOT NULL, value BLOB NOT NULL)"
        )

    def __exit__(self, *_args: object) -> None:
        self._connection.close()

    def embedding(self, cache_key: str) -> np.ndarray | None:
        """Return the cached vector, or `None` when the key is absent."""
        row = self._connection.execute(
            "SELECT dimensions, value FROM embeddings WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        if row is None:
            return None
        dimensions, value = row
        return np.frombuffer(value, dtype=np.float32, count=dimensions).copy()

    def put_embeddings(self, values: Sequence[tuple[str, np.ndarray]]) -> None:
        rows: list[tuple[str, int, bytes]] = []
        for cache_key, vector in values:
            vector = np.asarray(vector, dtype=np.float32)
            if vector.ndim != 1:
                raise ValueError(f"expected one-dimensional embedding, got {vector.shape}")
            rows.append((cache_key, vector.shape[0], vector.tobytes()))
        self._connection.executemany(
            "INSERT OR IGNORE INTO embeddings(cache_key, dimensions, value) VALUES (?, ?, ?)", rows
        )
        self._connection.commit()


def cached_embeddings(
    cache: EmbeddingCache,
    texts: Sequence[str],
    model: str,
    embed_missing: Callable[[Sequence[str]], np.ndarray],
) -> np.ndarray:
    """Return an embedding matrix, requesting only unseen texts."""
    if not texts:
        raise ValueError("at least one text is required")

    vectors: list[np.ndarray | None] = []
    missing: dict[str, tuple[str, list[int]]] = {}
    for position, value in enumerate(texts):
        key = embedding_cache_key(value, model)
        cached = cache.embedding(key)
        vectors.append(cached)
        if cached is None:
            if key not in missing:
                missing[key] = (value, [])
            missing[key][1].append(position)

    if missing:
        missing_items = list(missing.items())
        missing_texts = [text for _, (text, _) in missing_items]
        embedded = np.asarray(embed_missing(missing_texts), dtype=np.float32)
        if embedded.ndim != 2 or embedded.shape[0] != len(missing):
            raise ValueError(f"embedder returned shape {embedded.shape} for {len(missing)} texts")
        cache.put_embeddings([(key, vector) for (key, _), vector in zip(missing_items, embedded, strict=True)])
        for (_key, (_, positions)), vector in zip(missing_items, embedded, strict=True):
            for position in positions:
                vectors[position] = vector

    return np.stack([vector for vector in vectors if vector is not None])
