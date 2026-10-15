# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared, content-addressed embedding passes for task and curriculum descriptions."""

import json
import time
from collections.abc import Callable, Iterator
from functools import partial
from typing import Protocol

import numpy as np
import pyarrow as pa
from finestore.layout import BlobTables
from finestore.reader import ReadView
from finestore.store import DataStore
from rigging.filesystem.storage_path import StoragePath
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo
from zephyr.shard_keys import deterministic_hash

from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.curriculum_models import EmbeddingSpec
from taskcompendium.pipeline.semantics import DescriptionStatus

EMBEDDING_INSTRUCTION = (
    "Represent the core knowledge and decisive operation required by this work, ignoring story details "
    "and routine setup, capture, formatting or submission unless they are the main problem."
)
# Keep cache routing independent of the execution pool size.
EMBEDDING_CACHE_SHARDS = 1024


def embedding_cache_partition(key: str) -> int:
    return deterministic_hash(key) % EMBEDDING_CACHE_SHARDS


def embedding_cache_path(root: str, partition: int) -> str:
    return str(StoragePath(root) / f"part-{partition:04d}")


class TextEmbedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


def embedding_text(text: str) -> str:
    """Use the same instruction and rendering for both sides of each comparison."""
    return f"Instruct: {EMBEDDING_INSTRUCTION}\nQuery: {text}"


def text_record(text: str, spec: EmbeddingSpec) -> dict:
    rendered = embedding_text(text)
    return {
        "embedding_key": canonical_sha256({"spec": spec.model_dump(mode="json"), "text": rendered}),
        "text": rendered,
    }


def task_text_records(row: dict, spec: EmbeddingSpec) -> Iterator[dict]:
    if row["status"] != DescriptionStatus.READY:
        counters.current_stage().update_counter("embedding/skipped_embeddings", 1)
        return
    yield {"content_hash": row["content_hash"], **text_record(row["description"], spec)}


def normalized_vectors(values: np.ndarray, rows: int, dimensions: int) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.shape != (rows, dimensions) or not np.isfinite(vectors).all():
        raise ValueError(f"Expected {rows} finite vectors of dimension {dimensions}, got {vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding provider returned a zero vector")
    return vectors / norms


def _embed_batches(
    batches: Iterator[list[dict]],
    shard: ShardInfo,
    *,
    spec: EmbeddingSpec,
    embedder_factory: Callable[[], TextEmbedder],
    cache_root: str,
) -> Iterator[dict]:
    cache_root = embedding_cache_path(cache_root, shard.shard_idx)
    view = ReadView(cache_root)
    with DataStore.open(cache_root) as store:
        wrote_vectors = False
        for batch in batches:
            keys = [row["embedding_key"] for row in batch]
            cached = view.read_blobs(keys)
            missing = [row for row in batch if row["embedding_key"] not in cached]
            started = time.monotonic()
            vectors = (
                normalized_vectors(
                    embedder_factory().embed([row["text"] for row in missing]), len(missing), spec.dimensions
                )
                if missing
                else np.empty((0, spec.dimensions), dtype=np.float32)
            )
            counters.current_stage().update_counter("embedding/provider_seconds", time.monotonic() - started)
            counters.current_stage().update_counter(
                "embedding/input_bytes", sum(len(row["text"].encode()) for row in missing)
            )
            if missing:
                wrote_vectors = True
                with store.transaction() as transaction:
                    for row, vector in zip(missing, vectors, strict=True):
                        payload = vector.astype("<f4").tobytes()
                        cached[row["embedding_key"]] = payload
                        transaction.write_object(row["embedding_key"], payload)
            counters.current_stage().update_counter("embedding/cache_hits", len(batch) - len(missing))
            counters.current_stage().update_counter("embedding/cache_misses", len(missing))
            for row in batch:
                vector = np.frombuffer(cached[row["embedding_key"]], dtype="<f4")
                if vector.shape != (spec.dimensions,) or not np.isfinite(vector).all() or not np.any(vector):
                    raise ValueError(f"Invalid cached embedding {row['embedding_key']}")
                yield {"embedding_key": row["embedding_key"], "embedding": vector.tolist()}
        if wrote_vectors:
            store.compact(BlobTables.DESCRIPTORS)


def embed_texts(
    context: ZephyrContext,
    *,
    texts: Dataset,
    spec: EmbeddingSpec,
    embedder_factory: Callable[[], TextEmbedder],
    cache_root: str,
    output_root: str,
    batch_size: int = 256,
) -> list[str]:
    """Return vector Parquet paths for unique text keys, reusing cached vectors."""
    schema = pa.schema([("embedding_key", pa.string()), ("embedding", pa.list_(pa.float32(), spec.dimensions))])
    result = context.execute(
        texts.deduplicate(key=lambda row: row["embedding_key"], num_output_shards=EMBEDDING_CACHE_SHARDS)
        .window(batch_size)
        .map_shard(partial(_embed_batches, spec=spec, embedder_factory=embedder_factory, cache_root=cache_root))
        .write_parquet(str(StoragePath(output_root) / "vectors/part-{shard:05d}.parquet"), schema=schema),
        name="semantic-embeddings",
    )
    (StoragePath(output_root) / "embedding_spec.json").write_text(spec.model_dump_json(indent=2))
    (StoragePath(output_root) / "manifest.json").write_text(
        json.dumps({"paths": result.results, "counters": result.counters}, default=str)
    )
    return result.results
