# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared, content-addressed embedding passes for task and curriculum descriptions."""

import json
from collections.abc import Iterator
from functools import partial
from typing import Literal, Protocol

import numpy as np
import pyarrow as pa
from finestore.reader import ReadView
from finestore.store import DataStore
from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.storage_path import StoragePath
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo

from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.semantics import DescriptionStatus, SemanticKey, SemanticProjection

PROJECTION_INSTRUCTIONS = {
    SemanticProjection.SUBJECT_DOMAIN: "Represent the operational knowledge domain required to perform this work.",
    SemanticProjection.TASK_MECHANIC: (
        "Represent the domain-independent transformation or reasoning action required by this work."
    ),
    SemanticProjection.OPERATIONS: "Represent the operations and requested result of this work for capability matching.",
}


class EmbeddingSpec(BaseModel):
    """Numerical and text-rendering identity shared by tasks and curriculum anchors."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    revision: str
    tokenizer_revision: str
    dimensions: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    compute_dtype: Literal["float32", "float16", "bfloat16"]
    pooling: Literal["last"] = "last"
    normalization: Literal["l2"] = "l2"
    storage_dtype: Literal["float32"] = "float32"


class TextEmbedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


def embedding_text(text: str, projection: SemanticProjection) -> str:
    """Use the same instruction and rendering for both sides of each comparison."""
    return f"Instruct: {PROJECTION_INSTRUCTIONS[projection]}\nQuery: {text}"


def text_record(text: str, projection: SemanticProjection, spec: EmbeddingSpec) -> dict:
    rendered = embedding_text(text, projection)
    return {
        "embedding_key": canonical_sha256({"spec": spec.model_dump(mode="json"), "text": rendered}),
        "projection": projection.value,
        "text": rendered,
    }


def task_text_records(row: dict, spec: EmbeddingSpec) -> Iterator[dict]:
    if row["status"] != DescriptionStatus.READY:
        return
    key = SemanticKey.model_validate(row["key"])
    for projection in SemanticProjection:
        yield {"content_hash": row["content_hash"], **text_record(key.text(projection), projection, spec)}


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
    _shard: ShardInfo,
    *,
    spec: EmbeddingSpec,
    embedder: TextEmbedder,
    cache_root: str,
) -> Iterator[dict]:
    view = ReadView(cache_root)
    with DataStore.open(cache_root) as store:
        for batch in batches:
            keys = [row["embedding_key"] for row in batch]
            cached = view.read_blobs(keys)
            missing = [row for row in batch if row["embedding_key"] not in cached]
            vectors = (
                normalized_vectors(embedder.embed([row["text"] for row in missing]), len(missing), spec.dimensions)
                if missing
                else np.empty((0, spec.dimensions), dtype=np.float32)
            )
            if missing:
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


def embed_texts(
    context: ZephyrContext,
    *,
    texts: Dataset,
    spec: EmbeddingSpec,
    embedder: TextEmbedder,
    cache_root: str,
    output_root: str,
    batch_size: int = 128,
) -> list[str]:
    """Embed unique text keys with batched FineStore reads and transactional cache writes."""
    schema = pa.schema([("embedding_key", pa.string()), ("embedding", pa.list_(pa.float32(), spec.dimensions))])
    result = context.execute(
        texts.deduplicate(key=lambda row: row["embedding_key"])
        .window(batch_size)
        .map_shard(partial(_embed_batches, spec=spec, embedder=embedder, cache_root=cache_root))
        .write_parquet(str(StoragePath(output_root) / "vectors/part-{shard:05d}.parquet"), schema=schema),
        name="semantic-embeddings",
    )
    (StoragePath(output_root) / "embedding_spec.json").write_text(spec.model_dump_json(indent=2))
    (StoragePath(output_root) / "manifest.json").write_text(
        json.dumps({"paths": result.results, "counters": result.counters}, default=str)
    )
    return result.results
