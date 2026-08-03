# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Map a normalized source to quantized FastTransformer embeddings."""

import logging
import os
from collections.abc import Iterator
from functools import cache, partial
from typing import Any

import pyarrow as pa
from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import datakit_source_key
from marin.execution.artifact import write_artifact
from rigging.filesystem import StoragePath
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.readers import InputFileSpec, load_file
from zephyr.runners import InlineRunner

from experiments.datakit.embeddings.artifact import EmbeddingAttrData, quantize_to_int8
from experiments.datakit.embeddings.fast_transformer.embedder import (
    MANIFEST_FILENAME,
    FastEmbeddingBundleManifest,
    FastEmbeddingModel,
    verified_payload,
)

logger = logging.getLogger(__name__)


def embedding_schema(embedding_dim: int) -> pa.Schema:
    """Return the Parquet schema for one embedding dimension."""
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("embedding", pa.list_(pa.int8(), embedding_dim)),
        ]
    )


@cache
def _load_released_model(bundle_root: str, manifest_sha256: str) -> FastEmbeddingModel:
    return FastEmbeddingModel.load(bundle_root, manifest_sha256)


def embed_records(
    model: FastEmbeddingModel,
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    quantization_scale: float,
) -> list[dict[str, Any]]:
    """Return quantized embeddings while preserving the input IDs and order."""
    if batch_size < 1:
        raise ValueError("The batch size must be positive")
    if quantization_scale <= 0:
        raise ValueError("The quantization scale must be positive")
    if not records:
        return []
    ids = [record["id"] for record in records]
    texts = [record["text"] for record in records]
    vectors = model(texts, batch_size=batch_size)
    quantized = quantize_to_int8(vectors, quantization_scale)
    return [{"id": document_id, "embedding": quantized[index].tolist()} for index, document_id in enumerate(ids)]


def _embed_shard(
    batches: Iterator[list[dict[str, Any]]],
    shard: ShardInfo,
    *,
    bundle_root: str,
    manifest_sha256: str,
    batch_size: int,
    quantization_scale: float,
) -> Iterator[dict[str, Any]]:
    model = _load_released_model(bundle_root, manifest_sha256)
    document_count = 0
    byte_count = 0
    for batch in batches:
        document_count += len(batch)
        byte_count += sum(len(record["text"]) for record in batch)
        yield from embed_records(
            model,
            batch,
            batch_size=batch_size,
            quantization_scale=quantization_scale,
        )
    counters.pipeline.update_counter("embed/docs_in", document_count)
    counters.pipeline.update_counter("embed/bytes_in", byte_count)
    counters.pipeline.update_counter("embed/shards_in", 1)
    logger.info(
        "shard %d/%d: %d documents (%.1f MiB) encoded",
        shard.shard_idx,
        shard.total_shards,
        document_count,
        byte_count / 1024 / 1024,
    )


def embed_source(
    output_path: str,
    normalized: NormalizedData,
    *,
    bundle_root: str,
    manifest_sha256: str,
    batch_size: int,
    max_shards: int | None = None,
    worker_resources: ResourceConfig | None = None,
    max_workers: int = 128,
) -> EmbeddingAttrData:
    """Embed each source shard with one released FastTransformer bundle."""
    if batch_size < 1:
        raise ValueError("The batch size must be positive")
    if max_workers < 1:
        raise ValueError("The worker count must be positive")
    manifest_payload = verified_payload(StoragePath(bundle_root), MANIFEST_FILENAME, manifest_sha256)
    manifest = FastEmbeddingBundleManifest.model_validate_json(manifest_payload)

    source_shards = sorted(
        str(path) for path in StoragePath(f"{normalized.main_output_dir.rstrip('/')}/**/*.parquet").glob()
    )
    if max_shards is not None:
        source_shards = source_shards[:max_shards]
    if not source_shards:
        raise RuntimeError(f"No source Parquet shards under {normalized.main_output_dir}")

    output_basenames = tuple(os.path.basename(path) for path in source_shards)

    def _output_path(shard_index: int, _total: int, basenames: tuple[str, ...] = output_basenames) -> str:
        return f"{output_path.rstrip('/')}/{basenames[shard_index]}"

    source_specs = [InputFileSpec(path=path, columns=["id", "text"]) for path in source_shards]
    quantization_scale = manifest.quantization_scale
    dataset = (
        Dataset.from_list(source_specs)
        .flat_map(load_file)
        .window(batch_size)
        .map_shard(
            partial(
                _embed_shard,
                bundle_root=bundle_root,
                manifest_sha256=manifest_sha256,
                batch_size=batch_size,
                quantization_scale=quantization_scale,
            )
        )
        .write_parquet(
            _output_path,
            schema=embedding_schema(manifest.output_dimension),
            skip_existing=True,
        )
    )

    if worker_resources is None:
        worker_resources = ResourceConfig(cpu=8, ram="16g")
    context = ZephyrContext(
        resources=worker_resources,
        max_workers=min(max_workers, len(source_shards)),
        name=f"embed-fast-transformer-{os.path.basename(normalized.main_output_dir)[:8]}",
        stage_runner_factory=InlineRunner,
    )
    outcome = context.execute(dataset, verbose=True)

    artifact = EmbeddingAttrData(
        output_dir=output_path,
        source_key=datakit_source_key(normalized.main_output_dir),
        model_name=f"FastTransformer/{manifest.tokenizer_name}",
        model_revision=manifest_sha256,
        embedding_dim=manifest.output_dimension,
        quantization_scale=quantization_scale,
        quantization_range=manifest.quantization_range,
        batch_size=batch_size,
        counters=dict(outcome.counters),
    )
    write_artifact(artifact, output_path)
    return artifact
