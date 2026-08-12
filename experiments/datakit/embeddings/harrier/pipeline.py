# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Map normalized documents to co-partitioned int8 Harrier embeddings."""

import logging
import os
import shutil
import tarfile
import tempfile
from collections.abc import Iterator
from enum import StrEnum
from functools import cache
from pathlib import Path

import numpy as np
import pyarrow as pa
from fray.types import ResourceConfig
from huggingface_hub import snapshot_download
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import DatakitArtifactPath, datakit_source_key
from marin.execution.artifact import write_artifact
from pydantic import BaseModel, Field
from rigging.filesystem import StoragePath, marin_temp_bucket
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.readers import InputFileSpec, load_file
from zephyr.runners import InlineRunner
from zephyr.worker_context import zephyr_worker_ctx

from experiments.datakit.embeddings.harrier.tei_client import TeiEmbeddingClient

logger = logging.getLogger(__name__)

EMBEDDING_ATTR_DATA_VERSION = 2
HARRIER_REPO = "microsoft/harrier-oss-v1-0.6b"
HARRIER_REVISION = "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee"
HARRIER_DIM = 1_024
HARRIER_MAX_TOKENS = 8_192
HARRIER_MAX_RAW_TEXT_CHARS = 100_000
HARRIER_MAX_WORKERS = 256
HARRIER_STAGING_TTL_DAYS = 14

DEFAULT_BATCH_SIZE = 4_096

_MODEL_ARCHIVE_NAME = "model.tar"
_MODEL_DIRECTORY_NAME = "model"
_TEI_ENDPOINT_SHARED_KEY = "harrier_tei_endpoint_name"

# A sweep over 9,733 completed Harrier embeddings sampled across 100 shards of
# datakit/samples/harrier-50m found mean cosine 0.99976, minimum cosine 0.99909,
# and 0.00012% coordinate clipping at +/-0.3.
QUANT_RANGE = 0.3
QUANT_SCALE: float = QUANT_RANGE / 127

_EMBEDDING_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("embedding", pa.list_(pa.int8(), HARRIER_DIM)),
    ]
)


class EmbeddingAttrData(BaseModel):
    """Co-partitioned per-source embedding shards."""

    version: str = f"v{EMBEDDING_ATTR_DATA_VERSION}"
    output_dir: DatakitArtifactPath
    source_key: str
    dedup_attr_dir: DatakitArtifactPath | None = None
    model_name: str
    model_revision: str = ""
    embedding_dim: int
    quantization_scale: float
    quantization_range: float
    batch_size: int
    counters: dict[str, int | float] = Field(default_factory=dict)

    def shard_paths(self) -> list[str]:
        return sorted(str(path) for path in (StoragePath(self.output_dir) / "*.parquet").glob())


class EmbeddingDocumentSet(StrEnum):
    """Documents that a Harrier embedding job includes."""

    ALL = "all"
    DEDUPLICATED = "deduplicated"
    FUZZY_DUPLICATES = "fuzzy_duplicates"


def quantize_to_int8(arr: np.ndarray) -> np.ndarray:
    """Quantize fp32 values into the calibrated Harrier int8 range."""
    return np.clip(np.round(arr / QUANT_SCALE), -127, 127).astype(np.int8)


def dequantize_to_fp32(arr: np.ndarray, scale: float = QUANT_SCALE) -> np.ndarray:
    """Dequantize Harrier int8 values to fp32."""
    return arr.astype(np.float32) * scale


def stage_harrier(repo_id: str, revision: str, destination_path: str) -> str:
    """Download and archive a pinned model in the output region."""
    archive_url = str(
        StoragePath(
            marin_temp_bucket(
                ttl_days=HARRIER_STAGING_TTL_DAYS,
                prefix="harrier-staging",
                source_prefix=destination_path,
            )
        )
        / repo_id.replace("/", "__")
        / revision
        / _MODEL_ARCHIVE_NAME
    )
    staged_archive = StoragePath(archive_url)
    if staged_archive.exists() and staged_archive.size() > 0:
        logger.info("Harrier model already staged at %s", archive_url)
        return archive_url

    logger.info("Fetching Harrier model %s@%s", repo_id, revision)
    with tempfile.TemporaryDirectory() as temporary_directory:
        model_path = Path(temporary_directory) / _MODEL_DIRECTORY_NAME
        snapshot_download(repo_id=repo_id, revision=revision, local_dir=model_path)
        archive_path = Path(temporary_directory) / _MODEL_ARCHIVE_NAME
        with tarfile.open(archive_path, "w", dereference=True) as archive:
            archive.add(model_path, arcname=_MODEL_DIRECTORY_NAME)

        staged_archive.parent.mkdirs()
        with archive_path.open("rb") as source, staged_archive.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        if staged_archive.size() != archive_path.stat().st_size:
            raise ValueError(f"Staged Harrier archive at {archive_url} has the wrong size")
    return archive_url


@cache
def _load_tei_embedder(endpoint_name: str) -> TeiEmbeddingClient:
    return TeiEmbeddingClient(endpoint_name, HARRIER_DIM)


def _embed_shard(batches: Iterator[list[dict]], shard: ShardInfo) -> Iterator[dict]:
    endpoint_name: str = zephyr_worker_ctx().get_shared(_TEI_ENDPOINT_SHARED_KEY)
    embedder = _load_tei_embedder(endpoint_name)
    document_count = 0
    byte_count = 0
    for batch in batches:
        ids = [record["id"] for record in batch]
        texts = [record["text"][:HARRIER_MAX_RAW_TEXT_CHARS] for record in batch]
        document_count += len(ids)
        byte_count += sum(len(text.encode()) for text in texts)
        embeddings = embedder.embed(texts)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = quantize_to_int8(embeddings / norms)
        for document_id, embedding in zip(ids, embeddings, strict=True):
            yield {"id": document_id, "embedding": embedding.tolist()}

    counters.pipeline.update_counter("embed/docs_in", document_count)
    counters.pipeline.update_counter("embed/bytes_in", byte_count)
    counters.pipeline.update_counter("embed/shards_in", 1)
    logger.info(
        "shard %d/%d: %d docs (%.1f MB) encoded",
        shard.shard_idx,
        shard.total_shards,
        document_count,
        byte_count / 1024 / 1024,
    )


def select_document(document: dict, dedup: dict | None, document_set: EmbeddingDocumentSet) -> dict | None:
    """Return a document when it belongs to the selected document set."""
    if document_set == EmbeddingDocumentSet.ALL:
        return document

    is_fuzzy_duplicate = dedup is not None and not dedup["is_cluster_canonical"]
    if document_set == EmbeddingDocumentSet.FUZZY_DUPLICATES:
        return document if is_fuzzy_duplicate else None
    return None if is_fuzzy_duplicate else document


def _select_deduplicated_document(document: dict, dedup: dict | None) -> dict | None:
    selected = select_document(document, dedup, EmbeddingDocumentSet.DEDUPLICATED)
    if selected is not None:
        return selected
    counters.pipeline.update_counter("embed/docs_dedup_dropped", 1)
    return None


def _select_fuzzy_duplicate(document: dict, dedup: dict | None) -> dict | None:
    return select_document(document, dedup, EmbeddingDocumentSet.FUZZY_DUPLICATES)


def embed_source(
    output_path: str,
    normalized: NormalizedData,
    *,
    endpoint_name: str,
    document_set: EmbeddingDocumentSet,
    repo_id: str = HARRIER_REPO,
    revision: str = HARRIER_REVISION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_shards: int | None = None,
    dedup_attr_dir: str | None = None,
    worker_resources: ResourceConfig | None = None,
    max_workers: int = HARRIER_MAX_WORKERS,
) -> EmbeddingAttrData:
    """Embed the parquet shards in a normalized source."""
    if document_set == EmbeddingDocumentSet.ALL:
        if dedup_attr_dir is not None:
            raise ValueError("dedup_attr_dir must be None when document_set is all")
    elif dedup_attr_dir is None:
        raise ValueError(f"dedup_attr_dir is required when document_set is {document_set.value}")

    source_shards = sorted(str(path) for path in (StoragePath(normalized.main_output_dir) / "**" / "*.parquet").glob())
    if max_shards is not None:
        source_shards = source_shards[:max_shards]
    if not source_shards:
        raise RuntimeError(f"No source parquet shards under {normalized.main_output_dir}")

    output_basenames = tuple(os.path.basename(path) for path in source_shards)

    def _output_path(shard_index: int, _total: int, basenames: tuple[str, ...] = output_basenames) -> str:
        return str(StoragePath(output_path) / basenames[shard_index])

    source_specs = [InputFileSpec(path=path, columns=["id", "text"]) for path in source_shards]
    documents = Dataset.from_list(source_specs).flat_map(load_file)
    if dedup_attr_dir is not None:
        combiner = (
            _select_fuzzy_duplicate
            if document_set == EmbeddingDocumentSet.FUZZY_DUPLICATES
            else _select_deduplicated_document
        )
        dedup_specs = [
            InputFileSpec(
                path=str(StoragePath(dedup_attr_dir) / os.path.basename(path)),
                columns=["id", "is_cluster_canonical"],
            )
            for path in source_shards
        ]
        dedup_attrs = Dataset.from_list(dedup_specs).flat_map(load_file)
        documents = documents.sorted_merge_join(
            dedup_attrs,
            left_key=lambda record: record["id"],
            right_key=lambda record: record["id"],
            combiner=combiner,
            how="left",
        ).filter(lambda record: record is not None)

    dataset = (
        documents.window(batch_size)
        .map_shard(_embed_shard)
        .write_parquet(_output_path, schema=_EMBEDDING_SCHEMA, skip_existing=True)
    )
    resources = worker_resources or ResourceConfig.with_cpu(cpu=1, ram="16g", disk="16g")
    context = ZephyrContext(
        resources=resources,
        coordinator_resources=ResourceConfig(cpu=1, ram="8g", preemptible=False),
        max_workers=min(max_workers, HARRIER_MAX_WORKERS, len(source_shards)),
        chunk_storage_prefix=marin_temp_bucket(ttl_days=1, prefix="zephyr", source_prefix=output_path),
        name=f"embed-harrier-{os.path.basename(normalized.main_output_dir)[:8]}",
        stage_runner_factory=InlineRunner,
    )
    context.put(_TEI_ENDPOINT_SHARED_KEY, endpoint_name)
    outcome = context.execute(dataset, verbose=True)

    artifact = EmbeddingAttrData(
        output_dir=output_path,
        source_key=datakit_source_key(normalized.main_output_dir),
        dedup_attr_dir=dedup_attr_dir,
        model_name=repo_id,
        model_revision=revision,
        embedding_dim=HARRIER_DIM,
        quantization_scale=QUANT_SCALE,
        quantization_range=QUANT_RANGE,
        batch_size=batch_size,
        counters=dict(outcome.counters),
    )
    write_artifact(artifact, output_path)
    return artifact
