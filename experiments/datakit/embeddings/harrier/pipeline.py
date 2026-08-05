# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Map-only Zephyr Harrier pipeline: NormalizedData -> co-partitioned int8 embedding parquet.

Each source parquet shard becomes one Zephyr task producing one output
parquet shard with the same basename. Schema:

    id         string
    embedding  list<int8> length 1024  (Harrier quantized symmetrically)

Storage: ~1 byte per embedded value (no Parquet dictionary needed because we
write int8 directly). Consumers ``pq.read_table`` then dequantize with
``dequantize_to_fp32`` (``int8.astype(float32) * scale``).

Model staging: the driver downloads the pinned HuggingFace snapshot once,
uploads one archive to region-local TTL storage, and broadcasts only its URL.
Each H100 worker downloads and loads the model once, cached across shards by
``InlineRunner``. The driver job must enable ``marin-core:gpu`` so nested
Zephyr workers inherit the CUDA PyTorch environment.

Counters emitted: ``embed/docs_in``, ``embed/bytes_in``, ``embed/shards_in``.
"""

import logging
import os
import tarfile
import tempfile
from collections.abc import Iterator
from functools import cache
from pathlib import Path

import numpy as np
import pyarrow as pa
from fray.types import ResourceConfig
from huggingface_hub import snapshot_download
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import DatakitArtifactPath, datakit_source_key
from marin.execution.artifact import write_artifact
from pydantic import BaseModel
from rigging.filesystem import StoragePath, marin_temp_bucket
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.readers import InputFileSpec, load_file
from zephyr.runners import InlineRunner
from zephyr.worker_context import zephyr_worker_ctx

logger = logging.getLogger(__name__)
EMBEDDING_ATTR_DATA_VERSION = 2

HARRIER_REPO = "microsoft/harrier-oss-v1-0.6b"
HARRIER_REVISION = "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee"
HARRIER_DIM = 1_024
HARRIER_MAX_TOKENS = 8_192
HARRIER_TOKENIZE_BATCH_SIZE = 128
HARRIER_MAX_RAW_TEXT_CHARS = 1_048_576
HARRIER_MAX_INFERENCE_BATCH_TOKENS = 32_768
HARRIER_MAX_INFERENCE_BATCH_SIZE = 64
HARRIER_TARGET_CLUSTER = "cw-us-east-02a"
HARRIER_MAX_WORKERS = 256

# Records held by each Zephyr window. GPU batches are formed separately by
# padded-token budget inside :class:`_HarrierEmbedder`.
DEFAULT_BATCH_SIZE = 4096

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
    """Co-partitioned per-source embedding parquet shards.

    Mirrors :class:`~marin.processing.tokenize.attributes.TokenizedAttrData`.
    One output parquet shard per source shard, sharing basename and row order
    (sort-by-id invariant carries through). Persisted as the step's ``.artifact``.
    Load via ``read_artifact(step.output_path, EmbeddingAttrData)``.

    Attributes:
        output_dir: Directory containing the per-shard parquet outputs.
        source_key: Prefix-relative identity of the ``NormalizedData.main_output_dir`` this mirrors.
            Co-partitioning means consumers can join ``(basename, row_idx)``
            without an id index.
        model_name: HuggingFace model id.
        embedding_dim: Vector dimension (1024 for Harrier 0.6B).
        quantization_scale: ``fp32 = int8.astype(float32) * scale``.
        quantization_range: Original envelope before quantization.
        batch_size: Encode batch size used (informational; recoverable
            from logs, but recorded for reproducibility).
        counters: Aggregated zephyr counters from the embed pipeline.
    """

    version: str = f"v{EMBEDDING_ATTR_DATA_VERSION}"
    output_dir: DatakitArtifactPath
    source_key: str
    model_name: str
    model_revision: str = ""
    embedding_dim: int
    quantization_scale: float
    quantization_range: float
    batch_size: int
    counters: dict[str, int | float] = {}

    def shard_paths(self) -> list[str]:
        return sorted(str(m) for m in StoragePath(f"{self.output_dir.rstrip('/')}/*.parquet").glob())


def quantize_to_int8(arr: np.ndarray) -> np.ndarray:
    """Quantize fp32 to int8 with ``QUANT_SCALE`` (255 symmetric levels in [-0.3, 0.3])."""
    return np.clip(np.round(arr / QUANT_SCALE), -127, 127).astype(np.int8)


def dequantize_to_fp32(arr: np.ndarray, scale: float = QUANT_SCALE) -> np.ndarray:
    """Inverse of :func:`quantize_to_int8`. Consumers call this on the loaded int8 column."""
    return arr.astype(np.float32) * scale


_HARRIER_SHARED_KEY = "harrier_model_archive_url"


def _inference_groups(
    lengths: list[int],
    max_batch_tokens: int = HARRIER_MAX_INFERENCE_BATCH_TOKENS,
    max_batch_size: int = HARRIER_MAX_INFERENCE_BATCH_SIZE,
) -> list[list[int]]:
    """Group sequence indices into length-sorted padded batches."""
    if any(length <= 0 or length > HARRIER_MAX_TOKENS for length in lengths):
        raise ValueError(f"Token lengths must be in [1, {HARRIER_MAX_TOKENS}]")
    groups: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for index in sorted(range(len(lengths)), key=lambda i: (lengths[i], i)):
        next_max = max(current_max, lengths[index])
        if current and (len(current) >= max_batch_size or next_max * (len(current) + 1) > max_batch_tokens):
            groups.append(current)
            current = []
            current_max = 0
        current.append(index)
        current_max = max(current_max, lengths[index])
    if current:
        groups.append(current)
    return groups


class _HarrierEmbedder:
    def __init__(self, model_path: Path) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast  # noqa: PLC0415

        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.torch = torch
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise TypeError(f"Expected a fast tokenizer, got {type(tokenizer).__name__}")
        self.tokenizer = tokenizer
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")
        self.model.eval()
        if int(self.model.config.hidden_size) != HARRIER_DIM:
            raise ValueError(f"Harrier hidden size is {self.model.config.hidden_size}; expected {HARRIER_DIM}")

    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = np.empty((len(texts), HARRIER_DIM), dtype=np.float32)
        for chunk_start in range(0, len(texts), HARRIER_TOKENIZE_BATCH_SIZE):
            chunk = texts[chunk_start : chunk_start + HARRIER_TOKENIZE_BATCH_SIZE]
            tokenized = self.tokenizer(
                chunk,
                add_special_tokens=True,
                padding=False,
                truncation=True,
                max_length=HARRIER_MAX_TOKENS,
                return_attention_mask=False,
            )["input_ids"]
            lengths = [len(input_ids) for input_ids in tokenized]
            for group in _inference_groups(lengths):
                inputs = self.tokenizer.pad(
                    [{"input_ids": tokenized[index]} for index in group],
                    padding=True,
                    return_tensors="pt",
                )
                device_inputs = {name: value.to("cuda") for name, value in inputs.items()}
                with self.torch.inference_mode():
                    output = self.model(**device_inputs, use_cache=False)
                vectors = output.last_hidden_state[:, -1].float()
                if not self.torch.isfinite(vectors).all():
                    raise ValueError(f"Harrier returned non-finite vectors at input row {chunk_start}")
                embeddings[chunk_start + np.asarray(group)] = vectors.cpu().numpy()
        return embeddings


@cache
def _load_embedder_from_shared() -> _HarrierEmbedder:
    archive_url: str = zephyr_worker_ctx().get_shared(_HARRIER_SHARED_KEY)
    local_root = Path(tempfile.mkdtemp(prefix="harrier-"))
    local_archive = local_root / "model.tar"
    StoragePath(archive_url).download_to(local_archive)
    with tarfile.open(local_archive) as archive:
        archive.extractall(local_root, filter="data")
    logger.info("Loading Harrier from %s (staged at %s)", local_root / "model", archive_url)
    return _HarrierEmbedder(local_root / "model")


def _stage_harrier(repo_id: str, revision: str, output_path: str) -> str:
    """Download and archive the pinned model once in the output region."""
    sanitized_repo = repo_id.replace("/", "__")
    staged_url = (
        f"{marin_temp_bucket(ttl_days=1, prefix='harrier-staging', source_prefix=output_path)}"
        f"/{sanitized_repo}/{revision}/model.tar"
    )
    staged = StoragePath(staged_url)
    if staged.exists():
        logger.info("Harrier model already staged at %s (cache hit)", staged_url)
        return staged_url

    logger.info("Fetching Harrier model %s@%s on driver", repo_id, revision)
    with tempfile.TemporaryDirectory() as temporary_directory:
        model_path = Path(temporary_directory) / "model"
        snapshot_download(repo_id=repo_id, revision=revision, local_dir=model_path)
        archive_path = Path(temporary_directory) / "model.tar"
        with tarfile.open(archive_path, "w", dereference=True) as archive:
            archive.add(model_path, arcname="model")
        size_mb = archive_path.stat().st_size / 1e6
        logger.info("Uploading %.1f MB of Harrier weights to %s", size_mb, staged_url)
        staged.parent.mkdirs()
        staged.upload_from(archive_path)
    return staged_url


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize for cosine K-means and the int8 round trip."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _embed_shard(
    batches: Iterator[list[dict]],
    shard: ShardInfo,
) -> Iterator[dict]:
    """Per-shard map: each ``batches`` window is one ``embedder.embed(texts)`` call.

    Yields ``{id, embedding}`` records preserving input order. Emits zephyr
    counters with the totals for the shard.
    """
    embedder = _load_embedder_from_shared()
    n_docs = 0
    n_bytes = 0
    for batch in batches:
        ids = [r["id"] for r in batch]
        texts = [r["text"][:HARRIER_MAX_RAW_TEXT_CHARS] for r in batch]
        n_docs += len(ids)
        n_bytes += sum(len(t) for t in texts)
        raw = embedder.embed(texts)
        raw = _l2_normalize(raw)
        q = quantize_to_int8(raw)
        for i, did in enumerate(ids):
            yield {"id": did, "embedding": q[i].tolist()}
    counters.pipeline.update_counter("embed/docs_in", n_docs)
    counters.pipeline.update_counter("embed/bytes_in", n_bytes)
    counters.pipeline.update_counter("embed/shards_in", 1)
    logger.info(
        "shard %d/%d: %d docs (%.1f MB) encoded",
        shard.shard_idx,
        shard.total_shards,
        n_docs,
        n_bytes / 1024 / 1024,
    )


def embed_source(
    output_path: str,
    normalized: NormalizedData,
    *,
    repo_id: str = HARRIER_REPO,
    revision: str = HARRIER_REVISION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_shards: int | None = None,
    worker_resources: ResourceConfig | None = None,
    max_workers: int = HARRIER_MAX_WORKERS,
) -> EmbeddingAttrData:
    """Map-only Zephyr embed of every shard under ``NormalizedData.main_output_dir``.

    Each Zephyr task reads one source parquet shard, encodes records in
    ``batch_size``-sized windows via Harrier, L2-normalizes, quantizes to int8, and writes one output
    parquet shard with the same basename.
    """
    source_shards = sorted(str(m) for m in StoragePath(f"{normalized.main_output_dir.rstrip('/')}/**/*.parquet").glob())
    if max_shards is not None:
        source_shards = source_shards[:max_shards]
    if not source_shards:
        raise RuntimeError(f"No source parquet shards under {normalized.main_output_dir}")

    output_basenames = tuple(os.path.basename(p) for p in source_shards)

    def _output_path(shard_idx: int, _total: int, bn: tuple[str, ...] = output_basenames) -> str:
        return f"{output_path.rstrip('/')}/{bn[shard_idx]}"

    logger.info(
        "Embedding %d shards from %s with %s (batch=%d)",
        len(source_shards),
        normalized.main_output_dir,
        repo_id,
        batch_size,
    )

    staged_url = _stage_harrier(repo_id, revision, output_path)

    # Project columns at read time — partition_id (~8 B/row) is ~120 GB of
    # extra read at 15 B-doc scale, and we don't need it.
    source_specs = [InputFileSpec(path=p, columns=["id", "text"]) for p in source_shards]

    ds = (
        Dataset.from_list(source_specs)
        .flat_map(load_file)
        .window(batch_size)
        .map_shard(_embed_shard)
        .write_parquet(_output_path, schema=_EMBEDDING_SCHEMA, skip_existing=True)
    )

    if worker_resources is None:
        worker_resources = ResourceConfig.with_gpu(
            "H100",
            count=1,
            cpu=8,
            ram="32g",
            disk="32g",
            target_cluster=HARRIER_TARGET_CLUSTER,
        )

    ctx = ZephyrContext(
        resources=worker_resources,
        max_workers=min(max_workers, HARRIER_MAX_WORKERS, len(source_shards)),
        chunk_storage_prefix=marin_temp_bucket(ttl_days=1, prefix="zephyr", source_prefix=output_path),
        name=f"embed-harrier-{os.path.basename(normalized.main_output_dir)[:8]}",
        stage_runner_factory=InlineRunner,
    )
    ctx.put(_HARRIER_SHARED_KEY, staged_url)
    outcome = ctx.execute(ds, verbose=True)

    artifact = EmbeddingAttrData(
        output_dir=output_path,
        source_key=datakit_source_key(normalized.main_output_dir),
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
