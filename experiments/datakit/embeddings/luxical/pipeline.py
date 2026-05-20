# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Map-only Zephyr embed pipeline: NormalizedData -> co-partitioned int8 embedding parquet.

Each source parquet shard becomes one Zephyr task producing one output
parquet shard with the same basename. Schema:

    id         string
    embedding  list<int8> length 192   (Luxical-One quantized symmetrically
                                        to +/-0.6 via scale 0.6/127)

Storage: ~1 byte per embedded value (no Parquet dictionary needed because we
write int8 directly). Consumers ``pq.read_table`` then dequantize with
``dequantize_to_fp32`` (``int8.astype(float32) * scale``).

Model staging: the driver does one ``hf_hub_download`` (~880 MB) and
stages the raw ``.npz`` bytes via ``ZephyrContext.put`` -- workers then
``get_shared`` them from the (in-region) ``chunk_storage_prefix`` and
materialize an :class:`Embedder` from a per-process tempfile, cached
via ``@cache``. Combined with ``InlineRunner`` this gets the .npz
across the wire exactly once per pipeline execution and reused for
every shard. Without staging the bytes, every worker process would
re-pull from the HF Hub.

Counters emitted: ``embed/docs_in``, ``embed/bytes_in``, ``embed/shards_in``.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterator
from functools import cache
from typing import Any

import numpy as np
import pyarrow as pa
from fray import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import Artifact
from marin.utils import fsspec_glob
from pydantic import BaseModel
from zephyr import Dataset, InputFileSpec, ShardInfo, ZephyrContext, counters, load_file, zephyr_worker_ctx
from zephyr.runners import InlineRunner

logger = logging.getLogger(__name__)

LUXICAL_REPO = "DatologyAI/luxical-one"
LUXICAL_WEIGHTS_FILE = "luxical_one_rc4.npz"
LUXICAL_DIM = 192

# Records per ``embedder(texts)`` call. The native
# ``luxical.embedder.Embedder`` takes the whole list at once (no explicit
# batch_size arg), so this primarily bounds in-flight memory. Benched on
# Iris cpu=8 (see bench_batch_size.py): native throughput climbs from 218
# docs/s at batch=64 to 1516 docs/s at batch=4096 and plateaus there
# (batch=10000 was identical). Memory at batch=4096: ~48 MB of in-flight
# text + BoW state — trivial in a ram=16g worker. Bumping further wastes
# RAM with no throughput gain.
DEFAULT_BATCH_SIZE = 4096

# Quantization envelope. Sweep against real Luxical-One output (10K docs of
# nemotron_cc_v2/high_quality) showed +/-0.6 keeps mean cos sim 0.9998 with
# 0.001% clipping; +/-0.3 dropped cos sim to 0.98. See git history.
QUANT_RANGE = 0.6
QUANT_SCALE: float = QUANT_RANGE / 127  # int8 [-127, 127] -> fp32 [-0.6, 0.6] (255 levels, symmetric)

_EMBEDDING_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("embedding", pa.list_(pa.int8(), LUXICAL_DIM)),
    ]
)


class EmbeddingAttrData(BaseModel):
    """Co-partitioned per-source embedding parquet shards.

    Mirrors :class:`~marin.processing.tokenize.attributes.TokenizedAttrData`.
    One output parquet shard per source shard, sharing basename and row order
    (sort-by-id invariant carries through). Persisted as the step's ``.artifact``.
    Load via ``Artifact.from_path(step, EmbeddingAttrData)``.

    Attributes:
        output_dir: Directory containing the per-shard parquet outputs.
        source_main_dir: ``NormalizedData.main_output_dir`` this mirrors.
            Co-partitioning means consumers can join ``(basename, row_idx)``
            without an id index.
        model_name: HuggingFace model id.
        embedding_dim: Vector dimension (192 for Luxical-One).
        quantization_scale: ``fp32 = int8.astype(float32) * scale``.
        quantization_range: Original envelope before quantization.
        batch_size: Encode batch size used (informational; recoverable
            from logs, but recorded for reproducibility).
        counters: Aggregated zephyr counters from the embed pipeline.
    """

    version: str = "v1"
    output_dir: str
    source_main_dir: str
    model_name: str
    embedding_dim: int
    quantization_scale: float
    quantization_range: float
    batch_size: int
    counters: dict[str, int] = {}

    def shard_paths(self) -> list[str]:
        return sorted(fsspec_glob(f"{self.output_dir.rstrip('/')}/*.parquet"))


def quantize_to_int8(arr: np.ndarray) -> np.ndarray:
    """Quantize fp32 to int8 with ``QUANT_SCALE`` (255 symmetric levels in [-0.6, 0.6])."""
    return np.clip(np.round(arr / QUANT_SCALE), -127, 127).astype(np.int8)


def dequantize_to_fp32(arr: np.ndarray, scale: float = QUANT_SCALE) -> np.ndarray:
    """Inverse of :func:`quantize_to_int8`. Consumers call this on the loaded int8 column."""
    return arr.astype(np.float32) * scale


# Zephyr shared-data key under which the driver stages the .npz bytes.
_LUXICAL_SHARED_KEY = "luxical_npz_bytes"


# Per-process embedder cache — survives across map_shard calls under
# InlineRunner, so a worker materializes Luxical exactly once regardless
# of how many shards it handles. The .npz bytes are fetched once from
# Zephyr shared data (in-region GCS), written to a tempfile, and passed
# to ``Embedder.load`` (which expects a path -- the Cython ArrowTokenizer
# isn't picklable, so we can't ship the loaded Embedder directly).
@cache
def _load_embedder_from_shared() -> Any:
    """Return the native Luxical Embedder loaded from Zephyr-staged ``.npz`` bytes."""
    from luxical.embedder import Embedder

    npz_bytes: bytes = zephyr_worker_ctx().get_shared(_LUXICAL_SHARED_KEY)
    fd, local = tempfile.mkstemp(prefix="luxical-", suffix=".npz")
    os.close(fd)
    with open(local, "wb") as f:
        f.write(npz_bytes)
    logger.info("Loading native Luxical embedder from %s (%.1f MB)", local, len(npz_bytes) / 1e6)
    return Embedder.load(local)


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize. The native Embedder returns un-normalized vectors;
    we normalize ourselves so the downstream pipeline can treat embeddings as unit-norm
    (cosine K-means, int8 round trip)."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _embed_shard(
    batches: Iterator[list[dict]],
    shard: ShardInfo,
) -> Iterator[dict]:
    """Per-shard map: each ``batches`` window is one ``embedder(texts)`` call.

    Yields ``{id, embedding}`` records preserving input order. Emits zephyr
    counters with the totals for the shard.
    """
    embedder = _load_embedder_from_shared()
    n_docs = 0
    n_bytes = 0
    for batch in batches:
        ids = [r["id"] for r in batch]
        texts = [r["text"] for r in batch]
        n_docs += len(ids)
        n_bytes += sum(len(t) for t in texts)
        raw = np.asarray(embedder(texts, progress_bars=False), dtype=np.float32)
        raw = _l2_normalize(raw)
        q = quantize_to_int8(raw)
        for i, did in enumerate(ids):
            yield {"id": did, "embedding": q[i].tolist()}
    counters.increment("embed/docs_in", n_docs)
    counters.increment("embed/bytes_in", n_bytes)
    counters.increment("embed/shards_in", 1)
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
    repo_id: str = LUXICAL_REPO,
    weights_filename: str = LUXICAL_WEIGHTS_FILE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_shards: int | None = None,
    worker_resources: ResourceConfig | None = None,
    max_workers: int = 128,
) -> EmbeddingAttrData:
    """Map-only Zephyr embed of every shard under ``NormalizedData.main_output_dir``.

    Each Zephyr task reads one source parquet shard, encodes records in
    ``batch_size``-sized batches via the native :mod:`luxical.embedder`
    Embedder, L2-normalizes, quantizes to int8, and writes one output
    parquet shard with the same basename.
    """
    source_shards = sorted(fsspec_glob(f"{normalized.main_output_dir.rstrip('/')}/**/*.parquet"))
    if max_shards is not None:
        source_shards = source_shards[:max_shards]
    if not source_shards:
        raise RuntimeError(f"No source parquet shards under {normalized.main_output_dir}")

    output_basenames = tuple(os.path.basename(p) for p in source_shards)

    def _output_path(shard_idx: int, _total: int, bn: tuple[str, ...] = output_basenames) -> str:
        return f"{output_path.rstrip('/')}/{bn[shard_idx]}"

    logger.info(
        "Embedding %d shards from %s with %s/%s (batch=%d)",
        len(source_shards),
        normalized.main_output_dir,
        repo_id,
        weights_filename,
        batch_size,
    )

    # One HF Hub download on the driver. Workers will pull the bytes from
    # the in-region chunk_storage_prefix via zephyr_worker_ctx().get_shared.
    from huggingface_hub import hf_hub_download

    logger.info("Fetching luxical weights %s/%s on driver", repo_id, weights_filename)
    npz_path = hf_hub_download(repo_id=repo_id, filename=weights_filename)
    with open(npz_path, "rb") as f:
        npz_bytes = f.read()
    logger.info("Staging %.1f MB of luxical weights as Zephyr shared data", len(npz_bytes) / 1e6)

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
        worker_resources = ResourceConfig(cpu=8, ram="16g")

    ctx = ZephyrContext(
        resources=worker_resources,
        max_workers=min(max_workers, len(source_shards)),
        name=f"embed-luxical-{os.path.basename(normalized.main_output_dir)[:8]}",
        # Override Iris's default SubprocessRunner so the per-process embedder
        # cache actually gets reused across shards (huge win at this load cost).
        stage_runner_factory=InlineRunner,
    )
    ctx.put(_LUXICAL_SHARED_KEY, npz_bytes)
    outcome = ctx.execute(ds, verbose=True)

    artifact = EmbeddingAttrData(
        output_dir=output_path,
        source_main_dir=normalized.main_output_dir,
        model_name=f"{repo_id}/{weights_filename}",
        embedding_dim=LUXICAL_DIM,
        quantization_scale=QUANT_SCALE,
        quantization_range=QUANT_RANGE,
        batch_size=batch_size,
        counters=dict(outcome.counters),
    )
    Artifact.save(artifact, output_path)
    return artifact
