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

Model staging: the driver does one ``hf_hub_download`` (~880 MB), uploads
the file to an in-region TTL'd GCS path (``marin_temp_bucket``), and
broadcasts just the resulting URL via ``ZephyrContext.put``. Workers
``get_shared`` the URL (a string), ``fs.get`` the .npz to a per-process
tempfile, and load the :class:`Embedder` from there, cached via
``@cache``. Combined with ``InlineRunner`` this puts the .npz across the
wire exactly once per pipeline execution and reuses it for every shard
-- with none of the 880 MB ever traveling through cloudpickle or sitting
in coordinator RAM.

Counters emitted: ``embed/docs_in``, ``embed/bytes_in``, ``embed/bytes_embedded``,
``embed/docs_truncated``, ``embed/shards_in``.
"""

import logging
import os
import tempfile
from collections.abc import Iterator
from functools import cache
from typing import Any

import numpy as np
import pyarrow as pa
from fray.types import ResourceConfig
from huggingface_hub import hf_hub_download
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import DatakitArtifactPath, datakit_source_key
from marin.execution.artifact import write_artifact
from pydantic import BaseModel
from rigging.filesystem import StoragePath, marin_temp_bucket
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo
from zephyr.readers import InputFileSpec, load_parquet_batch
from zephyr.runners import InlineRunner
from zephyr.worker_context import zephyr_worker_ctx

logger = logging.getLogger(__name__)
EMBEDDING_ATTR_DATA_VERSION = 2

LUXICAL_REPO = "DatologyAI/luxical-one"
LUXICAL_WEIGHTS_FILE = "luxical_one_rc4.npz"
# Immutable HF commit for the weights. Passed to ``hf_hub_download`` so every region
# fetches identical bytes, and folded into the staging path + embed hash so a retag
# re-stages and re-keys rather than silently changing embeddings under a fixed hash.
LUXICAL_REVISION = "474cfeb959dd473b3d1cd61da630f566037e69e2"
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
    Load via ``read_artifact(step.output_path, EmbeddingAttrData)``.

    Attributes:
        output_dir: Directory containing the per-shard parquet outputs.
        source_key: Prefix-relative identity of the ``NormalizedData.main_output_dir`` this mirrors.
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

    version: str = f"v{EMBEDDING_ATTR_DATA_VERSION}"
    output_dir: DatakitArtifactPath
    source_key: str
    model_name: str
    model_revision: str = ""
    embedding_dim: int
    quantization_scale: float
    quantization_range: float
    batch_size: int
    doc_sample_chars: int = 0
    counters: dict[str, int | float] = {}

    def shard_paths(self) -> list[str]:
        return sorted(str(m) for m in StoragePath(f"{self.output_dir.rstrip('/')}/*.parquet").glob())


def quantize_to_int8(arr: np.ndarray) -> np.ndarray:
    """Quantize fp32 to int8 with ``QUANT_SCALE`` (255 symmetric levels in [-0.6, 0.6])."""
    return np.clip(np.round(arr / QUANT_SCALE), -127, 127).astype(np.int8)


def dequantize_to_fp32(arr: np.ndarray, scale: float = QUANT_SCALE) -> np.ndarray:
    """Inverse of :func:`quantize_to_int8`. Consumers call this on the loaded int8 column."""
    return arr.astype(np.float32) * scale


# Zephyr shared-data key under which the driver broadcasts the staged .npz URL.
_LUXICAL_SHARED_KEY = "luxical_npz_url"


# Per-process embedder cache — survives across map_shard calls under
# InlineRunner, so a worker materializes Luxical exactly once regardless
# of how many shards it handles. The URL is fetched once from Zephyr
# shared data (a tiny string), then the worker pulls the .npz from
# in-region GCS into a tempfile and hands the path to ``Embedder.load``
# (the Cython ArrowTokenizer isn't picklable, so we can't broadcast the
# loaded Embedder directly).
@cache
def _load_embedder_from_shared() -> Any:
    """Return the native Luxical Embedder loaded via the driver-staged GCS URL."""
    from luxical.embedder import Embedder  # noqa: PLC0415  # optional dep: luxical

    npz_url: str = zephyr_worker_ctx().get_shared(_LUXICAL_SHARED_KEY)
    fd, local = tempfile.mkstemp(prefix="luxical-", suffix=".npz")
    os.close(fd)
    StoragePath(npz_url).download_to(local)
    logger.info("Loading native Luxical embedder from %s (staged at %s)", local, npz_url)
    return Embedder.load(local)


def _stage_luxical_to_gcs(repo_id: str, weights_filename: str, revision: str) -> str:
    """Download the .npz from HF on the driver and upload it to an in-region TTL'd
    GCS path. Returns the GCS URL workers will read from. Stable per
    ``(region, repo_id, weights_filename, revision)`` so concurrent / consecutive
    pipeline runs share the staged file, and a revision bump stages to a distinct path."""
    sanitized_repo = repo_id.replace("/", "__")
    staged_url = (
        f"{marin_temp_bucket(ttl_days=1, prefix='luxical-staging')}/{sanitized_repo}/{revision}/{weights_filename}"
    )
    staged = StoragePath(staged_url)
    if staged.exists():
        logger.info("Luxical weights already staged at %s (cache hit)", staged_url)
        return staged_url

    logger.info("Fetching luxical weights %s/%s@%s on driver", repo_id, weights_filename, revision)
    npz_local = hf_hub_download(repo_id=repo_id, filename=weights_filename, revision=revision)
    size_mb = os.path.getsize(npz_local) / 1e6
    logger.info("Uploading %.1f MB of luxical weights to %s", size_mb, staged_url)
    staged.parent.mkdirs()
    staged.upload_from(npz_local)
    return staged_url


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize. The native Embedder returns un-normalized vectors;
    we normalize ourselves so the downstream pipeline can treat embeddings as unit-norm
    (cosine K-means, int8 round trip)."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


# Longest document text handed to the embedder. Code sources carry individual documents
# of many megabytes (stack-v3 shards reach 944 MB compressed), and one document that
# large defeats any batch-level bound: it must be embedded in a single call whatever the
# window size is. Truncating caps the per-document cost, so embed memory stops scaling
# with the largest document in the corpus -- and with each document bounded, a whole
# ``batch_size`` window is bounded for free.
#
# Head truncation matches the Harrier embedding pipeline (#7998), which truncates
# documents at 8,192 tokens -- roughly this many characters of code. Keeping both
# pipelines on the same rule means a document's embedded prefix does not depend on which
# embedder read it. 32 KiB is also far more text than a 192-dim lexical embedding can
# distinguish, so a truncated document's vector stays representative.
EMBED_DOC_SAMPLE_CHARS = 32 * 1024


def sample_document(text: str, max_chars: int = EMBED_DOC_SAMPLE_CHARS) -> str:
    """Return at most *max_chars* of *text*, taken from its beginning.

    A document at or below the cap is returned unchanged, which is the overwhelming
    majority of web text.
    """
    return text[:max_chars]


# Target Python-object bytes per read chunk. ``zephyr.readers.load_parquet`` does
# ``yield from batch.to_pylist()``, converting an ENTIRE Parquet row group into Python
# dicts before yielding the first record -- roughly 3-4x the Arrow size. Truncation
# happens after the read, so it cannot bound this: a 944 MB row group still materializes
# in full. Slice the batch first; ``RecordBatch.slice`` is zero-copy, so only the slice
# becomes Python objects.
_READ_CHUNK_TARGET_BYTES = 64 * 1024 * 1024


def rows_per_chunk(num_rows: int, nbytes: int, target_bytes: int = _READ_CHUNK_TARGET_BYTES) -> int:
    """Rows to materialize at once, from a batch's own average row size.

    Megabyte-document sources get small chunks and small-document sources get large
    ones, without either needing to know the other's shape.
    """
    if num_rows <= 0:
        return 1
    avg_row_bytes = max(1, nbytes // num_rows)
    return max(1, target_bytes // avg_row_bytes)


def _load_records_bounded(
    source: InputFileSpec | str,
    doc_sample_chars: int = EMBED_DOC_SAMPLE_CHARS,
) -> Iterator[dict]:
    """Yield Parquet records, truncated, without materializing a whole row group.

    Truncation happens **here** rather than in the map: the map runs downstream of
    ``window``, so truncating there would still let a window retain ``batch_size``
    untruncated documents and scale with raw document size. Truncating at read time
    bounds the reader, the window, and the embedder call at once.

    Same records and order as ``zephyr.readers.load_parquet``, with ``text`` truncated.
    """
    n_bytes_in = 0
    n_bytes_embedded = 0
    n_truncated = 0

    def prepare(records: list[dict]) -> Iterator[dict]:
        nonlocal n_bytes_in, n_bytes_embedded, n_truncated
        for record in records:
            raw = record.get("text") or ""
            kept = sample_document(raw, doc_sample_chars)
            n_bytes_in += len(raw)
            n_bytes_embedded += len(kept)
            n_truncated += len(kept) < len(raw)
            record["text"] = kept
            yield record

    for batch in load_parquet_batch(source):
        if batch.num_rows == 0:
            continue
        per_chunk = rows_per_chunk(batch.num_rows, batch.nbytes, _READ_CHUNK_TARGET_BYTES)
        for offset in range(0, batch.num_rows, per_chunk):
            yield from prepare(batch.slice(offset, per_chunk).to_pylist())

    counters.pipeline.update_counter("embed/bytes_in", n_bytes_in)
    counters.pipeline.update_counter("embed/bytes_embedded", n_bytes_embedded)
    counters.pipeline.update_counter("embed/docs_truncated", n_truncated)


def _embed_shard(
    batches: Iterator[list[dict]],
    shard: ShardInfo,
) -> Iterator[dict]:
    """Per-shard map: each ``batches`` window is one ``embedder(texts)`` call.

    Yields ``{id, embedding}`` records preserving input order. Emits zephyr
    counters with the totals for the shard. ``text`` arrives already truncated from
    :func:`_load_records_bounded`, so a window cannot retain oversized documents.
    """
    embedder = _load_embedder_from_shared()
    n_docs = 0
    for batch in batches:
        ids = [r["id"] for r in batch]
        texts = [r["text"] for r in batch]
        n_docs += len(ids)
        raw = np.asarray(embedder(texts, progress_bars=False), dtype=np.float32)
        raw = _l2_normalize(raw)
        q = quantize_to_int8(raw)
        for i, did in enumerate(ids):
            yield {"id": did, "embedding": q[i].tolist()}
    counters.pipeline.update_counter("embed/docs_in", n_docs)
    counters.pipeline.update_counter("embed/shards_in", 1)
    logger.info("shard %d/%d: %d docs encoded", shard.shard_idx, shard.total_shards, n_docs)


def embed_source(
    output_path: str,
    normalized: NormalizedData,
    *,
    repo_id: str = LUXICAL_REPO,
    weights_filename: str = LUXICAL_WEIGHTS_FILE,
    revision: str = LUXICAL_REVISION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    doc_sample_chars: int = EMBED_DOC_SAMPLE_CHARS,
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
    source_shards = sorted(str(m) for m in StoragePath(f"{normalized.main_output_dir.rstrip('/')}/**/*.parquet").glob())
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

    # One HF Hub download + one in-region GCS upload on the driver. Workers
    # will pull the file via fs.get; we only broadcast the URL through
    # Zephyr shared data, so no 880 MB ever touches coordinator RAM or
    # cloudpickle.
    staged_url = _stage_luxical_to_gcs(repo_id, weights_filename, revision)

    # Project columns at read time — partition_id (~8 B/row) is ~120 GB of
    # extra read at 15 B-doc scale, and we don't need it.
    source_specs = [InputFileSpec(path=p, columns=["id", "text"]) for p in source_shards]

    ds = (
        Dataset.from_list(source_specs)
        .flat_map(lambda spec, dsc=doc_sample_chars: _load_records_bounded(spec, dsc))
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
    ctx.put(_LUXICAL_SHARED_KEY, staged_url)
    outcome = ctx.execute(ds, verbose=True)

    artifact = EmbeddingAttrData(
        output_dir=output_path,
        source_key=datakit_source_key(normalized.main_output_dir),
        model_name=f"{repo_id}/{weights_filename}",
        model_revision=revision,
        embedding_dim=LUXICAL_DIM,
        quantization_scale=QUANT_SCALE,
        quantization_range=QUANT_RANGE,
        batch_size=batch_size,
        doc_sample_chars=doc_sample_chars,
        counters=dict(outcome.counters),
    )
    write_artifact(artifact, output_path)
    return artifact
