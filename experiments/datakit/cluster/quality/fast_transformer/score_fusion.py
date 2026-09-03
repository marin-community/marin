# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score normalized documents with the fusion quality scorer.

The fusion scorer reads a document two ways: its first ``max_tokens`` ids under the
corpus tokenizer, and its int8[1024] Harrier embedding. Both inputs are leaves of one
normalized source that share shard count and basenames, so one Zephyr task scores one
shard pair and writes one output shard under the same basename, one row per
normalized document in the normalized shard's row order. That order is what the
store walks positionally against decon and tokenize.

Tokenization runs inside the task through the tokenize stage's own core
(:func:`marin.processing.tokenize._core.tokenize_batches_with_id` with the text
format), so the ids equal those a tokenize leaf of this source would hold under the
same tokenizer. Text is capped at :data:`TEXT_CHAR_CAP` characters first: the scorer
reads ``max_tokens`` tokens, and the cap changes them only for a document whose first
65,536 characters tokenize to fewer.

The embedding side is matched on ``id``, since one Harrier leaf is a repartition whose
row order does not follow the normalized shard. A document without an embedding and an
embedding no document claims both fail the shard: either means the two leaves did not
come from one normalize run.

Every worker holds one scorer per process (``InlineRunner``) and runs
:data:`TASK_RESOURCES`-sized tasks concurrently in threads. Tokenization and parquet
decode release the GIL, so the concurrent tasks are what keep a worker's cores and its
accelerator busy; the forward is a small fraction of a task's time.
"""

import functools
import itertools
import logging
import os
import threading
from collections.abc import Iterator
from functools import partial

import numpy as np
import pyarrow as pa
from fray.types import ResourceConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.tokenizers import TokenizerBackend
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import DatakitArtifactPath, datakit_source_key
from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize._core import CHUNK_INDEX_FIELD, INPUT_IDS_FIELD, tokenize_batches_with_id
from pydantic import BaseModel
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo
from zephyr.runners import InlineRunner

from experiments.datakit.cluster.quality.fast_transformer.data import NUM_RESERVED, PAD_ID, UNK_ID
from experiments.datakit.cluster.quality.fast_transformer.inference import predict
from experiments.datakit.cluster.quality.fast_transformer.keyed_rows import KeyedRows, read_keyed_rows
from experiments.datakit.cluster.quality.fast_transformer.quality_model import (
    QualityPin,
    quality_model_dir,
    require_pinned_model,
)
from experiments.datakit.cluster.quality.fast_transformer.scorer import PooledScorer, load_pooled_scorer

logger = logging.getLogger(__name__)

FUSION_SCORES_VERSION = 1
# Documents per tokenize call and per forward. Padded to a constant shape so the
# forward compiles once; the largest shard holds 2.68M documents, so a batch
# bounds the resident tokens and embeddings at ~6 KB a document.
BATCH_DOCS = 4096
# 128 characters per token over the scorer's 512-token window.
TEXT_CHAR_CAP = 65_536
TEXT_FORMAT = TextLmDatasetFormat()
# One H100 node has 8 GPUs and 128 vCPUs. Sixteen concurrent tasks per worker keep
# its cores tokenizing while the forwards share one device.
WORKER_RESOURCES = ResourceConfig.with_gpu("H100", count=1, cpu=16, ram="96g", disk="64g")
TASK_RESOURCES = ResourceConfig.with_gpu("H100", count=1, cpu=1, ram="6g", disk="64g")
COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="8g", preemptible=False)
MAX_WORKERS = 256

SCORE_SCHEMA = pa.schema([pa.field("id", pa.string()), pa.field("score", pa.float32())])


class FusionScores(BaseModel):
    """Co-partitioned per-source raw fusion scores.

    ``output_dir`` holds one parquet shard per normalized shard, same basename, with
    ``id`` and ``score`` -- the scorer's sigmoid, uncalibrated -- one row per
    normalized document in the normalized shard's row order.
    """

    version: str = f"v{FUSION_SCORES_VERSION}"
    output_dir: DatakitArtifactPath
    source_key: str
    embedding_dir: DatakitArtifactPath
    model: str
    model_sha256: str
    tokenizer: str
    counters: dict[str, int | float]

    def shard_paths(self) -> list[str]:
        return sorted(str(path) for path in (StoragePath(self.output_dir) / "*.parquet").glob())


def fusion_hash_attrs(pin: QualityPin) -> dict[str, str | int]:
    """The identity of a fusion score step, shared by its producer and its consumers."""
    return {
        "model": pin.name,
        "model_sha256": pin.model_sha256,
        "tokenizer": pin.tokenizer,
        "text_char_cap": TEXT_CHAR_CAP,
        "v": FUSION_SCORES_VERSION,
    }


def paired_basenames(*dirs: str) -> list[str]:
    """The parquet basenames every directory holds, refusing any asymmetry.

    Co-partitioned leaves of one source share their complete basename sets. A leaf
    that carries a basename another lacks came from a different normalize run, and
    its documents would otherwise leave no trace in the output.
    """
    sets = {d: {os.path.basename(str(p)) for p in (StoragePath(d) / "*.parquet").glob()} for d in dirs}
    first_dir, first = next(iter(sets.items()))
    if not first:
        raise FileNotFoundError(f"no parquet shards under {first_dir}")
    for other_dir, other in sets.items():
        if other != first:
            missing = sorted(first - other)[:3]
            extra = sorted(other - first)[:3]
            raise ValueError(
                f"{other_dir} is not co-partitioned with {first_dir}: {len(first - other)} basenames missing "
                f"(e.g. {missing}) and {len(other - first)} unexpected (e.g. {extra})"
            )
    return sorted(first)


def verify_remap(remap: dict[int, int]) -> int:
    """Assert the remap is the full-vocab identity offset and return the vocab size.

    The fusion checkpoint maps every raw tokenizer id to ``id + NUM_RESERVED``, and
    scoring exploits that: the remap becomes an add rather than a per-token dict
    lookup. A checkpoint shipping a compacted remap fails here instead of silently
    scoring scrambled ids.
    """
    size = len(remap)
    wrong = [t for t in range(size) if remap.get(t) != t + NUM_RESERVED]
    if wrong:
        raise ValueError(
            f"remap is not the full-vocab identity offset ({len(wrong)} of {size} entries differ, "
            f"e.g. {wrong[:5]}); the fusion scorer assumes raw_id + {NUM_RESERVED}"
        )
    return size


_SCORER_LOCK = threading.Lock()


@functools.cache
def _load_pinned_scorer(model_dir: str, pin: QualityPin) -> PooledScorer:
    require_pinned_model(pin, model_dir)
    scorer = load_pooled_scorer(model_dir)
    verify_remap(scorer.remap)
    logger.info("loaded %s (%s): max_tokens=%d", pin.name, model_dir, scorer.max_tokens)
    return scorer


def pinned_scorer(model_dir: str, pin: QualityPin) -> PooledScorer:
    """The process's one scorer, digest-checked and loaded on first use.

    Concurrent tasks in a worker all miss the cache at start; the lock makes them
    load the 158 MB checkpoint once instead of once each.
    """
    with _SCORER_LOCK:
        return _load_pinned_scorer(model_dir, pin)


def pad_ids(rows: list[list[int]], max_tokens: int, vocab_size: int) -> np.ndarray:
    """Dense ``[n, max_tokens]`` compact ids: the first ``max_tokens`` of each row, remapped.

    The remap is ``raw + NUM_RESERVED`` (see :func:`verify_remap`). An id at or above
    the vocab becomes ``UNK_ID``: a jax gather clamps out-of-range indices rather than
    raising, which would score the row against an unrelated embedding row.
    """
    n = len(rows)
    lengths = np.fromiter((min(len(row), max_tokens) for row in rows), dtype=np.int64, count=n)
    flat = np.fromiter(
        itertools.chain.from_iterable(row[:max_tokens] for row in rows), dtype=np.int64, count=int(lengths.sum())
    )
    compact = np.where(flat < vocab_size - NUM_RESERVED, flat + NUM_RESERVED, UNK_ID).astype(np.int32)
    out = np.full((n, max_tokens), PAD_ID, dtype=np.int32)
    out[np.arange(max_tokens)[None, :] < lengths[:, None]] = compact
    return out


def normalize_embeddings(rows: np.ndarray) -> np.ndarray:
    """int8[1024] rows -> float32, L2-normalized, exactly as training fed them.

    A uniform quantization scale cancels under L2 normalization, so the int8 values
    are normalized directly rather than dequantized first.
    """
    x = rows.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)


def first_chunk_ids(ids: np.ndarray, texts: list[str]) -> list[list[int]]:
    """Tokenize documents through the tokenize stage's core; return each one's first chunk."""
    records = [{"id": doc_id, "text": text[:TEXT_CHAR_CAP]} for doc_id, text in zip(ids, texts, strict=True)]
    rows = [
        row[INPUT_IDS_FIELD]
        for row in tokenize_batches_with_id(data_format=TEXT_FORMAT, batches=iter([records]))
        if row[CHUNK_INDEX_FIELD] == 0
    ]
    if len(rows) != len(records):
        raise ValueError(f"tokenizer returned {len(rows)} documents for {len(records)}")
    return rows


def rebatch(batches: Iterator[pa.RecordBatch], rows_per_batch: int) -> Iterator[pa.RecordBatch]:
    """Regroup record batches into batches of exactly ``rows_per_batch`` rows, plus a tail."""
    pending: list[pa.RecordBatch] = []
    rows = 0
    for batch in batches:
        pending.append(batch)
        rows += batch.num_rows
        if rows < rows_per_batch:
            continue
        table = pa.Table.from_batches(pending)
        full = rows - rows % rows_per_batch
        for start in range(0, full, rows_per_batch):
            yield pa.concat_batches(table.slice(start, rows_per_batch).to_batches())
        pending = table.slice(full).to_batches()
        rows -= full
    if rows:
        yield pa.concat_batches(pa.Table.from_batches(pending).to_batches())


def _score_shard(
    batches: Iterator[pa.RecordBatch],
    shard: ShardInfo,
    *,
    embedding_paths: tuple[str, ...],
    model_dir: str,
    pin: QualityPin,
    batch_docs: int,
) -> Iterator[pa.RecordBatch]:
    """Score one normalized shard against its embedding shard, in the normalized order."""
    scorer = pinned_scorer(model_dir, pin)
    vocab_size = scorer.model.config.vocab_size
    embedding_path = embedding_paths[shard.shard_idx]
    where = f"shard {shard.shard_idx} ({embedding_path})"
    embeddings: KeyedRows = read_keyed_rows(embedding_path, "embedding")
    claimed = np.zeros(len(embeddings), dtype=bool)
    documents = 0
    for batch in rebatch(batches, batch_docs):
        ids = batch.column("id").to_numpy(zero_copy_only=False)
        tokens = pad_ids(first_chunk_ids(ids, batch.column("text").to_pylist()), scorer.max_tokens, vocab_size)
        embedding = normalize_embeddings(embeddings.values[embeddings.rows_for(ids, claimed, where)])
        scores = predict(scorer.model, tokens, batch_size=batch_docs, doc_embed=embedding)
        documents += len(ids)
        yield pa.RecordBatch.from_arrays([batch.column("id"), pa.array(scores, type=pa.float32())], schema=SCORE_SCHEMA)
    embeddings.require_all_claimed(claimed, documents, where)
    counters.pipeline.update_counter("fusion/docs_scored", documents)
    counters.pipeline.update_counter("fusion/shards", 1)
    logger.info("shard %d/%d: %d documents scored", shard.shard_idx, shard.total_shards, documents)


def score_fusion(
    output_path: str,
    *,
    normalized: NormalizedData,
    embedding_dir: str,
    quality_model: QualityPin,
    batch_docs: int = BATCH_DOCS,
    worker_resources: ResourceConfig = WORKER_RESOURCES,
    task_resources: ResourceConfig = TASK_RESOURCES,
    max_workers: int = MAX_WORKERS,
    zephyr_context: ZephyrContext | None = None,
) -> FusionScores:
    """Score every shard of one normalized source; one Zephyr task per shard pair.

    Output shards that already exist are skipped, so a rerun after a partial failure
    scores only the remainder.
    """
    model_dir = quality_model_dir(quality_model)
    text_dir = normalized.main_output_dir
    basenames = tuple(paired_basenames(text_dir, embedding_dir))
    embedding_paths = tuple(prefix_join(embedding_dir, name) for name in basenames)

    def _output_path(shard_idx: int, _total: int, names: tuple[str, ...] = basenames) -> str:
        return prefix_join(output_path, names[shard_idx])

    logger.info("scoring %d shards of %s against %s -> %s", len(basenames), text_dir, embedding_dir, output_path)
    pipeline = (
        Dataset.from_list([prefix_join(text_dir, name) for name in basenames])
        .load_parquet(columns=["id", "text"], batch_mode=True)
        .map_shard(
            partial(
                _score_shard,
                embedding_paths=embedding_paths,
                model_dir=model_dir,
                pin=quality_model,
                batch_docs=batch_docs,
            )
        )
        .write_parquet(_output_path, schema=SCORE_SCHEMA, skip_existing=True)
    )
    ctx = zephyr_context or ZephyrContext(
        name=f"fusion-{os.path.basename(text_dir.rstrip('/'))[:8]}",
        resources=worker_resources,
        coordinator_resources=COORDINATOR_RESOURCES,
        max_workers=min(max_workers, len(basenames)),
        chunk_storage_prefix=marin_temp_bucket(ttl_days=1, prefix="zephyr", source_prefix=output_path),
        stage_runner_factory=InlineRunner,
    )
    ctx.put("tokenizer_name", quality_model.tokenizer)
    ctx.put("tokenizer_backend", TokenizerBackend.HF)
    outcome = ctx.execute(pipeline, verbose=True, map_task_resources=task_resources)
    return FusionScores(
        output_dir=output_path,
        source_key=datakit_source_key(text_dir),
        embedding_dir=embedding_dir,
        model=quality_model.name,
        model_sha256=quality_model.model_sha256,
        tokenizer=quality_model.tokenizer,
        counters=dict(outcome.counters),
    )


def fusion_score_step(
    *,
    name: str,
    normalized: StepSpec,
    embedding: StepSpec,
    quality_model: QualityPin,
    batch_docs: int = BATCH_DOCS,
    worker_resources: ResourceConfig = WORKER_RESOURCES,
    task_resources: ResourceConfig = TASK_RESOURCES,
    max_workers: int = MAX_WORKERS,
    zephyr_context: ZephyrContext | None = None,
) -> StepSpec:
    """A step that scores ``normalized`` against ``embedding`` with ``quality_model``.

    ``embedding`` is the Harrier leaf of the same source; its ``output_path`` is the
    shard directory. The model bytes enter the identity through the pin's digest
    and are checked again by every worker before it writes.
    """
    return StepSpec(
        name=name,
        deps=[normalized, embedding],
        hash_attrs=fusion_hash_attrs(quality_model),
        fn=lambda output_path: score_fusion(
            output_path,
            normalized=read_artifact(normalized.output_path, NormalizedData),
            embedding_dir=embedding.output_path,
            quality_model=quality_model,
            batch_docs=batch_docs,
            worker_resources=worker_resources,
            task_resources=task_resources,
            max_workers=max_workers,
            zephyr_context=zephyr_context,
        ),
    )
