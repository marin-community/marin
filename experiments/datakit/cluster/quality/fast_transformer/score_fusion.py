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

The embedding side is read positionally: every leaf of a source holds the normalized
shard's documents in its row order. :class:`shards.AlignedColumn` checks each batch's ids
against the embedding shard's as it takes them, so an embedding leaf from a different
normalize run fails the shard instead of pairing documents with the wrong vectors.

Every worker holds one scorer per process (``InlineRunner``) and runs
:data:`POOL`'s task-sized tasks concurrently in threads. Tokenization and parquet
decode release the GIL, so the concurrent tasks are what keep a worker's cores and its
accelerator busy; the forward is a small fraction of a task's time.
"""

import functools
import itertools
import logging
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
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import ShardInfo
from zephyr.runners import InlineRunner

from experiments.datakit.cluster.quality.fast_transformer.data import NUM_RESERVED, PAD_ID, UNK_ID
from experiments.datakit.cluster.quality.fast_transformer.inference import predict
from experiments.datakit.cluster.quality.fast_transformer.quality_model import (
    QualityPin,
    quality_model_dir,
    require_pinned_model,
)
from experiments.datakit.cluster.quality.fast_transformer.scorer import PooledScorer, load_pooled_scorer
from experiments.datakit.cluster.quality.fast_transformer.shards import (
    ShardPool,
    map_normalized_shards,
    read_aligned_column,
    rebatch,
)

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
POOL = ShardPool(
    worker=ResourceConfig.with_gpu("H100", count=1, cpu=16, ram="96g", disk="64g"),
    task=ResourceConfig.with_gpu("H100", count=1, cpu=1, ram="6g", disk="64g"),
    max_workers=256,
)

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


def fusion_hash_attrs(pin: QualityPin) -> dict[str, str | int]:
    """The identity of a fusion score step, shared by its producer and its consumers."""
    return {
        "model": pin.name,
        "model_sha256": pin.model_sha256,
        "tokenizer": pin.tokenizer,
        "text_char_cap": TEXT_CHAR_CAP,
        "v": FUSION_SCORES_VERSION,
    }


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
    """The process's one scorer, digest-checked and loaded on first use."""
    # functools.cache is not atomic: the concurrent tasks of a fresh worker all
    # miss at once and would each load the 158 MB checkpoint without the lock.
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
    """int8[1024] rows -> float32, L2-normalized, exactly as training fed them."""
    # No dequantization step: a uniform scale cancels under L2 normalization, and
    # training normalized the raw int8 values the same way.
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


def _score_shard(
    batches: Iterator[pa.RecordBatch],
    shard: ShardInfo,
    side_paths: tuple[str, ...],
    *,
    model_dir: str,
    pin: QualityPin,
    batch_docs: int,
) -> Iterator[pa.RecordBatch]:
    """Score one normalized shard against its embedding shard, in the normalized order."""
    scorer = pinned_scorer(model_dir, pin)
    vocab_size = scorer.model.config.vocab_size
    (embedding_path,) = side_paths
    where = f"shard {shard.shard_idx} ({embedding_path})"
    embeddings = read_aligned_column(embedding_path, "embedding", where)
    documents = 0
    for batch in rebatch(batches, batch_docs):
        ids = batch.column("id").to_numpy(zero_copy_only=False)
        tokens = pad_ids(first_chunk_ids(ids, batch.column("text").to_pylist()), scorer.max_tokens, vocab_size)
        embedding = normalize_embeddings(embeddings.take(ids))
        scores = predict(scorer.model, tokens, batch_size=batch_docs, doc_embed=embedding)
        documents += len(ids)
        yield pa.RecordBatch.from_arrays([batch.column("id"), pa.array(scores, type=pa.float32())], schema=SCORE_SCHEMA)
    embeddings.require_consumed()
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
    pool: ShardPool = POOL,
    zephyr_context: ZephyrContext | None = None,
) -> FusionScores:
    """Score every shard of one normalized source; one Zephyr task per shard pair.

    Output shards that already exist are skipped, so a rerun after a partial failure
    scores only the remainder.
    """
    model_dir = quality_model_dir(quality_model)
    text_dir = normalized.main_output_dir
    outcome = map_normalized_shards(
        name="fusion",
        text_dir=text_dir,
        side_dirs=[embedding_dir],
        columns=["id", "text"],
        shard_fn=partial(_score_shard, model_dir=model_dir, pin=quality_model, batch_docs=batch_docs),
        output_path=output_path,
        schema=SCORE_SCHEMA,
        pool=pool,
        zephyr_context=zephyr_context,
        # InlineRunner keeps the per-process scorer alive across a worker's tasks.
        stage_runner_factory=InlineRunner,
        shared={"tokenizer_name": quality_model.tokenizer, "tokenizer_backend": TokenizerBackend.HF},
    )
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
    pool: ShardPool = POOL,
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
            pool=pool,
            zephyr_context=zephyr_context,
        ),
    )
