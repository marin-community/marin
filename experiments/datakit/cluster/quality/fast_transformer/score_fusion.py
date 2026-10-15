# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score normalized documents with the fusion quality scorer.

The fusion scorer reads a document two ways: its first ``max_tokens`` ids under the
corpus tokenizer, and its int8[1024] Harrier embedding. Both come from leaves of one
normalized source that share its shard count, basenames and row order: the tokenize
stage's attribute shards hold the ids and the Harrier leaf the embeddings. One Zephyr
task walks one normalized shard's ids, takes the token and embedding rows of the two
side shards in step with them, and writes one output shard under the same basename,
one row per normalized document in the normalized shard's row order. That order is
what the store walks positionally against decon and tokenize.

No tokenizer runs on the worker. The ids are the tokenize leaf's stored
``input_ids`` (:func:`marin.processing.tokenize.attributes.iter_tokenized_documents`
regroups a document's chunk rows into one array), of which the scorer reads the
first ``max_tokens``. A tokenize artifact built from another normalize output or
under another tokenizer repository is refused before anything is listed; the
tokenizer commit enters the output identity through the tokenize step the score
step depends on, and the pin names it a second time in the hash.

Both sides are read positionally. :class:`AlignedColumn` and
:class:`AlignedDocuments` check each batch's ids against the rows they hand back,
and a side with rows left over fails the shard, so a leaf from a different
normalize run fails the shard instead of pairing documents with the wrong vectors
or tokens.

Every worker holds one scorer per process (``InlineRunner``) and runs
:data:`TASK_RESOURCES`-sized tasks concurrently in threads. Parquet decode releases
the GIL, so the concurrent tasks are what keep a worker's cores and its accelerator
busy.
"""

import functools
import logging
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial

import numpy as np
import polars as pl
import pyarrow as pa
from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import DatakitArtifactPath, datakit_source_key
from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize.attributes import TokenizedAttrData, iter_tokenized_documents
from pydantic import BaseModel
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo
from zephyr.runners import InlineRunner

from experiments.datakit.cluster.quality.fast_transformer.data import PAD_ID
from experiments.datakit.cluster.quality.fast_transformer.inference import predict
from experiments.datakit.cluster.quality.fast_transformer.quality_model import (
    QualityPin,
    quality_model_dir,
    require_pinned_model,
)
from experiments.datakit.cluster.quality.fast_transformer.scorer import PooledScorer, load_pooled_scorer

logger = logging.getLogger(__name__)

FUSION_SCORES_VERSION = 2
# Documents per forward. Padded to a constant shape so the forward compiles once;
# the largest shard holds 2.68M documents, so a batch bounds the resident tokens
# and embeddings at ~6 KB a document. The batch shape moves the scores (up to
# 1.6e-3 between 4,096 and 32,768), so it is hashed into the step identity rather
# than exposed as a knob.
BATCH_DOCS = 4096
# One H100 node has 8 GPUs and 128 vCPUs. Sixteen concurrent tasks per worker keep
# its cores decoding parquet while the forwards share one device. Sized when the
# task also tokenized the text; the decode-only task leaves the cores lighter.
WORKER_RESOURCES = ResourceConfig.with_gpu("H100", count=1, cpu=16, ram="96g", disk="64g")
TASK_RESOURCES = ResourceConfig.with_gpu("H100", count=1, cpu=1, ram="6g", disk="64g")
COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="8g", preemptible=False)
MAX_WORKERS = 256

SCORE_SCHEMA = pa.schema([pa.field("id", pa.string()), pa.field("score", pa.float32())])


class FusionScores(BaseModel):
    """Co-partitioned per-source raw fusion scores.

    ``output_dir`` holds one parquet shard per normalized shard, same basename, with
    ``id`` and ``score`` -- the scorer's sigmoid, uncalibrated -- one row per
    normalized document in the normalized shard's row order. ``tokenize_dir`` and
    ``embedding_dir`` are the two side leaves the scores were computed from.
    """

    version: str = f"v{FUSION_SCORES_VERSION}"
    output_dir: DatakitArtifactPath
    source_key: str
    tokenize_dir: DatakitArtifactPath
    embedding_dir: DatakitArtifactPath
    model: str
    model_sha256: str
    tokenizer: str
    counters: dict[str, int | float]


def fusion_hash_attrs(pin: QualityPin) -> dict[str, str | int]:
    """The identity of a fusion score step, shared by its producer and its consumers.

    The ids and embeddings it reads enter through the step's dependencies; what is
    hashed here is the scorer applied to them and the batch shape it runs at.
    """
    return {
        "model": pin.name,
        "model_sha256": pin.model_sha256,
        "tokenizer": pin.tokenizer,
        "batch_docs": BATCH_DOCS,
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


@dataclass
class AlignedColumn:
    """One co-partitioned shard's column, taken in step with the normalized shard's rows.

    Each :meth:`take` checks the ids of the rows it returns against the documents they
    are paired with, and :meth:`require_consumed` checks that no row was left over, so a
    side written from a different normalize run fails the shard rather than misaligning it.
    """

    ids: np.ndarray
    values: np.ndarray
    where: str
    consumed: int = 0

    def take(self, doc_ids: np.ndarray) -> np.ndarray:
        """Return the values of the next ``len(doc_ids)`` rows, which must carry ``doc_ids``."""
        start, end = self.consumed, self.consumed + len(doc_ids)
        if end > len(self.ids) or not np.array_equal(self.ids[start:end], doc_ids):
            raise ValueError(
                f"{self.where}: rows {start}..{end} do not carry the normalized shard's ids; "
                f"the two sides did not come from one normalize run"
            )
        self.consumed = end
        return self.values[start:end]

    def require_consumed(self) -> None:
        if self.consumed != len(self.ids):
            raise ValueError(
                f"{self.where}: {len(self.ids)} rows against {self.consumed} documents; "
                f"the two sides did not come from one normalize run"
            )


def read_aligned_column(path: str, column: str, where: str) -> AlignedColumn:
    """Read ``id`` and ``column`` from one co-partitioned parquet shard, whole."""
    # polars types a fixed-width list column as an Array with no offsets buffer,
    # so the int32 offset ceiling that fails a whole-column pyarrow read of the
    # largest Harrier shards (2,682,446 documents x 1,024 values > 2^31-1) does
    # not apply, and to_numpy hands back one contiguous [n, width] block.
    with StoragePath(path).open("rb") as fh:
        frame = pl.read_parquet(fh, columns=["id", column])
    return AlignedColumn(frame.get_column("id").to_numpy(), frame.get_column(column).to_numpy(), where)


@dataclass
class AlignedDocuments:
    """One tokenize shard's documents, taken in step with the normalized shard's rows.

    The tokenize leaf is streamed rather than read whole: :meth:`take` pulls the next
    ``len(doc_ids)`` documents and checks each one's id against the normalized row it
    is paired with, returning each document's first ``max_tokens`` ids. A document
    the tokenize side lacks (the stage drops zero-token documents), one it holds in
    another order, or one left over after the normalized rows fails the shard.
    """

    documents: Iterator[tuple[str, np.ndarray]]
    max_tokens: int
    where: str
    consumed: int = 0

    def take(self, doc_ids: np.ndarray) -> list[np.ndarray]:
        """Return the first ``max_tokens`` ids of the next ``len(doc_ids)`` documents, which must carry ``doc_ids``."""
        rows: list[np.ndarray] = []
        for position, expected in enumerate(doc_ids, start=self.consumed):
            document = next(self.documents, None)
            if document is None or document[0] != expected:
                found = None if document is None else document[0]
                raise ValueError(
                    f"{self.where}: row {position}: normalized id {expected!r} but tokenize document {found!r}; "
                    f"the two sides did not come from one normalize run"
                )
            rows.append(document[1][: self.max_tokens])
        self.consumed += len(doc_ids)
        return rows

    def require_consumed(self) -> None:
        document = next(self.documents, None)
        if document is not None:
            raise ValueError(
                f"{self.where}: tokenize document {document[0]!r} left over after {self.consumed} normalized rows; "
                f"the two sides did not come from one normalize run"
            )


def read_aligned_documents(path: str, max_tokens: int, where: str) -> AlignedDocuments:
    """Stream the documents of one tokenize shard, chunk rows regrouped, as an :class:`AlignedDocuments`."""
    return AlignedDocuments(iter_tokenized_documents(path), max_tokens, where)


_SCORER_LOCK = threading.Lock()


@functools.cache
def _load_pinned_scorer(model_dir: str, pin: QualityPin) -> PooledScorer:
    require_pinned_model(pin, model_dir)
    scorer = load_pooled_scorer(model_dir)
    logger.info("loaded %s (%s): max_tokens=%d", pin.name, model_dir, scorer.max_tokens)
    return scorer


def pinned_scorer(model_dir: str, pin: QualityPin) -> PooledScorer:
    """The process's one scorer, digest-checked and loaded on first use."""
    # functools.cache is not atomic: the concurrent tasks of a fresh worker all
    # miss at once and would each load the 158 MB checkpoint without the lock.
    with _SCORER_LOCK:
        return _load_pinned_scorer(model_dir, pin)


def pad_ids(rows: list[np.ndarray], max_tokens: int, remap_table: np.ndarray) -> np.ndarray:
    """Dense ``[n, max_tokens]`` compact ids: the first ``max_tokens`` of each row through ``remap_table``.

    The table is the scorer's (:func:`data.remap_table`): a raw id above its largest
    known one lands on the ``UNK_ID`` sentinel in its last row. That clamp matters
    because a jax gather clamps out-of-range indices rather than raising, which would
    score the row against an unrelated embedding row.
    """
    out = np.full((len(rows), max_tokens), PAD_ID, dtype=np.int32)
    last = len(remap_table) - 1
    for i, row in enumerate(rows):
        window = np.asarray(row[:max_tokens])
        out[i, : len(window)] = remap_table[np.minimum(window, last)]
    return out


def normalize_embeddings(rows: np.ndarray) -> np.ndarray:
    """int8[1024] rows -> float32, L2-normalized, exactly as training fed them."""
    # No dequantization step: a uniform scale cancels under L2 normalization, and
    # training normalized the raw int8 values the same way.
    x = rows.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)


def shard_output_pattern(output_path: str, basenames: tuple[str, ...]):
    """The ``write_parquet`` pattern that names shard ``i`` after input basename ``i``."""

    def _output_path(shard_idx: int, _total: int) -> str:
        return prefix_join(output_path, basenames[shard_idx])

    return _output_path


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


def validated_tokenize_dir(pin: QualityPin, normalized: NormalizedData, tokenized: TokenizedAttrData, split: str) -> str:
    """The shard directory of ``tokenized``'s ``split``, once it is checked to be
    ``normalized``'s tokenize output under the tokenizer repository ``pin`` reads.

    The artifact records the repository, not the commit; the commit is pinned by the
    tokenize step's own identity, which the score step depends on.
    """
    repo, _, _ = pin.tokenizer.partition("@")
    if tokenized.tokenizer != repo:
        raise ValueError(
            f"{pin.name} reads {repo!r} ids but the tokenize stage wrote {tokenized.tokenizer!r} ids; "
            f"score from a tokenize step under {repo!r}"
        )
    source_key = datakit_source_key(normalized.main_output_dir)
    if tokenized.source_keys.get(split) != source_key:
        raise ValueError(
            f"tokenize artifact's {split!r} split was built from {tokenized.source_keys.get(split)!r}, "
            f"not from this source's normalize output {source_key!r}"
        )
    return tokenized.output_dirs[split]


def _score_shard(
    batches: Iterator[pa.RecordBatch],
    shard: ShardInfo,
    *,
    tokenize_paths: tuple[str, ...],
    embedding_paths: tuple[str, ...],
    model_dir: str,
    pin: QualityPin,
) -> Iterator[pa.RecordBatch]:
    """Score one normalized shard from its tokenize and embedding shards, in the normalized order."""
    scorer = pinned_scorer(model_dir, pin)
    tokenize_path = tokenize_paths[shard.shard_idx]
    embedding_path = embedding_paths[shard.shard_idx]
    documents = read_aligned_documents(tokenize_path, scorer.max_tokens, f"shard {shard.shard_idx} ({tokenize_path})")
    embeddings = read_aligned_column(embedding_path, "embedding", f"shard {shard.shard_idx} ({embedding_path})")
    scored = 0
    for batch in rebatch(batches, BATCH_DOCS):
        ids = batch.column("id").to_numpy(zero_copy_only=False)
        tokens = pad_ids(documents.take(ids), scorer.max_tokens, scorer.remap_table)
        embedding = normalize_embeddings(embeddings.take(ids))
        scores = predict(scorer.model, tokens, batch_size=BATCH_DOCS, doc_embed=embedding)
        scored += len(ids)
        yield pa.RecordBatch.from_arrays([batch.column("id"), pa.array(scores, type=pa.float32())], schema=SCORE_SCHEMA)
    documents.require_consumed()
    embeddings.require_consumed()
    counters.pipeline.update_counter("fusion/docs_scored", scored)
    counters.pipeline.update_counter("fusion/shards", 1)
    logger.info("shard %d/%d: %d documents scored", shard.shard_idx, shard.total_shards, scored)


def score_fusion(
    output_path: str,
    *,
    normalized: NormalizedData,
    tokenized: TokenizedAttrData,
    embedding_dir: str,
    quality_model: QualityPin,
    split: str = "train",
    worker_resources: ResourceConfig = WORKER_RESOURCES,
    task_resources: ResourceConfig = TASK_RESOURCES,
    max_workers: int = MAX_WORKERS,
    zephyr_context: ZephyrContext | None = None,
) -> FusionScores:
    """Score every shard of one normalized source; one Zephyr task per shard.

    ``tokenized`` must be the tokenize output of ``normalized`` (its ``split``
    source key names that normalize dir) under the tokenizer ``quality_model``
    reads; both are checked before anything is listed. Output shards that already
    exist are skipped, so a rerun after a partial failure scores only the remainder.
    """
    model_dir = quality_model_dir(quality_model)
    text_dir = normalized.main_output_dir
    tokenize_dir = validated_tokenize_dir(quality_model, normalized, tokenized, split)
    basenames = tuple(paired_basenames(text_dir, tokenize_dir, embedding_dir))
    tokenize_paths = tuple(prefix_join(tokenize_dir, name) for name in basenames)
    embedding_paths = tuple(prefix_join(embedding_dir, name) for name in basenames)
    logger.info(
        "scoring %d shards of %s from %s and %s -> %s",
        len(basenames),
        text_dir,
        tokenize_dir,
        embedding_dir,
        output_path,
    )
    pipeline = (
        Dataset.from_list([prefix_join(text_dir, name) for name in basenames])
        .load_parquet(columns=["id"], batch_mode=True)
        .map_shard(
            partial(
                _score_shard,
                tokenize_paths=tokenize_paths,
                embedding_paths=embedding_paths,
                model_dir=model_dir,
                pin=quality_model,
            )
        )
        .write_parquet(shard_output_pattern(output_path, basenames), schema=SCORE_SCHEMA, skip_existing=True)
    )
    ctx = zephyr_context or ZephyrContext(
        name=f"fusion-{os.path.basename(text_dir.rstrip('/'))[:8]}",
        resources=worker_resources,
        coordinator_resources=COORDINATOR_RESOURCES,
        max_workers=min(max_workers, len(basenames)),
        stage_runner_factory=InlineRunner,
    )
    outcome = ctx.execute(pipeline, verbose=True, map_task_resources=task_resources)
    return FusionScores(
        output_dir=output_path,
        source_key=datakit_source_key(text_dir),
        tokenize_dir=tokenize_dir,
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
    tokenized: StepSpec,
    embedding: StepSpec,
    quality_model: QualityPin,
    split: str = "train",
    worker_resources: ResourceConfig = WORKER_RESOURCES,
    task_resources: ResourceConfig = TASK_RESOURCES,
    max_workers: int = MAX_WORKERS,
    zephyr_context: ZephyrContext | None = None,
) -> StepSpec:
    """A step that scores ``normalized`` from ``tokenized`` and ``embedding`` with ``quality_model``.

    ``tokenized`` is the source's tokenize step under the tokenizer the pin reads
    (its artifact is a :class:`TokenizedAttrData`), and ``embedding`` the Harrier
    leaf of the same source, whose ``output_path`` is the shard directory. The model
    bytes enter the identity through the pin's digest and are checked again by
    every worker before it writes.
    """
    return StepSpec(
        name=name,
        deps=[normalized, tokenized, embedding],
        hash_attrs=fusion_hash_attrs(quality_model),
        fn=lambda output_path: score_fusion(
            output_path,
            normalized=read_artifact(normalized.output_path, NormalizedData),
            tokenized=read_artifact(tokenized.output_path, TokenizedAttrData),
            embedding_dir=embedding.output_path,
            quality_model=quality_model,
            split=split,
            worker_resources=worker_resources,
            task_resources=task_resources,
            max_workers=max_workers,
            zephyr_context=zephyr_context,
        ),
    )
