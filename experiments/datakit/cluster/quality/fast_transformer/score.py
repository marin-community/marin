# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-source quality scoring with the pooled fast-transformer.

:func:`score_normalized` is the reference pipeline's quality step: one zephyr
pipeline over a source's tokenize attribute shards (one shard per file). Each
document's stored ``input_ids`` (BOS/EOS included, chunk rows regrouped into one
array) is scored whole-doc -- bme: mean over begin/middle/end ``max_tokens``
windows, so a shared boilerplate prefix can't blind the score -- and the
monotonic calibration is applied so the fixed 0.2-bucket quantization is
quality-coherent across content types. No tokenizer runs on the worker: the
trainer encodes label text through the tokenize stage's own encoder, so the
stored ids are the ids the model was trained on.

Writes the lean scored records (``source``, ``id``, ``score`` calibrated in
``[0, 1]``, ``quality_bucket`` 0..4) to ``<output>/outputs/main/``. Each input
file maps 1:1 to one output file named after the input, and a tokenize shard
carries its normalize shard's basename, so the output is co-partitioned with the
source ``NormalizedData`` by basename *and row order* -- the store's positional
join relies on both. ``from_list`` + ``flat_map`` keeps one input file as exactly
one zephyr shard processed as a single sequential stream. The co-partitioned
normalize shard's ids are read in lockstep: a tokenize shard that does not line
up with it document-for-document is refused at the first row that differs, and
any shard that fails mid-stream (that mismatch, the streaming reader rejecting
out-of-order chunk rows, a writer error) commits no output, because
``ThreadedBatchWriter`` aborts its stream on an exception, so a re-run scores it
again instead of skipping it as done. The stage report reads its spot-check text
from the normalized shard of the same basename, so no text is written here.

The stage is forward-bound (~30 CPU-s per 35k docs at batch 64 on a laptop
CPU), not I/O-bound. The model dir holds the scorer artifacts (``*.eqx`` +
``*_remap.json`` + ``*_meta.json``) plus the calibration json. ``.eqx``
deserialisation needs a local path, so each worker streams it down once (cached
per process). Listing uses single-level ``*.parquet`` globs only: a recursive
glob makes s3fs ``HeadObject`` the prefix, which the CW object store answers
with a 400.
"""

import functools
import itertools
import json
import logging
import posixpath
from collections.abc import Iterator

import numpy as np
import pyarrow.parquet as pq
from fray.cluster import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import datakit_source_key
from marin.processing.tokenize.attributes import TokenizedAttrData, iter_tokenized_documents
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo
from zephyr.runners import InlineRunner
from zephyr.writers import ThreadedBatchWriter, write_parquet_file

from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES, MODEL_CALIB, QualityScores
from experiments.datakit.cluster.quality.fast_transformer.calibrate import Calibration, load_calibration
from experiments.datakit.cluster.quality.fast_transformer.data import bme_windows
from experiments.datakit.cluster.quality.fast_transformer.quality_model import artifact_names, model_stem
from experiments.datakit.cluster.quality.fast_transformer.scorer import (
    PooledScorer,
    load_pooled_scorer,
    score_windowed,
)

logger = logging.getLogger(__name__)

BATCH_SIZE = 512
# Scoring is forward-bound: a worker's CPU goes to the model forward over 64-window
# batches, not to parquet I/O. Resident memory is the model, one BATCH_SIZE batch of at
# most three max_tokens int32 windows per doc, and the compiled caches; 8g keeps the
# margin that 4g lacked when workers were OOM-killed on the 100B corpus.
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g")
_SHARD_FILE = "__shard_file"  # internal: input basename carried to the writer to name the output
_ID_BATCH_SIZE = 8192  # rows read from a normalize shard at once


@functools.cache
def _load_scorer(model_dir: str, calib_file: str = MODEL_CALIB) -> tuple[PooledScorer, Calibration]:
    """Load the scorer + calibration once per worker process."""
    scorer = load_pooled_scorer(model_dir)
    calibration = load_calibration(model_dir, calib_file)
    logger.info("loaded FT scorer + calibration (%s) from %s", calib_file, model_dir)
    return scorer, calibration


def _iter_ids(path: str) -> Iterator[str]:
    with StoragePath(path).open("rb") as fh:
        for batch in pq.ParquetFile(fh).iter_batches(batch_size=_ID_BATCH_SIZE, columns=["id"]):
            yield from batch.column("id").to_pylist()


def _load_documents(tok_path: str, *, text_dir: str, max_tokens: int) -> Iterator[dict]:
    """One record per document of a tokenize shard: its bme token windows, read in
    lockstep with the normalize shard's ids.

    The two shards must hold the same documents in the same order; the first row
    where they differ raises, and the shard's partial output is discarded.
    """
    basename = posixpath.basename(tok_path)
    documents = iter_tokenized_documents(tok_path)
    text_ids = _iter_ids(prefix_join(text_dir, basename))
    for position in itertools.count():
        document = next(documents, None)
        text_id = next(text_ids, None)
        if document is None and text_id is None:
            return
        doc_id = document[0] if document is not None else None
        if document is None or text_id is None or doc_id != text_id:
            raise RuntimeError(
                f"{basename}: row {position}: normalize id {text_id!r} but tokenize document {doc_id!r} "
                "(tokenize drops zero-token documents) -- co-partitioning broken"
            )
        yield {"id": doc_id, "windows": bme_windows(document[1], max_tokens), _SHARD_FILE: basename}


def _predict_batch(records: list[dict], *, source: str, model_dir: str, calib_file: str) -> Iterator[dict]:
    """Score a batch of records over their token windows; carry source/id/score/
    quality_bucket. ``_SHARD_FILE`` names the output file after the input."""
    scorer, calibration = _load_scorer(model_dir, calib_file)
    # This scorer reads tokens only, so a per-type calibration is refused here.
    cal = calibration.apply(score_windowed(scorer, [r["windows"] for r in records]), None)
    buckets = np.digitize(cal, BUCKET_EDGES)
    for r, c, b in zip(records, cal, buckets, strict=True):
        yield {
            "source": source,
            "id": r["id"],
            "score": float(c),
            "quality_bucket": int(b),
            _SHARD_FILE: r[_SHARD_FILE],
        }


def _output_file(output_path: str, shard_file: str) -> str:
    return prefix_join(output_path, f"outputs/main/{shard_file}")


def _make_scored_writer(output_path: str):
    """A ``map_shard`` writer. One input file per shard, so all its records share
    one output file under ``outputs/main/``, named after the input. A shard that
    raises mid-stream leaves no output behind: the writer aborts instead of
    committing."""

    def scored_writer(records: Iterator[dict], shard: ShardInfo) -> Iterator[dict]:
        records = iter(records)
        first = next(records, None)
        if first is None:
            return  # empty shard (e.g. all inputs skipped) -> nothing to write
        path = _output_file(output_path, first[_SHARD_FILE])
        results: dict[str, dict] = {}

        def write(items):
            results["main"] = write_parquet_file(items, output_path=path)

        with ThreadedBatchWriter(write) as writer:
            for r in itertools.chain((first,), records):
                writer.submit({k: r[k] for k in ("source", "id", "score", "quality_bucket")})
                counters.pipeline.update_counter("ft_quality/scored", 1)
        yield results

    return scored_writer


def _validated_max_tokens(model_dir: str, normalized: NormalizedData, tokenized: TokenizedAttrData, split: str) -> int:
    """The model's ``max_tokens``, once ``tokenized`` is checked to be ``normalized``'s
    tokenize output in the tokenizer the model was trained on."""
    _, _, meta_name = artifact_names(model_stem(model_dir))
    with (StoragePath(model_dir) / meta_name).open("r") as fh:
        meta = json.loads(fh.read())
    if meta["tokenizer"] != tokenized.tokenizer:
        raise ValueError(
            f"quality model {model_dir} was trained for tokenizer {meta['tokenizer']!r} but the tokenize "
            f"stage wrote {tokenized.tokenizer!r} ids; retrain with python -m "
            "experiments.datakit.cluster.quality.fast_transformer.train --labels <parquet> --out-dir <new dir>"
        )
    source_key = datakit_source_key(normalized.main_output_dir)
    if tokenized.source_keys.get(split) != source_key:
        raise ValueError(
            f"tokenize artifact's {split!r} split was built from {tokenized.source_keys.get(split)!r}, "
            f"not from this source's normalize output {source_key!r}"
        )
    return int(meta["max_tokens"])


def score_normalized(
    *,
    output_path: str,
    normalized: NormalizedData,
    tokenized: TokenizedAttrData,
    source: str,
    model_dir: str,
    split: str = "train",
    calib_file: str = MODEL_CALIB,
    max_workers: int | None = None,
    worker_resources: ResourceConfig = WORKER_RESOURCES,
) -> QualityScores:
    """Score one source from its tokenize shards; one zephyr shard per input parquet file.

    ``tokenized`` must be the tokenize output of ``normalized`` (its ``split``
    source key names that normalize dir) and must carry the tokenizer the model in
    ``model_dir`` was trained on; both are checked before anything is listed.

    Input files whose lean main output already exists are dropped up front (a
    shard that fails removes its output, so a main file means fully scored) --
    a re-run after a partial failure only scores the remainder.
    """
    max_tokens = _validated_max_tokens(model_dir, normalized, tokenized, split)

    inputs = tokenized.shard_paths(split)
    out_main = prefix_join(output_path, "outputs/main")
    done = {posixpath.basename(str(m)) for m in StoragePath(prefix_join(out_main, "*.parquet")).glob()}
    files = [f for f in inputs if posixpath.basename(f) not in done]
    logger.info("%s: scoring %d/%d files (max_workers=%s)", source, len(files), len(inputs), max_workers)

    aggregated: dict[str, int | float] = {}
    if files:
        pipeline = (
            Dataset.from_list(files)
            .flat_map(functools.partial(_load_documents, text_dir=normalized.main_output_dir, max_tokens=max_tokens))
            .window(BATCH_SIZE)
            .flat_map(functools.partial(_predict_batch, source=source, model_dir=model_dir, calib_file=calib_file))
            .map_shard(_make_scored_writer(output_path))
        )
        # InlineRunner keeps the per-process cached model alive across shards in a
        # worker. Iris job names reject '/', so the source is flattened.
        ctx = ZephyrContext(
            name=f"ft-quality-{source.replace('/', '-')}",
            resources=worker_resources,
            max_workers=max_workers,
            stage_runner_factory=InlineRunner,
        )
        aggregated = dict(ctx.execute(pipeline).counters)

    return QualityScores(
        main_output_dir=out_main,
        model_dir=model_dir,
        calib_file=calib_file,
        bucket_edges=list(BUCKET_EDGES),
        counters=aggregated,
    )
