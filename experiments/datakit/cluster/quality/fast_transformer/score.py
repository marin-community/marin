# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-source quality scoring with the pooled fast-transformer.

:func:`score_normalized` is the reference pipeline's quality step: one zephyr
pipeline over a source's tokenize attribute shards (one shard per file). Each
document's stored ``input_ids`` (BOS/EOS included, chunk rows regrouped into one
array) is scored whole-doc -- bme: mean over begin/middle/end ``max_tokens``
windows, so a shared boilerplate prefix can't blind the score -- and the
monotonic calibration is applied so the fixed 0.2-bucket quantization is
quality-coherent across content types. The co-partitioned normalize shard is
read in lockstep for the samples' text. No tokenizer runs on the worker: the
trainer encodes label text through the tokenize stage's own encoder, so the
stored ids are the ids the model was trained on.

Writes two outputs via a split-writer (like normalize's main/dups): the lean
scored records (``source``, ``id``, ``score`` calibrated in ``[0, 1]``,
``quality_bucket`` 0..4) to ``<output>/outputs/main/``, and a ~``sample_pct``
systematic sample *with text* to ``<output>/outputs/samples/`` that the stage
report reads directly. Each input file maps 1:1 to one output file named after
the input, and a tokenize shard carries its normalize shard's basename, so the
output is co-partitioned with the source ``NormalizedData`` by basename *and
row order* -- the store's positional join relies on both. ``from_list`` +
``flat_map`` keeps one input file as exactly one zephyr shard processed as a
single sequential stream. A tokenize shard that does not line up with its
normalize shard document-for-document is refused at the first row that
differs, and any shard that fails mid-stream (that mismatch, the streaming
reader rejecting out-of-order chunk rows, a writer error) commits no output,
because ``ThreadedBatchWriter`` aborts its stream on an exception, so a re-run
scores it again instead of skipping it as done.

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
from rigging.filesystem.factory import open_url
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo
from zephyr.runners import InlineRunner
from zephyr.writers import ThreadedBatchWriter, write_parquet_file

from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES, MODEL_CALIB, QualityScores
from experiments.datakit.cluster.quality.fast_transformer.data import bme_windows
from experiments.datakit.cluster.quality.fast_transformer.scorer import (
    MODEL_META,
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
SAMPLE_TEXT_CHARS = 4_000  # text kept per sampled doc for the report spot-check
SAMPLE_PCT = 0.02  # fraction of each shard emitted (with text) as the samples side output
_SHARD_FILE = "__shard_file"  # internal: input basename carried to the writer to name the output
_TEXT_BATCH_SIZE = 8192  # rows read from a normalize shard at once


@functools.cache
def _load_scorer(model_dir: str, calib_file: str = MODEL_CALIB) -> tuple[PooledScorer, np.ndarray, np.ndarray]:
    """Load the scorer + calibration once per worker process."""
    scorer = load_pooled_scorer(model_dir)
    with open_url(prefix_join(model_dir, calib_file), "r") as fh:
        calib = json.loads(fh.read())
    logger.info("loaded FT scorer + calibration (%s) from %s", calib_file, model_dir)
    return scorer, np.asarray(calib["xk"], dtype=np.float64), np.asarray(calib["yk"], dtype=np.float64)


def _iter_id_texts(path: str) -> Iterator[tuple[str, str]]:
    with StoragePath(path).open("rb") as fh:
        for batch in pq.ParquetFile(fh).iter_batches(batch_size=_TEXT_BATCH_SIZE, columns=["id", "text"]):
            yield from zip(batch.column("id").to_pylist(), batch.column("text").to_pylist(), strict=True)


def _load_documents(tok_path: str, *, text_dir: str, max_tokens: int) -> Iterator[dict]:
    """One record per document of a tokenize shard: its bme token windows plus the
    normalize shard's text for the samples, read in lockstep.

    The two shards must hold the same documents in the same order; the first row
    where they differ raises, and the shard's partial outputs are discarded.
    """
    basename = posixpath.basename(tok_path)
    documents = iter_tokenized_documents(tok_path)
    texts = _iter_id_texts(prefix_join(text_dir, basename))
    for position in itertools.count():
        document = next(documents, None)
        row = next(texts, None)
        if document is None and row is None:
            return
        doc_id = document[0] if document is not None else None
        text_id = row[0] if row is not None else None
        if document is None or row is None or doc_id != text_id:
            raise RuntimeError(
                f"{basename}: row {position}: normalize id {text_id!r} but tokenize document {doc_id!r} "
                "(tokenize drops zero-token documents) -- co-partitioning broken"
            )
        yield {
            "id": doc_id,
            "windows": bme_windows(document[1], max_tokens),
            "text": row[1][:SAMPLE_TEXT_CHARS],
            _SHARD_FILE: basename,
        }


def _predict_batch(records: list[dict], *, source: str, model_dir: str, calib_file: str) -> Iterator[dict]:
    """Score a batch of records over their token windows; carry source/id/score/
    quality_bucket + text. ``text`` is dropped for the lean main output and kept for
    the samples side output; ``_SHARD_FILE`` names the output file after the input."""
    scorer, xk, yk = _load_scorer(model_dir, calib_file)
    cal = np.interp(score_windowed(scorer, [r["windows"] for r in records]), xk, yk)
    buckets = np.digitize(cal, BUCKET_EDGES)
    for r, c, b in zip(records, cal, buckets, strict=True):
        yield {
            "source": source,
            "id": r["id"],
            "score": float(c),
            "quality_bucket": int(b),
            "text": r["text"],
            _SHARD_FILE: r[_SHARD_FILE],
        }


def _systematic_take(index: int, pct: float) -> bool:
    """Whether to keep record ``index`` (0-based) in a ~``pct`` sample.

    Deterministic and non-hashing: a systematic rule that keeps every ~1/pct-th record
    by position. No RNG and no id-hashing, so a given shard (records arrive in a stable
    order from the sorted input files) always yields exactly the same sample."""
    return int((index + 1) * pct) > int(index * pct)


def _output_paths(output_path: str, shard_file: str) -> tuple[str, str]:
    """(main, samples) output paths for one input file's scored records."""
    return prefix_join(output_path, f"outputs/main/{shard_file}"), prefix_join(
        output_path, f"outputs/samples/{shard_file}"
    )


def _make_scored_writer(output_path: str, sample_pct: float):
    """A ``map_shard`` split-writer. One input file per shard, so all its records share
    an output name: fan them to ``outputs/main/`` (lean) and a ~``sample_pct``
    systematic sample *with text* to ``outputs/samples/``. A shard that raises
    mid-stream leaves no output behind: the writers abort instead of committing."""

    def scored_writer(records: Iterator[dict], shard: ShardInfo) -> Iterator[dict]:
        records = iter(records)
        first = next(records, None)
        if first is None:
            return  # empty shard (e.g. all inputs skipped) -> nothing to write
        main_path, sample_path = _output_paths(output_path, first[_SHARD_FILE])

        results: dict[str, dict] = {}

        def write_to(path: str, key: str):
            def _fn(items):
                results[key] = write_parquet_file(items, output_path=path)

            return _fn

        with (
            ThreadedBatchWriter(write_to(main_path, "main")) as main_writer,
            ThreadedBatchWriter(write_to(sample_path, "samples")) as sample_writer,
        ):
            for i, r in enumerate(itertools.chain((first,), records)):
                main_writer.submit({k: r[k] for k in ("source", "id", "score", "quality_bucket")})
                counters.pipeline.update_counter("ft_quality/scored", 1)
                if _systematic_take(i, sample_pct):
                    sample_writer.submit({k: r[k] for k in ("source", "id", "score", "quality_bucket", "text")})
                    counters.pipeline.update_counter("ft_quality/sampled", 1)
        yield results

    return scored_writer


def _validated_max_tokens(model_dir: str, normalized: NormalizedData, tokenized: TokenizedAttrData, split: str) -> int:
    """The model's ``max_tokens``, once ``tokenized`` is checked to be ``normalized``'s
    tokenize output in the tokenizer the model was trained on."""
    with open_url(prefix_join(model_dir, MODEL_META), "r") as fh:
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
    sample_pct: float = SAMPLE_PCT,
    max_workers: int | None = None,
    worker_resources: ResourceConfig = WORKER_RESOURCES,
) -> QualityScores:
    """Score one source from its tokenize shards; one zephyr shard per input parquet file.

    ``tokenized`` must be the tokenize output of ``normalized`` (its ``split``
    source key names that normalize dir) and must carry the tokenizer the model in
    ``model_dir`` was trained on; both are checked before anything is listed.

    Input files whose lean main output already exists are dropped up front (a
    shard that fails removes its outputs, so a main file means fully scored) --
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
            .map_shard(_make_scored_writer(output_path, sample_pct))
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
        samples_output_dir=prefix_join(output_path, "outputs/samples"),
        model_dir=model_dir,
        calib_file=calib_file,
        bucket_edges=list(BUCKET_EDGES),
        counters=aggregated,
    )
