# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-source quality scoring with the pooled fast-transformer.

:func:`score_normalized` is the reference pipeline's quality step: one zephyr
pipeline over a source's normalized parquet files (one shard per file), scoring
each doc whole-doc (bme: mean over begin/middle/end ~512-token windows, so a
shared boilerplate prefix can't blind the score) and applying the monotonic
calibration so the fixed 0.2-bucket quantization is quality-coherent across
content types.

Writes two outputs via a split-writer (like normalize's main/dups): the lean
scored records (``source``, ``id``, ``score`` calibrated in ``[0, 1]``,
``quality_bucket`` 0..4) to ``<output>/outputs/main/``, and a ~``sample_pct``
systematic sample *with text* to ``<output>/outputs/samples/`` that the stage
report reads directly. Each input file maps 1:1 to one output file (named after
the input), so the output is co-partitioned with the source ``NormalizedData``
by basename *and row order* -- the store's positional join relies on both, which
holds because one input file is exactly one zephyr shard processed as a single
sequential stream. (Corollary: keep the ``flat_map(load_file)`` form; the
``.load_file()`` pushdown op splits large files into multiple shards, which
would break the 1:1 naming.)

The model dir holds the scorer artifacts (``*.eqx`` + ``*_remap.json`` +
``*_meta.json``) plus the calibration json. ``.eqx`` deserialisation needs a
local path, so each worker streams it down once (cached per process). Listing
uses single-level ``*.parquet`` globs only: a recursive glob makes s3fs
``HeadObject`` the prefix, which the CW object store answers with a 400.
"""

import functools
import json
import logging
import posixpath
from collections.abc import Iterator

import numpy as np
from fray.cluster import ResourceConfig
from marin.datakit.normalize import NormalizedData
from rigging.filesystem import StoragePath, open_url
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.readers import DEFAULT_FILE_PATH_COLUMN, load_file
from zephyr.runners import InlineRunner
from zephyr.writers import ThreadedBatchWriter, write_parquet_file

from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES, QualityScores
from experiments.datakit.cluster.quality.fast_transformer.scorer import (
    PooledScorer,
    load_pooled_scorer,
    score_bme,
)

logger = logging.getLogger(__name__)

BATCH_SIZE = 512
# Scoring is I/O-bound (workers sit ~25% CPU streaming parquet). A worker sits ~3 GiB
# resident (model + a batch's JAX activations + per-seq-len compiled caches); a heavier
# shard can spike transiently above that -- 4g OOM-killed workers on the 100B corpus.
# 8g covers the spike with margin; packing is CPU-bound (cpu=2), so the extra RAM is free.
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g")
MODEL_CALIB = "calib_bme.json"  # calibration json name in the model dir
SAMPLE_TEXT_CHARS = 4_000  # text kept per sampled doc for the report spot-check
SAMPLE_PCT = 0.02  # fraction of each shard emitted (with text) as the samples side output
_SHARD_FILE = "__shard_file"  # internal: input basename carried to the writer to name the output


@functools.cache
def _load_scorer(model_dir: str, calib_file: str = MODEL_CALIB) -> tuple[PooledScorer, np.ndarray, np.ndarray]:
    """Load the scorer + calibration once per worker process."""
    scorer = load_pooled_scorer(model_dir)
    with open_url(f"{model_dir.rstrip('/')}/{calib_file}", "r") as fh:
        calib = json.loads(fh.read())
    logger.info("loaded FT scorer + calibration (%s) from %s", calib_file, model_dir)
    return scorer, np.asarray(calib["xk"], dtype=np.float64), np.asarray(calib["yk"], dtype=np.float64)


def _predict_batch(records: list[dict], *, source: str, model_dir: str, calib_file: str) -> Iterator[dict]:
    """Score a batch of records with bme; carry source/id/score/quality_bucket + text.
    ``text`` is dropped for the lean main output and kept for the samples side
    output; ``_SHARD_FILE`` names the output file after the input file."""
    scorer, xk, yk = _load_scorer(model_dir, calib_file)
    cal = np.interp(score_bme(scorer, [r["text"] for r in records]), xk, yk)
    buckets = np.digitize(cal, BUCKET_EDGES)
    for r, c, b in zip(records, cal, buckets, strict=True):
        yield {
            "source": source,
            "id": r["id"],
            "score": float(c),
            "quality_bucket": int(b),
            "text": r["text"][:SAMPLE_TEXT_CHARS],
            _SHARD_FILE: posixpath.basename(r[DEFAULT_FILE_PATH_COLUMN]),
        }


def _systematic_take(index: int, pct: float) -> bool:
    """Whether to keep record ``index`` (0-based) in a ~``pct`` sample.

    Deterministic and non-hashing: a systematic rule that keeps every ~1/pct-th record
    by position. No RNG and no id-hashing, so a given shard (records arrive in a stable
    order from the sorted input files) always yields exactly the same sample."""
    return int((index + 1) * pct) > int(index * pct)


def _output_paths(output_path: str, shard_file: str) -> tuple[str, str]:
    """(main, samples) output paths for one input file's scored records."""
    base = output_path.rstrip("/")
    return f"{base}/outputs/main/{shard_file}", f"{base}/outputs/samples/{shard_file}"


def _make_scored_writer(output_path: str, sample_pct: float):
    """A ``map_shard`` split-writer. One input file per shard, so all its records share
    an output name: fan them to ``outputs/main/`` (lean) and a ~``sample_pct``
    systematic sample *with text* to ``outputs/samples/``."""

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
            for i, r in enumerate((first, *records)):
                main_writer.submit({k: r[k] for k in ("source", "id", "score", "quality_bucket")})
                counters.pipeline.update_counter("ft_quality/scored", 1)
                if _systematic_take(i, sample_pct):
                    sample_writer.submit({k: r[k] for k in ("source", "id", "score", "quality_bucket", "text")})
                    counters.pipeline.update_counter("ft_quality/sampled", 1)
        yield results

    return scored_writer


def score_normalized(
    *,
    output_path: str,
    normalized: NormalizedData,
    source: str,
    model_dir: str,
    calib_file: str = MODEL_CALIB,
    sample_pct: float = SAMPLE_PCT,
    max_workers: int | None = None,
    worker_resources: ResourceConfig = WORKER_RESOURCES,
) -> QualityScores:
    """Score one normalized source; one zephyr shard per input parquet file.

    Input files whose lean main output already exists are dropped up front
    (main is written last per shard, so its presence means fully scored) --
    a re-run after a partial failure only scores the remainder.
    """
    main_dir = normalized.main_output_dir.rstrip("/")
    inputs = sorted(str(m) for m in StoragePath(f"{main_dir}/*.parquet").glob())
    out_main = f"{output_path.rstrip('/')}/outputs/main"
    done = {posixpath.basename(str(m)) for m in StoragePath(f"{out_main}/*.parquet").glob()}
    files = [f for f in inputs if posixpath.basename(f) not in done]
    logger.info("%s: scoring %d/%d files (max_workers=%s)", source, len(files), len(inputs), max_workers)

    aggregated: dict[str, int | float] = {}
    if files:
        pipeline = (
            Dataset.from_list(files)
            .flat_map(functools.partial(load_file, include_file_paths=True))
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
        samples_output_dir=f"{output_path.rstrip('/')}/outputs/samples",
        model_dir=model_dir,
        calib_file=calib_file,
        bucket_edges=list(BUCKET_EDGES),
        counters=aggregated,
    )
