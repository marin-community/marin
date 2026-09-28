# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bucket the fusion quality scores under the per-type calibration.

The store partitions documents by ``quality_bucket``, and this step is where that
column is decided. It reads three co-partitioned leaves of one source: the
normalized shard for the row order, the fusion score leaf for the raw sigmoid,
and the content-type leaf for the type that selects the calibration curve. It
writes one output shard per basename with the store's columns, one row per
normalized document, in the normalized shard's row order.

All three leaves hold the normalized shard's documents in its row order, which is
the order the store walks positionally against decon and tokenize, so the score
and type sides are read by position. Each batch's ids are checked against both
sides' (:class:`shards.AlignedColumn`), and a side with rows left over fails
the shard, since either means the leaves came from different normalize runs.

Calibration is :meth:`calibrate.Calibration.apply`: a document routes through its
content type's curve when the calibration carries one and through the default
curve otherwise, and ``quality_bucket`` is the calibrated score digitized at
:data:`BUCKET_EDGES`. Splitting bucketing from scoring keeps a refit of the
cutpoints from rescoring the corpus.
"""

import functools
import logging
from collections.abc import Iterator
from functools import partial

import numpy as np
import pyarrow as pa
from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import ShardInfo

from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES, QualityScores
from experiments.datakit.cluster.quality.fast_transformer.calibrate import Calibration, load_calibration
from experiments.datakit.cluster.quality.fast_transformer.quality_model import (
    CALIBRATION_FILE,
    QualityPin,
    quality_model_dir,
    require_pinned_calibration,
)
from experiments.datakit.cluster.quality.fast_transformer.shards import (
    ShardPool,
    map_normalized_shards,
    read_aligned_column,
)

logger = logging.getLogger(__name__)

QUALITY_BUCKETS_VERSION = 1
# A shard is three narrow columns and about four seconds of work, so the pool
# is bounded by pod scheduling rather than compute. Each worker runs four shards
# at once (the cpu and ram ratios below), and 128 workers keep 512 shards in
# flight with a quarter of the pods a one-shard-per-worker pool needed; the
# task ram covers the largest shard's 2.68M ids across the three sides.
POOL = ShardPool(
    worker=ResourceConfig(cpu=4, ram="24g", disk="8g"),
    task=ResourceConfig(cpu=1, ram="6g", disk="8g"),
    max_workers=128,
)

QUALITY_SCHEMA = pa.schema(
    [
        pa.field("source", pa.string()),
        pa.field("id", pa.string()),
        pa.field("content_type", pa.string()),
        pa.field("raw_score", pa.float32()),
        pa.field("score", pa.float32()),
        pa.field("quality_bucket", pa.int32()),
    ]
)


def quality_hash_attrs(pin: QualityPin) -> dict[str, str | int | list[float]]:
    """The identity of a bucket step, shared by its producer and its consumers.

    The scores and types it reads enter through the step's dependencies; what is
    hashed here is the remap applied to them.
    """
    return {
        "model": pin.name,
        "calibration_sha256": pin.calibration_sha256,
        "bucket_edges": list(BUCKET_EDGES),
        "v": QUALITY_BUCKETS_VERSION,
    }


@functools.cache
def _pinned_calibration(model_dir: str, pin: QualityPin) -> Calibration:
    require_pinned_calibration(pin, model_dir)
    return load_calibration(model_dir)


def _bucket_shard(
    batches: Iterator[pa.RecordBatch],
    shard: ShardInfo,
    side_paths: tuple[str, ...],
    *,
    source: str,
    model_dir: str,
    pin: QualityPin,
) -> Iterator[pa.RecordBatch]:
    """Bucket one shard: walk the normalized ids, taking score and type row for row."""
    calibration = _pinned_calibration(model_dir, pin)
    where = f"{source} shard {shard.shard_idx}"
    score_path, type_path = side_paths
    scores = read_aligned_column(score_path, "score", f"{where} (scores)")
    types = read_aligned_column(type_path, "content_type", f"{where} (types)")
    documents = 0
    for batch in batches:
        ids = batch.column("id").to_numpy(zero_copy_only=False)
        raw = scores.take(ids).astype(np.float32)
        content_type = types.take(ids)
        calibrated = calibration.apply(raw, content_type).astype(np.float32)
        bucket = np.digitize(calibrated, BUCKET_EDGES).astype(np.int32)
        documents += len(ids)
        yield pa.RecordBatch.from_arrays(
            [
                pa.array([source] * len(ids), type=pa.string()),
                batch.column("id"),
                pa.array(content_type, type=pa.string()),
                pa.array(raw, type=pa.float32()),
                pa.array(calibrated, type=pa.float32()),
                pa.array(bucket, type=pa.int32()),
            ],
            schema=QUALITY_SCHEMA,
        )
    scores.require_consumed()
    types.require_consumed()
    counters.pipeline.update_counter("quality/docs_bucketed", documents)
    counters.pipeline.update_counter("quality/shards", 1)


def bucket_quality_scores(
    output_path: str,
    *,
    source: str,
    normalized: NormalizedData,
    scores_dir: str,
    content_type_dir: str,
    quality_model: QualityPin,
    pool: ShardPool = POOL,
    zephyr_context: ZephyrContext | None = None,
) -> QualityScores:
    """Bucket one source's fusion scores; one Zephyr task per shard, several per worker.

    Output shards that already exist are skipped, so a rerun after a partial
    failure buckets only the remainder.
    """
    model_dir = quality_model_dir(quality_model)
    outcome = map_normalized_shards(
        name="quality",
        text_dir=normalized.main_output_dir,
        side_dirs=[scores_dir, content_type_dir],
        columns=["id"],
        shard_fn=partial(_bucket_shard, source=source, model_dir=model_dir, pin=quality_model),
        output_path=output_path,
        schema=QUALITY_SCHEMA,
        pool=pool,
        zephyr_context=zephyr_context,
    )
    return QualityScores(
        main_output_dir=output_path,
        model_dir=model_dir,
        calib_file=CALIBRATION_FILE,
        bucket_edges=list(BUCKET_EDGES),
        counters=dict(outcome.counters),
    )


def quality_step(
    *,
    name: str,
    source: str,
    normalized: StepSpec,
    scores: StepSpec,
    content_type: StepSpec,
    quality_model: QualityPin,
    pool: ShardPool = POOL,
    zephyr_context: ZephyrContext | None = None,
) -> StepSpec:
    """A step that buckets ``scores`` by ``content_type`` under ``quality_model``'s calibration.

    ``scores`` and ``content_type`` are the source's fusion score and content-type
    leaves; each ``output_path`` is a shard directory co-partitioned with
    ``normalized``. The artifact is a :class:`QualityScores`, which is what the
    store consumes.
    """
    return StepSpec(
        name=name,
        deps=[normalized, scores, content_type],
        hash_attrs=quality_hash_attrs(quality_model),
        fn=lambda output_path: bucket_quality_scores(
            output_path,
            source=source,
            normalized=read_artifact(normalized.output_path, NormalizedData),
            scores_dir=scores.output_path,
            content_type_dir=content_type.output_path,
            quality_model=quality_model,
            pool=pool,
            zephyr_context=zephyr_context,
        ),
    )
