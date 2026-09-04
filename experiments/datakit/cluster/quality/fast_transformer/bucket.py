# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bucket the fusion quality scores under the per-type calibration.

The store partitions documents by ``quality_bucket``, and this step is where that
column is decided. It reads three co-partitioned leaves of one source: the
normalized shard for the row order, the fusion score leaf for the raw sigmoid,
and the content-type leaf for the type that selects the calibration curve. It
writes one output shard per basename with the store's columns, one row per
normalized document, in the normalized shard's row order.

Order comes from the normalized side because that is the order the store walks
positionally against decon and tokenize. The score and type sides are matched on
``id``: the fusion run wrote its rows sorted by id within each token batch, which
equals the normalized order only for sources whose shards are id-sorted, and the
type leaf followed the score leaf. A document that finds no score or no type, a
row that no document claims, and a row count that differs between the sides all
fail the shard, since each means the leaves came from different normalize runs.

Calibration is :func:`calibrate.apply_calibration`: a document routes through its
content type's curve when the calibration carries one and through the default
curve otherwise, and ``quality_bucket`` is the calibrated score digitized at
:data:`BUCKET_EDGES`. Splitting bucketing from scoring keeps a refit of the
cutpoints from rescoring the corpus.
"""

import functools
import logging
import os
from collections.abc import Iterator
from functools import partial

import numpy as np
import pyarrow as pa
from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo

from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES, QualityScores
from experiments.datakit.cluster.quality.fast_transformer.calibrate import apply_calibration
from experiments.datakit.cluster.quality.fast_transformer.keyed_rows import read_keyed_rows
from experiments.datakit.cluster.quality.fast_transformer.quality_model import (
    CALIBRATION_FILE,
    QualityPin,
    load_calibration,
    quality_model_dir,
    require_pinned_calibration,
)
from experiments.datakit.cluster.quality.fast_transformer.score_fusion import (
    COORDINATOR_RESOURCES,
    paired_basenames,
    shard_output_pattern,
)

logger = logging.getLogger(__name__)

QUALITY_BUCKETS_VERSION = 1
# A shard is three narrow columns and about four seconds of work, so the pool
# is bounded by pod scheduling rather than compute. Each worker runs four shards
# at once (the cpu and ram ratios below), and 128 workers keep 512 shards in
# flight with a quarter of the pods a one-shard-per-worker pool needed; the
# task ram covers the largest shard's 2.68M ids across the three sides.
WORKER_RESOURCES = ResourceConfig(cpu=4, ram="24g", disk="8g")
TASK_RESOURCES = ResourceConfig(cpu=1, ram="6g", disk="8g")
MAX_WORKERS = 128

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
def _pinned_calibration(model_dir: str, pin: QualityPin) -> dict:
    require_pinned_calibration(pin, model_dir)
    return load_calibration(model_dir)


def _bucket_shard(
    batches: Iterator[pa.RecordBatch],
    shard: ShardInfo,
    *,
    source: str,
    score_paths: tuple[str, ...],
    type_paths: tuple[str, ...],
    model_dir: str,
    pin: QualityPin,
) -> Iterator[pa.RecordBatch]:
    """Bucket one shard: walk the normalized ids, look up score and type by id."""
    knots = _pinned_calibration(model_dir, pin)
    where = f"{source} shard {shard.shard_idx}"
    scores = read_keyed_rows(score_paths[shard.shard_idx], "score")
    types = read_keyed_rows(type_paths[shard.shard_idx], "content_type")
    scores_claimed = np.zeros(len(scores), dtype=bool)
    types_claimed = np.zeros(len(types), dtype=bool)
    documents = 0
    for batch in batches:
        ids = batch.column("id").to_numpy(zero_copy_only=False)
        raw = scores.values[scores.rows_for(ids, scores_claimed, f"{where} (scores)")].astype(np.float32)
        content_type = types.values[types.rows_for(ids, types_claimed, f"{where} (types)")]
        calibrated = apply_calibration(raw, content_type, knots).astype(np.float32)
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
    scores.require_all_claimed(scores_claimed, documents, f"{where} (scores)")
    types.require_all_claimed(types_claimed, documents, f"{where} (types)")
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
    worker_resources: ResourceConfig = WORKER_RESOURCES,
    task_resources: ResourceConfig = TASK_RESOURCES,
    max_workers: int = MAX_WORKERS,
    zephyr_context: ZephyrContext | None = None,
) -> QualityScores:
    """Bucket one source's fusion scores; one Zephyr task per shard, several per worker.

    Output shards that already exist are skipped, so a rerun after a partial
    failure buckets only the remainder.
    """
    model_dir = quality_model_dir(quality_model)
    text_dir = normalized.main_output_dir
    basenames = tuple(paired_basenames(text_dir, scores_dir, content_type_dir))
    score_paths = tuple(prefix_join(scores_dir, name) for name in basenames)
    type_paths = tuple(prefix_join(content_type_dir, name) for name in basenames)
    logger.info(
        "%s: bucketing %d shards from %s and %s -> %s", source, len(basenames), scores_dir, content_type_dir, output_path
    )
    pipeline = (
        Dataset.from_list([prefix_join(text_dir, name) for name in basenames])
        .load_parquet(columns=["id"], batch_mode=True)
        .map_shard(
            partial(
                _bucket_shard,
                source=source,
                score_paths=score_paths,
                type_paths=type_paths,
                model_dir=model_dir,
                pin=quality_model,
            )
        )
        .write_parquet(shard_output_pattern(output_path, basenames), schema=QUALITY_SCHEMA, skip_existing=True)
    )
    ctx = zephyr_context or ZephyrContext(
        name=f"quality-{os.path.basename(text_dir.rstrip('/'))[:8]}",
        resources=worker_resources,
        coordinator_resources=COORDINATOR_RESOURCES,
        max_workers=min(max_workers, len(basenames)),
        chunk_storage_prefix=marin_temp_bucket(ttl_days=1, prefix="zephyr", source_prefix=output_path),
    )
    outcome = ctx.execute(pipeline, verbose=True, map_task_resources=task_resources)
    return QualityScores(
        main_output_dir=output_path,
        samples_output_dir=None,
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
    worker_resources: ResourceConfig = WORKER_RESOURCES,
    task_resources: ResourceConfig = TASK_RESOURCES,
    max_workers: int = MAX_WORKERS,
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
            worker_resources=worker_resources,
            task_resources=task_resources,
            max_workers=max_workers,
            zephyr_context=zephyr_context,
        ),
    )
