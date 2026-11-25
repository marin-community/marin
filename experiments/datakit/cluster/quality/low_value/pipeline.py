# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind structural low-value scoring to the normalized Datakit shard layout."""

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache, partial

import pyarrow as pa
from fray.types import ResourceConfig
from marin.datakit.low_value.model import (
    OperatingPoint,
    load_model,
    load_operating_point,
)
from marin.datakit.low_value.parquet import SCORE_SCHEMA, LowValueScores, score_batches
from marin.datakit.low_value.scoring import DocumentScorer, load_scorer
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo

from experiments.datakit.cluster.quality.fast_transformer.score_fusion import shard_output_pattern


@dataclass(frozen=True)
class LowValuePin:
    model_path: str
    model_sha256: str
    calibration_sha256: str


@dataclass(frozen=True)
class ScoringBundle:
    scorer: DocumentScorer
    operating_point: OperatingPoint


def _load_bundle(pin: LowValuePin, model_dir: str, target: float) -> ScoringBundle:
    model = load_model(prefix_join(model_dir, "model.json"), pin.model_sha256)
    operating_point = load_operating_point(
        prefix_join(model_dir, "calibration.json"), pin.calibration_sha256, pin.model_sha256, target
    )
    return ScoringBundle(load_scorer(model), operating_point)


_bundle = cache(_load_bundle)


def _score_shard(
    batches: Iterator[pa.RecordBatch], _shard: ShardInfo, *, source: str, pin: LowValuePin, model_dir: str, target: float
) -> Iterator[pa.RecordBatch]:
    bundle = _bundle(pin, model_dir, target)
    for batch in score_batches(batches, source=source, scorer=bundle.scorer, operating_point=bundle.operating_point):
        counters.pipeline.update_counter("low_value/documents", len(batch))
        counters.pipeline.update_counter("low_value/flagged", sum(batch.column("is_low_value").to_pylist()))
        yield batch


def score_source(
    output_path: str,
    *,
    normalized: NormalizedData,
    source: str,
    pin: LowValuePin,
    target_false_drop: float,
    zephyr_context: ZephyrContext,
) -> LowValueScores:
    """Write a resumable co-partitioned leaf of scores and cutoff decisions."""
    model_dir = prefix_join(marin_prefix(), pin.model_path)
    # Fail before output admission if either pinned artifact has changed.
    _load_bundle(pin, model_dir, target_false_drop)
    paths = sorted(str(path) for path in (StoragePath(normalized.main_output_dir) / "*.parquet").glob())
    if not paths:
        raise ValueError(f"No normalized shards under {normalized.main_output_dir}")
    basenames = tuple(StoragePath(path).name for path in paths)
    pipeline = (
        Dataset.from_list(paths)
        .load_parquet(columns=["id", "text"], batch_mode=True)
        .map_shard(partial(_score_shard, source=source, pin=pin, model_dir=model_dir, target=target_false_drop))
        .write_parquet(shard_output_pattern(output_path, basenames), schema=SCORE_SCHEMA, skip_existing=True)
    )
    outcome = zephyr_context.execute(pipeline, verbose=True, map_task_resources=ResourceConfig(cpu=1, ram="4g"))
    return LowValueScores(
        output_dir=output_path,
        source=source,
        model_sha256=pin.model_sha256,
        calibration_sha256=pin.calibration_sha256,
        target_false_drop=target_false_drop,
        documents_scored_this_run=int(outcome.counters.get("low_value/documents", 0)),
    )


def quality_step(
    *,
    name: str,
    source: str,
    normalized: StepSpec,
    pin: LowValuePin,
    target_false_drop: float,
    zephyr_context: ZephyrContext,
) -> StepSpec:
    """Create a binary quality side leaf, hashing model, calibration and cutoff choice."""
    return StepSpec(
        name=name,
        deps=[normalized],
        hash_attrs={
            "model_sha256": pin.model_sha256,
            "calibration_sha256": pin.calibration_sha256,
            "target_false_drop": target_false_drop,
            "source": source,
            "v": 1,
        },
        fn=lambda output_path: score_source(
            output_path,
            normalized=read_artifact(normalized.output_path, NormalizedData),
            source=source,
            pin=pin,
            target_false_drop=target_false_drop,
            zephyr_context=zephyr_context,
        ),
    )
