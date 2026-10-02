# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare fast-track training with shuffled tokens from a completed DataKit store."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import click
import numpy as np
from fray.types import ResourceConfig
from levanter.store.cache import TreeCache
from levanter.store.tree_store import TreeStore
from levanter.tokenizers import load_tokenizer
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from rigging.filesystem.storage_path import prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.runners import SubprocessRunner

from experiments.datakit.store.bucket_writer import write_bucket_cache
from experiments.datakit.store.datakit_store import BucketCacheStats
from experiments.datakit.store.mixture import MixtureWeighting, bucket_name, log_store_summary
from experiments.grug.fast_track.data_pipeline import DATA_POOL_RESOURCES, DATA_POOL_WORKERS, FastTrackDataStore
from experiments.grug.fast_track.launch import (
    H100_LADDER_SIZES,
    DataKitTrainingSource,
    Stage,
    ThroughputResult,
    build_h100_ladder_run,
)

logger = logging.getLogger(__name__)
READ_BATCH_DOCUMENTS = 256
SHUFFLE_TASK_RESOURCES = ResourceConfig(cpu=2, ram="8g", disk="8g")


def shuffle_bucket_tokens(
    bucket: BucketCacheStats,
    *,
    output_path: str,
    seed: int,
    special_ids: tuple[int, ...],
) -> BucketCacheStats:
    """Shuffle ordinary tokens within each document and retain special-token positions."""
    exemplar = {"input_ids": np.zeros(0, dtype=np.int32)}
    cache = TreeCache.load(bucket.path, exemplar)
    shard_paths = (
        [prefix_join(bucket.path, shard) for shard in cache.ledger.finished_shards]
        if cache.is_sharded
        else [bucket.path]
    )
    stores = [TreeStore.open(exemplar, path, mode="r", cache_metadata=True) for path in shard_paths]
    # The first offset holds the row count. Remaining offsets are cumulative token counts.
    lengths = np.concatenate(
        [np.diff(store.tree["input_ids"].offsets[1 : len(store) + 1].read().result(), prepend=0) for store in stores]
    )
    rng = np.random.default_rng(np.random.SeedSequence([seed, bucket.cluster_id, bucket.quality_bucket]))

    def shuffled_documents() -> Iterator[np.ndarray]:
        for store in stores:
            for start in range(0, len(store), READ_BATCH_DOCUMENTS):
                rows = store.get_batch_sync(range(start, min(start + READ_BATCH_DOCUMENTS, len(store))))
                for row in rows:
                    tokens = np.asarray(row["input_ids"], dtype=np.int32).copy()
                    ordinary = ~np.isin(tokens, special_ids)
                    tokens[ordinary] = rng.permutation(tokens[ordinary])
                    yield tokens

    ledger = write_bucket_cache(output_path, shuffled_documents(), lengths)
    if ledger.total_num_rows != bucket.total_elements or ledger.field_counts["input_ids"] != bucket.total_tokens:
        raise ValueError(f"Shuffled cache counts differ from the source bucket at {bucket.path}")
    logger.info("Shuffled %d documents and %d tokens into %s", bucket.total_elements, bucket.total_tokens, output_path)
    return bucket.model_copy(update={"path": output_path, "n_shards": 1})


@dataclass(frozen=True)
class ShuffleStoreConfig:
    source: str
    output_path: str
    seed: int


def shuffle_store(config: ShuffleStoreConfig) -> FastTrackDataStore:
    """Write a separate shuffled store with the baseline's document and token counts."""
    store = FastTrackDataStore.raw_load(config.source).store
    special_ids = tuple(load_tokenizer(store.tokenizer).all_special_ids)
    with ZephyrContext(
        name="fast-track-token-shuffle",
        resources=DATA_POOL_RESOURCES,
        max_workers=DATA_POOL_WORKERS,
        stage_runner_factory=SubprocessRunner,
    ) as pool:
        buckets = pool.execute(
            Dataset.from_list(store.buckets).map(
                lambda bucket: shuffle_bucket_tokens(
                    bucket,
                    output_path=prefix_join(config.output_path, bucket_name(bucket.cluster_id, bucket.quality_bucket)),
                    seed=config.seed,
                    special_ids=special_ids,
                )
            ),
            map_task_resources=SHUFFLE_TASK_RESOURCES,
        ).results
    shuffled = store.model_copy(update={"cache_path": config.output_path, "buckets": buckets})
    log_store_summary(shuffled)
    return FastTrackDataStore(store=shuffled)


def build_shuffled_store(source: str, seed: int) -> ArtifactStep[FastTrackDataStore]:
    """Bind token shuffling to a completed baseline data artifact."""
    name = "datakit/fast-track/shuffled-tokens"
    version = resolve_version(name, None)
    baseline = ArtifactStep.adopt(
        user_namespaced_name("datakit/fast-track/control-source", version),
        version,
        source,
        kind=FastTrackDataStore,
    )
    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=FastTrackDataStore,
        run=shuffle_store,
        build_config=lambda ctx: ShuffleStoreConfig(ctx.artifact_path(baseline), ctx.output_path, seed),
        deps=(baseline,),
    )


@click.command()
@click.option("--source-store", required=True, help="Completed FastTrackDataStore artifact from the baseline.")
@click.option("--run-id", required=True)
@click.option("--size", required=True, type=click.Choice(H100_LADDER_SIZES))
@click.option("--dense", is_flag=True)
@click.option("--seed", type=click.IntRange(min=0), default=0, show_default=True, help="Token permutation seed.")
@click.option("--stop-after", type=click.Choice([stage.value for stage in Stage]), default=Stage.TRAIN.value)
@build_options
def main(
    source_store: str, run_id: str, size: str, dense: bool, seed: int, stop_after: str
) -> ArtifactStep[ThroughputResult] | ArtifactStep[FastTrackDataStore]:
    store = build_shuffled_store(source_store, seed)
    if Stage(stop_after) is Stage.DATAKIT:
        return store
    return build_h100_ladder_run(
        run_id=run_id,
        size=size,
        dense=dense,
        save_checkpoints=True,
        training_source=DataKitTrainingSource(store, MixtureWeighting.TOKEN_PROPORTIONAL),
    )


if __name__ == "__main__":
    main()
