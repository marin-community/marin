# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DataKit steps for the fast-track end-to-end experiment."""

import logging
from dataclasses import replace

from fray.types import ResourceConfig
from levanter.data.text.datasets import DatasetComponentBase, LmDataConfig
from levanter.tokenizers import load_tokenizer
from marin.execution.artifact import Artifact, read_artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_spec import StepSpec
from marin.experiment.namespacing import user_namespaced_name
from zephyr.context import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.datakit.execution import run_steps_in_pool
from experiments.datakit.reference_pipeline import (
    SMOKE_SCALE,
    DriverPlacement,
    TokenizerSpec,
    reference_datakit_steps,
)
from experiments.datakit.store.datakit_store import ClusteredStoreData
from experiments.datakit.store.mixture import (
    FlatCacheComponent,
    MixtureWeighting,
    flat_cache_mixture,
    log_store_summary,
    store_mixture,
)
from experiments.grug.fast_track.contracts import add_dataset_mixture_weights

logger = logging.getLogger(__name__)


def add_prepared_dataset_component(
    baseline: LmDataConfig,
    *,
    name: str,
    component: DatasetComponentBase,
    fraction: float,
    max_train_batches: int,
) -> LmDataConfig:
    """Add a prepared dataset at its token share without a second simulated slice."""
    weights = baseline.train_weights
    if not isinstance(weights, dict):
        raise ValueError("add-dataset training requires fixed dictionary weights")
    if name in baseline.components:
        raise ValueError(f"new dataset component {name!r} already exists")

    return replace(
        baseline,
        components={**baseline.components, name: component},
        train_weights=add_dataset_mixture_weights(weights, new_component=name, fraction=fraction),
        max_train_batches={name: max_train_batches},
        target_budget=None,
        experiment_budget=None,
    )


class FastTrackDataStore(Artifact):
    """A cached DataKit store for a fast-track training step."""

    store: ClusteredStoreData


DATA_POOL_WORKERS = 1
# Leave capacity for system services on a 128-CPU, 2-TiB H100 node.
DATA_POOL_RESOURCES = ResourceConfig(cpu=120, ram="1900g", disk="25t")
# Most shard tasks request two CPUs. Keep enough steps active to fill the worker
# when each source has only one shard.
DATA_PIPELINE_CONCURRENCY = 64
FAST_TRACK_SAMPLE_PREFIX = "s3://marin-us-east-02a/marin/datakit/sample_25b_2026_10_02"


def data_pool(name: str) -> ZephyrContext:
    """Create the shared fast-track worker pool with room for nested sampling."""
    return ZephyrContext(
        name=name,
        resources=DATA_POOL_RESOURCES,
        max_workers=DATA_POOL_WORKERS,
        max_concurrent_pipelines=DATA_PIPELINE_CONCURRENCY + SMOKE_SCALE.sample_parallel_sources,
        stage_runner_factory=SubprocessRunner,
    )


def build_fast_track_data(
    *,
    run_id: str,
    sources: dict[str, StepSpec],
    quality_model: str,
    quality_model_version: str,
    tokenizer: TokenizerSpec,
    tokenizer_vocab: int,
    version: str | None = None,
) -> ArtifactStep[FastTrackDataStore]:
    """Build the DataKit recipe and cache its store with the full recipe identity."""
    datakit = reference_datakit_steps(
        sources,
        quality_model=quality_model,
        quality_model_version=quality_model_version,
        scale=SMOKE_SCALE,
        driver_placement=DriverPlacement.COORDINATOR,
        tokenizer=tokenizer,
    )

    def materialize(_config: object) -> FastTrackDataStore:
        loaded_tokenizer = load_tokenizer(tokenizer.name)
        if len(loaded_tokenizer) != tokenizer_vocab:
            raise ValueError(
                f"tokenizer {tokenizer.name!r} has {len(loaded_tokenizer)} entries; "
                f"the fast-track model requires {tokenizer_vocab}"
            )
        with data_pool(f"fast-track-{run_id}-data") as pool:
            run_steps_in_pool(datakit.all_steps, pool=pool, max_concurrent=DATA_PIPELINE_CONCURRENCY)
        store = read_artifact(datakit.output_buckets.output_path, ClusteredStoreData)
        log_store_summary(store)
        return FastTrackDataStore(store=store)

    name = f"datakit/fast-track/{run_id}"
    version = resolve_version(name, version)
    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=FastTrackDataStore,
        run=materialize,
        build_config=lambda _ctx: {
            "store": datakit.output_buckets.name_with_hash,
            "tokenizer_vocab": tokenizer_vocab,
        },
    )


def store_mixture_for_step(
    *,
    ctx: StepContext,
    store_step: ArtifactStep[FastTrackDataStore],
    weighting: MixtureWeighting,
    min_tokens_per_component: int,
    tokenizer: str,
    training_tokens: int,
) -> LmDataConfig:
    """Build a training mixture from a cached DataKit artifact."""
    if ctx.is_fingerprint:
        component_name = f"datakit-{weighting.value}"
        return flat_cache_mixture(
            tokenizer=tokenizer,
            caches={component_name: FlatCacheComponent(cache_dir=ctx.artifact_path(store_step), weight=1.0)},
        )

    store = ctx.resolved(store_step).store
    if store.tokenizer != tokenizer:
        raise ValueError(f"DataKit tokenizer {store.tokenizer!r} does not match requested tokenizer {tokenizer!r}")
    available_tokens = sum(
        bucket.total_tokens for bucket in store.buckets if bucket.total_tokens >= min_tokens_per_component
    )
    if available_tokens < training_tokens:
        raise ValueError(
            f"DataKit store has {available_tokens:,} usable tokens; this run requires {training_tokens:,}. "
            "Build a larger testbed sample."
        )
    log_store_summary(store)
    return store_mixture(
        store,
        weighting=weighting,
        min_tokens_per_component=min_tokens_per_component,
    )
