# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DataKit steps for the fast-track end-to-end experiment."""

import logging
from dataclasses import dataclass, replace
from typing import Protocol

from levanter.data.text.datasets import LmDataConfig
from levanter.tokenizers import load_tokenizer
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_spec import StepSpec
from marin.experiment.namespacing import user_namespaced_name
from zephyr.context import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.datakit.reference_pipeline import (
    DEFAULT_MAX_CONCURRENT,
    SMOKE_SCALE,
    PipelineScale,
    PoolConfig,
    TokenizerSpec,
    materialize_reference_store,
    sample_sources,
    select_sources,
)
from experiments.datakit.store.datakit_store import ClusteredStoreData
from experiments.datakit.store.mixture import (
    FlatCacheComponent,
    MixtureWeighting,
    flat_cache_mixture,
    log_store_summary,
    store_mixture,
)

logger = logging.getLogger(__name__)


class FastTrackDataStore(Artifact):
    """A cached DataKit store for a fast-track training step."""

    store: ClusteredStoreData


@dataclass(frozen=True)
class PreparedFastTrackData:
    """Normalized sources and the scale used to process them."""

    scale: PipelineScale
    sources: dict[str, StepSpec]


class FastTrackDataSource(Protocol):
    """Build one set of normalized sources for the small DataKit graph."""

    def prepare(self, pool_workers: int) -> PreparedFastTrackData: ...


def _smoke_scale(pool_workers: int) -> PipelineScale:
    return replace(
        SMOKE_SCALE,
        pool=PoolConfig(n_workers=pool_workers, worker=SMOKE_SCALE.pool.worker),
    )


@dataclass(frozen=True)
class SampleDataSource:
    """Read selected normalized sources from a DataKit sample."""

    sample_prefix: str
    source_names: tuple[str, ...] | None

    def prepare(self, pool_workers: int) -> PreparedFastTrackData:
        scale = _smoke_scale(pool_workers)
        sources = sample_sources(
            self.sample_prefix,
            None if self.source_names is None else list(self.source_names),
        )
        return PreparedFastTrackData(scale=scale, sources=sources)


@dataclass(frozen=True)
class RegistryDataSource:
    """Read selected raw sources from the DataKit registry."""

    source_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.source_names:
            raise ValueError("registry data requires at least one source")

    def prepare(self, pool_workers: int) -> PreparedFastTrackData:
        scale = _smoke_scale(pool_workers)
        return PreparedFastTrackData(scale=scale, sources=select_sources(list(self.source_names)))


@dataclass(frozen=True)
class FastTrackDataConfig:
    """Inputs for one small production DataKit graph."""

    run_id: str
    source: FastTrackDataSource
    quality_model: str
    quality_model_version: str
    pool_workers: int
    tokenizer: TokenizerSpec
    tokenizer_vocab: int


def materialize_fast_track_data(config: FastTrackDataConfig) -> FastTrackDataStore:
    """Run the DataKit graph and return its clustered store metadata."""
    tokenizer = load_tokenizer(config.tokenizer.name)
    if len(tokenizer) != config.tokenizer_vocab:
        raise ValueError(
            f"tokenizer {config.tokenizer.name!r} has {len(tokenizer)} entries; "
            f"the fast-track model requires {config.tokenizer_vocab}"
        )
    prepared = config.source.prepare(config.pool_workers)
    scale = prepared.scale
    with ZephyrContext(
        name=f"fast-track-{config.run_id}-data",
        resources=scale.pool.worker,
        max_workers=scale.pool.n_workers,
        stage_runner_factory=SubprocessRunner,
    ) as zephyr_context:
        store = materialize_reference_store(
            prepared.sources,
            quality_model=config.quality_model,
            quality_model_version=config.quality_model_version,
            scale=scale,
            zephyr_context=zephyr_context,
            tokenizer=config.tokenizer,
            max_concurrent=DEFAULT_MAX_CONCURRENT,
        )
    log_store_summary(store)
    return FastTrackDataStore(store=store)


def build_fast_track_data(
    config: FastTrackDataConfig, *, version: str | None = None
) -> ArtifactStep[FastTrackDataStore]:
    """Build the cached DataKit artifact for an end-to-end fast-track run."""
    name = f"datakit/fast-track/{config.run_id}"
    version = resolve_version(name, version)
    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=FastTrackDataStore,
        run=materialize_fast_track_data,
        build_config=lambda ctx: replace(config, quality_model=ctx.runtime_arg("quality_model")),
        runtime_args={"quality_model": config.quality_model},
    )


def store_mixture_for_step(
    *,
    ctx: StepContext,
    store_step: ArtifactStep[FastTrackDataStore],
    weighting: MixtureWeighting,
    min_tokens_per_component: int,
    tokenizer: str,
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
    log_store_summary(store)
    return store_mixture(
        store,
        weighting=weighting,
        min_tokens_per_component=min_tokens_per_component,
    )
