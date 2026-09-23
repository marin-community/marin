# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DataKit steps for the fast-track end-to-end experiment."""

import hashlib
import logging
from dataclasses import dataclass, replace
from typing import Protocol

import pyarrow as pa
from fray.cluster import ResourceConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.tokenizers import MarinTokenizer, load_tokenizer, tokenizer_content_hash
from marin.datakit.normalize import normalize_step
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_spec import StepSpec
from marin.experiment.namespacing import user_namespaced_name
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.runners import SubprocessRunner
from zephyr.writers import write_parquet_file

from experiments.datakit.reference_pipeline import (
    DEFAULT_MAX_CONCURRENT,
    SMOKE_SCALE,
    ClusterConfig,
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

_REPEATED_DOCUMENT_PARAGRAPH = """Data pipelines must preserve the intended distribution of source documents.
Each stage records its inputs and outputs so a later run can use the same data.
Exact duplicate removal keeps one copy of a repeated document. Domain and quality
labels then describe the remaining content. This text includes enough separate words
for tokenization, embedding, and MinHash processing in a small integration test."""
REPEATED_DOCUMENT = "\n\n".join([_REPEATED_DOCUMENT_PARAGRAPH] * 96)
_REPEATED_DOCUMENT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
    ]
)


class FastTrackDataStore(Artifact):
    """A cached DataKit store for a fast-track training step."""

    store: ClusteredStoreData


@dataclass(frozen=True)
class FastTrackDataEnvironment:
    """Runtime inputs that a fast-track data source can inspect."""

    pool_workers: int
    tokenizer: MarinTokenizer
    sequence_length: int


@dataclass(frozen=True)
class PreparedFastTrackData:
    """Normalized sources and the scale used to process them."""

    scale: PipelineScale
    sources: dict[str, StepSpec]


class FastTrackDataSource(Protocol):
    """Build one set of normalized sources for the small DataKit graph."""

    def prepare(self, environment: FastTrackDataEnvironment) -> PreparedFastTrackData: ...


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

    def prepare(self, environment: FastTrackDataEnvironment) -> PreparedFastTrackData:
        scale = _smoke_scale(environment.pool_workers)
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

    def prepare(self, environment: FastTrackDataEnvironment) -> PreparedFastTrackData:
        scale = _smoke_scale(environment.pool_workers)
        return PreparedFastTrackData(scale=scale, sources=select_sources(list(self.source_names)))


@dataclass(frozen=True)
class RepeatedDocumentDataSource:
    """Create one raw source with the same document in many rows."""

    count: int

    def prepare(self, environment: FastTrackDataEnvironment) -> PreparedFastTrackData:
        if len(environment.tokenizer.encode(REPEATED_DOCUMENT)) < environment.sequence_length:
            raise ValueError("the repeated document must contain at least one fast-track training sequence")
        scale = repeated_document_scale(environment.pool_workers)
        return PreparedFastTrackData(scale=scale, sources=repeated_document_sources(self.count, scale))


@dataclass(frozen=True)
class FastTrackDataConfig:
    """Inputs for one small production DataKit graph."""

    run_id: str
    source: FastTrackDataSource
    quality_model: str
    quality_model_version: str
    pool_workers: int
    tokenizer: str
    tokenizer_vocab: int
    sequence_length: int


def repeated_document_sources(count: int, scale: PipelineScale) -> dict[str, StepSpec]:
    """Build one raw source with the same document in many rows."""
    if count < 2:
        raise ValueError(f"repeated-document count must be at least 2, got {count}")

    document_hash = hashlib.sha256(REPEATED_DOCUMENT.encode()).hexdigest()

    def write_raw(output_path: str) -> None:
        destination = StoragePath(output_path) / "part-00000.parquet"
        records = ({"id": f"repeat-{index:06d}", "text": REPEATED_DOCUMENT} for index in range(count))
        write_parquet_file(records, str(destination), schema=_REPEATED_DOCUMENT_SCHEMA)

    raw = StepSpec(
        name="datakit/fast_track/repeated_document/raw",
        hash_attrs={"count": count, "document_sha256": document_hash},
        fn=write_raw,
    )
    normalized = normalize_step(
        name="datakit/fast_track/repeated_document/normalize",
        download=raw,
        file_extensions=(".parquet",),
        target_partition_bytes=1024 * 1024,
        max_workers=scale.pool.n_workers,
        worker_resources=scale.pool.worker,
    )
    return {"repeated_document": normalized}


def repeated_document_scale(pool_workers: int) -> PipelineScale:
    """Return a small pipeline shape for a one-document result."""
    return replace(
        _smoke_scale(pool_workers),
        cluster=ClusterConfig(k_train=1, k_views=(), cluster_view=1),
        n_per_source_for_sample=1,
        dedup_max_parallelism=1,
        train_centroids_resources=ResourceConfig.with_cpu(cpu=1, ram="2g"),
    )


def materialize_fast_track_data(config: FastTrackDataConfig) -> FastTrackDataStore:
    """Run the DataKit graph and return its clustered store metadata."""
    tokenizer = load_tokenizer(config.tokenizer)
    if len(tokenizer) != config.tokenizer_vocab:
        raise ValueError(
            f"tokenizer {config.tokenizer!r} has {len(tokenizer)} entries; "
            f"the fast-track model requires {config.tokenizer_vocab}"
        )
    prepared = config.source.prepare(
        FastTrackDataEnvironment(
            pool_workers=config.pool_workers,
            tokenizer=tokenizer,
            sequence_length=config.sequence_length,
        )
    )
    scale = prepared.scale
    tokenizer_spec = TokenizerSpec(config.tokenizer, tokenizer_content_hash(config.tokenizer))

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
            tokenizer=tokenizer_spec,
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
