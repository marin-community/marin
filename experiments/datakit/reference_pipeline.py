# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end reference DAG: Datakit sources → (cluster x quality) store.

This wires the existing per-stage building blocks into a single
StepRunner-walkable graph. Two modes (``--mode``), same DAG:

- ``full``: sources from :func:`marin.datakit.sources.all_sources`, K=5000.
- ``sample``: a pre-built testbed sample registered as already-normalized
  sources (``--sample-prefix``), K=64 -- a true end-to-end run on real data.

Decontamination (:mod:`marin.datakit.decon`) has three stages that you can run separately:

The eval item table is prepared once per region (``python -m experiments.datakit.decontam.items``); the
stages read it:

- ``decon-keys`` chooses each item's rarest key over reference shards of every source.
- ``decon-mark`` marks each selected source: key scan, encoder scoring on GPU workers, marks.
- ``decon-report`` reads completed marks and writes the validation report.

Run preparation first::

    python -m experiments.datakit.reference_pipeline \
        --mode full --target decon-keys --sources all \
        --pool-workers 60 --pool-cpu 16 --pool-ram 128g --pool-disk 128g

After it succeeds, start the marking job. Use ``--mark-sources`` to test a
source subset while the marks keep their full-pipeline identities::

    python -m experiments.datakit.reference_pipeline \
        --mode full --target decon-mark --sources all \
        --mark-sources all \
        --pool-workers 60 --pool-cpu 16 --pool-ram 128g --pool-disk 128g \
        --max-concurrent 16

Use the same source selection with ``--target decon-report`` after all marks
succeed.

Per source::

    normalize → tokenize             → quality (pooled fast-transformer on the tokenize ids, given model dir)
              → embed (luxical-one)   → assign (domain v0, given centroids)
              → decontam               (shared eval keys; encoder scoring on GPU)
              → minhash

Then:
    global_exact_dedup([<normalized source>])
    fuzzy_dups([<minhash per source>])
    large_clusters → cluster_text([<normalized source>], fuzzy_dups) → verify_fuzzy_clusters
    build_clustered_store(tokenize, decontam, cluster_assign, quality, exact_dedup, verified_dedup)
    one ``datakit/report/<stage>`` step per stage -- a single self-contained
    HTML page built from that stage's counters + site/sample outputs
    (:mod:`experiments.datakit.reports`)

Most stages keep one step per source with a separate output directory. Global
exact dedup, fuzzy dedup, the decontamination keys, and the store combine sources.

Worker fleet: subprocess-compatible stages share one Zephyr coordinator and
worker group. Steps that require process-local model caches use dedicated
``InlineRunner`` contexts. One :class:`PoolConfig` sets the worker count and
worker and task shapes. ``--max-concurrent`` limits concurrent StepRunner steps.
:func:`run_reference_steps` starts the shared pool after the normalize steps and
stops it when the last shared-pool stage succeeds. Thus idle shared workers do
not hold the hosts that the dedicated pools need.

Public API: :func:`reference_datakit_steps`. Pass ``sources`` (a ``{name:
normalize_step}`` mapping), a ``quality_model`` dir, and optionally pre-staged
domain centroids (``None`` trains them inline).

Region-agnostic: worker sizing is one :class:`PoolConfig`. ``MARIN_PREFIX`` is
resolved by :func:`rigging.filesystem.cluster_config.marin_prefix` -- unset (the normal iris-
worker case) it falls back to the in-region bucket, so source artifacts, the
eval corpus, the decontamination item table and models, and every output land in-region. Override via
``iris job run -e MARIN_PREFIX <bucket>``.

Submit the sample-mode end-to-end run on iris::

    uv run iris --cluster=cw-rno2a job run --priority interactive --cpu 2 --memory 8GB \\
        --enable-extra-resources -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
        -- python -m experiments.datakit.reference_pipeline \\
            --mode sample --sample-prefix s3://.../datakit/sample_100b_8ae7a94f \\
            --sources all --pool-workers 512 \\
            --quality-model-version pooled-junkgate2-marin \\
            --domain-centroids-version <run-id>

Reproducibility contract
------------------------
A step's ``hash_attrs`` is its cross-region identity: it must contain every
parameter its fn reads, and no region-specific ``gs://`` path (else byte-identical
data gets a different output path per region). Two consequences:

* External inputs enter the hash as a caller-supplied *version tag*, never their
  absolute path -- ``quality_model_version`` for the quality model dir and
  ``centroids_version`` for pre-staged centroids. The HF luxical weights and the
  tokenizer are pinned to immutable commits (``LUXICAL_REVISION`` /
  ``TOKENIZER_REVISION``).
* ``embed`` and ``train_centroids`` are *train-once / replicate* artifacts, not
  per-region recomputes: luxical inference (float + int8 quantization on
  heterogeneous CPUs) and faiss K-means (seeded but not bit-stable across machine
  types / thread counts) can differ bit-for-bit between regions. To reproduce a
  store exactly, replicate those bytes (pass the trained centroids as
  ``domain_centroids``) rather than recomputing inline.

The decontamination item table enters the hash by ``ITEMS_VERSION`` and the models by ``MODELS_VERSION``,
not by their paths.
"""

import argparse
import logging
import posixpath
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field, replace

from fray.types import ResourceConfig
from levanter.tokenizers import TokenizerBackend
from marin.datakit.decon import (
    DeconAttributes,
    ScoreWorkers,
    decon_keys_step,
    decon_mark_step,
    decon_scan_step,
    decon_score_step,
)
from marin.datakit.normalize import NormalizedData
from marin.datakit.sources import all_sources
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner, step_is_built
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.cluster_dedup import ClusterDedupParams
from marin.processing.classification.deduplication.cluster_text import ClusterTextParams
from marin.processing.classification.deduplication.cluster_verify import (
    ClusterVerificationLimits,
    ClusterVerifiedFuzzyDupsAttrData,
    cluster_verify_step,
)
from marin.processing.classification.deduplication.fuzzy_dups import (
    FUZZY_DUPS_ATTR_DATA_VERSION,
    FuzzyDupsAttrData,
    compute_fuzzy_dups_attrs,
)
from marin.processing.classification.deduplication.fuzzy_minhash import (
    MINHASH_ATTR_DATA_VERSION,
    MinHashAttrData,
    compute_minhash_attrs,
)
from marin.processing.classification.deduplication.large_clusters import LargeClusterParams, large_clusters_step
from marin.processing.classification.deduplication.materialize_cluster_text import cluster_text_step
from marin.processing.tokenize.attributes import (
    TokenizedAttrData,
    tokenize_attributes_step,
)
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging
from zephyr.context import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.datakit.cluster.domain.v0.assign import (
    AssignmentAttrData,
    assign_hash_attrs,
    assign_source,
)
from experiments.datakit.cluster.domain.v0.sample import sample_centroid_inputs
from experiments.datakit.cluster.domain.v0.train import train_centroids
from experiments.datakit.cluster.quality.fast_transformer.artifact import QualityScores
from experiments.datakit.cluster.quality.fast_transformer.score import score_normalized
from experiments.datakit.decontam.config import (
    CPU_SCORE_WORKERS,
    CUTOFF,
    FLAGGED_SAMPLE_SIZE,
    GPU_SCORE_WORKERS,
    REFERENCE_SEED,
    REFERENCE_SHARDS_PER_SOURCE,
    decon_models,
)
from experiments.datakit.decontam.items import prepared_items_step
from experiments.datakit.embeddings.luxical.pipeline import (
    EMBED_DOC_SAMPLE_CHARS,
    EMBEDDING_ATTR_DATA_VERSION,
    LUXICAL_REPO,
    LUXICAL_REVISION,
    LUXICAL_WEIGHTS_FILE,
    EmbeddingAttrData,
    embed_source,
)
from experiments.datakit.global_exact_dedup import (
    GLOBAL_EXACT_DEDUP_DATA_VERSION,
    GlobalExactDedupData,
    global_exact_deduplicate,
)
from experiments.datakit.reports.decontam import decontam_report
from experiments.datakit.reports.dedup import cluster_dedup_report
from experiments.datakit.reports.domain import assign_report
from experiments.datakit.reports.normalize import normalize_report
from experiments.datakit.reports.quality import quality_report
from experiments.datakit.reports.store import store_report
from experiments.datakit.reports.tokenize import tokenize_report
from experiments.datakit.store.datakit_store import (
    DEFAULT_PARALLEL_BUCKET_WRITES,
    DEFAULT_PARTITION_PROCESSES,
    ClusteredStoreData,
    build_clustered_store,
)

logger = logging.getLogger(__name__)

SHARED_POOL_RELEASE_POLL_INTERVAL = 60.0
"""Seconds between checks for the completion of every shared-pool stage."""


# Tokenize: canonical Marin tokenizer. Not scale-sensitive.
TOKENIZER = "marin-community/marin-tokenizer"
# Immutable HF commit for the tokenizer. Hashed into the tokenize step so a silent
# upstream retag invalidates the cache instead of changing token ids under a fixed
# hash. NOTE: ``levanter.tokenizers.load_tokenizer`` does not yet accept a revision
# (it stages from a Marin GCS mirror, then HF), so this pins the *recipe identity*;
# byte-level enforcement is tracked in a follow-up to thread ``revision`` through the
# loader.
TOKENIZER_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"
TOKENIZER_BACKEND = TokenizerBackend.HF
SPLIT = "train"


@dataclass(frozen=True)
class ClusterConfig:
    """Spherical-K-means knobs for the domain-clustering stage.

    ``cluster_view`` is the K the store partitions on (``cluster=<C>/quality=<Q>/``)
    and must be ``k_train`` or one of ``k_views`` -- the assign stage only
    materializes a ``cluster_<K>`` column for those. ``k_train`` must not exceed
    the centroid-training sample size, so shrink it for small inline runs.

    The ``*_seed`` / ``train_n_*`` fields pin the centroid-training recipe so it enters
    the sample/train hashes. faiss K-means is seeded but not bit-reproducible across
    machine types / thread counts, so identical bytes across regions require
    *replicating* the trained centroids (pass them as ``domain_centroids``), not
    recomputing inline -- see the module docstring's reproducibility note.
    """

    k_train: int = 5000
    k_views: tuple[int, ...] = (40, 1000)
    cluster_view: int = 40
    sample_seed: int = 42
    train_seed: int = 42
    train_n_iter: int = 20
    train_n_redo: int = 3

    def __post_init__(self) -> None:
        if self.cluster_view not in (self.k_train, *self.k_views):
            raise ValueError(
                f"cluster_view={self.cluster_view} must be k_train ({self.k_train}) or one of k_views ({self.k_views})"
            )


# Remote stage-driver jobs (embed / quality / assign / centroid-sample) submit a
# pipeline to their own dedicated coordinator and block, so they need almost
# nothing themselves.
DRIVER_RESOURCES = ResourceConfig(cpu=1, ram="2g")


@dataclass(frozen=True)
class PoolConfig:
    """The Zephyr worker fleet for the reference pipeline.

    ``n_workers`` sets the shared pool size. ``worker`` sets each worker shape.
    ``task`` sets the resource budget for each subprocess in the shared pool.
    ``coordinator`` sets the resources for the shared-pool coordinator. The
    coordinator holds shard metadata, so large pipelines need more than the
    Zephyr default. Subprocess-compatible stages share this pool. Inline stages
    create a dedicated pool with the worker settings. The worker must fit the
    largest task that can use the shared pool.
    """

    n_workers: int = 512
    worker: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=2, ram="16g", disk="16g"))
    task: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=2, ram="16g", disk="16g"))
    coordinator: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=1, ram="8g", preemptible=False))


@dataclass(frozen=True)
class MinhashConfig:
    """Content-determining MinHash / LSH knobs for the fuzzy-dedup stage.

    Not scale-sensitive (a smoke and a full run must agree), but every field shapes
    the emitted signatures and buckets, so all are hashed into the minhash step -- and
    thereby, via the minhash deps, into dedup and the store.
    """

    num_perms: int = 286
    num_bands: int = 26
    ngram_size: int = 5
    text_cap_chars: int | None = 500_000
    seed: int = 42


# Preserve selected source text that the word-shingle rule can misclassify.
# Exact dedup still applies. Policy: https://github.com/marin-community/marin/pull/8405
FUZZY_DEDUP_EXEMPT_SOURCES = (
    "biocollection/free_text_stream",
    "biocollection/instruction_stream",
    "biocorpus",
    "cp/data_provenance",
    "davinci-dev/ctx-native",
    "dna/functional-regions",
    "massive_function_calling",
    "nemotron_legal/globalcit",
    "nemotron_specialized_v1_1/code_concepts",
    "nemotron_specialized_v1_1/economics",
    "nemotron_specialized_v1_1/formal_logic",
    "nemotron_specialized_v1_1/multiple_choice",
    "nemotron_specialized_v1_2/fact_seeking",
    "nemotron_specialized_v1_2/generative",
    "nemotron_specialized_v1_2/multiple_choice",
    "swe-rebench-contree",
)


@dataclass(frozen=True)
class StoreConfig:
    """Execution shape for the final map-only clustered store.

    Production uses 192 task-local partitions, which is about 104B tokens per
    task for a 20T-token corpus. The dedicated worker shape keeps the store's
    large RAM and local-disk request out of the shared upstream worker pool.
    """

    fuzzy_exempt_sources: tuple[str, ...] = FUZZY_DEDUP_EXEMPT_SOURCES
    task_count: int | None = 192
    partition_processes: int = 32
    max_parallel_bucket_writes: int = DEFAULT_PARALLEL_BUCKET_WRITES
    worker: ResourceConfig = field(
        default_factory=lambda: ResourceConfig(cpu=96, ram="700g", disk="900g", preemptible=False)
    )


@dataclass(frozen=True)
class FuzzyClusterConfig:
    """Production duplicate rule, materialization policy, and worker shapes."""

    plan: LargeClusterParams = field(default_factory=LargeClusterParams)
    text: ClusterTextParams = field(default_factory=ClusterTextParams)
    rule: ClusterDedupParams = field(default_factory=ClusterDedupParams)
    limits: ClusterVerificationLimits = field(default_factory=ClusterVerificationLimits)
    worker: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=32, ram="192g", disk="512g"))
    map_task: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=1, ram="12g", disk="48g"))
    reduce_task: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=1, ram="26g", disk="48g"))
    max_workers: int = 64


@dataclass(frozen=True)
class PipelineScale:
    """Sizing for :func:`reference_datakit_steps`.

    Most worker CPU/RAM lives in :class:`PoolConfig`; the final store has a
    dedicated large worker in :class:`StoreConfig`. ``DEFAULT_SCALE`` is the
    production K=5000 shape; ``SMOKE_SCALE`` is K=64 for a quick end-to-end run.
    """

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)
    minhash: MinhashConfig = field(default_factory=MinhashConfig)
    fuzzy: FuzzyClusterConfig = field(default_factory=FuzzyClusterConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    embed_batch_size: int = 4096
    assign_batch_size: int = 4096
    # Inline domain training: ~100 sources x 100k = ~10M-row centroid sample.
    n_per_source_for_sample: int = 100_000
    # Concurrent per-source sampler pipelines run inside the centroid-sample
    # stage's coordinator; kept modest so it isn't overwhelmed.
    sample_parallel_sources: int = 4
    dedup_max_parallelism: int = 4096
    # Centroid training is single-process FAISS K-means, not a pool stage.
    train_centroids_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig.with_cpu(cpu=32, ram="64g"))
    # Decontamination scores its pairs on its own workers: ``GPU_SCORE_WORKERS``, or ``CPU_SCORE_WORKERS`` on a
    # cluster without GPUs (the GCP cluster). Every scale names them.
    decon_score: ScoreWorkers = field(kw_only=True)


DEFAULT_SCALE = PipelineScale(decon_score=GPU_SCORE_WORKERS)
"""Production full-fleet sizing (every ``all_sources()`` entry, K=5000)."""

SMOKE_SCALE = PipelineScale(
    fuzzy=FuzzyClusterConfig(
        text=ClusterTextParams(output_shards=64),
        worker=ResourceConfig(cpu=2, ram="8g", disk="16g"),
        map_task=ResourceConfig(cpu=1, ram="4g", disk="8g"),
        reduce_task=ResourceConfig(cpu=1, ram="4g", disk="8g"),
        max_workers=16,
    ),
    cluster=ClusterConfig(k_train=64, k_views=(8, 16), cluster_view=8),
    pool=PoolConfig(
        n_workers=16,
        worker=ResourceConfig(cpu=2, ram="8g", disk="8g"),
        task=ResourceConfig(cpu=2, ram="8g", disk="8g"),
    ),
    store=StoreConfig(
        task_count=None,
        partition_processes=DEFAULT_PARTITION_PROCESSES,
        worker=ResourceConfig(cpu=2, ram="8g", disk="8g"),
    ),
    n_per_source_for_sample=20_000,
    dedup_max_parallelism=64,
    train_centroids_resources=ResourceConfig.with_cpu(cpu=4, ram="8g"),
    decon_score=CPU_SCORE_WORKERS,
)
"""Small K + a small pool -- a true end-to-end run on a testbed sample."""


def select_sources(names: list[str] | None = None) -> dict[str, StepSpec]:
    """Map source names to their normalize StepSpec; ``None`` selects every source.

    Raises ``KeyError`` (listing the unknown names) if any requested name isn't
    in :func:`marin.datakit.sources.all_sources`.
    """
    registry = all_sources()
    if names is None:
        selected = registry
    else:
        unknown = [n for n in names if n not in registry]
        if unknown:
            raise KeyError(f"unknown sources {unknown}; known: {sorted(registry)}")
        selected = {n: registry[n] for n in names}
    sources = {name: src.normalized for name, src in selected.items()}
    logger.info("select_sources: %d sources (%s)", len(sources), "all" if names is None else ", ".join(names))
    return sources


def _build_embed_step(name: str, normalize_step: StepSpec, scale: PipelineScale, output_prefix: str | None) -> StepSpec:
    return StepSpec(
        name=f"datakit/embed/{name}",
        output_path_prefix=output_prefix,
        deps=[normalize_step],
        hash_attrs={
            "luxical_repo": LUXICAL_REPO,
            "luxical_weights": LUXICAL_WEIGHTS_FILE,
            "luxical_revision": LUXICAL_REVISION,
            "batch_size": scale.embed_batch_size,
            # Truncation changes the vector for any document above the cap, so it is
            # part of the embed step identity per the contract above.
            "doc_sample_chars": EMBED_DOC_SAMPLE_CHARS,
            "v": EMBEDDING_ATTR_DATA_VERSION,
        },
        fn=remote(
            lambda output_path, np=normalize_step.output_path: embed_source(
                output_path=output_path,
                normalized=read_artifact(np, NormalizedData),
                revision=LUXICAL_REVISION,
                batch_size=scale.embed_batch_size,
                doc_sample_chars=EMBED_DOC_SAMPLE_CHARS,
                worker_resources=scale.pool.worker,
                max_workers=scale.pool.n_workers,
            ),
            resources=DRIVER_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )


def build_per_source_embed_steps(
    sources: dict[str, StepSpec], scale: PipelineScale = DEFAULT_SCALE, output_prefix: str | None = None
) -> dict[str, StepSpec]:
    """Build the Luxical embed StepSpec for each source.

    ``sources`` maps source name → normalize StepSpec; the embed step is built
    against that step as its dep. Exposed so callers that also want to build
    the domain training subgraph (via :func:`build_train_centroids_step`) can
    share the same embeds across both wirings.
    """
    return {name: _build_embed_step(name, step, scale, output_prefix) for name, step in sources.items()}


def build_train_centroids_step(
    embed_steps: dict[str, StepSpec], scale: PipelineScale = DEFAULT_SCALE, output_prefix: str | None = None
) -> StepSpec:
    """Build the K-means training StepSpec for the domain centroids.

    The returned step's ``output_path`` contains ``centroids_<k_train>.npy``
    plus ``lookup_<k_train>_to_<k>.npy`` for each ``k`` in ``scale.cluster.k_views``
    -- the same layout :func:`reference_datakit_steps` consumes via its
    ``domain_centroids`` parameter when given a centroids path.
    """
    cluster = scale.cluster
    sample_step = StepSpec(
        name="datakit/cluster/sample_centroids",
        output_path_prefix=output_prefix,
        deps=list(embed_steps.values()),
        hash_attrs={
            "n_per_source": scale.n_per_source_for_sample,
            "seed": cluster.sample_seed,
            "format": "parquet",
            "v": 1,
        },
        fn=remote(
            lambda output_path, es={n: s.output_path for n, s in embed_steps.items()}: sample_centroid_inputs(
                output_path=output_path,
                embeddings={n: read_artifact(p, EmbeddingAttrData) for n, p in es.items()},
                n_per_source=scale.n_per_source_for_sample,
                seed=cluster.sample_seed,
                worker_resources=scale.pool.worker,
                max_workers=scale.pool.n_workers,
                parallel_sources=scale.sample_parallel_sources,
            ),
            resources=DRIVER_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )
    # Pin the K-means/BLAS thread count to the allocated CPUs so centroid training
    # is reproducible independent of which node (and how many physical cores) the
    # single-process training pod lands on (marin#6798).
    n_threads = int(scale.train_centroids_resources.cpu)
    return StepSpec(
        name="datakit/cluster/train_centroids",
        output_path_prefix=output_prefix,
        deps=[sample_step],
        hash_attrs={
            "k_train": cluster.k_train,
            "k_views": list(cluster.k_views),
            "n_threads": n_threads,
            "seed": cluster.train_seed,
            "n_iter": cluster.train_n_iter,
            "n_redo": cluster.train_n_redo,
            "v": 1,
        },
        fn=remote(
            lambda output_path, sp=sample_step.output_path: train_centroids(
                output_path=output_path,
                sample_path=sp,
                k_train=cluster.k_train,
                k_views=cluster.k_views,
                n_threads=n_threads,
                n_iter=cluster.train_n_iter,
                n_redo=cluster.train_n_redo,
                seed=cluster.train_seed,
            ),
            resources=scale.train_centroids_resources,
            pip_dependency_groups=["datakit"],
        ),
    )


def _resolve_centroids(
    domain_centroids: str | StepSpec,
    cluster: ClusterConfig,
    centroids_version: str | None,
) -> tuple[str, dict[int, str], list[StepSpec], str]:
    """Return ``(centroids_uri, lookup_uris, extra_deps, hash_value)`` for assign.

    ``hash_value`` is the region-independent identity string mixed into the assign
    hash by construction: an inline StepSpec contributes its ``name_with_hash``
    (identity, not the ``MARIN_PREFIX``-rooted output path), and a pre-staged path
    contributes the caller's ``centroids_version`` tag -- never the absolute ``gs://``
    path -- so identical centroids resolve to the same assign output path in any region.
    """
    if isinstance(domain_centroids, StepSpec):
        base = domain_centroids.output_path
        return (
            f"{base}/centroids_{cluster.k_train}.npy",
            {k: f"{base}/lookup_{cluster.k_train}_to_{k}.npy" for k in cluster.k_views},
            [domain_centroids],
            domain_centroids.name_with_hash,  # region-independent identity (also captured via the dep)
        )
    if not centroids_version:
        raise ValueError(
            "centroids_version is required when domain_centroids is a pre-staged path: "
            "the absolute path is region-specific and must not enter the cache hash. "
            "Pass a stable tag identifying the centroid bytes (e.g. the training run id)."
        )
    base = domain_centroids.rstrip("/")
    return (
        f"{base}/centroids_{cluster.k_train}.npy",
        {k: f"{base}/lookup_{cluster.k_train}_to_{k}.npy" for k in cluster.k_views},
        [],
        centroids_version,
    )


def _resolve_quality_model_version(quality_model: str, quality_model_version: str | None) -> str:
    """Return the region-independent identity tag hashed into the quality step."""
    if not quality_model_version:
        raise ValueError(
            f"quality_model_version is required: the quality model dir ({quality_model}) is "
            "region-specific and must not enter the cache hash. Pass a stable tag "
            "identifying the model bytes (e.g. 'pooled-junkgate2-marin')."
        )
    return quality_model_version


def _assign_embedding(
    output_path: str,
    embed_path: str,
    centroids_uri: str,
    lookup_uris: dict[int, str],
    scale: PipelineScale,
) -> AssignmentAttrData:
    """Assign one source, reading its shape from the luxical embedding artifact."""
    embedding = read_artifact(embed_path, EmbeddingAttrData)
    return assign_source(
        output_path=output_path,
        embedding_dir=embedding.output_dir,
        embedding_dim=embedding.embedding_dim,
        quantization_scale=embedding.quantization_scale,
        source_key=embedding.source_key,
        centroids_uri=centroids_uri,
        lookup_uris=lookup_uris,
        batch_size=scale.assign_batch_size,
        worker_resources=scale.pool.worker,
        max_workers=scale.pool.n_workers,
    )


@dataclass(frozen=True)
class DatakitSteps:
    """Result of :func:`reference_datakit_steps`."""

    sources: dict[str, StepSpec]
    """Echo of the input sources mapping (``{name: normalize_step}``)."""

    output_buckets: StepSpec
    """Final store StepSpec. Its ``output_path`` is the per-(cluster, quality)
    bucket directory the downstream training mixture reads from."""

    all_steps: list[StepSpec]
    """Every StepSpec the runner needs (shared upstream, per-source, dedup, store)."""

    shared_pool_steps: list[StepSpec]
    """The steps that execute on the shared Zephyr pool when one is supplied."""


@dataclass(frozen=True)
class DecontaminationSteps:
    """Stages for a restartable decontamination run."""

    items: StepSpec
    keys: StepSpec
    scans: dict[str, StepSpec]
    scores: dict[str, StepSpec]
    marks: dict[str, StepSpec]
    report: StepSpec

    @property
    def shared_pool_steps(self) -> list[StepSpec]:
        """The stages that receive the shared Zephyr context; scoring runs on its own workers."""
        return [self.keys, *self.scans.values(), *self.marks.values()]


@dataclass(frozen=True)
class ZephyrDatakitSteps:
    """Storage-backed Datakit stages implemented as Zephyr pipelines."""

    exact_dedup: StepSpec
    tokenize: dict[str, StepSpec]
    minhash: dict[str, StepSpec]
    fuzzy_dedup: StepSpec

    @property
    def shared_pool_steps(self) -> list[StepSpec]:
        """The stages that receive the shared Zephyr context."""
        return [self.exact_dedup, *self.tokenize.values(), *self.minhash.values(), self.fuzzy_dedup]


def zephyr_datakit_steps(
    sources: dict[str, StepSpec],
    scale: PipelineScale = DEFAULT_SCALE,
    zephyr_context: ZephyrContext | None = None,
    output_prefix: str | None = None,
) -> ZephyrDatakitSteps:
    """Build exact-dedup, tokenize, MinHash, and fuzzy-dedup stages.

    ``output_prefix`` roots every stage output in place of ``MARIN_PREFIX``.
    """
    source_names = sorted(sources)
    worker_resources = scale.pool.task if zephyr_context is not None else scale.pool.worker
    exact_dedup = StepSpec(
        name="datakit/global_exact_dedup",
        output_path_prefix=output_prefix,
        deps=[sources[name] for name in source_names],
        hash_attrs={"sources": source_names, "v": GLOBAL_EXACT_DEDUP_DATA_VERSION},
        fn=lambda output_path: global_exact_deduplicate(
            sources={name: read_artifact(sources[name].output_path, NormalizedData) for name in source_names},
            output_path=output_path,
            worker_resources=worker_resources,
            max_workers=scale.pool.n_workers,
            zephyr_context=zephyr_context,
        ),
    )

    mh = scale.minhash
    tokenize_steps: dict[str, StepSpec] = {}
    minhash_steps: dict[str, StepSpec] = {}
    for name, normalize_step in sources.items():
        tokenize_steps[name] = tokenize_attributes_step(
            name=f"datakit/tokenize/{name}",
            output_path_prefix=output_prefix,
            train_normalize=normalize_step,
            tokenizer=TOKENIZER,
            tokenizer_backend=TOKENIZER_BACKEND,
            tokenizer_revision=TOKENIZER_REVISION,
            max_workers=scale.pool.n_workers,
            worker_resources=worker_resources,
            zephyr_context=zephyr_context,
        )
        minhash_steps[name] = StepSpec(
            name=f"datakit/minhash/{name}",
            output_path_prefix=output_prefix,
            deps=[normalize_step],
            hash_attrs={
                "num_perms": mh.num_perms,
                "num_bands": mh.num_bands,
                "ngram_size": mh.ngram_size,
                "text_cap_chars": mh.text_cap_chars,
                "seed": mh.seed,
                "v": MINHASH_ATTR_DATA_VERSION,
            },
            fn=lambda output_path, n=normalize_step: compute_minhash_attrs(
                source=read_artifact(n.output_path, NormalizedData),
                output_path=output_path,
                num_perms=mh.num_perms,
                num_bands=mh.num_bands,
                ngram_size=mh.ngram_size,
                text_cap_chars=mh.text_cap_chars,
                seed=mh.seed,
                worker_resources=worker_resources,
                zephyr_context=zephyr_context,
            ),
        )

    fuzzy_dedup = StepSpec(
        name="datakit/dedup",
        output_path_prefix=output_prefix,
        deps=list(minhash_steps.values()),
        hash_attrs={"v": FUZZY_DUPS_ATTR_DATA_VERSION},
        fn=lambda output_path: compute_fuzzy_dups_attrs(
            inputs=[read_artifact(step.output_path, MinHashAttrData) for step in minhash_steps.values()],
            output_path=output_path,
            max_parallelism=scale.dedup_max_parallelism,
            cc_resume=True,
            worker_resources=worker_resources,
            zephyr_context=zephyr_context,
        ),
    )
    return ZephyrDatakitSteps(
        exact_dedup=exact_dedup,
        tokenize=tokenize_steps,
        minhash=minhash_steps,
        fuzzy_dedup=fuzzy_dedup,
    )


def decontamination_steps(
    sources: dict[str, StepSpec],
    *,
    scale: PipelineScale = DEFAULT_SCALE,
    zephyr_context: ZephyrContext | None = None,
    mark_source_names: list[str] | None = None,
    output_prefix: str | None = None,
) -> DecontaminationSteps:
    """Build decontamination stages with one full-source preparation.

    ``mark_source_names`` selects marks without changing the source set used to
    choose the shared keys. Thus, a subset mark has the same identity as its
    mark in the complete reference graph.
    """
    selected_names = set(sources if mark_source_names is None else mark_source_names)
    unknown_names = selected_names - sources.keys()
    if unknown_names:
        raise ValueError(f"unknown mark sources: {sorted(unknown_names)}")

    worker_resources = scale.pool.task if zephyr_context is not None else scale.pool.worker
    items = prepared_items_step()
    keys = decon_keys_step(
        name="datakit/decontam/_keys",
        items=items,
        sources=sources,
        shards_per_source=REFERENCE_SHARDS_PER_SOURCE,
        seed=REFERENCE_SEED,
        worker_resources=worker_resources,
        max_workers=scale.pool.n_workers,
        zephyr_context=zephyr_context,
        output_path_prefix=output_prefix,
    )
    selected = {name: step for name, step in sources.items() if name in selected_names}
    scans = {
        name: decon_scan_step(
            name=f"datakit/decontam/scan/{name}",
            normalized=normalize_step,
            keys=keys,
            worker_resources=worker_resources,
            max_workers=scale.pool.n_workers,
            zephyr_context=zephyr_context,
            output_path_prefix=output_prefix,
        )
        for name, normalize_step in selected.items()
    }
    scores = {
        name: decon_score_step(
            name=f"datakit/decontam/score/{name}",
            scan=scans[name],
            keys=keys,
            scorer=decon_models(),
            score_workers=scale.decon_score,
            output_path_prefix=output_prefix,
        )
        for name in selected
    }
    marks = {
        name: decon_mark_step(
            name=f"datakit/decontam/{name}",
            normalized=normalize_step,
            scores=scores[name],
            cutoff=CUTOFF,
            flagged_sample_size=FLAGGED_SAMPLE_SIZE,
            worker_resources=worker_resources,
            max_workers=scale.pool.n_workers,
            zephyr_context=zephyr_context,
            output_path_prefix=output_prefix,
        )
        for name, normalize_step in selected.items()
    }
    report = StepSpec(
        name="datakit/report/decontam",
        output_path_prefix=output_prefix,
        deps=list(marks.values()),
        hash_attrs={"v": 1},
        fn=lambda output_path: decontam_report(
            output_path,
            {name: read_artifact(step.output_path, DeconAttributes) for name, step in marks.items()},
        ),
    )
    return DecontaminationSteps(items=items, keys=keys, scans=scans, scores=scores, marks=marks, report=report)


def reference_datakit_steps(
    sources: dict[str, StepSpec],
    *,
    quality_model: str,
    quality_model_version: str | None = None,
    domain_centroids: str | StepSpec | None = None,
    centroids_version: str | None = None,
    scale: PipelineScale = DEFAULT_SCALE,
    zephyr_context: ZephyrContext | None = None,
    output_prefix: str | None = None,
) -> DatakitSteps:
    """Build the reference Datakit DAG over the given normalize steps.

    Every step's output lands at ``<output_prefix>/<step_name>_<hash>/``.
    ``output_prefix`` defaults to ``MARIN_PREFIX``, so changing the deploy region
    is just a matter of changing ``MARIN_PREFIX``. Inputs such as the eval corpus
    always resolve against ``MARIN_PREFIX``.

    Args:
        sources: ``{name: normalize_step}``. Each step must produce a
            :class:`marin.datakit.normalize.NormalizedData` artifact;
            misuse fails loudly the first time a downstream step tries
            ``read_artifact(step.output_path, NormalizedData)``.
        quality_model: Directory holding the pooled fast-transformer scorer
            artifacts plus the calibration json (immutable by convention).
        quality_model_version: Required. A stable tag identifying the model bytes
            (e.g. ``'pooled-junkgate2-marin'``) -- hashed into the quality step in place
            of the region-specific ``quality_model`` dir, so the same scorer
            resolves to one output path across regions.
        domain_centroids: A GCS directory holding ``centroids_<k_train>.npy``
            and ``lookup_<k_train>_to_<k>.npy`` for each ``k`` in
            ``scale.cluster.k_views``; a StepSpec whose ``output_path`` will
            contain that layout once it runs (see
            :func:`build_train_centroids_step`); or ``None`` to train inline
            from the per-source embeds. When training inline, ``scale.cluster.k_train``
            must not exceed the centroid sample size -- use a smaller K
            (e.g. ``SMOKE_SCALE``) on small source sets.
        centroids_version: Required when ``domain_centroids`` is a pre-staged
            path. A stable tag identifying the centroid bytes (e.g. the training
            run id) -- hashed into the assign step in place of the region-specific
            path so identical centroids resolve to one output path across regions.
            Ignored (and unneeded) for the StepSpec / inline-training case, whose
            identity comes from the step hash.
        scale: K / fan-out sizing plus the per-stage worker :class:`PoolConfig`.
            ``DEFAULT_SCALE`` is the production full-fleet shape; ``SMOKE_SCALE``
            runs the same DAG end-to-end on a testbed sample.
        zephyr_context: Optional shared context for subprocess-compatible stages.
        output_prefix: Root for every step output, for example a temporary
            prefix below ``MARIN_PREFIX``. ``None`` uses ``MARIN_PREFIX``.
    """
    cluster = scale.cluster
    fuzzy = scale.fuzzy
    if fuzzy.plan.minimum_size > fuzzy.text.max_cluster_size:
        raise ValueError(
            f"Fuzzy cluster plan minimum_size ({fuzzy.plan.minimum_size}) exceeds "
            f"text max_cluster_size ({fuzzy.text.max_cluster_size})"
        )
    unknown_exempt = set(scale.store.fuzzy_exempt_sources) - (all_sources().keys() | sources.keys())
    if unknown_exempt:
        raise ValueError(f"Unknown fuzzy-exempt sources: {sorted(unknown_exempt)!r}")
    zephyr_steps = zephyr_datakit_steps(sources, scale, zephyr_context, output_prefix)
    exact_dedup = zephyr_steps.exact_dedup
    embed_steps = build_per_source_embed_steps(sources, scale, output_prefix)
    if domain_centroids is None:
        domain_centroids = build_train_centroids_step(embed_steps, scale, output_prefix)

    centroids_uri, lookup_uris, centroids_deps, centroids_hash = _resolve_centroids(
        domain_centroids, cluster, centroids_version
    )
    quality_model_hash = _resolve_quality_model_version(quality_model, quality_model_version)
    decontamination = decontamination_steps(
        sources, scale=scale, zephyr_context=zephyr_context, output_prefix=output_prefix
    )

    # ---- Per-source steps ------------------------------------------------------
    per_source: dict[str, dict[str, StepSpec]] = {}

    for name, normalize_step in sources.items():
        embed = embed_steps[name]
        tokenize = zephyr_steps.tokenize[name]

        # Domain assign: consumes the embed + the (given or trained) centroids.
        # ``centroids_hash`` feeds hash_attrs so re-pointing at a new model
        # invalidates already-assigned outputs.
        assign = StepSpec(
            name=f"datakit/cluster_assign/{name}",
            output_path_prefix=output_prefix,
            deps=[embed, *centroids_deps],
            hash_attrs=assign_hash_attrs(centroids_hash, cluster.k_train, cluster.k_views, scale.assign_batch_size),
            fn=remote(
                lambda output_path, ep=embed.output_path: _assign_embedding(
                    output_path=output_path,
                    embed_path=ep,
                    centroids_uri=centroids_uri,
                    lookup_uris=lookup_uris,
                    scale=scale,
                ),
                resources=DRIVER_RESOURCES,
                pip_dependency_groups=["datakit"],
            ),
        )

        quality = StepSpec(
            name=f"datakit/quality/{name}",
            output_path_prefix=output_prefix,
            deps=[normalize_step, tokenize],
            hash_attrs={"model_version": quality_model_hash, "v": 2},
            fn=remote(
                lambda output_path, np=normalize_step.output_path, tp=tokenize.output_path, src=name: score_normalized(
                    output_path=output_path,
                    normalized=read_artifact(np, NormalizedData),
                    tokenized=read_artifact(tp, TokenizedAttrData),
                    source=src,
                    model_dir=quality_model,
                    split=SPLIT,
                    max_workers=scale.pool.n_workers,
                    worker_resources=scale.pool.worker,
                ),
                resources=DRIVER_RESOURCES,
            ),
        )

        decontam = decontamination.marks[name]

        minhash = zephyr_steps.minhash[name]

        per_source[name] = {
            "tokenize": tokenize,
            "embed": embed,
            "assign": assign,
            "quality": quality,
            "decontam": decontam,
            "minhash": minhash,
        }

    dedup = zephyr_steps.fuzzy_dedup

    cluster_plan = large_clusters_step(
        name="datakit/large_fuzzy_clusters",
        output_path_prefix=output_prefix,
        candidates=dedup,
        params=fuzzy.plan,
        max_workers=fuzzy.max_workers,
        worker_resources=fuzzy.worker,
        task_resources=fuzzy.map_task,
    )
    cluster_text = cluster_text_step(
        name="datakit/fuzzy_cluster_text",
        output_path_prefix=output_prefix,
        plan=cluster_plan,
        params=fuzzy.text,
        max_workers=fuzzy.max_workers,
        worker_resources=fuzzy.worker,
        map_task_resources=fuzzy.map_task,
        reduce_task_resources=fuzzy.reduce_task,
    )
    verified_dedup = cluster_verify_step(
        name="datakit/verify_fuzzy_clusters",
        output_path_prefix=output_prefix,
        cluster_text=cluster_text,
        params=fuzzy.rule,
        limits=fuzzy.limits,
        max_workers=fuzzy.max_workers,
        worker_resources=fuzzy.worker,
        map_task_resources=fuzzy.map_task,
        reduce_task_resources=fuzzy.reduce_task,
    )

    # ---- Final store: attribute join + per-bucket Levanter cache ---------------
    def _store_fn(output_path: str) -> ClusteredStoreData:
        return build_clustered_store(
            tokenize={n: read_artifact(s["tokenize"].output_path, TokenizedAttrData) for n, s in per_source.items()},
            decontam={n: read_artifact(s["decontam"].output_path, DeconAttributes) for n, s in per_source.items()},
            cluster_assign={
                n: read_artifact(s["assign"].output_path, AssignmentAttrData) for n, s in per_source.items()
            },
            quality={n: read_artifact(s["quality"].output_path, QualityScores) for n, s in per_source.items()},
            exact_dedup=read_artifact(exact_dedup.output_path, GlobalExactDedupData),
            dedup=read_artifact(verified_dedup.output_path, ClusterVerifiedFuzzyDupsAttrData),
            output_path=output_path,
            cluster_view=cluster.cluster_view,
            split=SPLIT,
            fuzzy_exempt_sources=frozenset(scale.store.fuzzy_exempt_sources) & per_source.keys(),
            worker_resources=scale.store.worker,
            max_workers=scale.pool.n_workers,
            task_count=scale.store.task_count,
            partition_processes=scale.store.partition_processes,
            max_parallel_bucket_writes=scale.store.max_parallel_bucket_writes,
        )

    store_deps: list[StepSpec] = []
    for s in per_source.values():
        store_deps += [s["tokenize"], s["decontam"], s["assign"], s["quality"]]
    store_deps += [exact_dedup, verified_dedup]

    # Task partitioning determines the number and contents of leaf caches.
    # Tokenizer and quality bucket edges are already captured by dependencies.
    store = StepSpec(
        name="datakit/store",
        output_path_prefix=output_prefix,
        deps=store_deps,
        hash_attrs={
            "cluster_view": cluster.cluster_view,
            "split": SPLIT,
            "task_count": scale.store.task_count,
            "fuzzy_exempt_sources": sorted(scale.store.fuzzy_exempt_sources),
            "v": 4,
        },
        fn=_store_fn,
    )

    # ---- Per-stage reports --------------------------------------------------
    # One single-page HTML per stage, aggregated across sources from the stage's
    # site/sample outputs and counters. Plain callables: bounded reads, run
    # inline in the driver. Bump "v" to regenerate reports over cached stage outputs.
    normalize_paths = {n: s.output_path for n, s in sources.items()}
    tokenize_paths = {n: s["tokenize"].output_path for n, s in per_source.items()}
    quality_paths = {n: s["quality"].output_path for n, s in per_source.items()}
    assign_paths = {n: s["assign"].output_path for n, s in per_source.items()}
    reports = [
        StepSpec(
            name="datakit/report/normalize",
            output_path_prefix=output_prefix,
            deps=list(sources.values()),
            hash_attrs={"v": 2},
            fn=lambda op: normalize_report(
                op, {n: read_artifact(p, NormalizedData) for n, p in normalize_paths.items()}
            ),
        ),
        StepSpec(
            name="datakit/report/tokenize",
            output_path_prefix=output_prefix,
            deps=[s["tokenize"] for s in per_source.values()],
            hash_attrs={"v": 1, "split": SPLIT},
            fn=lambda op: tokenize_report(
                op, {n: read_artifact(p, TokenizedAttrData) for n, p in tokenize_paths.items()}, SPLIT
            ),
        ),
        StepSpec(
            name="datakit/report/quality",
            output_path_prefix=output_prefix,
            deps=[*sources.values(), *(s["quality"] for s in per_source.values())],
            hash_attrs={"v": 2},
            fn=lambda op: quality_report(
                op,
                {n: read_artifact(p, QualityScores) for n, p in quality_paths.items()},
                {n: read_artifact(normalize_paths[n], NormalizedData) for n in quality_paths},
            ),
        ),
        StepSpec(
            name="datakit/report/domain",
            output_path_prefix=output_prefix,
            deps=[s["assign"] for s in per_source.values()],
            hash_attrs={"v": 1, "cluster_view": cluster.cluster_view},
            fn=lambda op: assign_report(
                op, {n: read_artifact(p, AssignmentAttrData) for n, p in assign_paths.items()}, cluster.cluster_view
            ),
        ),
        decontamination.report,
        StepSpec(
            name="datakit/report/dedup",
            output_path_prefix=output_prefix,
            deps=[dedup, verified_dedup],
            hash_attrs={"v": 2},
            fn=lambda op: cluster_dedup_report(
                op,
                read_artifact(dedup.output_path, FuzzyDupsAttrData),
                read_artifact(verified_dedup.output_path, ClusterVerifiedFuzzyDupsAttrData),
            ),
        ),
        StepSpec(
            name="datakit/report/store",
            output_path_prefix=output_prefix,
            deps=[store],
            hash_attrs={"v": 1},
            fn=lambda op: store_report(op, read_artifact(store.output_path, ClusteredStoreData)),
        ),
    ]

    all_steps: list[StepSpec] = [
        exact_dedup,
        decontamination.items,
        decontamination.keys,
        *decontamination.scans.values(),
        *decontamination.scores.values(),
    ]
    if isinstance(domain_centroids, StepSpec):
        all_steps.append(domain_centroids)
    for s in per_source.values():
        all_steps += list(s.values())
    all_steps += [dedup, cluster_plan, cluster_text, verified_dedup, store, *reports]
    return DatakitSteps(
        sources=sources,
        output_buckets=store,
        all_steps=all_steps,
        shared_pool_steps=zephyr_steps.shared_pool_steps + decontamination.shared_pool_steps,
    )


SAMPLE_PREFIX = "s3://marin-us-east-02a/marin/datakit/sample_0.1b_7d7d8fd7"

QUALITY_MODEL = "datakit/models/quality/pooled_junkgate2_marin"
"""Pooled fast-transformer scorer directory, relative to ``MARIN_PREFIX``."""

QUALITY_MODEL_VERSION = "pooled-junkgate2-marin"
"""Identity tag for the bytes at :data:`QUALITY_MODEL`."""


def shared_zephyr_context(scale: PipelineScale, name: str = "datakit-reference") -> ZephyrContext:
    """Return the worker pool that the subprocess-compatible stages share."""
    return ZephyrContext(
        name=name,
        resources=scale.pool.worker,
        coordinator_resources=scale.pool.coordinator,
        max_workers=scale.pool.n_workers,
        stage_runner_factory=SubprocessRunner,
    )


def require_prebuilt_inputs(steps: Iterable[StepSpec]) -> None:
    """Fail before a run starts when a pre-built input (a step without a function, such as the prepared item
    table or a registered sample source) is missing; the runner would fail only when it reached the step."""
    missing = [f"{step.name} at {step.output_path}" for step in steps if step.fn is None and not step_is_built(step)]
    if missing:
        raise RuntimeError(
            f"pre-built inputs are missing: {missing}. The decontamination item table is prepared with "
            "`python -m experiments.datakit.decontam.items` (see its docstring)."
        )


def run_reference_steps(steps: DatakitSteps, zephyr_context: ZephyrContext, *, max_concurrent: int) -> None:
    """Run the reference DAG and hold the shared pool only while its stages need it.

    The shared pool keeps every worker that it gets until it stops. Thus it starts
    after the normalize steps succeed and stops when every shared-pool stage
    succeeds. The dedicated normalize, embed, quality, fuzzy-cluster, and store
    pools then do not wait for hosts that idle shared workers hold.
    """
    require_prebuilt_inputs(steps.all_steps)
    runner = StepRunner()
    runner.run(steps.sources.values(), max_concurrent=max_concurrent)
    stop = threading.Event()
    with zephyr_context, ThreadPoolExecutor(max_workers=1, thread_name_prefix="datakit-shared-pool-release") as watcher:
        # A failed watcher leaves the pool to hold its workers until the DAG ends.
        # ``result()`` sends the error to the caller, so the run does not pass silently.
        release = watcher.submit(_release_shared_pool, zephyr_context, steps.shared_pool_steps, stop)
        try:
            runner.run(steps.all_steps, max_concurrent=max_concurrent)
        finally:
            stop.set()
        release.result()


def _release_shared_pool(zephyr_context: ZephyrContext, pool_steps: list[StepSpec], stop: threading.Event) -> None:
    """Stop the shared pool when every step in ``pool_steps`` has succeeded."""
    remaining = list(pool_steps)
    while not stop.wait(SHARED_POOL_RELEASE_POLL_INTERVAL):
        remaining = [step for step in remaining if not step_is_built(step)]
        if not remaining:
            logger.info("All %d shared-pool stages succeeded; stopping the shared Zephyr pool", len(pool_steps))
            zephyr_context.shutdown()
            return


def quality_model_path() -> str:
    """Resolve :data:`QUALITY_MODEL` against the active cluster prefix.

    Kept a call, not a module constant, so importing this module never reads the
    environment: a caller that sets ``MARIN_PREFIX`` after import still gets the
    prefix it asked for.
    """
    return f"{marin_prefix()}/{QUALITY_MODEL}"


# A content-diverse subset of a testbed sample (wiki / academic / reference /
# code / math / multilingual / web / sft / agent-trajectory), small enough for a
# quick end-to-end run. nsf_awards is a 3-row edge case on purpose.
SAMPLE_SOURCES = (
    "cp/wikiteam",
    "cp/arxiv_abstracts",
    "nsf_awards",
    "starcoder2/ir_python",
    "numinamath-1.5",
    "finepdfs/spa_Latn",
    "nemotron_cc_v2/medium_quality",
    "nemotron_sft/sft_general",
    "hplt_v3",
    "swe-rebench-openhands",
)


def sample_sources(sample_prefix: str, names: list[str] | None = None, run_tag: str = "") -> dict[str, StepSpec]:
    """Map testbed-sample source names to StepSpecs registered on their existing dirs.

    ``sample`` mode: each source is a completed normalize-step output in the
    sample tree, so the returned steps are already ``SUCCESS`` on storage; the
    runner skips them and downstream steps read their ``NormalizedData``. The
    sample id lives in the step *name* so downstream hashes re-key per sample.
    ``None`` discovers every source (an ``.artifact.json`` at source depth 1-3).

    ``run_tag`` enters each source step's ``hash_attrs`` (not its output path, which
    stays the shared sample dir), so a fresh tag re-keys the whole downstream DAG --
    every stage recomputes on the *same* input, for benchmarking a from-scratch run.
    """
    prefix = sample_prefix.rstrip("/")
    sample_id = posixpath.basename(prefix)
    discovered = [
        str(m)[len(prefix) + 1 : -len("/.artifact.json")]
        for depth in ("*", "*/*", "*/*/*")
        for m in StoragePath(f"{prefix}/{depth}/.artifact.json").glob()
    ]
    if names is None:
        names = discovered
    else:
        unknown = sorted(set(names) - set(discovered))
        if unknown:
            raise KeyError(f"sources {unknown} not in {sample_id}; known: {sorted(discovered)}")
    logger.info(
        "sample_sources: %d sources from %s%s", len(names), sample_id, f" (run_tag={run_tag})" if run_tag else ""
    )
    tag_attrs = {"run_tag": run_tag} if run_tag else {}
    return {
        name: StepSpec(
            name=f"datakit/sample/{sample_id}/{name}",
            override_output_path=f"{prefix}/{name}",
            hash_attrs=tag_attrs,
        )
        for name in sorted(names)
    }


def _select_pipeline_sources(args: argparse.Namespace) -> dict[str, StepSpec]:
    """Build the ``{name: normalize_step}`` mapping for the chosen mode."""
    if args.mode == "sample":
        default = ",".join(SAMPLE_SOURCES)
        names = None if args.sources == "all" else [s.strip() for s in (args.sources or default).split(",") if s.strip()]
        return sample_sources(args.sample_prefix, names, args.run_tag)
    names = None if args.sources in (None, "all") else [s.strip() for s in args.sources.split(",") if s.strip()]
    return select_sources(names)


def _apply_pool_overrides(scale: PipelineScale, args: argparse.Namespace) -> PipelineScale:
    """Override the scale's worker fleet from ``--pool-*`` flags."""
    worker_device = (
        ResourceConfig.with_gpu(args.pool_gpu).device if args.pool_gpu is not None else scale.pool.worker.device
    )
    worker = replace(
        scale.pool.worker,
        device=worker_device,
        **{k: v for k, v in (("cpu", args.pool_cpu), ("ram", args.pool_ram), ("disk", args.pool_disk)) if v is not None},
    )
    task = replace(
        scale.pool.task,
        **{
            k: v
            for k, v in (
                ("cpu", args.pool_task_cpu),
                ("ram", args.pool_task_ram),
                ("disk", args.pool_task_disk),
            )
            if v is not None
        },
    )
    coordinator = replace(
        scale.pool.coordinator,
        **{
            k: v
            for k, v in (
                ("cpu", args.pool_coordinator_cpu),
                ("ram", args.pool_coordinator_ram),
            )
            if v is not None
        },
    )
    n_workers = args.pool_workers if args.pool_workers is not None else scale.pool.n_workers
    return replace(
        scale,
        pool=replace(scale.pool, n_workers=n_workers, worker=worker, task=task, coordinator=coordinator),
    )


def _require_built_steps(steps: dict[str, StepSpec], label: str) -> None:
    missing = [name for name, step in steps.items() if not step_is_built(step)]
    if not missing:
        return
    shown = ", ".join(missing[:20])
    remainder = f" and {len(missing) - 20} more" if len(missing) > 20 else ""
    raise RuntimeError(f"missing {label} for {len(missing)} sources: {shown}{remainder}")


def _decontamination_target_steps(result: DecontaminationSteps, target: str) -> list[StepSpec]:
    if target == "decon-keys":
        require_prebuilt_inputs([result.items])
        return [result.keys]
    if target == "decon-mark":
        missing = [step.name for step in (result.items, result.keys) if not step_is_built(step)]
        if missing:
            raise RuntimeError(f"decontamination preparation is not complete: {', '.join(missing)}")
        return list(result.marks.values())
    if target == "decon-report":
        _require_built_steps(result.marks, "decontamination marks")
        return [result.report]
    raise ValueError(f"unknown decontamination target: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("full", "sample"),
        default="full",
        help="full: registry sources, K=5000. sample: a pre-built testbed sample (see --sample-prefix), K=64.",
    )
    parser.add_argument(
        "--target",
        choices=("all", "decon-keys", "decon-mark", "decon-report"),
        default="all",
        help=(
            "all: complete reference DAG. decon-keys: item keys over reference "
            "shards. decon-mark: per-source marks from completed preparation stages. decon-report: report "
            "from completed marks."
        ),
    )
    parser.add_argument("--sample-prefix", default=SAMPLE_PREFIX, help="testbed sample root (--mode sample)")
    parser.add_argument(
        "--quality-model", default=quality_model_path(), help="pooled fast-transformer scorer + calib dir"
    )
    parser.add_argument(
        "--domain-centroids",
        default=None,
        help="dir with centroids_<K>.npy + lookup_<K>_to_<k>.npy. Omit to train centroids inline from the embeds.",
    )
    parser.add_argument(
        "--domain-centroids-version",
        default=None,
        help=(
            "Stable identity tag for --domain-centroids (e.g. the training run id). "
            "Required with --domain-centroids: hashed in place of the region-specific "
            "path so identical centroids resolve to one output path across regions."
        ),
    )
    parser.add_argument(
        "--quality-model-version",
        help=(
            "Stable identity tag for --quality-model (e.g. 'pooled-junkgate2-marin'). Required for --target all. "
            "Hashed in place of the region-specific model dir for cross-region reproducibility."
        ),
    )
    parser.add_argument(
        "--sources",
        default=None,
        help="comma-separated source names, or 'all' for every source. Omit: full=all, sample=curated subset.",
    )
    parser.add_argument(
        "--mark-sources",
        default=None,
        help="comma-separated sources to mark from the full --sources preparation, or 'all'; decon-mark/report only",
    )
    parser.add_argument("--pool-workers", type=int, default=None, help="Zephyr worker count (override scale)")
    parser.add_argument("--pool-cpu", type=float, default=None, help="per-worker CPUs (override scale)")
    parser.add_argument("--pool-ram", default=None, help="per-worker RAM, e.g. 16g (override scale)")
    parser.add_argument("--pool-disk", default=None, help="per-worker disk, e.g. 16g (override scale)")
    parser.add_argument("--pool-gpu", default=None, help="place shared-pool workers on nodes with one GPU of this type")
    parser.add_argument("--pool-task-cpu", type=float, default=None, help="CPUs for each shared-pool task")
    parser.add_argument("--pool-task-ram", default=None, help="RAM for each shared-pool task, e.g. 32g")
    parser.add_argument("--pool-task-disk", default=None, help="disk for each shared-pool task, e.g. 16g")
    parser.add_argument("--pool-coordinator-cpu", type=float, default=None, help="CPUs for the pool coordinator")
    parser.add_argument("--pool-coordinator-ram", default=None, help="RAM for the pool coordinator, e.g. 8g")
    parser.add_argument("--max-concurrent", type=int, default=8, metavar="N", help="max steps StepRunner runs at once")
    parser.add_argument(
        "--run-tag",
        default="",
        help="sample mode: mix this into every step's hash so the whole DAG recomputes fresh (benchmarking)",
    )
    args = parser.parse_args()

    configure_logging(logging.INFO)

    if args.target == "all" and not args.quality_model_version:
        parser.error("--quality-model-version is required with --target all")
    if args.mark_sources is not None and args.target not in ("decon-mark", "decon-report"):
        parser.error("--mark-sources can only be used with --target decon-mark or decon-report")

    scale = _apply_pool_overrides(SMOKE_SCALE if args.mode == "sample" else DEFAULT_SCALE, args)
    sources = _select_pipeline_sources(args)
    if args.target in ("decon-keys", "decon-mark"):
        _require_built_steps(sources, "normalized artifacts")

    mark_source_names = None
    if args.mark_sources not in (None, "all"):
        mark_source_names = [name.strip() for name in args.mark_sources.split(",") if name.strip()]
        if not mark_source_names:
            parser.error("--mark-sources must name at least one source or use 'all'")

    if args.target == "all":
        zephyr_context = shared_zephyr_context(scale)
        result = reference_datakit_steps(
            sources,
            quality_model=args.quality_model,
            quality_model_version=args.quality_model_version,
            domain_centroids=args.domain_centroids,
            centroids_version=args.domain_centroids_version,
            scale=scale,
            zephyr_context=zephyr_context,
        )
        run_reference_steps(result, zephyr_context, max_concurrent=args.max_concurrent)
        return

    zephyr_context = None if args.target == "decon-report" else shared_zephyr_context(scale)
    decontamination = decontamination_steps(
        sources,
        scale=scale,
        zephyr_context=zephyr_context,
        mark_source_names=mark_source_names,
    )
    target_steps = _decontamination_target_steps(decontamination, args.target)
    with zephyr_context if zephyr_context is not None else nullcontext():
        StepRunner().run(target_steps, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
