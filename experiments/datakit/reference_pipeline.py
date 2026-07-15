# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end reference DAG: Datakit sources → (cluster x quality) store.

This wires the existing per-stage building blocks into a single
StepRunner-walkable graph. Two modes (``--mode``), same DAG:

- ``full``: sources from :func:`marin.datakit.sources.all_sources`, K=5000.
- ``sample``: a pre-built testbed sample registered as already-normalized
  sources (``--sample-prefix``), K=64 -- a true end-to-end run on real data.

Per source::

    normalize → tokenize
              → embed (luxical-one)   → assign (domain v0, given centroids)
              → quality                (pooled fast-transformer, given model dir)
              → decontam               (shared eval bloom)
              → minhash

Then:
    fuzzy_dups([<minhash per source>])
    build_clustered_store(tokenize, decontam, cluster_assign, quality, dedup)
    one ``datakit/report/<stage>`` step per stage -- a single self-contained
    HTML page built from that stage's counters + site/sample outputs
    (:mod:`experiments.datakit.reports`)

Every stage keeps one step per source with its own output dir; only dedup and
the store combine sources, by design.

Worker fleet: every stage runs its pipeline on its own dedicated Zephyr
coordinator + workers (vanilla ``ZephyrContext``, built inside the stage
functions), sized by one :class:`PoolConfig` (``n_workers`` x ``worker``) shared
across the stages -- so the *only* resource knob is that fleet.
``--max-concurrent`` bounds how many stages the StepRunner walks at once.

Public API: :func:`reference_datakit_steps`. Pass ``sources`` (a ``{name:
normalize_step}`` mapping), a ``quality_model`` dir, and optionally pre-staged
domain centroids (``None`` trains them inline).

Region-agnostic: worker sizing is one :class:`PoolConfig`. ``MARIN_PREFIX`` is
resolved by :func:`rigging.filesystem.marin_prefix` -- unset (the normal iris-
worker case) it falls back to the in-region bucket, so source artifacts, the
eval corpus (``EVAL_ROOT``), and every output land in-region. Override via
``iris job run -e MARIN_PREFIX <bucket>``.

Submit the sample-mode end-to-end run on iris::

    uv run iris --cluster=cw-rno2a job run --priority interactive --cpu 2 --memory 8GB \\
        --enable-extra-resources -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
        -- python -m experiments.datakit.reference_pipeline \\
            --mode sample --sample-prefix s3://.../datakit/sample_100b_8ae7a94f \\
            --sources all --pool-workers 512
"""

import argparse
import logging
import posixpath
from dataclasses import dataclass, field, replace

from fray.types import ResourceConfig
from levanter.tokenizers import TokenizerBackend
from marin.datakit.decon import (
    DeconAttributes,
    build_eval_bloom_step,
    decon_step,
)
from marin.datakit.normalize import NormalizedData
from marin.datakit.sources import all_sources
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.fuzzy_dups import (
    FuzzyDupsAttrData,
    compute_fuzzy_dups_attrs,
)
from marin.processing.classification.deduplication.fuzzy_minhash import (
    MinHashAttrData,
    compute_minhash_attrs,
)
from marin.processing.tokenize.attributes import (
    TokenizedAttrData,
    tokenize_attributes_step,
)
from rigging.filesystem import StoragePath, marin_prefix
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.domain.v0.assign import (
    AssignmentAttrData,
    assign_source,
)
from experiments.datakit.cluster.domain.v0.sample import sample_centroid_inputs
from experiments.datakit.cluster.domain.v0.train import train_centroids
from experiments.datakit.cluster.quality.fast_transformer.artifact import QualityScores
from experiments.datakit.cluster.quality.fast_transformer.score import score_normalized
from experiments.datakit.decontam.prepare_eval_corpus import DECON_EXCLUDED_EVAL_TASKS
from experiments.datakit.embeddings.luxical.pipeline import (
    LUXICAL_REPO,
    LUXICAL_WEIGHTS_FILE,
    EmbeddingAttrData,
    embed_source,
)
from experiments.datakit.reports.decontam import decontam_report
from experiments.datakit.reports.dedup import dedup_report
from experiments.datakit.reports.domain import assign_report
from experiments.datakit.reports.normalize import normalize_report
from experiments.datakit.reports.quality import quality_report
from experiments.datakit.reports.store import store_report
from experiments.datakit.reports.tokenize import tokenize_report
from experiments.datakit.store.datakit_store import (
    ClusteredStoreData,
    build_clustered_store,
)

logger = logging.getLogger(__name__)


# Tokenize: canonical Marin tokenizer. Not scale-sensitive.
TOKENIZER = "marin-community/marin-tokenizer"
TOKENIZER_BACKEND = TokenizerBackend.HF
SPLIT = "train"

# Decontam. The combined eval corpus written by decontam/prepare_eval_corpus.py,
# staged per-region (``aa/<eval>/<split>.parquet`` + ``lmh/<task>/eval.parquet``).
EVAL_ROOT = f"{marin_prefix()}/datakit/decontam/evals"
# Bloom capacity -- unique ngram hashes the filter must hold: ~21.78M unique
# hashes across the AA + LMH corpus, with 2.3x headroom. At FPR=1e-9 this is a
# ~270 MB filter.
ESTIMATED_DOC_COUNT = 50_000_000
FALSE_POSITIVE_RATE = 1e-9
NGRAM_LENGTH = 13
OVERLAP_THRESHOLD = 0.5
# Contaminated docs reservoir-sampled per shard into the flagged side output
# the decontam stage report reads.
FLAGGED_SAMPLE_SIZE = 8


@dataclass(frozen=True)
class ClusterConfig:
    """Spherical-K-means knobs for the domain-clustering stage.

    ``cluster_view`` is the K the store partitions on (``cluster=<C>/quality=<Q>/``)
    and must be ``k_train`` or one of ``k_views`` -- the assign stage only
    materializes a ``cluster_<K>`` column for those. ``k_train`` must not exceed
    the centroid-training sample size, so shrink it for small inline runs.
    """

    k_train: int = 5000
    k_views: tuple[int, ...] = (40, 1000)
    cluster_view: int = 40

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
    """The dedicated Zephyr worker fleet each stage runs on.

    ``n_workers`` workers, each sized ``worker`` (cpu / ram / disk). Every stage
    spins up its own coordinator with this fleet and takes one whole worker per
    task, so ``n_workers`` is the per-stage task concurrency and ``worker`` must
    be large enough for the heaviest per-shard stage (embed model load, minhash /
    dedup / store buffers).
    """

    n_workers: int = 512
    worker: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=2, ram="16g", disk="16g"))


@dataclass(frozen=True)
class PipelineScale:
    """Non-resource sizing for :func:`reference_datakit_steps`.

    Worker CPU/RAM lives in one :class:`PoolConfig` (:attr:`pool`); the rest is
    content-shaping (cluster K, batch sizes, dedup fan-out). ``DEFAULT_SCALE`` is
    production K=5000; ``SMOKE_SCALE`` is K=64 for a quick end-to-end run.
    """

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)
    embed_batch_size: int = 4096
    assign_batch_size: int = 4096
    # Inline domain training: ~100 sources x 100k = ~10M-row centroid sample.
    n_per_source_for_sample: int = 100_000
    # Concurrent per-source sampler pipelines run inside the centroid-sample
    # stage's coordinator; kept modest so it isn't overwhelmed.
    sample_parallel_sources: int = 4
    dedup_max_parallelism: int = 4096
    store_shards_per_task: int = 1
    # Centroid training is single-process FAISS K-means, not a pool stage.
    train_centroids_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig.with_cpu(cpu=32, ram="64g"))


DEFAULT_SCALE = PipelineScale()
"""Production full-fleet sizing (every ``all_sources()`` entry, K=5000)."""

SMOKE_SCALE = PipelineScale(
    cluster=ClusterConfig(k_train=64, k_views=(8, 16), cluster_view=8),
    pool=PoolConfig(n_workers=16, worker=ResourceConfig(cpu=2, ram="8g", disk="8g")),
    n_per_source_for_sample=20_000,
    dedup_max_parallelism=64,
    train_centroids_resources=ResourceConfig.with_cpu(cpu=4, ram="8g"),
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


def default_sources() -> dict[str, StepSpec]:
    """Every ``all_sources()`` entry, mapped to its normalize StepSpec."""
    return select_sources(None)


def _build_embed_step(name: str, normalize_step: StepSpec, scale: PipelineScale) -> StepSpec:
    return StepSpec(
        name=f"datakit/embed/{name}",
        deps=[normalize_step],
        hash_attrs={
            "luxical_repo": LUXICAL_REPO,
            "luxical_weights": LUXICAL_WEIGHTS_FILE,
            "batch_size": scale.embed_batch_size,
            "v": 1,
        },
        fn=remote(
            lambda output_path, np=normalize_step.output_path: embed_source(
                output_path=output_path,
                normalized=read_artifact(np, NormalizedData),
                batch_size=scale.embed_batch_size,
                worker_resources=scale.pool.worker,
                max_workers=scale.pool.n_workers,
            ),
            resources=DRIVER_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )


def build_per_source_embed_steps(
    sources: dict[str, StepSpec], scale: PipelineScale = DEFAULT_SCALE
) -> dict[str, StepSpec]:
    """Build the Luxical embed StepSpec for each source.

    ``sources`` maps source name → normalize StepSpec; the embed step is built
    against that step as its dep. Exposed so callers that also want to build
    the domain training subgraph (via :func:`build_train_centroids_step`) can
    share the same embeds across both wirings.
    """
    return {name: _build_embed_step(name, step, scale) for name, step in sources.items()}


def build_train_centroids_step(embed_steps: dict[str, StepSpec], scale: PipelineScale = DEFAULT_SCALE) -> StepSpec:
    """Build the K-means training StepSpec for the domain centroids.

    The returned step's ``output_path`` contains ``centroids_<k_train>.npy``
    plus ``lookup_<k_train>_to_<k>.npy`` for each ``k`` in ``scale.cluster.k_views``
    -- the same layout :func:`reference_datakit_steps` consumes via its
    ``domain_centroids`` parameter when given a centroids path.
    """
    cluster = scale.cluster
    sample_step = StepSpec(
        name="datakit/cluster/sample_centroids",
        deps=list(embed_steps.values()),
        hash_attrs={"n_per_source": scale.n_per_source_for_sample, "format": "parquet", "v": 1},
        fn=remote(
            lambda output_path, es={n: s.output_path for n, s in embed_steps.items()}: sample_centroid_inputs(
                output_path=output_path,
                embeddings={n: read_artifact(p, EmbeddingAttrData) for n, p in es.items()},
                n_per_source=scale.n_per_source_for_sample,
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
        deps=[sample_step],
        hash_attrs={"k_train": cluster.k_train, "k_views": list(cluster.k_views), "n_threads": n_threads, "v": 1},
        fn=remote(
            lambda output_path, sp=sample_step.output_path: train_centroids(
                output_path=output_path,
                sample_path=sp,
                k_train=cluster.k_train,
                k_views=cluster.k_views,
                n_threads=n_threads,
            ),
            resources=scale.train_centroids_resources,
            pip_dependency_groups=["datakit"],
        ),
    )


def _resolve_centroids(
    domain_centroids: str | StepSpec,
    cluster: ClusterConfig,
) -> tuple[str, dict[int, str], list[StepSpec], object]:
    """Return ``(centroids_uri, lookup_uris, extra_deps, hash_value)`` for assign."""
    if isinstance(domain_centroids, StepSpec):
        base = domain_centroids.output_path
        return (
            f"{base}/centroids_{cluster.k_train}.npy",
            {k: f"{base}/lookup_{cluster.k_train}_to_{k}.npy" for k in cluster.k_views},
            [domain_centroids],
            base,  # already includes a content hash; safe in hash_attrs
        )
    base = domain_centroids.rstrip("/")
    return (
        f"{base}/centroids_{cluster.k_train}.npy",
        {k: f"{base}/lookup_{cluster.k_train}_to_{k}.npy" for k in cluster.k_views},
        [],
        domain_centroids,
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


def reference_datakit_steps(
    sources: dict[str, StepSpec],
    *,
    quality_model: str,
    domain_centroids: str | StepSpec | None = None,
    scale: PipelineScale = DEFAULT_SCALE,
) -> DatakitSteps:
    """Build the reference Datakit DAG over the given normalize steps.

    Every step's output lands at ``<MARIN_PREFIX>/<step_name>_<hash>/`` via
    the default StepSpec routing -- this pipeline never sets
    ``output_path_prefix``, so changing the deploy region is just a matter
    of changing ``MARIN_PREFIX``.

    Args:
        sources: ``{name: normalize_step}``. Each step must produce a
            :class:`marin.datakit.normalize.NormalizedData` artifact;
            misuse fails loudly the first time a downstream step tries
            ``read_artifact(step.output_path, NormalizedData)``.
        quality_model: Directory holding the pooled fast-transformer scorer
            artifacts plus the calibration json (immutable by convention --
            the step hash covers the path, not the bytes).
        domain_centroids: A GCS directory holding ``centroids_<k_train>.npy``
            and ``lookup_<k_train>_to_<k>.npy`` for each ``k`` in
            ``scale.cluster.k_views``; a StepSpec whose ``output_path`` will
            contain that layout once it runs (see
            :func:`build_train_centroids_step`); or ``None`` to train inline
            from the per-source embeds. When training inline, ``scale.cluster.k_train``
            must not exceed the centroid sample size -- use a smaller K
            (e.g. ``SMOKE_SCALE``) on small source sets.
        scale: K / fan-out sizing plus the per-stage worker :class:`PoolConfig`.
            ``DEFAULT_SCALE`` is the production full-fleet shape; ``SMOKE_SCALE``
            runs the same DAG end-to-end on a testbed sample.
    """
    cluster = scale.cluster
    embed_steps = build_per_source_embed_steps(sources, scale)
    if domain_centroids is None:
        domain_centroids = build_train_centroids_step(embed_steps, scale)

    centroids_uri, lookup_uris, centroids_deps, centroids_hash = _resolve_centroids(domain_centroids, cluster)

    # One combined decontam bloom (no merge step); every per-source decon
    # consumes it directly. Same name/params as the testbed decon arm, so runs
    # sharing a prefix share the built bloom.
    decon_bloom_step = build_eval_bloom_step(
        name="datakit/bloom/_combined_fixed",
        eval_data_sources=[EVAL_ROOT],
        ngram_length=NGRAM_LENGTH,
        overlap_threshold=OVERLAP_THRESHOLD,
        estimated_doc_count=ESTIMATED_DOC_COUNT,
        false_positive_rate=FALSE_POSITIVE_RATE,
        exclude_eval_dirs=DECON_EXCLUDED_EVAL_TASKS,
    )

    # ---- Per-source steps ------------------------------------------------------
    per_source: dict[str, dict[str, StepSpec]] = {}
    minhash_steps: list[StepSpec] = []

    for name, normalize_step in sources.items():
        embed = embed_steps[name]

        tokenize = tokenize_attributes_step(
            name=f"datakit/tokenize/{name}",
            train_normalize=normalize_step,
            tokenizer=TOKENIZER,
            tokenizer_backend=TOKENIZER_BACKEND,
            max_workers=scale.pool.n_workers,
            worker_resources=scale.pool.worker,
        )

        # Domain assign: consumes the embed + the (given or trained) centroids.
        # ``centroids_hash`` feeds hash_attrs so re-pointing at a new model
        # invalidates already-assigned outputs.
        assign = StepSpec(
            name=f"datakit/cluster_assign/{name}",
            deps=[embed, *centroids_deps],
            hash_attrs={
                "centroids_dir": centroids_hash,
                "k_train": cluster.k_train,
                "k_views": list(cluster.k_views),
                "batch_size": scale.assign_batch_size,
                "v": 1,
            },
            fn=remote(
                lambda output_path, ep=embed.output_path: assign_source(
                    output_path=output_path,
                    embedding=read_artifact(ep, EmbeddingAttrData),
                    centroids_uri=centroids_uri,
                    lookup_uris=lookup_uris,
                    window_size=scale.assign_batch_size,
                    worker_resources=scale.pool.worker,
                    max_workers=scale.pool.n_workers,
                ),
                resources=DRIVER_RESOURCES,
                pip_dependency_groups=["datakit"],
            ),
        )

        quality = StepSpec(
            name=f"datakit/quality/{name}",
            deps=[normalize_step],
            hash_attrs={"model_dir": quality_model, "v": 1},
            fn=remote(
                lambda output_path, np=normalize_step.output_path, src=name: score_normalized(
                    output_path=output_path,
                    normalized=read_artifact(np, NormalizedData),
                    source=src,
                    model_dir=quality_model,
                    max_workers=scale.pool.n_workers,
                    worker_resources=scale.pool.worker,
                ),
                resources=DRIVER_RESOURCES,
            ),
        )

        decontam = decon_step(
            name=f"datakit/decontam/{name}",
            normalized=normalize_step,
            prebuilt_bloom=decon_bloom_step,
            ngram_length=NGRAM_LENGTH,
            overlap_threshold=OVERLAP_THRESHOLD,
            estimated_doc_count=ESTIMATED_DOC_COUNT,
            false_positive_rate=FALSE_POSITIVE_RATE,
            flagged_sample_size=FLAGGED_SAMPLE_SIZE,
            worker_resources=scale.pool.worker,
        )

        minhash = StepSpec(
            name=f"datakit/minhash/{name}",
            deps=[normalize_step],
            fn=lambda op, n=normalize_step: compute_minhash_attrs(
                source=read_artifact(n.output_path, NormalizedData),
                output_path=op,
                worker_resources=scale.pool.worker,
            ),
        )
        minhash_steps.append(minhash)

        per_source[name] = {
            "tokenize": tokenize,
            "embed": embed,
            "assign": assign,
            "quality": quality,
            "decontam": decontam,
            "minhash": minhash,
        }

    # ---- Cross-source dedup ----------------------------------------------------
    dedup = StepSpec(
        name="datakit/dedup",
        deps=minhash_steps,
        fn=lambda op: compute_fuzzy_dups_attrs(
            inputs=[read_artifact(s.output_path, MinHashAttrData) for s in minhash_steps],
            output_path=op,
            max_parallelism=scale.dedup_max_parallelism,
            cc_resume=True,
            worker_resources=scale.pool.worker,
        ),
    )

    # ---- Final store: 5-way join + per-bucket Levanter cache ------------------
    def _store_fn(output_path: str) -> ClusteredStoreData:
        return build_clustered_store(
            tokenize={n: read_artifact(s["tokenize"].output_path, TokenizedAttrData) for n, s in per_source.items()},
            decontam={n: read_artifact(s["decontam"].output_path, DeconAttributes) for n, s in per_source.items()},
            cluster_assign={
                n: read_artifact(s["assign"].output_path, AssignmentAttrData) for n, s in per_source.items()
            },
            quality={n: read_artifact(s["quality"].output_path, QualityScores) for n, s in per_source.items()},
            dedup=read_artifact(dedup.output_path, FuzzyDupsAttrData),
            output_path=output_path,
            cluster_view=cluster.cluster_view,
            split=SPLIT,
            worker_resources=scale.pool.worker,
            max_workers=scale.pool.n_workers,
            shards_per_task=scale.store_shards_per_task,
        )

    store_deps: list[StepSpec] = []
    for s in per_source.values():
        store_deps += [s["tokenize"], s["decontam"], s["assign"], s["quality"]]
    store_deps.append(dedup)

    store = StepSpec(
        name="datakit/store",
        deps=store_deps,
        hash_attrs={
            "shards_per_task": scale.store_shards_per_task,
            "cluster_view": cluster.cluster_view,
            "split": SPLIT,
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
    decontam_paths = {n: s["decontam"].output_path for n, s in per_source.items()}
    reports = [
        StepSpec(
            name="datakit/report/normalize",
            deps=list(sources.values()),
            hash_attrs={"v": 2},
            fn=lambda op: normalize_report(
                op, {n: read_artifact(p, NormalizedData) for n, p in normalize_paths.items()}
            ),
        ),
        StepSpec(
            name="datakit/report/tokenize",
            deps=[s["tokenize"] for s in per_source.values()],
            hash_attrs={"v": 1, "split": SPLIT},
            fn=lambda op: tokenize_report(
                op, {n: read_artifact(p, TokenizedAttrData) for n, p in tokenize_paths.items()}, SPLIT
            ),
        ),
        StepSpec(
            name="datakit/report/quality",
            deps=[s["quality"] for s in per_source.values()],
            hash_attrs={"v": 1},
            fn=lambda op: quality_report(op, {n: read_artifact(p, QualityScores) for n, p in quality_paths.items()}),
        ),
        StepSpec(
            name="datakit/report/domain",
            deps=[s["assign"] for s in per_source.values()],
            hash_attrs={"v": 1, "cluster_view": cluster.cluster_view},
            fn=lambda op: assign_report(
                op, {n: read_artifact(p, AssignmentAttrData) for n, p in assign_paths.items()}, cluster.cluster_view
            ),
        ),
        StepSpec(
            name="datakit/report/decontam",
            deps=[s["decontam"] for s in per_source.values()],
            hash_attrs={"v": 1},
            fn=lambda op: decontam_report(op, {n: read_artifact(p, DeconAttributes) for n, p in decontam_paths.items()}),
        ),
        StepSpec(
            name="datakit/report/dedup",
            deps=[dedup],
            hash_attrs={"v": 1},
            fn=lambda op: dedup_report(op, read_artifact(dedup.output_path, FuzzyDupsAttrData)),
        ),
        StepSpec(
            name="datakit/report/store",
            deps=[store],
            hash_attrs={"v": 1},
            fn=lambda op: store_report(op, read_artifact(store.output_path, ClusteredStoreData)),
        ),
    ]

    all_steps: list[StepSpec] = [decon_bloom_step]
    if isinstance(domain_centroids, StepSpec):
        all_steps.append(domain_centroids)
    for s in per_source.values():
        all_steps += list(s.values())
    all_steps += [dedup, store, *reports]
    return DatakitSteps(sources=sources, output_buckets=store, all_steps=all_steps)


SAMPLE_PREFIX = "s3://marin-us-east-02a/marin/datakit/sample_0.1b_7d7d8fd7"
QUALITY_MODEL = "s3://marin-us-east-02a/marin/user/rav/quality/pooled_junkgate2"

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
    worker = replace(
        scale.pool.worker,
        **{k: v for k, v in (("cpu", args.pool_cpu), ("ram", args.pool_ram), ("disk", args.pool_disk)) if v is not None},
    )
    n_workers = args.pool_workers if args.pool_workers is not None else scale.pool.n_workers
    return replace(scale, pool=PoolConfig(n_workers=n_workers, worker=worker))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("full", "sample"),
        default="full",
        help="full: registry sources, K=5000. sample: a pre-built testbed sample (see --sample-prefix), K=64.",
    )
    parser.add_argument("--sample-prefix", default=SAMPLE_PREFIX, help="testbed sample root (--mode sample)")
    parser.add_argument("--quality-model", default=QUALITY_MODEL, help="pooled fast-transformer scorer + calib dir")
    parser.add_argument(
        "--domain-centroids",
        default=None,
        help="dir with centroids_<K>.npy + lookup_<K>_to_<k>.npy. Omit to train centroids inline from the embeds.",
    )
    parser.add_argument(
        "--sources",
        default=None,
        help="comma-separated source names, or 'all' for every source. Omit: full=all, sample=curated subset.",
    )
    parser.add_argument("--pool-workers", type=int, default=None, help="per-stage worker count (override scale)")
    parser.add_argument("--pool-cpu", type=float, default=None, help="per-worker CPUs (override scale)")
    parser.add_argument("--pool-ram", default=None, help="per-worker RAM, e.g. 16g (override scale)")
    parser.add_argument("--pool-disk", default=None, help="per-worker disk, e.g. 16g (override scale)")
    parser.add_argument("--max-concurrent", type=int, default=8, metavar="N", help="max steps StepRunner runs at once")
    parser.add_argument(
        "--run-tag",
        default="",
        help="sample mode: mix this into every step's hash so the whole DAG recomputes fresh (benchmarking)",
    )
    args = parser.parse_args()

    configure_logging(logging.INFO)

    scale = _apply_pool_overrides(SMOKE_SCALE if args.mode == "sample" else DEFAULT_SCALE, args)
    sources = _select_pipeline_sources(args)

    # Each stage runs its pipeline on its own dedicated Zephyr coordinator +
    # worker fleet (vanilla ``ZephyrContext``, built inside the stage functions),
    # sized by ``scale.pool``. ``--max-concurrent`` bounds how many stages the
    # StepRunner walks at once.
    result = reference_datakit_steps(
        sources,
        quality_model=args.quality_model,
        domain_centroids=args.domain_centroids,
        scale=scale,
    )
    StepRunner().run(result.all_steps, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
