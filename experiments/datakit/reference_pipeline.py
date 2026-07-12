# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end reference DAG: Datakit sources → (cluster x quality) store.

This wires the existing per-stage building blocks into a single
StepRunner-walkable graph. The source set is a parameter
(:func:`select_sources` / ``--sources``), and all resource / K / fan-out
sizing lives in a :class:`PipelineScale` (``--scale full|smoke``), so the
*same* DAG runs at full-fleet scale or as a quick e2e smoke on a few small
sources without forking the pipeline. Per source (default: every entry from
:func:`marin.datakit.sources.all_sources` — no carve-outs):

    normalize → tokenize
              → embed (luxical-one)   → assign (domain v0, given centroids)
              → classify_quality       (quality v0, given .bin)
              → decontam               (shared eval bloom)
              → minhash

Then:
    fuzzy_dups([<minhash per source>])
    build_clustered_store(tokenize, decontam, cluster_assign, quality, dedup)

Public API: :func:`reference_datakit_steps`. Pass ``sources`` (a ``{name:
normalize_step}`` mapping) and either a path or a pre-built StepSpec for each
model. Each model accepts ``None`` to train inline -- supported today only
for the domain centroids; ``quality_model=None`` raises until v0 quality
training is wrapped as StepSpecs.

CLI flags map to those parameters:

  ``--domain-centroids``    Optional. Directory containing K-means output:
                            ``centroids_<k_train>.npy`` plus
                            ``lookup_<k_train>_to_<k>.npy`` for every K view
                            (``k_train`` / views come from ``scale.cluster``;
                            the layout produced by
                            :mod:`experiments.datakit.cluster.domain.v0.exp_full_clusters`).
                            If omitted, the CLI lets
                            :func:`reference_datakit_steps` build the
                            inline-training subgraph via
                            :func:`build_train_centroids_step`.

  ``--quality-model``       Required. Trained fasttext quality ``.bin`` from
                            :mod:`experiments.datakit.cluster.quality.v0.train`.
                            The CLI requires a path because the v0 quality
                            training flow (sample → LLM-score → fasttext-train)
                            is not yet wrapped as StepSpecs. Programmatic
                            callers can still pass a pre-built StepSpec
                            producing a ``FastTextModel`` artifact.

Region-agnostic: every worker resource is unpinned, so iris's ``--region``
flag drives scheduling. ``MARIN_PREFIX`` is resolved by
:func:`rigging.filesystem.marin_prefix` -- when unset (the normal iris-
worker case) it falls back to ``gs://marin-<region>`` from GCS metadata,
so the upstream source artifacts, the eval-corpus path
(``EVAL_ROOT`` in :mod:`decontam.all_sources_decon`, also derived from
``marin_prefix()``), and every step's output land in-region automatically.
Override by exporting ``MARIN_PREFIX`` or passing it via
``iris job run --env-vars MARIN_PREFIX <bucket>``.
``EVAL_ROOT`` is hardcoded to ``gs://marin-eu-west4/.../evals``: the eval
corpus is read once to build the decontam bloom, after which workers read
the bloom in-region -- the cross-region cost is one bloom build, not
per-shard.

Submit on iris (full)::

    uv run iris --cluster=marin job run --region <region> --extra=cpu \\
        --priority production --cpu 2 --memory 8GB \\
        -- python -m experiments.datakit.reference_pipeline \\
            --domain-centroids gs://.../cluster/train_centroids_<hash> \\
            --quality-model gs://.../quality/model.bin

Smoke (a few small sources, K=64, inline-trained centroids)::

    uv run iris --cluster=marin job run --region europe-west4 --extra=cpu \\
        --priority interactive --cpu 2 --memory 8GB \\
        -- python -m experiments.datakit.reference_pipeline \\
            --scale smoke --sources cp/peps,cp/foodista,nsf_awards \\
            --quality-model gs://.../quality/model.bin

or via the curated wrapper :mod:`experiments.datakit.reference_pipeline_smoke`,
which pins the source list and ``SMOKE_SCALE`` for you.
"""

import argparse
import logging
from dataclasses import dataclass, field

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
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.domain.v0.assign import (
    AssignmentAttrData,
    assign_source,
)
from experiments.datakit.cluster.domain.v0.sample import sample_centroid_inputs
from experiments.datakit.cluster.domain.v0.train import train_centroids
from experiments.datakit.cluster.quality.v0.all_sources_quality_llm import (
    LlmQualityOutput,
    _register_model_step,
    classify_llm_quality_step,
)
from experiments.datakit.decontam.all_sources_decon import (
    ESTIMATED_DOC_COUNT,
    EVAL_ROOT,
    FALSE_POSITIVE_RATE,
    NGRAM_LENGTH,
    OVERLAP_THRESHOLD,
)
from experiments.datakit.embeddings.luxical.pipeline import (
    LUXICAL_REPO,
    LUXICAL_WEIGHTS_FILE,
    EmbeddingAttrData,
    embed_source,
)
from experiments.datakit.store.datakit_store import (
    ClusteredStoreData,
    build_clustered_store,
)

logger = logging.getLogger(__name__)


# Tokenize: canonical Marin tokenizer. Not scale-sensitive.
TOKENIZER = "marin-community/marin-tokenizer"
TOKENIZER_BACKEND = TokenizerBackend.HF
SPLIT = "train"


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


@dataclass(frozen=True)
class PipelineScale:
    """All scale-sensitive knobs for :func:`reference_datakit_steps`.

    ``DEFAULT_SCALE`` reproduces the production full-fleet sizing; ``SMOKE_SCALE``
    shrinks K, sample size, worker resources, and fan-out so the same DAG runs
    end-to-end on a handful of small sources. Resource/worker fields are
    deliberately excluded from every step's ``hash_attrs`` (they are execution
    policy, not content), so a smoke run shares the per-source embed / tokenize /
    decontam / minhash / quality caches with a full run -- only the K-dependent
    steps (sample / train / assign / store / dedup) get distinct output paths.
    """

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    embed_batch_size: int = 4096
    assign_batch_size: int = 4096

    # Inline domain training (when the caller does not pre-stage centroids).
    # ~100 sources x 100k = ~10M-row centroid sample, matching exp_full_clusters.
    n_per_source_for_sample: int = 100_000
    sample_worker_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=2, ram="4g"))
    sample_max_workers: int = 256
    # FAISS K=5000 on 10M x 192 wants every core it can get.
    train_centroids_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig.with_cpu(cpu=32, ram="64g"))

    # Embed / assign mirror exp_full_clusters defaults (region unpinned).
    embed_worker_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=8, ram="64g"))
    assign_worker_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=4, ram="32g"))
    coordinator_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig.with_cpu(cpu=2, ram="4g"))
    embed_max_workers_per_source: int = 512
    assign_max_workers_per_source: int = 512

    # Tokenize.
    tokenize_worker_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(ram="10g", disk="5g"))
    tokenize_max_workers: int = 1024

    # Minhash / dedup mirror all_sources_fuzzy defaults.
    minhash_worker_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=5, ram="32g", disk="5g"))
    dedup_max_parallelism: int = 4096
    dedup_worker_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=3, ram="32g", disk="5g"))
    dedup_coordinator_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig(cpu=1, ram="3.5g", preemptible=False)
    )

    # Decontam shares all knobs with all_sources_decon (imported above).
    decontam_worker_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=2, ram="16g"))

    # Store.
    store_worker_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=2, ram="32g"))
    store_max_workers: int = 2048 + 1024


DEFAULT_SCALE = PipelineScale()
"""Production full-fleet sizing (every ``all_sources()`` entry, K=5000)."""

SMOKE_SCALE = PipelineScale(
    cluster=ClusterConfig(k_train=64, k_views=(8, 16), cluster_view=8),
    n_per_source_for_sample=20_000,
    sample_max_workers=16,
    # K=64 on a few-thousand-row sample is seconds on a handful of cores.
    train_centroids_resources=ResourceConfig.with_cpu(cpu=4, ram="8g"),
    embed_worker_resources=ResourceConfig(cpu=8, ram="16g"),
    assign_worker_resources=ResourceConfig(cpu=2, ram="8g"),
    embed_max_workers_per_source=32,
    assign_max_workers_per_source=32,
    tokenize_max_workers=32,
    minhash_worker_resources=ResourceConfig(cpu=2, ram="8g", disk="5g"),
    dedup_max_parallelism=64,
    dedup_worker_resources=ResourceConfig(cpu=2, ram="8g", disk="5g"),
    decontam_worker_resources=ResourceConfig(cpu=2, ram="8g"),
    store_worker_resources=ResourceConfig(cpu=2, ram="8g"),
    store_max_workers=64,
)
"""Small-data sizing: a few sources, K=64, modest workers -- a true e2e smoke."""


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
                worker_resources=scale.embed_worker_resources,
                max_workers=scale.embed_max_workers_per_source,
            ),
            resources=scale.coordinator_resources,
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
                worker_resources=scale.sample_worker_resources,
                max_workers=scale.sample_max_workers,
            ),
            resources=scale.coordinator_resources,
            pip_dependency_groups=["datakit"],
        ),
    )
    return StepSpec(
        name="datakit/cluster/train_centroids",
        deps=[sample_step],
        hash_attrs={"k_train": cluster.k_train, "k_views": list(cluster.k_views), "v": 1},
        fn=remote(
            lambda output_path, sp=sample_step.output_path: train_centroids(
                output_path=output_path,
                sample_path=sp,
                k_train=cluster.k_train,
                k_views=cluster.k_views,
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


def _resolve_quality_model(quality_model: str | StepSpec | None) -> StepSpec:
    if quality_model is None:
        raise NotImplementedError(
            "quality_model=None means train inline, but v0 quality training "
            "(sample → LLM-score → fasttext-train) is not yet wrapped as "
            "StepSpecs. Pass a GCS path to a trained model.bin or a pre-built "
            "StepSpec producing a FastTextModel artifact."
        )
    if isinstance(quality_model, StepSpec):
        return quality_model
    return _register_model_step(
        name="datakit/quality_model/reference",
        model_bin_path=quality_model,
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
    domain_centroids: str | StepSpec | None = None,
    quality_model: str | StepSpec | None = None,
    store_shards_per_task: int = 1,
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
        domain_centroids: A GCS directory holding ``centroids_<k_train>.npy``
            and ``lookup_<k_train>_to_<k>.npy`` for each ``k`` in
            ``scale.cluster.k_views``; a StepSpec whose ``output_path`` will
            contain that layout once it runs (see
            :func:`build_train_centroids_step`); or ``None`` to train inline
            from the per-source embeds. When training inline, ``scale.cluster.k_train``
            must not exceed the centroid sample size -- use a smaller K
            (e.g. ``SMOKE_SCALE``) on small source sets.
        quality_model: A GCS path to a trained fasttext quality ``.bin``,
            or a StepSpec producing a ``FastTextModel`` artifact. ``None``
            is reserved for a future inline-training mode and currently
            raises :class:`NotImplementedError`.
        store_shards_per_task: Tuning knob for the final store step.
        scale: Resource / K / fan-out sizing. ``DEFAULT_SCALE`` is the
            production full-fleet shape; ``SMOKE_SCALE`` runs the same DAG
            end-to-end on a few small sources.
    """
    cluster = scale.cluster
    embed_steps = build_per_source_embed_steps(sources, scale)
    if domain_centroids is None:
        domain_centroids = build_train_centroids_step(embed_steps, scale)

    centroids_uri, lookup_uris, centroids_deps, centroids_hash = _resolve_centroids(domain_centroids, cluster)
    quality_model_step = _resolve_quality_model(quality_model)

    # One combined decontam bloom (no merge step); every per-source decon
    # consumes it directly.
    decon_bloom_step = build_eval_bloom_step(
        name="datakit/bloom/_combined",
        eval_data_sources=[EVAL_ROOT],
        ngram_length=NGRAM_LENGTH,
        overlap_threshold=OVERLAP_THRESHOLD,
        estimated_doc_count=ESTIMATED_DOC_COUNT,
        false_positive_rate=FALSE_POSITIVE_RATE,
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
            max_workers=scale.tokenize_max_workers,
            worker_resources=scale.tokenize_worker_resources,
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
                    worker_resources=scale.assign_worker_resources,
                    max_workers=scale.assign_max_workers_per_source,
                ),
                resources=scale.coordinator_resources,
                pip_dependency_groups=["datakit"],
            ),
        )

        quality = classify_llm_quality_step(
            name=f"datakit/quality/{name}",
            normalized=normalize_step,
            model_step=quality_model_step,
            max_workers=scale.assign_max_workers_per_source,
        )

        decontam = decon_step(
            name=f"datakit/decontam/{name}",
            normalized=normalize_step,
            prebuilt_bloom=decon_bloom_step,
            ngram_length=NGRAM_LENGTH,
            overlap_threshold=OVERLAP_THRESHOLD,
            estimated_doc_count=ESTIMATED_DOC_COUNT,
            false_positive_rate=FALSE_POSITIVE_RATE,
            worker_resources=scale.decontam_worker_resources,
        )

        minhash = StepSpec(
            name=f"datakit/minhash/{name}",
            deps=[normalize_step],
            fn=lambda op, n=normalize_step: compute_minhash_attrs(
                source=read_artifact(n.output_path, NormalizedData),
                output_path=op,
                worker_resources=scale.minhash_worker_resources,
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
            worker_resources=scale.dedup_worker_resources,
            coordinator_resources=scale.dedup_coordinator_resources,
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
            quality={n: read_artifact(s["quality"].output_path, LlmQualityOutput) for n, s in per_source.items()},
            dedup=read_artifact(dedup.output_path, FuzzyDupsAttrData),
            output_path=output_path,
            cluster_view=cluster.cluster_view,
            split=SPLIT,
            worker_resources=scale.store_worker_resources,
            max_workers=scale.store_max_workers,
            shards_per_task=store_shards_per_task,
        )

    store_deps: list[StepSpec] = []
    for s in per_source.values():
        store_deps += [s["tokenize"], s["decontam"], s["assign"], s["quality"]]
    store_deps.append(dedup)

    store = StepSpec(
        name="datakit/store",
        deps=store_deps,
        hash_attrs={"shards_per_task": store_shards_per_task},
        fn=_store_fn,
    )

    all_steps: list[StepSpec] = [quality_model_step, decon_bloom_step]
    if isinstance(domain_centroids, StepSpec):
        all_steps.append(domain_centroids)
    for s in per_source.values():
        all_steps += list(s.values())
    all_steps += [dedup, store]
    return DatakitSteps(sources=sources, output_buckets=store, all_steps=all_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain-centroids",
        default=None,
        help=(
            "GCS dir containing centroids_<K>.npy + lookup_<K>_to_<k>.npy from "
            "cluster/domain/v0. If omitted, the pipeline trains centroids inline "
            "from the per-source embeds."
        ),
    )
    parser.add_argument(
        "--quality-model",
        required=True,
        help="GCS path to the trained fasttext quality model.bin from cluster/quality/v0",
    )
    parser.add_argument(
        "--sources",
        default=None,
        help=(
            "Comma-separated source names (keys of marin.datakit.sources.all_sources) to run. Omit to run every source."
        ),
    )
    parser.add_argument(
        "--scale",
        choices=("full", "smoke"),
        default="full",
        help=(
            "full: production sizing over all sources (K=5000). smoke: small resources + K=64 "
            "for a quick e2e run -- pair with --sources and a few small sources."
        ),
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        metavar="N",
        help="Max steps StepRunner runs concurrently. Defaults to StepRunner's default (8).",
    )
    parser.add_argument(
        "--store-shards-per-task",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Group N source-shard tuples into one store task so each task writes one part file "
            "per (cluster, quality) it touches instead of one per input shard. Reduces output "
            "file count by ~N x at the cost of N x longer per-task runtime and proportionally "
            "more in-flight per-bucket writer state. Defaults to 1 (no batching)."
        ),
    )
    args = parser.parse_args()

    configure_logging(logging.INFO)

    names = [s.strip() for s in args.sources.split(",") if s.strip()] if args.sources else None
    scale = SMOKE_SCALE if args.scale == "smoke" else DEFAULT_SCALE

    result = reference_datakit_steps(
        select_sources(names),
        domain_centroids=args.domain_centroids,
        quality_model=args.quality_model,
        store_shards_per_task=args.store_shards_per_task,
        scale=scale,
    )
    StepRunner().run(result.all_steps, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
