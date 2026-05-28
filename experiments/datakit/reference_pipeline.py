# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end reference DAG: every Datakit source → (cluster x quality) store.

This wires the existing per-stage building blocks into a single
StepRunner-walkable graph. Per source (every entry from
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
                            ``centroids_<K_TRAIN>.npy`` plus
                            ``lookup_<K_TRAIN>_to_<k>.npy`` for every K view
                            (the layout produced by
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

Submit on iris::

    uv run iris --cluster=marin job run --region <region> --extra=cpu \\
        --priority production --cpu 2 --memory 8GB \\
        -- python -m experiments.datakit.reference_pipeline \\
            --domain-centroids gs://.../cluster/train_centroids_<hash> \\
            --quality-model gs://.../quality/model.bin
"""

import argparse
import logging
from dataclasses import dataclass

from fray import ResourceConfig
from levanter.tokenizers import TokenizerBackend
from marin.datakit.decon import (
    DeconAttributes,
    build_eval_bloom_step,
    decon_step,
)
from marin.datakit.normalize import NormalizedData
from marin.datakit.sources import all_sources
from marin.execution.artifact import Artifact
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


# Clustering knobs. Must match the centroids file passed in.
K_TRAIN = 5000
K_VIEWS: tuple[int, ...] = (40, 1000)
CLUSTER_VIEW = 40  # the K-view the store partitions on (cluster=<C>/quality=<Q>/)
EMBED_BATCH_SIZE = 4096
ASSIGN_BATCH_SIZE = 4096

# Tokenize: canonical Marin tokenizer.
TOKENIZER = "marin-community/marin-tokenizer"
TOKENIZER_BACKEND = TokenizerBackend.HF
TOKENIZE_WORKER_RESOURCES = ResourceConfig(ram="10g", disk="5g")
TOKENIZE_MAX_WORKERS = 1024

# Embed / assign resource shapes mirror exp_full_clusters defaults
# (region is unpinned -- iris's --region flag drives scheduling).
EMBED_WORKER_RESOURCES = ResourceConfig(cpu=8, ram="64g")
ASSIGN_WORKER_RESOURCES = ResourceConfig(cpu=4, ram="32g")
COORDINATOR_RESOURCES = ResourceConfig.with_cpu(cpu=2, ram="4g")
EMBED_MAX_WORKERS_PER_SOURCE = 512
ASSIGN_MAX_WORKERS_PER_SOURCE = 512

# Inline domain training (used when the caller does not pre-stage centroids).
# ~100 sources x 100k = ~10M-row centroid sample, matching exp_full_clusters.
N_PER_SOURCE_FOR_SAMPLE = 100_000
SAMPLE_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="4g")
SAMPLE_MAX_WORKERS = 256
# FAISS K=5000 on 10M x 192 wants every core it can get.
TRAIN_CENTROIDS_RESOURCES = ResourceConfig.with_cpu(cpu=32, ram="64g")

# Minhash / dedup mirror all_sources_fuzzy defaults.
MINHASH_WORKER_RESOURCES = ResourceConfig(cpu=5, ram="32g", disk="5g")
DEDUP_MAX_PARALLELISM = 4096
DEDUP_WORKER_RESOURCES = ResourceConfig(cpu=3, ram="32g", disk="5g")
DEDUP_COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="3.5g", preemptible=False)

# Decontam shares all knobs with all_sources_decon (imported above).
DECONTAM_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="16g")

# Store.
STORE_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="32g")
STORE_MAX_WORKERS = 2048 + 1024
SPLIT = "train"


def default_sources() -> dict[str, StepSpec]:
    """Default Datakit source set: every ``all_sources()`` entry, mapped to its normalize StepSpec."""
    sources = {name: src.normalized for name, src in all_sources().items()}
    logger.info("default_sources: %d sources", len(sources))
    return sources


def _build_embed_step(name: str, normalize_step: StepSpec) -> StepSpec:
    return StepSpec(
        name=f"datakit/embed/{name}",
        deps=[normalize_step],
        hash_attrs={
            "luxical_repo": LUXICAL_REPO,
            "luxical_weights": LUXICAL_WEIGHTS_FILE,
            "batch_size": EMBED_BATCH_SIZE,
            "v": 1,
        },
        fn=remote(
            lambda output_path, np=normalize_step.output_path: embed_source(
                output_path=output_path,
                normalized=Artifact.from_path(np, NormalizedData),
                batch_size=EMBED_BATCH_SIZE,
                worker_resources=EMBED_WORKER_RESOURCES,
                max_workers=EMBED_MAX_WORKERS_PER_SOURCE,
            ),
            resources=COORDINATOR_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )


def build_per_source_embed_steps(sources: dict[str, StepSpec]) -> dict[str, StepSpec]:
    """Build the Luxical embed StepSpec for each source.

    ``sources`` maps source name → normalize StepSpec; the embed step is built
    against that step as its dep. Exposed so callers that also want to build
    the domain training subgraph (via :func:`build_train_centroids_step`) can
    share the same embeds across both wirings.
    """
    return {name: _build_embed_step(name, step) for name, step in sources.items()}


def build_train_centroids_step(embed_steps: dict[str, StepSpec]) -> StepSpec:
    """Build the K-means training StepSpec for the domain centroids.

    The returned step's ``output_path`` contains ``centroids_<K_TRAIN>.npy``
    plus ``lookup_<K_TRAIN>_to_<k>.npy`` for each ``k`` in ``K_VIEWS`` -- the
    same layout :func:`reference_datakit_steps` consumes via its
    ``domain_centroids`` parameter when given a centroids path.
    """
    sample_step = StepSpec(
        name="datakit/cluster/sample_centroids",
        deps=list(embed_steps.values()),
        hash_attrs={"n_per_source": N_PER_SOURCE_FOR_SAMPLE, "format": "parquet", "v": 1},
        fn=remote(
            lambda output_path, es={n: s.output_path for n, s in embed_steps.items()}: sample_centroid_inputs(
                output_path=output_path,
                embeddings={n: Artifact.from_path(p, EmbeddingAttrData) for n, p in es.items()},
                n_per_source=N_PER_SOURCE_FOR_SAMPLE,
                worker_resources=SAMPLE_WORKER_RESOURCES,
                max_workers=SAMPLE_MAX_WORKERS,
            ),
            resources=COORDINATOR_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )
    return StepSpec(
        name="datakit/cluster/train_centroids",
        deps=[sample_step],
        hash_attrs={"k_train": K_TRAIN, "k_views": list(K_VIEWS), "v": 1},
        fn=remote(
            lambda output_path, sp=sample_step.output_path: train_centroids(
                output_path=output_path,
                sample_path=sp,
                k_train=K_TRAIN,
                k_views=K_VIEWS,
            ),
            resources=TRAIN_CENTROIDS_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )


def _resolve_centroids(
    domain_centroids: str | StepSpec,
) -> tuple[str, dict[int, str], list[StepSpec], object]:
    """Return ``(centroids_uri, lookup_uris, extra_deps, hash_value)`` for assign."""
    if isinstance(domain_centroids, StepSpec):
        base = domain_centroids.output_path
        return (
            f"{base}/centroids_{K_TRAIN}.npy",
            {k: f"{base}/lookup_{K_TRAIN}_to_{k}.npy" for k in K_VIEWS},
            [domain_centroids],
            base,  # already includes a content hash; safe in hash_attrs
        )
    base = domain_centroids.rstrip("/")
    return (
        f"{base}/centroids_{K_TRAIN}.npy",
        {k: f"{base}/lookup_{K_TRAIN}_to_{k}.npy" for k in K_VIEWS},
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
            ``Artifact.from_path(step, NormalizedData)``.
        domain_centroids: A GCS directory holding ``centroids_<K_TRAIN>.npy``
            and ``lookup_<K_TRAIN>_to_<k>.npy`` for each ``k`` in
            ``K_VIEWS``; a StepSpec whose ``output_path`` will contain that
            layout once it runs (see :func:`build_train_centroids_step`);
            or ``None`` to train inline from the per-source embeds.
        quality_model: A GCS path to a trained fasttext quality ``.bin``,
            or a StepSpec producing a ``FastTextModel`` artifact. ``None``
            is reserved for a future inline-training mode and currently
            raises :class:`NotImplementedError`.
        store_shards_per_task: Tuning knob for the final store step.
    """
    embed_steps = build_per_source_embed_steps(sources)
    if domain_centroids is None:
        domain_centroids = build_train_centroids_step(embed_steps)

    centroids_uri, lookup_uris, centroids_deps, centroids_hash = _resolve_centroids(domain_centroids)
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
            max_workers=TOKENIZE_MAX_WORKERS,
            worker_resources=TOKENIZE_WORKER_RESOURCES,
        )

        # Domain assign: consumes the embed + the (given or trained) centroids.
        # ``centroids_hash`` feeds hash_attrs so re-pointing at a new model
        # invalidates already-assigned outputs.
        assign = StepSpec(
            name=f"datakit/cluster_assign/{name}",
            deps=[embed, *centroids_deps],
            hash_attrs={
                "centroids_dir": centroids_hash,
                "k_train": K_TRAIN,
                "k_views": list(K_VIEWS),
                "batch_size": ASSIGN_BATCH_SIZE,
                "v": 1,
            },
            fn=remote(
                lambda output_path, ep=embed.output_path: assign_source(
                    output_path=output_path,
                    embedding=Artifact.from_path(ep, EmbeddingAttrData),
                    centroids_uri=centroids_uri,
                    lookup_uris=lookup_uris,
                    window_size=ASSIGN_BATCH_SIZE,
                    worker_resources=ASSIGN_WORKER_RESOURCES,
                    max_workers=ASSIGN_MAX_WORKERS_PER_SOURCE,
                ),
                resources=COORDINATOR_RESOURCES,
                pip_dependency_groups=["datakit"],
            ),
        )

        quality = classify_llm_quality_step(
            name=f"datakit/quality/{name}",
            normalized=normalize_step,
            model_step=quality_model_step,
            max_workers=ASSIGN_MAX_WORKERS_PER_SOURCE,
        )

        decontam = decon_step(
            name=f"datakit/decontam/{name}",
            normalized=normalize_step,
            prebuilt_bloom=decon_bloom_step,
            ngram_length=NGRAM_LENGTH,
            overlap_threshold=OVERLAP_THRESHOLD,
            estimated_doc_count=ESTIMATED_DOC_COUNT,
            false_positive_rate=FALSE_POSITIVE_RATE,
            worker_resources=DECONTAM_WORKER_RESOURCES,
        )

        minhash = StepSpec(
            name=f"datakit/minhash/{name}",
            deps=[normalize_step],
            fn=lambda op, n=normalize_step: compute_minhash_attrs(
                source=Artifact.from_path(n, NormalizedData),
                output_path=op,
                worker_resources=MINHASH_WORKER_RESOURCES,
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
            inputs=[Artifact.from_path(s, MinHashAttrData) for s in minhash_steps],
            output_path=op,
            max_parallelism=DEDUP_MAX_PARALLELISM,
            cc_resume=True,
            worker_resources=DEDUP_WORKER_RESOURCES,
            coordinator_resources=DEDUP_COORDINATOR_RESOURCES,
        ),
    )

    # ---- Final store: 5-way join + per-bucket Levanter cache ------------------
    def _store_fn(output_path: str) -> ClusteredStoreData:
        return build_clustered_store(
            tokenize={n: Artifact.from_path(s["tokenize"], TokenizedAttrData) for n, s in per_source.items()},
            decontam={n: Artifact.from_path(s["decontam"], DeconAttributes) for n, s in per_source.items()},
            cluster_assign={n: Artifact.from_path(s["assign"], AssignmentAttrData) for n, s in per_source.items()},
            quality={n: Artifact.from_path(s["quality"], LlmQualityOutput) for n, s in per_source.items()},
            dedup=Artifact.from_path(dedup, FuzzyDupsAttrData),
            output_path=output_path,
            cluster_view=CLUSTER_VIEW,
            split=SPLIT,
            worker_resources=STORE_WORKER_RESOURCES,
            max_workers=STORE_MAX_WORKERS,
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

    result = reference_datakit_steps(
        default_sources(),
        domain_centroids=args.domain_centroids,
        quality_model=args.quality_model,
        store_shards_per_task=args.store_shards_per_task,
    )
    StepRunner().run(result.all_steps, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
