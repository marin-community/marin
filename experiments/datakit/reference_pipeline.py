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

Assumes two pre-trained artifacts already exist on GCS — this pipeline does
not train either:

  ``--domain-centroids``    A directory containing the K-means output from
                            :mod:`experiments.datakit.cluster.domain.v0.exp_full_clusters`:
                            ``centroids_<K_TRAIN>.npy`` plus
                            ``lookup_<K_TRAIN>_to_<k>.npy`` for every K view.

  ``--quality-model-bin``   A trained fasttext quality ``.bin`` from
                            :mod:`experiments.datakit.cluster.quality.v0.train`.

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
            --quality-model-bin gs://.../quality/model.bin
"""

import argparse
import logging

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


# Sources to skip. ``numinamath-1.5`` is a fresh add (#5841) and not yet
# included in the trained domain centroids -- assigning it would just produce
# garbage cluster ids until we retrain.
_EXCLUDE_SOURCES: frozenset[str] = frozenset({"numinamath-1.5"})

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
ASSIGN_MAX_WORKERS_PER_SOURCE = 128

# Minhash / dedup mirror all_sources_fuzzy defaults.
MINHASH_WORKER_RESOURCES = ResourceConfig(cpu=5, ram="32g", disk="5g")
DEDUP_MAX_PARALLELISM = 4096
DEDUP_WORKER_RESOURCES = ResourceConfig(cpu=3, ram="32g", disk="5g")
DEDUP_COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="3.5g", preemptible=False)

# Decontam shares all knobs with all_sources_decon (imported above).
DECONTAM_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="16g")

# Store.
STORE_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g")
STORE_MAX_WORKERS = 4096
SPLIT = "train"


def build_steps(
    *,
    domain_centroids_dir: str,
    quality_model_bin: str,
    store_shards_per_task: int = 1,
) -> list[StepSpec]:
    """Build the full DAG of StepSpecs.

    Every step's output lands at ``<MARIN_PREFIX>/<step_name>_<hash>/`` via
    the default StepSpec routing -- this pipeline never sets
    ``output_path_prefix``, so changing the deploy region is just a matter
    of changing ``MARIN_PREFIX``.

    Args:
        domain_centroids_dir: GCS directory holding ``centroids_<K_TRAIN>.npy``
            and ``lookup_<K_TRAIN>_to_<k>.npy`` for each K in ``K_VIEWS``.
        quality_model_bin: GCS path to the trained fasttext quality ``.bin``.
    """
    sources = {name: src for name, src in all_sources().items() if name not in _EXCLUDE_SOURCES}
    logger.info(
        "Reference pipeline: %d sources (excluded: %s)",
        len(sources),
        sorted(_EXCLUDE_SOURCES),
    )

    centroids_uri = f"{domain_centroids_dir.rstrip('/')}/centroids_{K_TRAIN}.npy"
    lookup_uris = {k: f"{domain_centroids_dir.rstrip('/')}/lookup_{K_TRAIN}_to_{k}.npy" for k in K_VIEWS}

    # ---- Shared upstream steps -------------------------------------------------
    # Quality model wrapper: emits a FastTextModel artifact pointing at the
    # already-staged .bin (no download/training).
    quality_model_step = _register_model_step(
        name="datakit/quality_model/reference",
        model_bin_path=quality_model_bin,
    )

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

    for name, source in sources.items():
        normalize_step = source.normalized

        tokenize = tokenize_attributes_step(
            name=f"datakit/tokenize/{name}",
            train_normalize=normalize_step,
            tokenizer=TOKENIZER,
            tokenizer_backend=TOKENIZER_BACKEND,
            max_workers=TOKENIZE_MAX_WORKERS,
            worker_resources=TOKENIZE_WORKER_RESOURCES,
        )

        embed = StepSpec(
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
                pip_dependency_groups=["embed"],
            ),
        )

        # Domain assign: consumes the embed + the (given) centroids.
        # The centroids URIs feed hash_attrs so re-pointing at a new model
        # invalidates already-assigned outputs.
        assign = StepSpec(
            name=f"datakit/cluster_assign/{name}",
            deps=[embed],
            hash_attrs={
                "centroids_dir": domain_centroids_dir,
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
                pip_dependency_groups=["cluster"],
            ),
        )

        quality = classify_llm_quality_step(
            name=f"datakit/quality/{name}",
            normalized=normalize_step,
            model_step=quality_model_step,
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
    for s in per_source.values():
        all_steps += list(s.values())
    all_steps += [dedup, store]
    return all_steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain-centroids",
        required=True,
        help="GCS dir containing centroids_<K>.npy + lookup_<K>_to_<k>.npy from cluster/domain/v0",
    )
    parser.add_argument(
        "--quality-model-bin",
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
    steps = build_steps(
        domain_centroids_dir=args.domain_centroids,
        quality_model_bin=args.quality_model_bin,
        store_shards_per_task=args.store_shards_per_task,
    )
    StepRunner().run(steps, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
