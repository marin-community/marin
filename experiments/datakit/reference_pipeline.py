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

Submit on iris (eu-west4 pinned via ``MARIN_PREFIX``)::

    uv run iris --cluster=marin job run --region europe-west4 --extra=cpu \\
        --priority production --cpu 2 --memory 8GB \\
        -- python -m experiments.datakit.reference_pipeline \\
            --domain-centroids gs://.../cluster/train_centroids_<hash> \\
            --quality-model-bin gs://.../quality/model.bin \\
            --output-prefix gs://.../datakit/reference/<run-id>
"""

import argparse
import logging
import os

DATA_REGION = "europe-west4"
os.environ.setdefault("MARIN_PREFIX", "gs://marin-eu-west4")

from fray import ResourceConfig  # noqa: E402
from levanter.tokenizers import TokenizerBackend  # noqa: E402
from marin.datakit.decon import (  # noqa: E402
    DeconAttributes,
    build_eval_bloom_step,
    decon_step,
)
from marin.datakit.normalize import NormalizedData  # noqa: E402
from marin.datakit.sources import all_sources  # noqa: E402
from marin.execution.artifact import Artifact  # noqa: E402
from marin.execution.remote import remote  # noqa: E402
from marin.execution.step_runner import StepRunner  # noqa: E402
from marin.execution.step_spec import StepSpec  # noqa: E402
from marin.processing.classification.deduplication.fuzzy_dups import (  # noqa: E402
    FuzzyDupsAttrData,
    compute_fuzzy_dups_attrs,
)
from marin.processing.classification.deduplication.fuzzy_minhash import (  # noqa: E402
    MinHashAttrData,
    compute_minhash_attrs,
)
from marin.processing.tokenize.attributes import (  # noqa: E402
    TokenizedAttrData,
    tokenize_attributes_step,
)
from rigging.log_setup import configure_logging  # noqa: E402

from experiments.datakit.cluster.domain.v0.assign import (  # noqa: E402
    AssignmentAttrData,
    assign_source,
)
from experiments.datakit.cluster.quality.v0.all_sources_quality_llm import (  # noqa: E402
    LlmQualityOutput,
    _register_model_step,
    classify_llm_quality_step,
)
from experiments.datakit.decontam.all_sources_decon import (  # noqa: E402
    ESTIMATED_DOC_COUNT,
    EVAL_ROOT,
    FALSE_POSITIVE_RATE,
    NGRAM_LENGTH,
    OVERLAP_THRESHOLD,
)
from experiments.datakit.embeddings.luxical.pipeline import (  # noqa: E402
    LUXICAL_REPO,
    LUXICAL_WEIGHTS_FILE,
    EmbeddingAttrData,
    embed_source,
)
from experiments.datakit.store.datakit_store import (  # noqa: E402
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

# Embed / assign resource shapes mirror exp_full_clusters defaults.
EMBED_WORKER_RESOURCES = ResourceConfig(cpu=8, ram="16g", regions=[DATA_REGION])
ASSIGN_WORKER_RESOURCES = ResourceConfig(cpu=4, ram="8g", regions=[DATA_REGION])
COORDINATOR_RESOURCES = ResourceConfig.with_cpu(cpu=2, ram="4g", regions=[DATA_REGION])
EMBED_MAX_WORKERS_PER_SOURCE = 128
ASSIGN_MAX_WORKERS_PER_SOURCE = 128

# Minhash / dedup mirror all_sources_fuzzy defaults.
MINHASH_WORKER_RESOURCES = ResourceConfig(cpu=5, ram="32g", disk="5g")
DEDUP_MAX_PARALLELISM = 2048
DEDUP_WORKER_RESOURCES = ResourceConfig(cpu=3, ram="32g", disk="5g")
DEDUP_COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="3.5g", preemptible=False)

# Decontam shares all knobs with all_sources_decon (imported above).
DECONTAM_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g")

# Store.
STORE_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g", regions=[DATA_REGION])
STORE_MAX_WORKERS = 2048
SPLIT = "train"


def build_steps(
    *,
    domain_centroids_dir: str,
    quality_model_bin: str,
    output_prefix: str,
    inference_name: str = "reference",
) -> list[StepSpec]:
    """Build the full DAG of StepSpecs.

    Args:
        domain_centroids_dir: GCS directory holding ``centroids_<K_TRAIN>.npy``
            and ``lookup_<K_TRAIN>_to_<k>.npy`` for each K in ``K_VIEWS``.
        quality_model_bin: GCS path to the trained fasttext quality ``.bin``.
        output_prefix: GCS prefix for every step's output (e.g.
            ``gs://marin-eu-west4/datakit/reference/<run-id>``).
        inference_name: Sub-namespace under ``quality-llm/`` for the model
            wrapper step (so different model versions don't collide).
    """
    sources = all_sources()
    logger.info("Reference pipeline: %d sources", len(sources))

    centroids_uri = f"{domain_centroids_dir.rstrip('/')}/centroids_{K_TRAIN}.npy"
    lookup_uris = {k: f"{domain_centroids_dir.rstrip('/')}/lookup_{K_TRAIN}_to_{k}.npy" for k in K_VIEWS}

    # ---- Shared upstream steps -------------------------------------------------
    # Quality model wrapper: emits a FastTextModel artifact pointing at the
    # already-staged .bin (no download/training).
    quality_model_step = _register_model_step(
        name=f"datakit/quality_model/{inference_name}",
        model_bin_path=quality_model_bin,
        output_path_prefix=output_prefix,
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
            output_path_prefix=output_prefix,
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
            output_path_prefix=output_prefix,
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
            output_path_prefix=output_prefix,
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
            output_path_prefix=output_prefix,
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
        output_path_prefix=output_prefix,
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
        )

    store_deps: list[StepSpec] = []
    for s in per_source.values():
        store_deps += [s["tokenize"], s["decontam"], s["assign"], s["quality"]]
    store_deps.append(dedup)

    store = StepSpec(
        name="datakit/store",
        output_path_prefix=output_prefix,
        deps=store_deps,
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
        "--output-prefix",
        required=True,
        help="GCS prefix for all step outputs from this run",
    )
    parser.add_argument(
        "--inference-name",
        default="reference",
        help="Sub-namespace for the quality model wrapper step (defaults to 'reference')",
    )
    args = parser.parse_args()

    configure_logging(logging.INFO)
    steps = build_steps(
        domain_centroids_dir=args.domain_centroids,
        quality_model_bin=args.quality_model_bin,
        output_prefix=args.output_prefix,
        inference_name=args.inference_name,
    )
    StepRunner().run(steps)


if __name__ == "__main__":
    main()
