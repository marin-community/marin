# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the datakit (cluster x quality) Levanter store via the SHUFFLE path.

Drop-in alternative to :mod:`experiments.datakit.store.all_sources_store` that
calls :func:`build_clustered_store_shuffle` instead of the map-side builder, so
the store is born compact -- one large materialized cache per
``(cluster, quality, sub)`` instead of ~14.2M tiny leaf caches / ~70M GCS
objects (marin#6687). See ``datakit_store_shuffle`` for the pipeline and the
measured cost (~17 TB in-region scatter, ~2 h at ~2k workers).

Same inputs/roots as ``all_sources_store``::

    tokenize    gs://marin-eu-west4/datakit/tokenize/<src>_<hash>/
    decontam    gs://marin-eu-west4/datakit/decontam/<src>_<hash>/
    cluster     gs://marin-eu-west4/datakit/cluster/assign/<src>_<hash>/
    quality     gs://marin-eu-west4/datakit/llm-quality-classifier/inference/sonnet46-thr05/quality-llm/<src>_<hash>/
    dedup       gs://marin-eu-west4/datakit/dedup/dedup_v0_manual/

Skew: pass ``--hint-from`` a prior store artifact (default: the v0.1 map-side
store) so hot buckets (top bucket ~651B tokens) split across many reducers while
small buckets stay at one cache.

Submit on iris (eu-west4 pinned by the worker's ``MARIN_PREFIX``)::

    uv run iris --cluster=marin job run --region europe-west4 --extra=cpu \\
        --priority production --cpu 2 --memory 8GB --enable-extra-resources \\
        --no-preemptible \\
        -- python experiments/datakit/store/all_sources_store_shuffle.py
"""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor

from fray import ResourceConfig
from marin.datakit.decon import DeconAttributes
from marin.datakit.sources import all_sources
from marin.execution.artifact import Artifact
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from marin.processing.tokenize.attributes import TokenizedAttrData
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.domain.v0.assign import AssignmentAttrData
from experiments.datakit.cluster.quality.v0.all_sources_quality_llm import LlmQualityOutput
from experiments.datakit.store.all_sources_store import (
    CLUSTER_ASSIGN_ROOT,
    DECONTAM_ROOT,
    DEDUP_PATH,
    QUALITY_ROOT,
    TOKENIZE_ROOT,
    _build_resolution_index,
)
from experiments.datakit.store.datakit_store_shuffle import (
    _DEFAULT_TARGET_TOKENS_PER_SUBSHARD,
    bucket_token_hint_from_artifact,
    build_clustered_store_shuffle,
)

logger = logging.getLogger(__name__)

OUTPUT_PATH = "gs://marin-eu-west4/datakit/store/v0.2_shuffle_20260518"
# Prior map-side store: its per-bucket token mass sizes each bucket's subshards.
DEFAULT_HINT_FROM = "gs://marin-eu-west4/datakit/store/v0.1_20260518"

CLUSTER_VIEW = 40
SPLIT = "train"

# 16g: the reduce side streams each bucket-shard's docs through a tensorstore
# write buffer (~512 MB write-chunk) while the map holds numpy token payloads.
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="16g", disk="16g")
MAX_WORKERS = 2048


def main() -> None:
    configure_logging(logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT_PATH, help="Output store root.")
    parser.add_argument(
        "--hint-from",
        default=DEFAULT_HINT_FROM,
        help="Prior store artifact whose per-bucket token mass sizes subshards. Empty string -> no hint.",
    )
    parser.add_argument("--reduce-shards", type=int, default=2048, help="num_output_shards for the group_by.")
    parser.add_argument("--shards-per-task", type=int, default=1, help="Source shards batched per map task.")
    parser.add_argument("--max-subshards", type=int, default=128, help="Cap on subshards for the hottest bucket.")
    parser.add_argument(
        "--target-tokens-per-subshard",
        type=int,
        default=_DEFAULT_TARGET_TOKENS_PER_SUBSHARD,
        help="Aim for ~this many tokens per reduce cache when sizing subshards from the hint.",
    )
    parser.add_argument(
        "--default-subshards",
        type=int,
        default=1,
        help="Subshards for buckets absent from the hint (or all buckets if no hint).",
    )
    args = parser.parse_args()

    dedup = Artifact.from_path(DEDUP_PATH, FuzzyDupsAttrData)
    source_names = list(all_sources())

    logger.info("indexing 4 roots via shallow fs.ls")
    tokenize_index = _build_resolution_index(TOKENIZE_ROOT)
    decontam_index = _build_resolution_index(DECONTAM_ROOT)
    cluster_assign_index = _build_resolution_index(CLUSTER_ASSIGN_ROOT)
    quality_index = _build_resolution_index(QUALITY_ROOT)

    def _resolve(name: str) -> tuple[str, TokenizedAttrData, DeconAttributes, AssignmentAttrData, LlmQualityOutput]:
        return (
            name,
            Artifact.from_path(tokenize_index[name], TokenizedAttrData),
            Artifact.from_path(decontam_index[name], DeconAttributes),
            Artifact.from_path(cluster_assign_index[name], AssignmentAttrData),
            Artifact.from_path(quality_index[name], LlmQualityOutput),
        )

    tokenize: dict[str, TokenizedAttrData] = {}
    decontam: dict[str, DeconAttributes] = {}
    cluster_assign: dict[str, AssignmentAttrData] = {}
    quality: dict[str, LlmQualityOutput] = {}

    logger.info("loading typed artifacts for %d sources (ThreadPoolExecutor=16)", len(source_names))
    with ThreadPoolExecutor(max_workers=16) as pool:
        for name, tok, decon, cluster_a, qual in pool.map(_resolve, source_names):
            tokenize[name] = tok
            decontam[name] = decon
            cluster_assign[name] = cluster_a
            quality[name] = qual
    logger.info("resolved %d sources", len(tokenize))

    bucket_token_hint = bucket_token_hint_from_artifact(args.hint_from) if args.hint_from else None
    if bucket_token_hint is not None:
        logger.info("loaded subshard hint over %d buckets from %s", len(bucket_token_hint), args.hint_from)

    artifact = build_clustered_store_shuffle(
        tokenize=tokenize,
        decontam=decontam,
        cluster_assign=cluster_assign,
        quality=quality,
        dedup=dedup,
        output_path=args.output,
        cluster_view=CLUSTER_VIEW,
        split=SPLIT,
        worker_resources=WORKER_RESOURCES,
        max_workers=MAX_WORKERS,
        shards_per_task=args.shards_per_task,
        reduce_shards=args.reduce_shards,
        bucket_token_hint=bucket_token_hint,
        target_tokens_per_subshard=args.target_tokens_per_subshard,
        max_subshards=args.max_subshards,
        default_subshards=args.default_subshards,
    )

    logger.info(
        "done: %d buckets, %d total docs, %d total tokens, %d subshard caches -> %s",
        len(artifact.buckets),
        sum(b.total_elements for b in artifact.buckets),
        sum(b.total_tokens for b in artifact.buckets),
        sum(b.n_shards for b in artifact.buckets),
        artifact.cache_path,
    )


if __name__ == "__main__":
    main()
