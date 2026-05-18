# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Larger smoke: a handful of sources including some big ones.

Stresses the pipeline at meaningful scale before the full fleet run --
~780B tokens (~5-8% of the full registry), with two big nemotron sources
to surface any per-shard bucket OOM in ``_join_filter_bucket_shard`` and
GCS write-bandwidth issues at higher fan-out.

Source mix:

* ``cp/foodista`` -- tiny baseline (already validated).
* ``coderforge`` -- small/medium code corpus.
* ``nemotron_cc_v2_1/high_quality`` -- 25B, medium nemotron.
* ``nemotron_specialized/rqa`` -- 135B, biggish synth.
* ``nemotron_cc_v2/high_quality`` -- 608B, BIG.

Submit on iris (eu-west4) -- bump driver memory (default 1G OOMs while
building shard_specs over ~10K+ nemotron shards)::

    uv run iris --cluster=marin job run --region europe-west4 --extra=cpu \\
        --priority interactive --cpu 2 --memory 8GB --enable-extra-resources \\
        -- python experiments/datakit/store/smoke_test_mixed.py
"""

import logging

from fray import ResourceConfig
from marin.datakit.decon import DeconAttributes
from marin.execution.artifact import Artifact
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from marin.processing.tokenize.attributes import TokenizedAttrData
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.v0.assign import AssignmentAttrData
from experiments.datakit.store.all_sources_store import (
    CLUSTER_ASSIGN_ROOT,
    DECONTAM_ROOT,
    DEDUP_PATH,
    TOKENIZE_ROOT,
    _resolve_artifact_dir,
)
from experiments.datakit.store.datakit_store import build_clustered_store

logger = logging.getLogger(__name__)


SMOKE_SOURCES = (
    "cp/foodista",  # 0.02B -- tiny baseline
    "coderforge",  # 10B -- small/medium code
    "nemotron_cc_v2_1/high_quality",  # 25B -- medium nemotron
    "nemotron_specialized/rqa",  # 135B -- biggish synth
    "nemotron_cc_v2/high_quality",  # 608B -- BIG
)
OUTPUT_PATH = "gs://marin-eu-west4/datakit/store/_smoke_v0_mixed"

CLUSTER_VIEW = 40
SPLIT = "train"

# 32g to absorb in-memory per-cluster buckets on big nemotron shards
# (each cluster_40 partition can buffer up to ~shard-sized input_ids).
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="32g", disk="10g")
MAX_WORKERS = 1024


def main() -> None:
    configure_logging(logging.INFO)

    dedup = Artifact.from_path(DEDUP_PATH, FuzzyDupsAttrData)

    tokenize: dict[str, TokenizedAttrData] = {}
    decontam: dict[str, DeconAttributes] = {}
    cluster_assign: dict[str, AssignmentAttrData] = {}

    for source_name in SMOKE_SOURCES:
        tokenize[source_name] = Artifact.from_path(_resolve_artifact_dir(TOKENIZE_ROOT, source_name), TokenizedAttrData)
        decontam[source_name] = Artifact.from_path(_resolve_artifact_dir(DECONTAM_ROOT, source_name), DeconAttributes)
        cluster_assign[source_name] = Artifact.from_path(
            _resolve_artifact_dir(CLUSTER_ASSIGN_ROOT, source_name), AssignmentAttrData
        )

    logger.info("mixed-smoke: %d sources -> %s", len(tokenize), OUTPUT_PATH)

    artifact = build_clustered_store(
        tokenize=tokenize,
        decontam=decontam,
        cluster_assign=cluster_assign,
        dedup=dedup,
        output_path=OUTPUT_PATH,
        cluster_view=CLUSTER_VIEW,
        split=SPLIT,
        worker_resources=WORKER_RESOURCES,
        max_workers=MAX_WORKERS,
    )

    logger.info(
        "mixed-smoke done: %d clusters, %d total docs, %d total tokens -> %s",
        len(artifact.clusters),
        sum(c.total_elements for c in artifact.clusters.values()),
        sum(c.total_tokens for c in artifact.clusters.values()),
        artifact.cache_path,
    )


if __name__ == "__main__":
    main()
